import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from models import IssuesPayload
from vision import analyze
from rag_qdrant import search_snippets, ingest_seed_json
from api_live import get_live_facts
from plan_engine import compose_plan
from llm_writer import generate_answer
from pdf_report import build_pdf, simple_html

load_dotenv()

app = FastAPI(title="Retrofit Chatbot")

# serve static UI
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home():
    with open("static/index.html","r",encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.post("/chatbot")
async def chatbot(
    message: str = Form(""),
    rgb: UploadFile = File(...),
    thermal: UploadFile = File(...),
    address: str = Form(""),
    postcode: str = Form(""),
    inside: bool = Form(True),
    t_in: float = Form(20.0),
    t_out: float = Form(8.0),
    side: str = Form("front")
):
    # 1) Vision
    rgb_bytes = await rgb.read()
    thermal_bytes = await thermal.read()
    elements, findings, cap = analyze(rgb_bytes, thermal_bytes, inside, t_in, t_out)
    for e in elements: e.side = side if side in ("front","rear") else "unknown"
    issues = IssuesPayload(elements=elements, findings=findings, capture=cap)

    # 2) RAG
    try:
        snippets = search_snippets(message or "front window draught conservation Part F")
    except Exception:
        # first run bootstrap
        ingest_seed_json()
        snippets = search_snippets(message or "front window draught conservation Part F")

    # 3) APIs
    live = get_live_facts(address or None, postcode or None)

    # 4) Plan
    plan = compose_plan(issues, snippets, live)

    # 5) LLM summary
    text = generate_answer(issues, snippets, plan)

    # 6) PDF
    os.makedirs("out", exist_ok=True)
    pdf_path = "out/plan.pdf"
    build_pdf(simple_html(text), pdf_path)

    return JSONResponse({"reply": text, "plan": plan.model_dump(), "pdf": "/static/../"+pdf_path})

@app.get("/download-pdf")
def download_pdf():
    path = "out/plan.pdf"
    if os.path.exists(path):
        return FileResponse(path, media_type="application/pdf", filename="plan.pdf")
    return JSONResponse({"error":"PDF not found"}, status_code=404)
