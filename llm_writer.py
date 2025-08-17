import os, httpx
from models import Plan, RAGResponse, IssuesPayload

def render_prompt(issues: IssuesPayload, snippets: RAGResponse, plan: Plan) -> str:
    cites = "\n".join([f'{s.id} {s.source} p.{s.page}' if s.page else f'{s.id} {s.source}' for s in snippets.snippets])
    return f"""You are a retrofit advisor. Use ONLY the provided snippets; cite with [id].
Write ≤200 words: summary → actions (phase, cost band, impact, confidence, reason) → permissions/risks.
Findings: {issues.model_dump()}
Snippets:
{cites}
Plan: {plan.model_dump()}
If info is missing, say so briefly.
"""

def generate_answer(issues: IssuesPayload, snippets: RAGResponse, plan: Plan) -> str:
    prompt = render_prompt(issues, snippets, plan)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Fallback text if no LLM key
        lines = ["LLM not configured. Here are the recommended actions:"]
        for a in plan.actions:
            cost = a.cost_gbp or a.cost_per_window_gbp or a.cost_per_m2_gbp
            lines.append(f"- {a.name} (Phase {a.phase}) Cost: {cost} Impact: {a.impact} Confidence: {a.confidence}")
        return "\n".join(lines)
    body = {
        "model":"gpt-4o-mini",
        "messages":[
            {"role":"system","content":"You are a concise, trustworthy retrofit advisor."},
            {"role":"user","content": prompt}
        ],
        "temperature":0.2,
        "max_tokens":450
    }
    headers={"Authorization":f"Bearer {api_key}","Content-Type":"application/json"}
    try:
        r = httpx.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body, timeout=60)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception:
        return "LLM unavailable. " + "\n".join([f"- {a.name} (Phase {a.phase})" for a in plan.actions])
