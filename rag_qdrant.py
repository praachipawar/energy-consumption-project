import os, uuid, json
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from models import Snippet, RAGResponse

COLL = "retrofit_kb"
EMB = "BAAI/bge-small-en-v1.5"
DIM = 384

_client = QdrantClient(url=os.getenv("QDRANT_URL","http://localhost:6333"), api_key=os.getenv("QDRANT_API_KEY") or None)
_model = SentenceTransformer(EMB)

def ensure_collection():
    cols = [c.name for c in _client.get_collections().collections]
    if COLL not in cols:
        _client.create_collection(COLL, vectors_config=VectorParams(size=DIM, distance=Distance.COSINE))

def ingest_seed_json(path="data/kb_seed.json"):
    ensure_collection()
    with open(path,"r") as f:
        pairs = json.load(f)  # [{"text":..., "source":..., "page":...}]
    vecs = _model.encode([p["text"] for p in pairs], normalize_embeddings=True).tolist()
    points=[]
    for v,p in zip(vecs, pairs):
        points.append(PointStruct(id=uuid.uuid4().hex, vector=v, payload=p))
    _client.upsert(collection_name=COLL, points=points)

def search_snippets(query: str, k=3) -> RAGResponse:
    ensure_collection()
    v = _model.encode([query], normalize_embeddings=True)[0].tolist()
    res = _client.search(collection_name=COLL, query_vector=v, limit=k)
    out=[]
    for i,hit in enumerate(res, start=1):
        pl = hit.payload
        out.append(Snippet(id=f"[{i}]", text=pl["text"], source=pl.get("source",""), page=pl.get("page")))
    return RAGResponse(snippets=out)
