import os, json
import redis
from typing import Optional, Dict, Any
from models import LiveFacts

r = redis.from_url(os.getenv("REDIS_URL","redis://localhost:6379/0"))

def _fake_epc(address: Optional[str], postcode: Optional[str]) -> Dict[str, Any]:
    if not (address or postcode): return {}
    return {"rating":"E","rrn":"1234-5678-0000-0000-0000"}

def _fake_planning(address: Optional[str], postcode: Optional[str]) -> Dict[str, Any]:
    return {"conservation_area":"Moore Park"}

def _fake_grants(postcode: Optional[str]) -> Dict[str, Any]:
    return {"BUS":{"eligible": True, "amount_gbp": 7500}}

def get_live_facts(address: Optional[str], postcode: Optional[str]) -> LiveFacts:
    key = f"livefacts:{address}:{postcode}"
    cached = r.get(key)
    if cached:
        return LiveFacts(**json.loads(cached))
    try:
        epc = _fake_epc(address, postcode)
        planning = _fake_planning(address, postcode)
        grants = _fake_grants(postcode)
        data = LiveFacts(epc=epc or None, planning=planning or None, grants=grants or None,
                         status="ok", sources=["EPC","Council","GrantsAPI"])
        r.setex(key, 24*3600, json.dumps(data.model_dump()))
        return data
    except Exception:
        return LiveFacts(status="unavailable", sources=[])
