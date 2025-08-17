from pydantic import BaseModel
from typing import List, Optional, Literal, Tuple, Dict, Any

class Element(BaseModel):
    id: str
    type: Literal["window", "door", "chimney", "skylight", "roof_edge", "wall"]
    box: Tuple[int, int, int, int]  # x, y, w, h
    side: Optional[Literal["front", "rear", "unknown"]] = "unknown"

class Finding(BaseModel):
    element_id: str
    issue: Literal["draught", "thermal_bridge", "insulation_gap"]
    delta_c: float
    impact: Literal["low", "medium", "medium_large", "large"]
    confidence: Literal["low", "medium", "high"]
    reason: Optional[str] = None

class Capture(BaseModel):
    inside: Optional[bool] = None
    t_in: Optional[float] = None
    t_out: Optional[float] = None
    deltaT_env_c: Optional[float] = None

class IssuesPayload(BaseModel):
    elements: List[Element]
    findings: List[Finding]
    capture: Capture

class Snippet(BaseModel):
    id: str
    text: str
    source: str
    page: Optional[int] = None

class RAGResponse(BaseModel):
    snippets: List[Snippet]

class LiveFacts(BaseModel):
    epc: Optional[Dict[str, Any]] = None
    planning: Optional[Dict[str, Any]] = None
    grants: Optional[Dict[str, Any]] = None
    status: Literal["ok", "unavailable"] = "ok"
    sources: List[str] = []

class Action(BaseModel):
    name: str
    phase: int
    cost_gbp: Optional[Tuple[int, int]] = None
    cost_per_window_gbp: Optional[Tuple[int, int]] = None
    cost_per_m2_gbp: Optional[Tuple[int, int]] = None
    impact: Literal["low", "medium", "medium_large", "large"]
    confidence: Literal["low", "medium", "high"]
    reason: str
    citations: List[str] = []

class Plan(BaseModel):
    actions: List[Action]
