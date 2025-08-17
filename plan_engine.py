import json, os
from typing import List
from models import IssuesPayload, Action, Plan, RAGResponse, LiveFacts

COSTS_PATH = os.path.join("data","costs.json")
DEFAULT_COSTS = {
  "draughtproof_door":[150,300],
  "secondary_glazing_per_window":[500,900],
  "double_glazing_per_window":[500,900],
  "loft_insulation_per_m2":[18,35],
  "ashp":[9000,14000],
  "pv_3_4kw":[5000,8000]
}

def load_costs():
    if os.path.exists(COSTS_PATH):
        with open(COSTS_PATH,"r") as f: return json.load(f)
    return DEFAULT_COSTS

def compose_plan(issues: IssuesPayload, snippets: RAGResponse, live: LiveFacts) -> Plan:
    costs = load_costs()
    actions: List[Action] = []

    # Draughts
    if any(f.issue=="draught" for f in issues.findings):
        actions.append(Action(
            name="Draughtproof door & letterbox",
            phase=1,
            cost_gbp=tuple(costs["draughtproof_door"]),
            impact="medium_large",
            confidence=max([f.confidence for f in issues.findings if f.issue=="draught"], default="medium"),
            reason="Cold streaks detected near opening",
            citations=[s.id for s in snippets.snippets if "vent" in s.text.lower() or "draught" in s.text.lower()][:1]
        ))

    # Glazing
    front = any(e.side=="front" for e in issues.elements)
    cons = (live.planning or {}).get("conservation_area") is not None
    if any(f.issue in ("thermal_bridge","draught") for f in issues.findings):
        if front and cons:
            actions.append(Action(
                name="Secondary glazing (front) / Double (rear)",
                phase=2,
                cost_per_window_gbp=tuple(costs["secondary_glazing_per_window"]),
                impact="medium",
                confidence="medium",
                reason="Front in conservation area → secondary glazing; add trickle vents (Part F)",
                citations=[s.id for s in snippets.snippets][:2]
            ))
        else:
            actions.append(Action(
                name="Double glazing",
                phase=2,
                cost_per_window_gbp=tuple(costs["double_glazing_per_window"]),
                impact="medium",
                confidence="medium",
                reason="Reduce heat loss through glazing; add trickle vents (Part F)",
                citations=[s.id for s in snippets.snippets][:1]
            ))

    # Insulation
    if any(f.issue=="insulation_gap" for f in issues.findings):
        actions.append(Action(
            name="Top-up loft insulation (~270mm) & seal hatch",
            phase=2,
            cost_per_m2_gbp=tuple(costs["loft_insulation_per_m2"]),
            impact="large",
            confidence=max([f.confidence for f in issues.findings if f.issue=="insulation_gap"], default="medium"),
            reason="Large cold patch suggests missing insulation",
            citations=[s.id for s in snippets.snippets][:1]
        ))

    # Heating
    actions.append(Action(
        name="ASHP (after fabric upgrades)",
        phase=3,
        cost_gbp=tuple(costs["ashp"]),
        impact="medium",
        confidence="low",
        reason="Heat pump after tightening fabric; do heat-loss calc & noise siting",
        citations=[]
    ))

    # Solar
    actions.append(Action(
        name="Solar PV (3–4kW)",
        phase=4,
        cost_gbp=tuple(costs["pv_3_4kw"]),
        impact="medium",
        confidence="low",
        reason="Generate electricity once demand is reduced",
        citations=[]
    ))

    # Keep top 3 by impact, then earlier phase
    order = {"large":3, "medium_large":2, "medium":1, "low":0}
    actions = sorted(actions, key=lambda a: (order[a.impact], -a.phase), reverse=True)[:3]
    return Plan(actions=actions)
