from __future__ import annotations
from typing import Any

SCHEMA_VERSION = "1.0.0"

def _stage(round_name: Any) -> str:
    r=str(round_name or "").strip().lower()
    if not r: return "NOT_VERIFIED"
    if "semi" in r and "final" in r: return "SEMIFINAL"
    if "quarter" in r and "final" in r: return "QUARTERFINAL"
    if "round of 16" in r or "last 16" in r: return "ROUND_OF_16"
    if "final" in r: return "FINAL"
    if "playoff" in r or "play-off" in r: return "PLAYOFF"
    if "group" in r: return "GROUP_STAGE"
    if "regular" in r or "matchday" in r or "round" in r: return "LEAGUE_OR_REGULAR_ROUND"
    return "PROVIDER_ROUND_LABEL_ONLY"

def build(event: dict[str, Any]) -> dict[str, Any]:
    fixture=event.get("fixture") if isinstance(event.get("fixture"),dict) else {}
    round_name=fixture.get("round")
    descriptor=_stage(round_name)
    knockout=descriptor in {"SEMIFINAL","QUARTERFINAL","ROUND_OF_16","FINAL","PLAYOFF"}
    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": fixture.get("fixture_id"),
        "status": "LIVE_VERIFIED_COMPETITION_CONTEXT",
        "competition": fixture.get("league"),
        "country": fixture.get("country"),
        "season": fixture.get("season"),
        "provider_round": round_name,
        "stage_descriptor": descriptor,
        "knockout_label_from_round": knockout,
        "aggregate_score": "NOT_VERIFIED",
        "table_positions": "NOT_VERIFIED",
        "qualification_math": "NOT_MODELED",
        "objective_incentive": "NOT_INFERRED",
        "motivation": "NOT_INFERRED",
        "actionable": False,
        "decision_weight": 0.0,
        "policy": (
            "ROUND/STAGE LABELS MAY DESCRIBE COMPETITION STRUCTURE ONLY. "
            "NO MOTIVATION, MUST-WIN, ROTATION OR QUALIFICATION CLAIM WITHOUT VERIFIED TABLE/AGGREGATE DATA."
        ),
    }

def attach(payload: dict[str, Any]) -> dict[str,int]:
    attached=0
    for event in payload.get("events") or []:
        if not isinstance(event,dict) or event.get("event_type")!="SOCCER_REFRESH":
            continue
        intel=build(event); event["competition_context_intelligence"]=intel; attached+=1
        mi=event.get("match_intelligence")
        if isinstance(mi,dict) and isinstance(mi.get("areas"),dict): mi["areas"]["competition_context"]=intel
    return {"context_events":attached,"provider_requests_added":0}
