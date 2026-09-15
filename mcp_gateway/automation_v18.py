from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v5 as v5
from mcp_gateway import automation_v17 as v17
from mcp_gateway import soccer_model_relative_strength as rs

MODEL_VERSION = "SOCCER EDGE ENGINE v1.7"
AUTOMATION_VERSION = "2.7.0"

_LEGACY_BUILD = v5.build_raw_projection
_LEGACY_PUBLIC = v5.public_raw_projection


def _shadow_build(fx: dict[str, Any], sporting: dict[str, Any], availability_confidence: float | None = None) -> dict[str, Any]:
    baseline = _LEGACY_BUILD(fx, sporting, availability_confidence)
    challenger = rs.build_raw_projection(fx, sporting, availability_confidence)
    out = dict(baseline) if isinstance(baseline, dict) else {}
    if not isinstance(challenger, dict) or challenger.get("relative_strength_status") != "RESEARCH_CHALLENGER_ACTIVE":
        out["relative_strength_shadow"] = {
            "status": (challenger or {}).get("relative_strength_status", "NOT_AVAILABLE") if isinstance(challenger, dict) else "NOT_AVAILABLE",
            "actionable": False,
        }
        return out

    keys = (
        "raw_home_goal_rate", "raw_away_goal_rate", "raw_total_goals",
        "raw_home_win_prob", "raw_draw_prob", "raw_away_win_prob",
        "raw_btts_yes_prob", "raw_over_1_5_prob", "raw_over_2_5_prob", "raw_over_3_5_prob",
    )
    out["relative_strength_shadow"] = {
        "status": "RESEARCH_ONLY_SHADOW",
        "actionable": False,
        "projection_model": challenger.get("projection_model"),
        "baseline_source": challenger.get("relative_strength_baseline_source"),
        "sample": challenger.get("sample"),
        "strengths": challenger.get("strengths"),
        "challenger": {k: challenger.get(k) for k in keys},
        "baseline": {k: baseline.get(k) for k in keys} if isinstance(baseline, dict) else {},
        "policy": "OBSERVE_ONLY; DOES_NOT_CHANGE_SHORTLIST_MARKET_DECISION_CLASSIFICATION_OR_STAKE",
    }
    return out


async def run_tick() -> dict[str, Any]:
    previous_build = v5.build_raw_projection
    previous_public = v5.public_raw_projection
    v5.build_raw_projection = _shadow_build
    # Legacy public projection strips private distributions but otherwise retains
    # the shadow diagnostic object. No market evaluator sees challenger values.
    v5.public_raw_projection = _LEGACY_PUBLIC
    try:
        payload = await v17.run_tick()
    finally:
        v5.build_raw_projection = previous_build
        v5.public_raw_projection = previous_public

    shadow_count = 0
    for event in payload.get("events") or []:
        raw = event.get("raw_projection") if isinstance(event, dict) else None
        if isinstance(raw, dict) and isinstance(raw.get("relative_strength_shadow"), dict):
            if raw["relative_strength_shadow"].get("status") == "RESEARCH_ONLY_SHADOW":
                shadow_count += 1

    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    payload["relative_strength_1x2_shadow_count_this_tick"] = shadow_count
    payload["relative_strength_1x2_policy"] = (
        "RESEARCH_ONLY_SHADOW; LEGACY_OR_VERIFIED_GALAXY_PRODUCTION_PATH_UNCHANGED; NO_BET_UPGRADE"
    )
    return payload
