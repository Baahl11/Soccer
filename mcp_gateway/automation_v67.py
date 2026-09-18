from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v66 as v66
from mcp_gateway import xg_intelligence

MODEL_VERSION = v66.MODEL_VERSION
AUTOMATION_VERSION = "3.43.2"


async def run_tick() -> dict[str, Any]:
    payload = await v66.run_tick()
    metrics = xg_intelligence.attach(payload)
    ready = int(metrics.get("live_research_events") or 0) > 0
    accumulating = int(metrics.get("data_accumulating_events") or 0) > 0
    payload["xg_xga_intelligence"] = {
        "schema_version": xg_intelligence.SCHEMA_VERSION,
        **metrics,
        "provider_requests_added": 0,
        "canonical_model_weights_changed": False,
        "canonical_thresholds_changed": False,
        "canonical_bet_logic_changed": False,
        "production_status": (
            "LIVE_RESEARCH_NOT_ACTIONABLE"
            if ready
            else "HISTORICAL_API_FOOTBALL_XG_ACCUMULATING"
            if accumulating
            else "DORMANT_WAITING_FOR_NATURAL_POSTGAME_XG"
        ),
        "decision_weight": 0.0,
        "policy": (
            "API-FOOTBALL /fixtures/statistics expected_goals FROM FINALIZED FIXTURES ONLY; "
            "SAME-FIXTURE REALIZED xG NEVER USED PREGAME; HISTORICAL xGF/xGA RESEARCH ONLY; "
            "INTERNAL GOAL LAMBDAS NEVER RELABELED xG"
        ),
    }
    payload["v343_provider_requests_added"] = 0
    payload["v343_model_weights_changed"] = False
    payload["v343_canonical_bet_logic_changed"] = False
    payload["v343_xg_xga_checkpoint"] = (
        "API_FOOTBALL_HISTORICAL_XG_LIVE_RESEARCH_ONLY"
        if ready
        else "API_FOOTBALL_POSTGAME_XG_CAPTURE_BUILT; HISTORICAL_REGISTRY_ACCUMULATING"
    )
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
