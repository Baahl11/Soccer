from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v54 as v54
from mcp_gateway import formation_live_intelligence

MODEL_VERSION = v54.MODEL_VERSION
AUTOMATION_VERSION = "3.31.0"


async def run_tick() -> dict[str, Any]:
    payload = await v54.run_tick()
    metrics = formation_live_intelligence.attach(payload)
    payload["formation_live_intelligence"] = {
        "schema_version": formation_live_intelligence.SCHEMA_VERSION,
        **metrics,
        "provider_requests_added": 0,
        "canonical_model_weights_changed": False,
        "canonical_thresholds_changed": False,
        "canonical_bet_logic_changed": False,
        "production_status": "LIVE_RESEARCH_FEATURE_ONLY",
        "decision_weight": 0.0,
        "policy": "CONFIRMED_XI_FORMATION_PAIR + HISTORICAL_MATCHUP + OOS_RESIDUAL_GATE; DESCRIPTIVE CORRELATION NEVER UPGRADES BET",
    }
    payload["v331_provider_requests_added"] = 0
    payload["v331_model_weights_changed"] = False
    payload["v331_canonical_bet_logic_changed"] = False
    payload["v331_formations_checkpoint"] = "LIVE_CONFIRMED_FORMATION_MATCHUP_RESEARCH_ADDED; PRODUCTION_UNCHANGED"
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
