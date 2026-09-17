from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v55 as v55
from mcp_gateway import coach_intelligence

MODEL_VERSION = v55.MODEL_VERSION
AUTOMATION_VERSION = "3.32.0"


async def run_tick() -> dict[str, Any]:
    payload = await v55.run_tick()
    metrics = coach_intelligence.attach(payload)
    payload["coach_intelligence"] = {
        "schema_version": coach_intelligence.SCHEMA_VERSION,
        **metrics,
        "provider_requests_added": 0,
        "canonical_model_weights_changed": False,
        "canonical_thresholds_changed": False,
        "canonical_bet_logic_changed": False,
        "production_status": "LIVE_RESEARCH_CONTEXT_ONLY",
        "decision_weight": 0.0,
        "policy": "CONFIRMED_CURRENT_COACH + HISTORICAL_REGIME_CONTEXT; DESCRIPTIVE BEFORE/AFTER ONLY; NO CAUSAL BET UPGRADE",
    }
    payload["v332_provider_requests_added"] = 0
    payload["v332_model_weights_changed"] = False
    payload["v332_canonical_bet_logic_changed"] = False
    payload["v332_coach_checkpoint"] = "LIVE_COACH_REGIME_CONTEXT_ADDED; PRODUCTION_UNCHANGED"
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
