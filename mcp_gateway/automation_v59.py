from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v58 as v58
from mcp_gateway import injury_intelligence

MODEL_VERSION = v58.MODEL_VERSION
AUTOMATION_VERSION = "3.35.0"


async def run_tick() -> dict[str, Any]:
    payload = await v58.run_tick()
    metrics = injury_intelligence.attach(payload)
    payload["injury_intelligence"] = {
        "schema_version": injury_intelligence.SCHEMA_VERSION,
        **metrics,
        "provider_requests_added": 0,
        "canonical_model_weights_changed": False,
        "canonical_thresholds_changed": False,
        "canonical_bet_logic_changed": False,
        "canonical_availability_confidence_changed": False,
        "production_status": "LIVE_RESEARCH_CONTEXT_PLUS_RECHECK_FLAGS",
        "decision_weight": 0.0,
        "policy": "PROVIDER REPORTS + HISTORICAL ROLE EXPOSURE; NO REPLACEMENT QUALITY OR GOAL IMPACT INVENTED; XI CONFLICT REQUIRES RECHECK",
    }
    payload["v335_provider_requests_added"] = 0
    payload["v335_model_weights_changed"] = False
    payload["v335_canonical_bet_logic_changed"] = False
    payload["v335_injuries_checkpoint"] = "ROLE_EXPOSURE_AND_CONFLICT_CONTEXT_ADDED; PLAYER_IMPACT_MODEL_NOT_CLAIMED"
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
