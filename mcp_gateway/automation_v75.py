from __future__ import annotations
from typing import Any
from mcp_gateway import automation_v74 as v74
from mcp_gateway import rest_intelligence

MODEL_VERSION = v74.MODEL_VERSION
AUTOMATION_VERSION = "3.51.0"

async def run_tick() -> dict[str, Any]:
    payload = await v74.run_tick()
    metrics = rest_intelligence.attach(payload)
    payload["rest_intelligence"] = {
        "schema_version": rest_intelligence.SCHEMA_VERSION,
        **metrics,
        "provider_requests_added": 0,
        "decision_weight": 0.0,
        "canonical_model_weights_changed": False,
        "canonical_bet_logic_changed": False,
        "production_status": "CONTEXT_ONLY_NOT_ACTIONABLE",
    }
    payload["v351_provider_requests_added"] = 0
    payload["v351_model_weights_changed"] = False
    payload["v351_canonical_bet_logic_changed"] = False
    payload["v351_rest_checkpoint"] = "VERIFIED_RECENT_FIXTURE_REST_DAYS_LIVE; ZERO_WEIGHT_PENDING_OOS"
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
