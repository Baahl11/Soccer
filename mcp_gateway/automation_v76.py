from __future__ import annotations
from typing import Any
from mcp_gateway import automation_v75 as v75
from mcp_gateway import congestion_intelligence

MODEL_VERSION = v75.MODEL_VERSION
AUTOMATION_VERSION = "3.52.0"

async def run_tick() -> dict[str, Any]:
    payload = await v75.run_tick()
    metrics = congestion_intelligence.attach(payload)
    payload["congestion_intelligence"] = {
        "schema_version": congestion_intelligence.SCHEMA_VERSION,
        **metrics,
        "provider_requests_added": 0,
        "decision_weight": 0.0,
        "canonical_model_weights_changed": False,
        "canonical_bet_logic_changed": False,
        "production_status": "CONTEXT_ONLY_NOT_ACTIONABLE",
    }
    payload["v352_provider_requests_added"] = 0
    payload["v352_model_weights_changed"] = False
    payload["v352_canonical_bet_logic_changed"] = False
    payload["v352_congestion_checkpoint"] = "VERIFIED_7_14_21_DAY_FIXTURE_DENSITY_LIVE; MINUTES_ROTATION_PENDING"
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
