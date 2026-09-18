from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v67 as v67
from mcp_gateway import npxg_intelligence

MODEL_VERSION = v67.MODEL_VERSION
AUTOMATION_VERSION = "3.44.0"


async def run_tick() -> dict[str, Any]:
    payload = await v67.run_tick()
    metrics = npxg_intelligence.attach(payload)
    payload["npxg_npxga_intelligence"] = {
        "schema_version": npxg_intelligence.SCHEMA_VERSION,
        **metrics,
        "provider_requests_added": 0,
        "canonical_model_weights_changed": False,
        "canonical_thresholds_changed": False,
        "canonical_bet_logic_changed": False,
        "production_status": "DORMANT_APPROVED_SOURCE_DATA_BLOCKED",
        "decision_weight": 0.0,
        "policy": (
            "API-FOOTBALL + GALAXYPARLAY ONLY; NATIVE/VERIFIED npxG REQUIRED; "
            "NO FIXED PENALTY xG SUBTRACTION; NO GOAL-LAMBDA RELABEL"
        ),
    }
    payload["v344_provider_requests_added"] = 0
    payload["v344_model_weights_changed"] = False
    payload["v344_canonical_bet_logic_changed"] = False
    payload["v344_npxg_npxga_checkpoint"] = (
        "NPXG_NPXGA_GUARD_BUILT; API_FOOTBALL_AND_CURRENT_GALAXYPARLAY_CONTRACT_DO_NOT_EXPOSE_VERIFIED_NPXG"
    )
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
