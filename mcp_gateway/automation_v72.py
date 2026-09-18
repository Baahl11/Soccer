from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v71 as v71
from mcp_gateway import big_chances_intelligence

MODEL_VERSION = v71.MODEL_VERSION
AUTOMATION_VERSION = "3.48.0"


async def run_tick() -> dict[str, Any]:
    payload = await v71.run_tick()
    metrics = big_chances_intelligence.attach(payload)
    payload["big_chances_intelligence"] = {
        "schema_version": big_chances_intelligence.SCHEMA_VERSION,
        **metrics,
        "provider_requests_added": 0,
        "canonical_model_weights_changed": False,
        "canonical_thresholds_changed": False,
        "canonical_bet_logic_changed": False,
        "production_status": "PROVIDER_DEPENDENT_DATA_ACCUMULATING",
        "decision_weight": 0.0,
        "policy": (
            "API-FOOTBALL EXPLICIT BIG CHANCES FIELD ONLY WHEN PRESENT; "
            "NO SHOTS/xG/GOALS PROXY; ZERO WEIGHT UNTIL CONSISTENT HISTORY + OOS VALIDATION"
        ),
    }
    payload["v348_provider_requests_added"] = 0
    payload["v348_model_weights_changed"] = False
    payload["v348_canonical_bet_logic_changed"] = False
    payload["v348_big_chances_checkpoint"] = (
        "BIG_CHANCES_PROVIDER_DEPENDENT_CAPTURE_BUILT; CONSISTENT_HISTORY_REQUIRED"
    )
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
