from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v69 as v69
from mcp_gateway import field_tilt_intelligence

MODEL_VERSION = v69.MODEL_VERSION
AUTOMATION_VERSION = "3.46.0"


async def run_tick() -> dict[str, Any]:
    payload = await v69.run_tick()
    metrics = field_tilt_intelligence.attach(payload)
    payload["field_tilt_intelligence"] = {
        "schema_version": field_tilt_intelligence.SCHEMA_VERSION,
        **metrics,
        "provider_requests_added": 0,
        "canonical_model_weights_changed": False,
        "canonical_thresholds_changed": False,
        "canonical_bet_logic_changed": False,
        "production_status": "DORMANT_APPROVED_SOURCE_DATA_BLOCKED",
        "decision_weight": 0.0,
        "policy": (
            "API-FOOTBALL + GALAXYPARLAY ONLY; AUDITABLE TERRITORIAL DEFINITION REQUIRED; "
            "TOTAL POSSESSION MUST NEVER BE RELABELED AS FIELD TILT"
        ),
    }
    payload["v346_provider_requests_added"] = 0
    payload["v346_model_weights_changed"] = False
    payload["v346_canonical_bet_logic_changed"] = False
    payload["v346_field_tilt_checkpoint"] = (
        "FIELD_TILT_GUARD_BUILT; APPROVED_SOURCES_DO_NOT_EXPOSE_REQUIRED_TERRITORIAL_INPUTS"
    )
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
