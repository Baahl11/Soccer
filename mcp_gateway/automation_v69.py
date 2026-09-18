from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v68 as v68
from mcp_gateway import ppda_intelligence

MODEL_VERSION = v68.MODEL_VERSION
AUTOMATION_VERSION = "3.45.0"


async def run_tick() -> dict[str, Any]:
    payload = await v68.run_tick()
    metrics = ppda_intelligence.attach(payload)
    payload["ppda_intelligence"] = {
        "schema_version": ppda_intelligence.SCHEMA_VERSION,
        **metrics,
        "provider_requests_added": 0,
        "canonical_model_weights_changed": False,
        "canonical_thresholds_changed": False,
        "canonical_bet_logic_changed": False,
        "production_status": "DORMANT_APPROVED_SOURCE_DATA_BLOCKED",
        "decision_weight": 0.0,
        "policy": (
            "API-FOOTBALL + GALAXYPARLAY ONLY; NORMALIZED PPDA DEFINITION REQUIRED; "
            "NO TOTAL-PASS/TACKLE PROXY RELABELED AS PPDA"
        ),
    }
    payload["v345_provider_requests_added"] = 0
    payload["v345_model_weights_changed"] = False
    payload["v345_canonical_bet_logic_changed"] = False
    payload["v345_ppda_checkpoint"] = (
        "PPDA_GUARD_BUILT; APPROVED_SOURCES_DO_NOT_EXPOSE_NORMALIZED_PPDA"
    )
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
