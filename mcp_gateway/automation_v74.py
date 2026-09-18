from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v73 as v73
from mcp_gateway import tactical_style_intelligence

MODEL_VERSION = v73.MODEL_VERSION
AUTOMATION_VERSION = "3.50.0"


async def run_tick() -> dict[str, Any]:
    payload = await v73.run_tick()
    metrics = tactical_style_intelligence.attach(payload)
    payload["tactical_style_intelligence"] = {
        "schema_version": tactical_style_intelligence.SCHEMA_VERSION,
        **metrics,
        "provider_requests_added": 0,
        "canonical_model_weights_changed": False,
        "canonical_thresholds_changed": False,
        "canonical_bet_logic_changed": False,
        "decision_weight": 0.0,
        "production_status": "CONTEXT_ONLY_NOT_ACTIONABLE",
        "policy": (
            "FORMATION/POSSESSION/SHOT/SOT/CORNER/DISCIPLINE DESCRIPTORS ONLY; "
            "PRESS/BLOCK/TRANSITION/WIDTH REMAIN NOT_VERIFIED"
        ),
    }
    payload["v350_provider_requests_added"] = 0
    payload["v350_model_weights_changed"] = False
    payload["v350_canonical_bet_logic_changed"] = False
    payload["v350_tactical_style_checkpoint"] = (
        "DEFENSIBLE_TACTICAL_DESCRIPTORS_LIVE; SPATIAL/PRESSING CLAIMS BLOCKED"
    )
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
