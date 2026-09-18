from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v70 as v70
from mcp_gateway import box_entries_intelligence

MODEL_VERSION = v70.MODEL_VERSION
AUTOMATION_VERSION = "3.47.0"


async def run_tick() -> dict[str, Any]:
    payload = await v70.run_tick()
    metrics = box_entries_intelligence.attach(payload)
    payload["box_entries_intelligence"] = {
        "schema_version": box_entries_intelligence.SCHEMA_VERSION,
        **metrics,
        "provider_requests_added": 0,
        "canonical_model_weights_changed": False,
        "canonical_thresholds_changed": False,
        "canonical_bet_logic_changed": False,
        "production_status": "DORMANT_APPROVED_SOURCE_DATA_BLOCKED",
        "decision_weight": 0.0,
        "policy": (
            "API-FOOTBALL + GALAXYPARLAY ONLY; EXPLICIT BOX-ENTRY METRIC/EVENT REQUIRED; "
            "SHOTS INSIDE BOX/CROSSES/CORNERS MUST NEVER BE RELABELED AS BOX ENTRIES"
        ),
    }
    payload["v347_provider_requests_added"] = 0
    payload["v347_model_weights_changed"] = False
    payload["v347_canonical_bet_logic_changed"] = False
    payload["v347_box_entries_checkpoint"] = (
        "BOX_ENTRIES_GUARD_BUILT; APPROVED_SOURCES_DO_NOT_EXPOSE_EXPLICIT_BOX_ENTRIES"
    )
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
