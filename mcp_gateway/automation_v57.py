from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v56 as v56
from mcp_gateway import xi_intelligence

MODEL_VERSION = v56.MODEL_VERSION
AUTOMATION_VERSION = "3.33.0"


async def run_tick() -> dict[str, Any]:
    payload = await v56.run_tick()
    metrics = xi_intelligence.attach(payload)
    payload["xi_intelligence"] = {
        "schema_version": xi_intelligence.SCHEMA_VERSION,
        **metrics,
        "provider_requests_added": 0,
        "canonical_model_weights_changed": False,
        "canonical_thresholds_changed": False,
        "canonical_bet_logic_changed": False,
        "availability_gate_changed": False,
        "policy": "PERSIST XI FINGERPRINTS AND FLAG MATERIAL CHANGES; DO NOT INVENT PLAYER IMPACT OR CHANGE CANONICAL PROBABILITY",
    }
    payload["v333_provider_requests_added"] = 0
    payload["v333_model_weights_changed"] = False
    payload["v333_canonical_bet_logic_changed"] = False
    payload["v333_xi_checkpoint"] = "XI_PERSISTENCE_AND_CHANGE_DETECTION_ADDED; EXISTING_AVAILABILITY_GATE_UNCHANGED"
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
