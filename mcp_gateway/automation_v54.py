from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v53 as v53
from mcp_gateway import referee_intelligence

MODEL_VERSION = v53.MODEL_VERSION
AUTOMATION_VERSION = "3.30.0"


async def run_tick() -> dict[str, Any]:
    payload = await v53.run_tick()
    metrics = referee_intelligence.attach(payload)
    payload["referee_intelligence"] = {
        "schema_version": referee_intelligence.SCHEMA_VERSION,
        **metrics,
        "provider_requests_added": 0,
        "canonical_model_weights_changed": False,
        "canonical_thresholds_changed": False,
        "canonical_bet_logic_changed": False,
        "standalone_bet_signal_allowed": False,
        "production_status": "LIVE_RESEARCH_FEATURE_ONLY",
        "policy": "REFEREE PROFILE CONDITIONS CARDS MODELS ONLY WHEN SAMPLE GATES PASS; NEVER A STANDALONE PICK",
    }
    payload["v330_provider_requests_added"] = 0
    payload["v330_model_weights_changed"] = False
    payload["v330_canonical_bet_logic_changed"] = False
    payload["v330_referee_checkpoint"] = "LIVE_RESEARCH_REFEREE_PROFILE_ADDED; PRODUCTION_UNCHANGED"
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
