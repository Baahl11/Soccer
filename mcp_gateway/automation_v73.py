from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v72 as v72
from mcp_gateway import set_piece_intelligence

MODEL_VERSION = v72.MODEL_VERSION
AUTOMATION_VERSION = "3.49.0"


async def run_tick() -> dict[str, Any]:
    payload = await v72.run_tick()
    metrics = set_piece_intelligence.attach(payload)
    payload["set_piece_intelligence"] = {
        "schema_version": set_piece_intelligence.SCHEMA_VERSION,
        **metrics,
        "provider_requests_added": 0,
        "canonical_model_weights_changed": False,
        "canonical_thresholds_changed": False,
        "canonical_bet_logic_changed": False,
        "production_status": "CONTEXT_ONLY_NOT_ACTIONABLE",
        "decision_weight": 0.0,
        "policy": (
            "EXPLICIT API-FOOTBALL CORNER/FREE-KICK COMPONENTS ONLY; "
            "NO SET-PIECE GOAL/xG FABRICATION; OOS TARGET REQUIRED BEFORE ANY WEIGHT"
        ),
    }
    payload["v349_provider_requests_added"] = 0
    payload["v349_model_weights_changed"] = False
    payload["v349_canonical_bet_logic_changed"] = False
    payload["v349_set_piece_checkpoint"] = (
        "SET_PIECE_COMPONENT_CAPTURE_AND_LIVE_VOLUME_CONTEXT_BUILT; "
        "GOAL_CONVERSION_MODEL_BLOCKED_PENDING_VERIFIED_TARGET"
    )
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
