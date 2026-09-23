from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v117 as v117
from mcp_gateway import live_engine_v4

MODEL_VERSION = v117.MODEL_VERSION
AUTOMATION_VERSION = "4.27.0-phase22.live_engine"


def _annotate_phase22(payload: dict[str, Any]) -> None:
    payload["phase22_live_inplay_engine"] = {
        "schema_version": "1.0.0",
        "status": "INPUT_FRAMEWORK_IMPLEMENTED_RESEARCH_ONLY",
        "model_version": live_engine_v4.MODEL_VERSION,
        "required_live_fields": list(live_engine_v4.REQUIRED_LIVE_FIELDS),
        "targets": list(live_engine_v4.LIVE_TARGETS),
        "verified_xg_only": True,
        "pregame_prior_required": True,
        "numeric_posterior_update_enabled": False,
        "numeric_posterior_blocker": "LIVE_MODEL_NOT_OOS_CALIBRATED",
        "production_action_enabled": False,
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
        "runtime_prediction_weight": 0.0,
        "model_weights_changed": False,
        "canonical_bet_logic_changed": False,
        "note": (
            "Phase 22 defines verified live score/minute/red-card/shots/SOT/corners/possession/game-state "
            "inputs and preserves the pregame prior. It deliberately does not invent a numeric posterior "
            "until a live model has OOS calibration; next-goal and live-corners remain later targets."
        ),
    }
    payload["phase22_provider_requests_added"] = 0
    payload["phase22_model_weights_changed"] = False
    payload["phase22_canonical_bet_logic_changed"] = False
    payload["phase22_checkpoint"] = (
        "LIVE/IN-PLAY INPUT FRAMEWORK IMPLEMENTED IN RESEARCH MODE. Verified live inputs and "
        "pregame prior are mandatory; no numeric posterior, live BET/LEAN, next-goal or live-corners "
        "promotion occurs before dedicated OOS calibration."
    )


async def run_tick() -> dict[str, Any]:
    payload = await v117.run_tick()
    _annotate_phase22(payload)
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
