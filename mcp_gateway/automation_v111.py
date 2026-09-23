from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v110 as v110
from mcp_gateway import player_props_phase15_v4

MODEL_VERSION = v110.MODEL_VERSION
AUTOMATION_VERSION = "4.20.0-phase15.player_props"


def _annotate_phase15(payload: dict[str, Any]) -> None:
    payload["phase15_player_props_validation"] = {
        "schema_version": "1.0.0",
        "status": "VALIDATION_GATE_IMPLEMENTED",
        "model_version": player_props_phase15_v4.MODEL_VERSION,
        "prop_families": list(player_props_phase15_v4.PROP_KEYS),
        "minimum_prop_true_clv_rows": player_props_phase15_v4.MIN_PROP_TRUE_CLV,
        "minimum_gk_profiles": player_props_phase15_v4.MIN_GK_PROFILES,
        "confirmed_xi_required": True,
        "expected_minutes_required": True,
        "exact_observed_line_required": True,
        "prop_specific_calibration_required": True,
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
        "runtime_prediction_weight": 0.0,
        "model_weights_changed": False,
        "canonical_bet_logic_changed": False,
        "note": (
            "Phase 15 keeps Shots, SOT, Goalscorer, Assists, Player Cards and GK Saves "
            "research-only until OOS validation, confirmed XI/role/minutes, exact observed "
            "prop lines, prop-specific calibration and family-specific true CLV are materialized."
        ),
    }
    payload["phase15_provider_requests_added"] = 0
    payload["phase15_model_weights_changed"] = False
    payload["phase15_canonical_bet_logic_changed"] = False
    payload["phase15_checkpoint"] = (
        "PLAYER PROPS VALIDATION GATE IMPLEMENTED. Structural model sanity is not treated as "
        "performance evidence. Confirmed XI, expected minutes, exact observed line, OOS calibration "
        "and prop-family true CLV are mandatory. No production promotion or BET logic change."
    )


async def run_tick() -> dict[str, Any]:
    payload = await v110.run_tick()
    _annotate_phase15(payload)
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
