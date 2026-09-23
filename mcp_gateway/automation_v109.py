from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v108 as v108
from mcp_gateway import corners_oos_v4

MODEL_VERSION = v108.MODEL_VERSION
AUTOMATION_VERSION = "4.18.0-v4.022"


def _annotate_v4_022(payload: dict[str, Any]) -> None:
    payload["v4_022_corners_oos_validation"] = {
        "schema_version": "1.0.0",
        "status": "VALIDATION_GATE_IMPLEMENTED",
        "model_version": corners_oos_v4.MODEL_VERSION,
        "minimum_ft_oos": corners_oos_v4.MIN_FT_OOS,
        "minimum_formation_adjusted": corners_oos_v4.MIN_FORMATION_ADJUSTED,
        "minimum_team_rows": corners_oos_v4.MIN_TEAM_ROWS,
        "minimum_family_specific_true_clv_rows": corners_oos_v4.MIN_TRUE_CLV_ROWS,
        "ft_required_lines": list(corners_oos_v4.FT_LINES),
        "team_required_lines": list(corners_oos_v4.TEAM_LINES),
        "family_specific_true_clv_required": True,
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
        "runtime_prediction_weight": 0.0,
        "model_weights_changed": False,
        "canonical_bet_logic_changed": False,
        "note": (
            "V4-022 validates FT and Team Corners OOS separately. Formation-adjusted FT corners "
            "must sustain Brier/log-loss/MAE lift with adequate sample, while Team Corners require "
            "stable role/line evidence and dedicated corners true CLV before manual promotion review."
        ),
    }
    payload["v4_022_provider_requests_added"] = 0
    payload["v4_022_model_weights_changed"] = False
    payload["v4_022_canonical_bet_logic_changed"] = False
    payload["v4_022_checkpoint"] = (
        "CORNERS OOS VALIDATION GATE IMPLEMENTED. FT 8.5/9.5/10.5 plus Team 3.5/4.5/5.5 "
        "samples, formation lift and family-specific true CLV are reviewed conservatively. "
        "No automatic promotion, tier, stake, model-weight or canonical BET change."
    )


async def run_tick() -> dict[str, Any]:
    payload = await v108.run_tick()
    _annotate_v4_022(payload)
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
