from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v102 as v102
from mcp_gateway import ft_totals_validation_v4

MODEL_VERSION = v102.MODEL_VERSION
AUTOMATION_VERSION = "4.12.0-v4.016"


def _annotate_v4_016(payload: dict[str, Any]) -> None:
    payload["v4_016_ft_totals_production_validation"] = {
        "schema_version": "1.0.0",
        "status": "VALIDATION_GATE_IMPLEMENTED",
        "model_version": ft_totals_validation_v4.MODEL_VERSION,
        "supported_lines": list(ft_totals_validation_v4.SUPPORTED_LINES),
        "asian_quarter_settlement_supported": True,
        "minimum_directional_settled": ft_totals_validation_v4.MIN_DIRECTIONAL_SETTLED,
        "minimum_actionable_review_settled": ft_totals_validation_v4.MIN_ACTIONABLE_REVIEW_SETTLED,
        "minimum_family_specific_true_clv_rows": ft_totals_validation_v4.MIN_TRUE_CLV_ROWS,
        "family_specific_true_clv_required": True,
        "v4_oos_calibration_required": True,
        "market_fields_used_for_validation_only": True,
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
        "runtime_prediction_weight": 0.0,
        "model_weights_changed": False,
        "canonical_bet_logic_changed": False,
        "note": (
            "V4-016 adds the FT totals production-validation gate and correct Asian quarter-line "
            "settlement. Promotion remains blocked until family-specific settled sample, V4 OOS "
            "calibration, exact-line coverage and FT-totals true CLV satisfy review gates."
        ),
    }
    payload["v4_016_provider_requests_added"] = 0
    payload["v4_016_model_weights_changed"] = False
    payload["v4_016_canonical_bet_logic_changed"] = False
    payload["v4_016_checkpoint"] = (
        "FT TOTALS PRODUCTION VALIDATION GATE IMPLEMENTED. Asian 2.0/2.25/2.5/2.75/3.0/3.25 "
        "settlement is explicit; family-specific true CLV and V4 OOS calibration are mandatory. "
        "No automatic promotion, model-weight, tier, stake or canonical BET change."
    )


async def run_tick() -> dict[str, Any]:
    payload = await v102.run_tick()
    _annotate_v4_016(payload)
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
