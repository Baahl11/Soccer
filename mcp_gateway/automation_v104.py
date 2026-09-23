from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v103 as v103
from mcp_gateway import one_x_two_validation_v4

MODEL_VERSION = v103.MODEL_VERSION
AUTOMATION_VERSION = "4.13.0-v4.017"


def _annotate_v4_017(payload: dict[str, Any]) -> None:
    payload["v4_017_1x2_calibration_validation"] = {
        "schema_version": "1.0.0",
        "status": "VALIDATION_GATE_IMPLEMENTED",
        "model_version": one_x_two_validation_v4.MODEL_VERSION,
        "minimum_calibration_sample": one_x_two_validation_v4.MIN_CALIBRATION_SAMPLE,
        "minimum_family_specific_true_clv_rows": one_x_two_validation_v4.MIN_TRUE_CLV_ROWS,
        "requires_multiclass_brier_improvement": True,
        "requires_multiclass_log_loss_improvement": True,
        "family_specific_true_clv_required": True,
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
        "runtime_prediction_weight": 0.0,
        "model_weights_changed": False,
        "canonical_bet_logic_changed": False,
        "note": (
            "V4-017 validates 1X2 calibration evidence only. A challenger must improve both "
            "multiclass Brier and log loss and retain family-specific true CLV support before "
            "manual production review. No runtime probability replacement occurs here."
        ),
    }
    payload["v4_017_provider_requests_added"] = 0
    payload["v4_017_model_weights_changed"] = False
    payload["v4_017_canonical_bet_logic_changed"] = False
    payload["v4_017_checkpoint"] = (
        "1X2 CALIBRATION VALIDATION GATE IMPLEMENTED. Calibration sample, multiclass Brier, "
        "log loss and family-specific true CLV are reviewed conservatively. No automatic "
        "promotion or canonical BET/model-weight change."
    )


async def run_tick() -> dict[str, Any]:
    payload = await v103.run_tick()
    _annotate_v4_017(payload)
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
