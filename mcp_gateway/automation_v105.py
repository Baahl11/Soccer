from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v104 as v104
from mcp_gateway import btts_validation_v4

MODEL_VERSION = v104.MODEL_VERSION
AUTOMATION_VERSION = "4.14.0-v4.018"


def _annotate_v4_018(payload: dict[str, Any]) -> None:
    payload["v4_018_btts_calibration_validation"] = {
        "schema_version": "1.0.0",
        "status": "VALIDATION_GATE_IMPLEMENTED",
        "model_version": btts_validation_v4.MODEL_VERSION,
        "minimum_calibration_sample": btts_validation_v4.MIN_CALIBRATION_SAMPLE,
        "minimum_family_specific_true_clv_rows": btts_validation_v4.MIN_TRUE_CLV_ROWS,
        "family_specific_true_clv_required": True,
        "calibration_metrics": ["brier", "log_loss", "ece", "mce"],
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
        "runtime_prediction_weight": 0.0,
        "model_weights_changed": False,
        "canonical_bet_logic_changed": False,
        "note": (
            "V4-018 validates BTTS calibration and family-specific true CLV only. "
            "Existing historical probability diagnostics remain research evidence until the "
            "promotion gate is explicitly satisfied and manually reviewed."
        ),
    }
    payload["v4_018_provider_requests_added"] = 0
    payload["v4_018_model_weights_changed"] = False
    payload["v4_018_canonical_bet_logic_changed"] = False
    payload["v4_018_checkpoint"] = (
        "BTTS CALIBRATION VALIDATION GATE IMPLEMENTED. Brier/log-loss/ECE/MCE and family-specific "
        "true CLV are evaluated independently. No automatic production promotion, tier, stake, "
        "model-weight or canonical BET change."
    )


async def run_tick() -> dict[str, Any]:
    payload = await v104.run_tick()
    _annotate_v4_018(payload)
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
