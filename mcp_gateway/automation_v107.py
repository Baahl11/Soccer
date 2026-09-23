from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v106 as v106
from mcp_gateway import one_h_oos_v4

MODEL_VERSION = v106.MODEL_VERSION
AUTOMATION_VERSION = "4.16.0-v4.020"


def _annotate_v4_020(payload: dict[str, Any]) -> None:
    payload["v4_020_1h_oos_validation"] = {
        "schema_version": "1.0.0",
        "status": "VALIDATION_GATE_IMPLEMENTED",
        "model_version": one_h_oos_v4.MODEL_VERSION,
        "minimum_calibrated_oos": one_h_oos_v4.MIN_CALIBRATED_OOS,
        "minimum_family_specific_true_clv_rows": one_h_oos_v4.MIN_TRUE_CLV_ROWS,
        "required_lines": list(one_h_oos_v4.REQUIRED_LINES),
        "family_specific_true_clv_required": True,
        "asian_settlement_required": True,
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
        "runtime_prediction_weight": 0.0,
        "model_weights_changed": False,
        "canonical_bet_logic_changed": False,
        "note": (
            "V4-020 validates 1H OOS/calibration evidence and exact period-specific market support. "
            "The current persisted challenger is review-only and cannot replace runtime probabilities "
            "without better Brier/log-loss plus dedicated 1H market/CLV evidence."
        ),
    }
    payload["v4_020_provider_requests_added"] = 0
    payload["v4_020_model_weights_changed"] = False
    payload["v4_020_canonical_bet_logic_changed"] = False
    payload["v4_020_checkpoint"] = (
        "1H OOS VALIDATION GATE IMPLEMENTED. O0.5/O1.0/O1.25/O1.5/O1.75/O2.0 line coverage, "
        "calibration and dedicated 1H true CLV are required. No automatic promotion, tier, stake, "
        "model-weight or canonical BET change."
    )


async def run_tick() -> dict[str, Any]:
    payload = await v106.run_tick()
    _annotate_v4_020(payload)
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
