from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v99 as v99
from mcp_gateway import calibration_v4

MODEL_VERSION = v99.MODEL_VERSION
AUTOMATION_VERSION = "4.9.0-v4.013"


def _annotate_v4_013(payload: dict[str, Any]) -> None:
    dataset_meta = payload.get("v4_008_reproducible_training_dataset")
    if not isinstance(dataset_meta, dict):
        dataset_meta = {}
    current_rows = int(dataset_meta.get("current_tick_valid_feature_snapshots") or 0)

    payload["v4_013_calibration_engine_v1"] = {
        "schema_version": "1.0.0",
        "status": "RESEARCH_FRAMEWORK_IMPLEMENTED",
        "model_version": calibration_v4.MODEL_VERSION,
        "supported_targets": list(calibration_v4.SUPPORTED_TARGETS),
        "method": "PLATT_LOGIT",
        "metrics": ["brier", "log_loss", "ece", "mce", "reliability_bins"],
        "minimum_oos_rows": calibration_v4.MIN_OOS_ROWS,
        "minimum_positives": calibration_v4.MIN_POSITIVES,
        "minimum_negatives": calibration_v4.MIN_NEGATIVES,
        "same_fixture_oos_prediction_ledger_required": True,
        "current_tick_feature_snapshot_count": current_rows,
        "market_fields_used": False,
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
        "runtime_prediction_weight": 0.0,
        "model_weights_changed": False,
        "canonical_bet_logic_changed": False,
        "note": (
            "Calibration fitting is research-only until model probabilities and final outcomes "
            "exist on the same point-in-time OOS fixture cohort. No in-sample calibration is "
            "treated as production evidence."
        ),
    }
    payload["v4_013_provider_requests_added"] = 0
    payload["v4_013_model_weights_changed"] = False
    payload["v4_013_canonical_bet_logic_changed"] = False
    payload["v4_013_checkpoint"] = (
        "CALIBRATION ENGINE v1 IMPLEMENTED: deterministic Platt-logit research calibrator "
        "plus Brier, Log Loss, ECE, MCE and reliability bins, with OOS/sample gates. "
        "No market input, provider calls, runtime model weight, thresholds, tiers, stakes "
        "or canonical BET logic changed."
    )


async def run_tick() -> dict[str, Any]:
    payload = await v99.run_tick()
    _annotate_v4_013(payload)
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
