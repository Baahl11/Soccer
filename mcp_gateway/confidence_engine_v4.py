from __future__ import annotations

import math
from typing import Any

MODEL_VERSION = "SOCCER_CONFIDENCE_ENGINE_V4_1.0.0"
MIN_OOS_ROWS = 200
HIGH_CONFIDENCE_SCORE = 75.0
MODERATE_CONFIDENCE_SCORE = 55.0


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _clamp01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def score(
    *,
    oos_rows: Any,
    calibration_ece: Any,
    disagreement_max_range: Any,
    feature_missing_rate: Any,
) -> dict[str, Any]:
    try:
        rows = int(oos_rows)
    except (TypeError, ValueError):
        rows = 0
    ece = _num(calibration_ece)
    disagreement = _num(disagreement_max_range)
    missing = _num(feature_missing_rate)

    blockers: list[str] = []
    if rows < MIN_OOS_ROWS:
        blockers.append(f"OOS_ROWS_{rows}_LT_{MIN_OOS_ROWS}")
    if ece is None or ece < 0.0 or ece > 1.0:
        blockers.append("CALIBRATION_ECE_REQUIRED")
    if disagreement is None or disagreement < 0.0 or disagreement > 1.0:
        blockers.append("DISAGREEMENT_RANGE_REQUIRED")
    if missing is None or missing < 0.0 or missing > 1.0:
        blockers.append("FEATURE_MISSING_RATE_REQUIRED")

    if blockers:
        return {
            "schema_version": "1.0.0",
            "model_version": MODEL_VERSION,
            "status": "INSUFFICIENT_EVIDENCE",
            "confidence_score": None,
            "confidence_band": "INSUFFICIENT_DATA",
            "blockers": blockers,
            "components": None,
            "market_fields_used": False,
            "production_promotion_allowed": False,
            "runtime_prediction_weight": 0.0,
        }

    sample_component = _clamp01(rows / 1000.0)
    calibration_component = _clamp01(1.0 - (float(ece) / 0.15))
    disagreement_component = _clamp01(1.0 - (float(disagreement) / 0.20))
    completeness_component = _clamp01(1.0 - float(missing))

    weighted = (
        0.30 * sample_component
        + 0.30 * calibration_component
        + 0.25 * disagreement_component
        + 0.15 * completeness_component
    )
    confidence_score = round(weighted * 100.0, 3)

    if confidence_score >= HIGH_CONFIDENCE_SCORE:
        band = "HIGH"
    elif confidence_score >= MODERATE_CONFIDENCE_SCORE:
        band = "MODERATE"
    else:
        band = "LOW"

    return {
        "schema_version": "1.0.0",
        "model_version": MODEL_VERSION,
        "status": "RESEARCH_CONFIDENCE_AVAILABLE",
        "confidence_score": confidence_score,
        "confidence_band": band,
        "blockers": [],
        "components": {
            "sample_strength": round(sample_component, 6),
            "calibration_quality": round(calibration_component, 6),
            "model_agreement": round(disagreement_component, 6),
            "feature_completeness": round(completeness_component, 6),
        },
        "weights": {
            "sample_strength": 0.30,
            "calibration_quality": 0.30,
            "model_agreement": 0.25,
            "feature_completeness": 0.15,
        },
        "inputs": {
            "oos_rows": rows,
            "calibration_ece": round(float(ece), 8),
            "disagreement_max_range": round(float(disagreement), 8),
            "feature_missing_rate": round(float(missing), 8),
        },
        "market_fields_used": False,
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
        "runtime_prediction_weight": 0.0,
        "canonical_bet_logic_changed": False,
        "policy": (
            "RESEARCH_ONLY_CONFIDENCE_FROM_OOS_SAMPLE_CALIBRATION_MODEL_AGREEMENT_AND_FEATURE_COMPLETENESS;"
            "NO_ODDS_NO_PRICE_NO_EXECUTION_STATUS"
        ),
    }
