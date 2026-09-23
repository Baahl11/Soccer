from __future__ import annotations

import math
from typing import Any, Iterable

MODEL_VERSION = "SOCCER_CALIBRATION_V4_1.0.0"
MIN_OOS_ROWS = 200
MIN_POSITIVES = 25
MIN_NEGATIVES = 25
DEFAULT_BINS = 10
EPS = 1e-6
SUPPORTED_TARGETS = (
    "home_win",
    "draw",
    "away_win",
    "btts",
    "over_1_5",
    "over_2_5",
    "over_3_5",
)


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _clip_probability(value: float) -> float:
    return min(max(float(value), EPS), 1.0 - EPS)


def _logit(probability: float) -> float:
    p = _clip_probability(probability)
    return math.log(p / (1.0 - p))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def normalize_observations(observations: Iterable[dict[str, Any]]) -> tuple[list[tuple[float, int]], list[str]]:
    rows: list[tuple[float, int]] = []
    errors: list[str] = []
    for index, row in enumerate(observations):
        if not isinstance(row, dict):
            errors.append(f"ROW_{index}_NOT_OBJECT")
            continue
        probability = _num(row.get("probability"))
        outcome = row.get("outcome")
        try:
            outcome_int = int(outcome)
        except (TypeError, ValueError):
            outcome_int = -1
        if probability is None or probability < 0.0 or probability > 1.0:
            errors.append(f"ROW_{index}_INVALID_PROBABILITY")
            continue
        if outcome_int not in (0, 1):
            errors.append(f"ROW_{index}_INVALID_OUTCOME")
            continue
        rows.append((_clip_probability(probability), outcome_int))
    return rows, errors


def readiness(
    observations: Iterable[dict[str, Any]],
    *,
    min_oos_rows: int = MIN_OOS_ROWS,
    min_positives: int = MIN_POSITIVES,
    min_negatives: int = MIN_NEGATIVES,
) -> dict[str, Any]:
    rows, errors = normalize_observations(observations)
    positives = sum(outcome for _, outcome in rows)
    negatives = len(rows) - positives
    blockers: list[str] = []
    if errors:
        blockers.append("INVALID_OBSERVATIONS")
    if len(rows) < int(min_oos_rows):
        blockers.append(f"OOS_ROWS_{len(rows)}_LT_{int(min_oos_rows)}")
    if positives < int(min_positives):
        blockers.append(f"POSITIVES_{positives}_LT_{int(min_positives)}")
    if negatives < int(min_negatives):
        blockers.append(f"NEGATIVES_{negatives}_LT_{int(min_negatives)}")
    return {
        "ready": not blockers,
        "row_count": len(rows),
        "positive_count": positives,
        "negative_count": negatives,
        "minimum_oos_rows": int(min_oos_rows),
        "minimum_positives": int(min_positives),
        "minimum_negatives": int(min_negatives),
        "blockers": blockers,
        "validation_errors": errors[:20],
    }


def reliability_metrics(
    observations: Iterable[dict[str, Any]],
    *,
    bins: int = DEFAULT_BINS,
) -> dict[str, Any]:
    rows, errors = normalize_observations(observations)
    if errors or not rows:
        return {
            "status": "INVALID_OR_EMPTY",
            "row_count": len(rows),
            "validation_errors": errors[:20],
            "brier": None,
            "log_loss": None,
            "ece": None,
            "mce": None,
            "bins": [],
        }

    bin_count = max(int(bins), 2)
    brier = sum((p - y) ** 2 for p, y in rows) / len(rows)
    log_loss = -sum(y * math.log(p) + (1 - y) * math.log(1.0 - p) for p, y in rows) / len(rows)

    bucket_rows: list[list[tuple[float, int]]] = [[] for _ in range(bin_count)]
    for p, y in rows:
        index = min(int(p * bin_count), bin_count - 1)
        bucket_rows[index].append((p, y))

    reliability: list[dict[str, Any]] = []
    weighted_gap = 0.0
    max_gap = 0.0
    for index, bucket in enumerate(bucket_rows):
        if not bucket:
            continue
        mean_prediction = sum(p for p, _ in bucket) / len(bucket)
        observed_rate = sum(y for _, y in bucket) / len(bucket)
        gap = abs(mean_prediction - observed_rate)
        weighted_gap += gap * len(bucket)
        max_gap = max(max_gap, gap)
        reliability.append({
            "bin_index": index,
            "lower": round(index / bin_count, 6),
            "upper": round((index + 1) / bin_count, 6),
            "n": len(bucket),
            "mean_prediction": round(mean_prediction, 8),
            "observed_rate": round(observed_rate, 8),
            "absolute_gap": round(gap, 8),
        })

    return {
        "status": "OK",
        "row_count": len(rows),
        "brier": round(brier, 8),
        "log_loss": round(log_loss, 8),
        "ece": round(weighted_gap / len(rows), 8),
        "mce": round(max_gap, 8),
        "bins": reliability,
        "validation_errors": [],
    }


def fit_platt(
    observations: Iterable[dict[str, Any]],
    *,
    min_oos_rows: int = MIN_OOS_ROWS,
    min_positives: int = MIN_POSITIVES,
    min_negatives: int = MIN_NEGATIVES,
    iterations: int = 600,
    learning_rate: float = 0.05,
    l2: float = 0.001,
) -> dict[str, Any]:
    source = list(observations)
    gate = readiness(
        source,
        min_oos_rows=min_oos_rows,
        min_positives=min_positives,
        min_negatives=min_negatives,
    )
    if not gate["ready"]:
        return {
            "schema_version": "1.0.0",
            "model_version": MODEL_VERSION,
            "status": "INSUFFICIENT_OOS_FOR_CALIBRATION",
            "readiness": gate,
            "method": "PLATT_LOGIT",
            "parameters": None,
            "production_promotion_allowed": False,
        }

    rows, _ = normalize_observations(source)
    intercept = 0.0
    slope = 1.0
    n = float(len(rows))
    for _ in range(max(int(iterations), 1)):
        grad_intercept = 0.0
        grad_slope = 0.0
        for p, y in rows:
            x = _logit(p)
            fitted = _sigmoid(intercept + slope * x)
            error = fitted - y
            grad_intercept += error
            grad_slope += error * x
        grad_intercept = grad_intercept / n + l2 * intercept
        grad_slope = grad_slope / n + l2 * (slope - 1.0)
        intercept -= learning_rate * grad_intercept
        slope -= learning_rate * grad_slope

    return {
        "schema_version": "1.0.0",
        "model_version": MODEL_VERSION,
        "status": "RESEARCH_CALIBRATOR_FITTED",
        "readiness": gate,
        "method": "PLATT_LOGIT",
        "parameters": {
            "intercept": round(intercept, 10),
            "slope": round(slope, 10),
        },
        "iterations": max(int(iterations), 1),
        "learning_rate": learning_rate,
        "l2": l2,
        "market_fields_used": False,
        "production_promotion_allowed": False,
        "runtime_prediction_weight": 0.0,
    }


def calibrate_probability(probability: Any, calibration: dict[str, Any]) -> float | None:
    p = _num(probability)
    if p is None or p < 0.0 or p > 1.0:
        return None
    params = calibration.get("parameters") if isinstance(calibration, dict) else None
    if not isinstance(params, dict):
        return None
    intercept = _num(params.get("intercept"))
    slope = _num(params.get("slope"))
    if intercept is None or slope is None:
        return None
    return round(_sigmoid(intercept + slope * _logit(p)), 8)


def calibration_report(
    observations: Iterable[dict[str, Any]],
    *,
    bins: int = DEFAULT_BINS,
    min_oos_rows: int = MIN_OOS_ROWS,
) -> dict[str, Any]:
    source = list(observations)
    raw = reliability_metrics(source, bins=bins)
    fitted = fit_platt(source, min_oos_rows=min_oos_rows)
    if fitted.get("status") != "RESEARCH_CALIBRATOR_FITTED":
        return {
            "schema_version": "1.0.0",
            "model_version": MODEL_VERSION,
            "status": "DATA_BLOCKED",
            "raw_metrics": raw,
            "calibrator": fitted,
            "calibrated_metrics": None,
            "production_promotion_allowed": False,
        }

    calibrated_rows = [
        {
            "probability": calibrate_probability(row.get("probability"), fitted),
            "outcome": row.get("outcome"),
        }
        for row in source
    ]
    calibrated = reliability_metrics(calibrated_rows, bins=bins)
    return {
        "schema_version": "1.0.0",
        "model_version": MODEL_VERSION,
        "status": "RESEARCH_CALIBRATION_AVAILABLE",
        "raw_metrics": raw,
        "calibrator": fitted,
        "calibrated_metrics": calibrated,
        "brier_delta": (
            round(float(calibrated["brier"]) - float(raw["brier"]), 8)
            if raw.get("brier") is not None and calibrated.get("brier") is not None
            else None
        ),
        "log_loss_delta": (
            round(float(calibrated["log_loss"]) - float(raw["log_loss"]), 8)
            if raw.get("log_loss") is not None and calibrated.get("log_loss") is not None
            else None
        ),
        "market_fields_used": False,
        "production_promotion_allowed": False,
        "runtime_prediction_weight": 0.0,
    }
