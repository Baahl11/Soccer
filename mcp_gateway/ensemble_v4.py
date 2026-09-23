from __future__ import annotations

import math
from typing import Any

MODEL_VERSION = "SOCCER_ENSEMBLE_V4_1.0.0"
MIN_COMPONENT_OOS = 100
REQUIRED_COMPONENTS = ("DIXON_COLES", "BIVARIATE_POISSON", "LIGHTGBM_GOALS")
PROBABILITY_KEYS = (
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


def _normalized_1x2(prediction: dict[str, Any]) -> dict[str, float] | None:
    values = {key: _num(prediction.get(key)) for key in ("home_win", "draw", "away_win")}
    if any(value is None or value < 0 for value in values.values()):
        return None
    total = sum(float(value) for value in values.values())
    if total <= 0:
        return None
    return {key: float(value) / total for key, value in values.items()}


def validate_component_prediction(prediction: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    one_x_two = _normalized_1x2(prediction)
    if one_x_two is None:
        errors.append("INVALID_1X2_PROBABILITIES")
    for key in ("btts", "over_1_5", "over_2_5", "over_3_5"):
        value = _num(prediction.get(key))
        if value is None or value < 0 or value > 1:
            errors.append(f"INVALID_{key.upper()}_PROBABILITY")
    return errors


def readiness(
    component_oos_n: dict[str, int],
    *,
    min_component_oos: int = MIN_COMPONENT_OOS,
) -> dict[str, Any]:
    blockers: list[str] = []
    counts: dict[str, int] = {}
    for name in REQUIRED_COMPONENTS:
        try:
            count = int(component_oos_n.get(name, 0) or 0)
        except (TypeError, ValueError):
            count = 0
        counts[name] = count
        if count < min_component_oos:
            blockers.append(f"{name}_OOS_{count}_LT_{min_component_oos}")

    return {
        "ready": not blockers,
        "required_components": list(REQUIRED_COMPONENTS),
        "component_oos_counts": counts,
        "minimum_component_oos": min_component_oos,
        "blockers": blockers,
    }


def combine(
    component_predictions: dict[str, dict[str, Any]],
    component_oos_n: dict[str, int],
    *,
    weights: dict[str, float] | None = None,
    min_component_oos: int = MIN_COMPONENT_OOS,
) -> dict[str, Any]:
    gate = readiness(component_oos_n, min_component_oos=min_component_oos)
    if not gate["ready"]:
        return {
            "schema_version": "1.0.0",
            "model_version": MODEL_VERSION,
            "status": "INSUFFICIENT_COMPONENT_OOS",
            "readiness": gate,
            "probabilities": None,
            "weights": None,
            "market_fields_used": False,
            "production_promotion_allowed": False,
        }

    missing_components = [name for name in REQUIRED_COMPONENTS if name not in component_predictions]
    if missing_components:
        return {
            "schema_version": "1.0.0",
            "model_version": MODEL_VERSION,
            "status": "MISSING_COMPONENT_PREDICTIONS",
            "readiness": gate,
            "missing_components": missing_components,
            "probabilities": None,
            "weights": None,
            "market_fields_used": False,
            "production_promotion_allowed": False,
        }

    normalized: dict[str, dict[str, float]] = {}
    for name in REQUIRED_COMPONENTS:
        prediction = component_predictions[name]
        errors = validate_component_prediction(prediction)
        if errors:
            return {
                "schema_version": "1.0.0",
                "model_version": MODEL_VERSION,
                "status": "INVALID_COMPONENT_PREDICTION",
                "component": name,
                "errors": errors,
                "probabilities": None,
                "weights": None,
                "market_fields_used": False,
                "production_promotion_allowed": False,
            }
        p = dict(prediction)
        one_x_two = _normalized_1x2(prediction)
        assert one_x_two is not None
        p.update(one_x_two)
        normalized[name] = {key: float(p[key]) for key in PROBABILITY_KEYS}

    raw_weights = weights or {name: 1.0 for name in REQUIRED_COMPONENTS}
    cleaned: dict[str, float] = {}
    for name in REQUIRED_COMPONENTS:
        value = _num(raw_weights.get(name))
        cleaned[name] = max(float(value or 0.0), 0.0)
    total_weight = sum(cleaned.values())
    if total_weight <= 0:
        cleaned = {name: 1.0 for name in REQUIRED_COMPONENTS}
        total_weight = float(len(REQUIRED_COMPONENTS))
    normalized_weights = {name: value / total_weight for name, value in cleaned.items()}

    probabilities = {
        key: round(
            sum(normalized[name][key] * normalized_weights[name] for name in REQUIRED_COMPONENTS),
            8,
        )
        for key in PROBABILITY_KEYS
    }
    one_x_two_total = sum(probabilities[key] for key in ("home_win", "draw", "away_win"))
    if one_x_two_total > 0:
        for key in ("home_win", "draw", "away_win"):
            probabilities[key] = round(probabilities[key] / one_x_two_total, 8)

    return {
        "schema_version": "1.0.0",
        "model_version": MODEL_VERSION,
        "status": "RESEARCH_ENSEMBLE_AVAILABLE",
        "readiness": gate,
        "probabilities": probabilities,
        "weights": normalized_weights,
        "weight_policy": (
            "CALLER_SUPPLIED_RESEARCH_WEIGHTS"
            if weights
            else "EQUAL_RESEARCH_WEIGHTS_NO_META_MODEL_YET"
        ),
        "market_fields_used": False,
        "production_promotion_allowed": False,
        "runtime_prediction_weight": 0.0,
    }


def current_gate(dataset_rows: int) -> dict[str, Any]:
    # Until component walk-forward reports are materialized on the same fixture cohort,
    # dataset size is only an upper bound. Never pretend those rows are already OOS.
    upper_bound = max(int(dataset_rows or 0) - 50, 0)
    counts = {name: upper_bound for name in REQUIRED_COMPONENTS}
    gate = readiness(counts)
    return {
        "status": "READY_FOR_SAME_COHORT_ENSEMBLE_RESEARCH" if gate["ready"] else "DATA_BLOCKED",
        "dataset_rows": int(dataset_rows or 0),
        "component_oos_upper_bound": upper_bound,
        "readiness": gate,
        "production_promotion_allowed": False,
        "note": "OOS counts are conservative upper bounds until same-fixture component prediction ledgers are persisted.",
    }
