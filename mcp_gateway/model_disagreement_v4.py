from __future__ import annotations

import math
from typing import Any

from mcp_gateway import ensemble_v4

MODEL_VERSION = "SOCCER_MODEL_DISAGREEMENT_V4_1.0.0"
REQUIRED_COMPONENTS = ensemble_v4.REQUIRED_COMPONENTS
PROBABILITY_KEYS = ensemble_v4.PROBABILITY_KEYS
LOW_MAX_RANGE = 0.05
MODERATE_MAX_RANGE = 0.10


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _normalize_1x2(prediction: dict[str, Any]) -> dict[str, Any]:
    out = dict(prediction)
    keys = ("home_win", "draw", "away_win")
    values = [_num(out.get(key)) for key in keys]
    if any(value is None or value < 0 for value in values):
        return out
    total = sum(float(value) for value in values)
    if total > 0:
        for key, value in zip(keys, values):
            out[key] = float(value) / total
    return out


def analyze(component_predictions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    missing = [name for name in REQUIRED_COMPONENTS if name not in component_predictions]
    if missing:
        return {
            "schema_version": "1.0.0",
            "model_version": MODEL_VERSION,
            "status": "MISSING_COMPONENT_PREDICTIONS",
            "missing_components": missing,
            "disagreement": None,
            "production_promotion_allowed": False,
        }

    normalized: dict[str, dict[str, Any]] = {}
    for name in REQUIRED_COMPONENTS:
        prediction = component_predictions.get(name)
        if not isinstance(prediction, dict):
            return {
                "schema_version": "1.0.0",
                "model_version": MODEL_VERSION,
                "status": "INVALID_COMPONENT_PREDICTION",
                "component": name,
                "errors": ["PREDICTION_NOT_OBJECT"],
                "disagreement": None,
                "production_promotion_allowed": False,
            }
        errors = ensemble_v4.validate_component_prediction(prediction)
        if errors:
            return {
                "schema_version": "1.0.0",
                "model_version": MODEL_VERSION,
                "status": "INVALID_COMPONENT_PREDICTION",
                "component": name,
                "errors": errors,
                "disagreement": None,
                "production_promotion_allowed": False,
            }
        normalized[name] = _normalize_1x2(prediction)

    per_target: dict[str, dict[str, Any]] = {}
    ranges: list[float] = []
    for key in PROBABILITY_KEYS:
        values = [float(normalized[name][key]) for name in REQUIRED_COMPONENTS]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        value_range = max(values) - min(values)
        ranges.append(value_range)
        per_target[key] = {
            "mean": round(mean, 8),
            "stddev": round(math.sqrt(variance), 8),
            "range": round(value_range, 8),
            "min": round(min(values), 8),
            "max": round(max(values), 8),
            "component_values": {
                name: round(float(normalized[name][key]), 8)
                for name in REQUIRED_COMPONENTS
            },
        }

    max_range = max(ranges) if ranges else 0.0
    mean_range = sum(ranges) / len(ranges) if ranges else 0.0
    if max_range <= LOW_MAX_RANGE:
        level = "LOW"
    elif max_range <= MODERATE_MAX_RANGE:
        level = "MODERATE"
    else:
        level = "HIGH"

    worst_target = max(PROBABILITY_KEYS, key=lambda key: per_target[key]["range"])
    return {
        "schema_version": "1.0.0",
        "model_version": MODEL_VERSION,
        "status": "RESEARCH_DISAGREEMENT_AVAILABLE",
        "required_components": list(REQUIRED_COMPONENTS),
        "disagreement": {
            "level": level,
            "max_probability_range": round(max_range, 8),
            "mean_probability_range": round(mean_range, 8),
            "worst_target": worst_target,
            "per_target": per_target,
            "thresholds": {
                "low_max_range": LOW_MAX_RANGE,
                "moderate_max_range": MODERATE_MAX_RANGE,
            },
        },
        "market_fields_used": False,
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
        "runtime_prediction_weight": 0.0,
        "canonical_bet_logic_changed": False,
    }
