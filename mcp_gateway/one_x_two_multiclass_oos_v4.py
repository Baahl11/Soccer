from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter
from datetime import datetime
from typing import Any, Iterable

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_1X2_MULTICLASS_OOS_V4_1.0.3"
MIN_TRAIN_ROWS = 200
MIN_EVAL_ROWS = 100
BATCH_SIZE = 50
EPS = 1e-12


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        out = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return out if out.tzinfo is not None else None


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _extract(row: dict[str, Any]) -> tuple[tuple[float, float, float], int] | None:
    predictions = row.get("predictions") if isinstance(row.get("predictions"), dict) else {}
    outcomes = row.get("outcomes") if isinstance(row.get("outcomes"), dict) else {}

    raw = (
        _num(predictions.get("home_win")),
        _num(predictions.get("draw")),
        _num(predictions.get("away_win")),
    )
    if any(value is None or value < 0 for value in raw):
        return None

    total = sum(float(value) for value in raw)
    if total <= 0:
        return None
    probs = tuple(max(EPS, float(value) / total) for value in raw)
    renorm = sum(probs)
    probs = tuple(value / renorm for value in probs)

    outcome_values = (
        outcomes.get("home_win"),
        outcomes.get("draw"),
        outcomes.get("away_win"),
    )
    if outcome_values.count(1) != 1:
        return None
    target = outcome_values.index(1)
    return (probs[0], probs[1], probs[2]), target


def temperature_scale(probs: tuple[float, float, float], temperature: float) -> tuple[float, float, float]:
    temperature = max(0.05, float(temperature))
    logits = [math.log(max(EPS, p)) / temperature for p in probs]
    peak = max(logits)
    exps = [math.exp(value - peak) for value in logits]
    total = sum(exps)
    return (exps[0] / total, exps[1] / total, exps[2] / total)


def _metrics(items: Iterable[tuple[tuple[float, float, float], int]]) -> dict[str, Any]:
    rows = list(items)
    if not rows:
        return {"n": 0, "multiclass_brier": None, "multiclass_log_loss": None, "top1_accuracy": None}

    brier = 0.0
    log_loss = 0.0
    correct = 0
    for probs, target in rows:
        brier += sum((prob - (1.0 if idx == target else 0.0)) ** 2 for idx, prob in enumerate(probs))
        log_loss -= math.log(max(EPS, probs[target]))
        if max(range(3), key=lambda idx: probs[idx]) == target:
            correct += 1

    n = len(rows)
    return {
        "n": n,
        "multiclass_brier": round(brier / n, 8),
        "multiclass_log_loss": round(log_loss / n, 8),
        "top1_accuracy": round(correct / n, 8),
    }


def fit_temperature(train: list[tuple[tuple[float, float, float], int]]) -> float:
    if not train:
        return 1.0

    def loss(temp: float) -> float:
        total = 0.0
        for probs, target in train:
            scaled = temperature_scale(probs, temp)
            total -= math.log(max(EPS, scaled[target]))
        return total / len(train)

    coarse = [0.30 + 0.05 * i for i in range(55)]
    best = min(coarse, key=lambda temp: (loss(temp), abs(temp - 1.0)))
    lo = max(0.05, best - 0.06)
    fine = [lo + 0.005 * i for i in range(25)]
    return round(min(fine, key=lambda temp: (loss(temp), abs(temp - 1.0))), 6)


def _semantic_version(value: str) -> tuple[int, ...] | None:
    match = re.search(r"v([0-9]+(?:[.][0-9]+)*)", str(value), re.I)
    if not match:
        return None
    try:
        return tuple(int(part) for part in match.group(1).split("."))
    except ValueError:
        return None


def _current_model_rows(rows: Iterable[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]], dict[str, int]]:
    valid = []
    version_counts: Counter[str] = Counter()
    latest_timestamp_by_version: dict[str, datetime] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        timestamp = _parse_dt(row.get("run_timestamp"))
        extracted = _extract(row)
        if timestamp is None or extracted is None:
            continue
        version = str(row.get("model_version") or "UNKNOWN")
        version_counts[version] += 1
        prior = latest_timestamp_by_version.get(version)
        if prior is None or timestamp > prior:
            latest_timestamp_by_version[version] = timestamp
        valid.append((timestamp, version, row))

    if not valid:
        return None, [], dict(version_counts)

    semantic_versions = [
        (parsed, version)
        for version in version_counts
        if (parsed := _semantic_version(version)) is not None
    ]
    if semantic_versions:
        current_version = max(semantic_versions, key=lambda item: item[0])[1]
    else:
        current_version = max(
            latest_timestamp_by_version,
            key=lambda version: latest_timestamp_by_version[version],
        )

    selected = [row for _, version, row in valid if version == current_version]
    selected.sort(key=lambda row: _parse_dt(row.get("run_timestamp")) or datetime.min)
    return current_version, selected, dict(sorted(version_counts.items()))


def build_report(
    rows: Iterable[dict[str, Any]],
    *,
    min_train_rows: int = MIN_TRAIN_ROWS,
    batch_size: int = BATCH_SIZE,
) -> dict[str, Any]:
    current_version, current_rows, version_counts = _current_model_rows(rows)
    extracted = [_extract(row) for row in current_rows]
    samples = [item for item in extracted if item is not None]

    baseline_eval: list[tuple[tuple[float, float, float], int]] = []
    calibrated_eval: list[tuple[tuple[float, float, float], int]] = []
    folds: list[dict[str, Any]] = []

    for start in range(min_train_rows, len(samples), max(1, int(batch_size))):
        train = samples[:start]
        test = samples[start:start + max(1, int(batch_size))]
        if not test:
            continue
        temperature = fit_temperature(train)
        baseline_eval.extend(test)
        calibrated_batch = [(temperature_scale(probs, temperature), target) for probs, target in test]
        calibrated_eval.extend(calibrated_batch)
        folds.append({
            "train_rows": len(train),
            "eval_rows": len(test),
            "temperature": temperature,
            "baseline": _metrics(test),
            "calibrated": _metrics(calibrated_batch),
        })

    baseline = _metrics(baseline_eval)
    calibrated = _metrics(calibrated_eval)
    brier_delta = (
        round(float(calibrated["multiclass_brier"]) - float(baseline["multiclass_brier"]), 8)
        if calibrated.get("multiclass_brier") is not None and baseline.get("multiclass_brier") is not None
        else None
    )
    log_loss_delta = (
        round(float(calibrated["multiclass_log_loss"]) - float(baseline["multiclass_log_loss"]), 8)
        if calibrated.get("multiclass_log_loss") is not None and baseline.get("multiclass_log_loss") is not None
        else None
    )
    improves_both = (
        brier_delta is not None and log_loss_delta is not None
        and brier_delta < 0 and log_loss_delta < 0
    )

    evaluated_rows = int(baseline.get("n") or 0)
    if len(samples) < min_train_rows:
        status = "DATA_BLOCKED"
    elif evaluated_rows < MIN_EVAL_ROWS:
        status = "INSUFFICIENT_WALK_FORWARD_EVAL"
    elif improves_both:
        status = "RESEARCH_MULTICLASS_CALIBRATION_AVAILABLE"
    else:
        status = "RESEARCH_MULTICLASS_CHALLENGER_NOT_BETTER"

    deployment_temperature = fit_temperature(samples) if len(samples) >= min_train_rows else None
    deployment_calibrator = {
        "method": "TEMPERATURE_SCALING",
        "temperature": deployment_temperature,
        "training_rows": len(samples),
        "source_model_version": current_version,
        "status": (
            "RESEARCH_DEPLOYMENT_CALIBRATOR_FITTED"
            if deployment_temperature is not None and status == "RESEARCH_MULTICLASS_CALIBRATION_AVAILABLE"
            else "RESEARCH_DEPLOYMENT_CALIBRATOR_BLOCKED"
        ),
        "production_promotion_allowed": False,
        "runtime_prediction_weight": 0.0,
        "market_fields_used": False,
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": status,
        "source_model_version": current_version,
        "source_model_version_counts": version_counts,
        "source_rows_current_model": len(samples),
        "minimum_train_rows": int(min_train_rows),
        "minimum_eval_rows": MIN_EVAL_ROWS,
        "batch_size": int(batch_size),
        "walk_forward_folds": len(folds),
        "evaluated_rows": evaluated_rows,
        "baseline": baseline,
        "temperature_scaled": calibrated,
        "research_deployment_calibrator": deployment_calibrator,
        "brier_delta": brier_delta,
        "log_loss_delta": log_loss_delta,
        "improves_brier_and_log_loss": improves_both,
        "folds": folds,
        "anti_leakage": {
            "chronological_order_required": True,
            "temperature_fit_uses_prior_rows_only": True,
            "evaluation_rows_never_used_to_fit_their_fold": True,
            "current_model_version_only": True,
            "market_fields_used": False,
        },
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
        "runtime_prediction_weight": 0.0,
        "canonical_bet_logic_changed": False,
        "notes": [
            "Temperature scaling preserves the 1X2 simplex and class ranking while adjusting confidence.",
            "Only the latest observed model_version is evaluated to avoid mixing calibration regimes across runtime versions.",
            "Each fold fits temperature only on earlier rows and evaluates on later rows.",
            "This report is research-only and cannot change production probabilities by itself.",
            "A full-current-OOS temperature is persisted only as a research deployment calibrator for downstream Phase16 ranking; walk-forward metrics remain the validation evidence.",
        ],
    }


def _load_oos_report(path: str) -> list[dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict) or not isinstance(value.get("rows"), list):
        return []
    return [row for row in value["rows"] if isinstance(row, dict)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward multiclass 1X2 temperature calibration on canonical OOS rows.")
    parser.add_argument("--oos-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = build_report(_load_oos_report(args.oos_report))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps({
        "model_version": report["model_version"],
        "status": report["status"],
        "source_model_version": report["source_model_version"],
        "source_rows_current_model": report["source_rows_current_model"],
        "evaluated_rows": report["evaluated_rows"],
        "walk_forward_folds": report["walk_forward_folds"],
        "brier_delta": report["brier_delta"],
        "log_loss_delta": report["log_loss_delta"],
        "improves_brier_and_log_loss": report["improves_brier_and_log_loss"],
        "provider_requests_added": report["provider_requests_added"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
