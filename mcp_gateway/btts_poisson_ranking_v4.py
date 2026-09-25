from __future__ import annotations

import argparse
import json
from typing import Any

from mcp_gateway import calibration_v4
from mcp_gateway.goals_binary_challenger_v4 import (
    MIN_CLASS_ROWS,
    MIN_FOLD_ROWS,
    MIN_OOS_ROWS,
    MIN_TRAIN_ROWS,
    _binary_metrics,
    _effective_train_rows,
    _fold_boundaries,
    _goal_rates,
    _target,
    eligible_rows,
    poisson_probability,
)

MODEL_VERSION = "BTTS_POISSON_RANKING_V4_1.0.0"
SCHEMA_VERSION = "1.0.0"
TARGET = "btts"


def _platt_for_train(train: list[dict[str, Any]]) -> dict[str, Any]:
    observations = []
    for row in train:
        home, away = _goal_rates(row)
        outcome = _target(row, TARGET)
        if home is None or away is None or outcome is None:
            continue
        observations.append({
            "probability": poisson_probability(home, away, TARGET),
            "outcome": int(outcome),
        })
    return calibration_v4.fit_platt(
        observations,
        min_oos_rows=MIN_TRAIN_ROWS,
        min_positives=MIN_CLASS_ROWS,
        min_negatives=MIN_CLASS_ROWS,
    )


def walk_forward(
    rows: list[dict[str, Any]],
    *,
    min_train_rows: int = MIN_TRAIN_ROWS,
    min_oos_rows: int = MIN_OOS_ROWS,
) -> dict[str, Any]:
    ordered = eligible_rows(rows, TARGET)
    effective = _effective_train_rows(ordered, TARGET, int(min_train_rows))
    base = {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "target": TARGET,
        "status": "RESEARCH_ONLY",
        "eligible_rows": len(ordered),
        "minimum_training_rows": int(min_train_rows),
        "minimum_oos_rows": int(min_oos_rows),
        "minimum_class_rows": MIN_CLASS_ROWS,
        "market_fields_used": False,
        "post_kickoff_features_used": False,
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
        "runtime_prediction_weight": 0,
        "model_weights_changed": False,
        "canonical_bet_logic_changed": False,
        "ranking_signal": "INDEPENDENT_POISSON_BTTS_FROM_PREKICKOFF_HOME_AWAY_GOAL_RATES",
        "calibration_method": "EXPANDING_WALK_FORWARD_PLATT_LOGIT_TRAIN_ONLY",
        "eligibility_gate": "RAW_AUC_LOWER_95_GT_0_50_AND_PLATT_POSITIVE_SLOPE_AND_BRIER_LOGLOSS_IMPROVE",
    }
    if len(ordered) < int(min_train_rows) + int(min_oos_rows):
        return {
            **base,
            "status": "INSUFFICIENT_TRAINING_SAMPLE",
            "walk_forward_evaluated": 0,
            "raw_metrics": _binary_metrics([], []),
            "calibrated_metrics": _binary_metrics([], []),
            "research_eligible": False,
        }
    if effective is None:
        return {
            **base,
            "status": "INSUFFICIENT_CLASS_SUPPORT",
            "walk_forward_evaluated": 0,
            "raw_metrics": _binary_metrics([], []),
            "calibrated_metrics": _binary_metrics([], []),
            "research_eligible": False,
        }

    raw_probs: list[float] = []
    calibrated_probs: list[float] = []
    outcomes: list[int] = []
    folds: list[dict[str, Any]] = []
    positive_slope_all_folds = True

    for fold_index, (start, end) in enumerate(_fold_boundaries(len(ordered), effective), start=1):
        train = ordered[:start]
        test = ordered[start:end]
        calibration = _platt_for_train(train)
        params = calibration.get("parameters") if isinstance(calibration.get("parameters"), dict) else {}
        slope = params.get("slope")
        slope_positive = isinstance(slope, (int, float)) and float(slope) > 0
        positive_slope_all_folds = positive_slope_all_folds and slope_positive

        fold_raw: list[float] = []
        fold_cal: list[float] = []
        fold_y: list[int] = []
        for row in test:
            home, away = _goal_rates(row)
            outcome = _target(row, TARGET)
            if home is None or away is None or outcome is None:
                continue
            raw = poisson_probability(home, away, TARGET)
            calibrated = calibration_v4.calibrate_probability(raw, calibration) if slope_positive else None
            if calibrated is None:
                continue
            fold_raw.append(raw)
            fold_cal.append(float(calibrated))
            fold_y.append(int(outcome))

        raw_probs.extend(fold_raw)
        calibrated_probs.extend(fold_cal)
        outcomes.extend(fold_y)
        folds.append({
            "fold": fold_index,
            "train_rows": len(train),
            "test_rows": len(fold_y),
            "test_positives": sum(fold_y),
            "platt_status": calibration.get("status"),
            "platt_parameters": params or None,
            "positive_slope": slope_positive,
            "raw_metrics": _binary_metrics(fold_raw, fold_y),
            "calibrated_metrics": _binary_metrics(fold_cal, fold_y),
        })

    raw_metrics = _binary_metrics(raw_probs, outcomes)
    calibrated_metrics = _binary_metrics(calibrated_probs, outcomes)
    raw_auc = (raw_metrics.get("discrimination") or {}).get("auc")
    cal_auc = (calibrated_metrics.get("discrimination") or {}).get("auc")
    raw_l95 = (raw_metrics.get("discrimination") or {}).get("auc_lower_95")
    cal_l95 = (calibrated_metrics.get("discrimination") or {}).get("auc_lower_95")
    brier_delta = (
        round(float(calibrated_metrics["brier"]) - float(raw_metrics["brier"]), 8)
        if calibrated_metrics.get("brier") is not None and raw_metrics.get("brier") is not None
        else None
    )
    log_loss_delta = (
        round(float(calibrated_metrics["log_loss"]) - float(raw_metrics["log_loss"]), 8)
        if calibrated_metrics.get("log_loss") is not None and raw_metrics.get("log_loss") is not None
        else None
    )
    auc_delta = (
        round(float(cal_auc) - float(raw_auc), 8)
        if cal_auc is not None and raw_auc is not None
        else None
    )
    ranking_preserved = bool(
        positive_slope_all_folds
        and auc_delta is not None
        and abs(float(auc_delta)) <= 1e-6
    )
    sample_ready = len(outcomes) >= int(min_oos_rows)
    discrimination_ready = isinstance(raw_l95, (int, float)) and float(raw_l95) > 0.50
    calibration_improves = (
        brier_delta is not None and brier_delta < 0
        and log_loss_delta is not None and log_loss_delta < 0
    )
    research_eligible = bool(
        sample_ready
        and discrimination_ready
        and ranking_preserved
        and calibration_improves
    )

    return {
        **base,
        "effective_training_rows": effective,
        "available_oos_rows": len(ordered) - effective,
        "walk_forward_evaluated": len(outcomes),
        "fitted_folds": len(folds),
        "folds": folds,
        "raw_metrics": raw_metrics,
        "calibrated_metrics": calibrated_metrics,
        "comparison": {
            "auc_delta": auc_delta,
            "raw_auc_lower_95": raw_l95,
            "calibrated_auc_lower_95": cal_l95,
            "brier_delta": brier_delta,
            "log_loss_delta": log_loss_delta,
            "positive_slope_all_folds": positive_slope_all_folds,
            "ranking_preserved": ranking_preserved,
            "sample_ready": sample_ready,
            "discrimination_ready": discrimination_ready,
            "calibration_improves": calibration_improves,
        },
        "research_eligible": research_eligible,
        "next_gate": (
            "RESEARCH_ELIGIBLE_FOR_SEPARATE_SHADOW_REVIEW"
            if research_eligible
            else "REMAIN_RESEARCH_HOLD"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="BTTS Poisson ranking + train-only Platt calibration audit.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with open(args.dataset, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("rows") if isinstance(payload, dict) else []
    report = walk_forward(rows if isinstance(rows, list) else [])
    report["dataset_version"] = payload.get("dataset_version") if isinstance(payload, dict) else None
    report["feature_schema_version"] = payload.get("feature_schema_version") if isinstance(payload, dict) else None
    report["dataset_fingerprint"] = payload.get("dataset_fingerprint") if isinstance(payload, dict) else None
    report["dataset_row_count"] = payload.get("row_count") if isinstance(payload, dict) else None
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
