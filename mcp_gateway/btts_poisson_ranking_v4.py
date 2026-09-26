from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
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

MODEL_VERSION = "BTTS_POISSON_RANKING_V4_1.1.0"
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
        "selection_ranking_key": "RAW_POISSON_BTTS_PROBABILITY",
        "calibrated_probability_role": "EDGE_ESTIMATION_ONLY_NOT_SELECTION_SORT_KEY",
        "eligibility_gate": "RAW_AUC_LOWER_95_GT_0_50_AND_WITHIN_FOLD_MONOTONIC_PLATT_AND_BRIER_LOGLOSS_IMPROVE",
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
    within_fold_ranking_preserved = bool(
        positive_slope_all_folds
        and all(
            abs(
                float((fold.get("calibrated_metrics") or {}).get("discrimination", {}).get("auc") or 0.0)
                - float((fold.get("raw_metrics") or {}).get("discrimination", {}).get("auc") or 0.0)
            ) <= 1e-6
            for fold in folds
            if (fold.get("raw_metrics") or {}).get("discrimination", {}).get("auc") is not None
            and (fold.get("calibrated_metrics") or {}).get("discrimination", {}).get("auc") is not None
        )
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
        and within_fold_ranking_preserved
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
            "within_fold_ranking_preserved": within_fold_ranking_preserved,
            "pooled_calibrated_auc_delta_not_used_as_gate": auc_delta,
            "raw_poisson_is_selection_ranking_signal": True,
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


def _parse_dt(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _lineage_diagnostic(rows: list[dict[str, Any]]) -> dict[str, Any]:
    dataset_fixtures = {
        int(row["fixture_id"])
        for row in rows
        if isinstance(row, dict) and row.get("fixture_id") is not None
    }
    dataset_model_counts = Counter(
        str(row.get("model_version") or "UNKNOWN")
        for row in rows
        if isinstance(row, dict)
    )
    dataset_stage_counts = Counter(
        str(row.get("stage") or "UNKNOWN").upper()
        for row in rows
        if isinstance(row, dict)
    )

    ledger_path = Path("state/soccer_edge_state/analysis/oos_prediction_ledger_v4.jsonl")
    diagnostics_path = Path("state/soccer_edge_state/analysis/oos_stage_diagnostics_v4.json")
    if not ledger_path.exists():
        return {
            "status": "LEDGER_NOT_AVAILABLE",
            "dataset_fixture_count": len(dataset_fixtures),
            "dataset_model_version_counts": dict(sorted(dataset_model_counts.items())),
            "dataset_stage_counts": dict(sorted(dataset_stage_counts.items())),
        }

    current_model = None
    if diagnostics_path.exists():
        try:
            diag = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            current_model = str(diag.get("current_source_model_version") or "").strip() or None
        except Exception:
            current_model = None

    ledger_model_counts: Counter[str] = Counter()
    ledger_fixture_sets: dict[str, set[int]] = defaultdict(set)
    ledger_stage_counts: Counter[str] = Counter()
    exact_model_fixture_overlap = 0
    exact_model_stage_fixture_overlap = 0
    current_model_valid_rows = 0

    dataset_by_fixture = {
        int(row["fixture_id"]): row
        for row in rows
        if isinstance(row, dict) and row.get("fixture_id") is not None
    }

    for raw in ledger_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except Exception:
            continue
        if not isinstance(row, dict) or row.get("anti_leakage") is not True:
            continue
        try:
            fixture_id = int(row.get("fixture_id"))
        except (TypeError, ValueError):
            continue
        kickoff = _parse_dt(row.get("kickoff"))
        run_dt = _parse_dt(row.get("run_timestamp"))
        if kickoff is None or run_dt is None or run_dt >= kickoff:
            continue

        model_version = str(row.get("model_version") or "UNKNOWN")
        stage = str(row.get("run_type") or "UNKNOWN").upper()
        ledger_model_counts[model_version] += 1
        ledger_fixture_sets[model_version].add(fixture_id)
        ledger_stage_counts[stage] += 1
        if current_model and model_version == current_model:
            current_model_valid_rows += 1

        drow = dataset_by_fixture.get(fixture_id)
        if drow is None:
            continue
        if str(drow.get("model_version") or "UNKNOWN") == model_version:
            exact_model_fixture_overlap += 1
            if str(drow.get("stage") or "UNKNOWN").upper() == stage:
                exact_model_stage_fixture_overlap += 1

    overlap_by_model = {
        model: len(fixtures & dataset_fixtures)
        for model, fixtures in sorted(ledger_fixture_sets.items())
    }
    current_model_overlap = (
        overlap_by_model.get(current_model, 0)
        if current_model is not None
        else None
    )

    return {
        "status": "DIAGNOSTIC_ONLY",
        "current_source_model_version": current_model,
        "dataset_fixture_count": len(dataset_fixtures),
        "dataset_model_version_counts": dict(sorted(dataset_model_counts.items())),
        "dataset_stage_counts": dict(sorted(dataset_stage_counts.items())),
        "ledger_valid_prekickoff_rows": int(sum(ledger_model_counts.values())),
        "ledger_model_version_counts": dict(sorted(ledger_model_counts.items())),
        "ledger_stage_counts": dict(sorted(ledger_stage_counts.items())),
        "fixture_overlap_by_ledger_model_version": overlap_by_model,
        "current_model_valid_prekickoff_rows": current_model_valid_rows,
        "current_model_fixture_overlap": current_model_overlap,
        "exact_dataset_model_and_ledger_model_overlap_rows": exact_model_fixture_overlap,
        "exact_dataset_model_stage_and_ledger_model_stage_overlap_rows": exact_model_stage_fixture_overlap,
        "model_weights_changed": False,
        "thresholds_changed": False,
        "canonical_bet_logic_changed": False,
        "production_promotion_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="BTTS Poisson ranking + train-only Platt calibration audit.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with open(args.dataset, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("rows") if isinstance(payload, dict) else []
    safe_rows = rows if isinstance(rows, list) else []
    report = walk_forward(safe_rows)
    report["dataset_version"] = payload.get("dataset_version") if isinstance(payload, dict) else None
    report["feature_schema_version"] = payload.get("feature_schema_version") if isinstance(payload, dict) else None
    report["dataset_fingerprint"] = payload.get("dataset_fingerprint") if isinstance(payload, dict) else None
    report["dataset_row_count"] = payload.get("row_count") if isinstance(payload, dict) else None
    report["lineage_diagnostic"] = _lineage_diagnostic(safe_rows)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
