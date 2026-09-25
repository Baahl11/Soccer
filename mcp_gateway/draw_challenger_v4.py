from __future__ import annotations

import argparse
import json
import math
from typing import Any

from mcp_gateway.oos_stage_diagnostics_v4 import _auc_discrimination

MODEL_VERSION = "LIGHTGBM_DRAW_V4_CHALLENGER_1.0.0"
SCHEMA_VERSION = "1.0.0"
MIN_TRAIN_ROWS = 100
MIN_OOS_ROWS = 50
MIN_FOLD_ROWS = 25
MAX_GOALS = 15

FEATURE_KEYS = (
    "team_performance.home_goal_rate_blend",
    "team_performance.away_goal_rate_blend",
    "team_performance.total_goal_rate_blend",
    "availability.both_xi_confirmed",
    "availability.both_goalkeepers_confirmed",
    "availability.injury_report_count",
)


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(int(value))
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def poisson_draw_probability(home_lambda: float, away_lambda: float) -> float:
    home_lambda = max(float(home_lambda), 1e-9)
    away_lambda = max(float(away_lambda), 1e-9)
    probability = 0.0
    for goals in range(MAX_GOALS + 1):
        ph = math.exp(-home_lambda) * (home_lambda**goals) / math.factorial(goals)
        pa = math.exp(-away_lambda) * (away_lambda**goals) / math.factorial(goals)
        probability += ph * pa
    return min(max(probability, 1e-9), 1.0 - 1e-9)


def _target(row: dict[str, Any]) -> int | None:
    targets = row.get("targets") if isinstance(row.get("targets"), dict) else {}
    value = targets.get("draw")
    return int(value) if value in (0, 1, False, True) else None


def _features(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("features") if isinstance(row.get("features"), dict) else {}


def eligible_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("market_fields_included") is not False:
            continue
        if _target(row) is None:
            continue
        features = _features(row)
        home = _num(features.get("team_performance.home_goal_rate_blend"))
        away = _num(features.get("team_performance.away_goal_rate_blend"))
        if home is None or away is None or home <= 0 or away <= 0:
            continue
        captured = str(row.get("feature_captured_at") or "")
        kickoff = str(row.get("kickoff") or "")
        if not captured or not kickoff or captured >= kickoff:
            continue
        out.append(row)
    out.sort(key=lambda row: (
        str(row.get("feature_captured_at") or ""),
        int(row.get("fixture_id") or 0),
    ))
    return out


def vectorize(row: dict[str, Any]) -> list[float]:
    features = _features(row)
    home = float(_num(features.get("team_performance.home_goal_rate_blend")) or 0.0)
    away = float(_num(features.get("team_performance.away_goal_rate_blend")) or 0.0)
    total = _num(features.get("team_performance.total_goal_rate_blend"))
    total = float(total) if total is not None else home + away
    gap = abs(home - away)
    high = max(home, away, 1e-9)
    balance = min(home, away) / high
    geometric = math.sqrt(max(home * away, 0.0))
    baseline = poisson_draw_probability(home, away)

    values = [
        home,
        away,
        total,
        gap,
        balance,
        geometric,
        baseline,
    ]
    for key in (
        "availability.both_xi_confirmed",
        "availability.both_goalkeepers_confirmed",
        "availability.injury_report_count",
    ):
        value = _num(features.get(key))
        values.append(float("nan") if value is None else float(value))
        values.append(float(value is None))
    return values


def expanded_feature_names() -> list[str]:
    return [
        "home_goal_rate",
        "away_goal_rate",
        "total_goal_rate",
        "goal_rate_abs_gap",
        "goal_rate_balance_ratio",
        "goal_rate_geometric_mean",
        "poisson_draw_probability",
        "both_xi_confirmed",
        "both_xi_confirmed__missing",
        "both_goalkeepers_confirmed",
        "both_goalkeepers_confirmed__missing",
        "injury_report_count",
        "injury_report_count__missing",
    ]


def _binary_metrics(probabilities: list[float], outcomes: list[int]) -> dict[str, Any]:
    if not probabilities or len(probabilities) != len(outcomes):
        return {
            "rows": 0,
            "positive_count": 0,
            "negative_count": 0,
            "brier": None,
            "log_loss": None,
            "discrimination": _auc_discrimination([]),
        }
    observations = [
        {"probability": float(p), "outcome": int(y)}
        for p, y in zip(probabilities, outcomes)
    ]
    n = len(observations)
    brier = sum((row["probability"] - row["outcome"]) ** 2 for row in observations) / n
    log_loss = -sum(
        row["outcome"] * math.log(max(min(row["probability"], 1.0 - 1e-15), 1e-15))
        + (1 - row["outcome"]) * math.log(max(min(1.0 - row["probability"], 1.0 - 1e-15), 1e-15))
        for row in observations
    ) / n
    positives = sum(row["outcome"] for row in observations)
    return {
        "rows": n,
        "positive_count": positives,
        "negative_count": n - positives,
        "brier": round(brier, 8),
        "log_loss": round(log_loss, 8),
        "discrimination": _auc_discrimination(observations),
    }


def _fold_boundaries(n_rows: int, min_train_rows: int) -> list[tuple[int, int]]:
    remaining = max(n_rows - min_train_rows, 0)
    if remaining <= 0:
        return []
    fold_size = max(MIN_FOLD_ROWS, math.ceil(remaining / 5))
    return [
        (start, min(start + fold_size, n_rows))
        for start in range(min_train_rows, n_rows, fold_size)
    ]


def walk_forward(
    rows: list[dict[str, Any]],
    *,
    min_train_rows: int = MIN_TRAIN_ROWS,
    min_oos_rows: int = MIN_OOS_ROWS,
) -> dict[str, Any]:
    ordered = eligible_rows(rows)
    base = {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "eligible_rows": len(ordered),
        "minimum_training_rows": int(min_train_rows),
        "minimum_oos_rows": int(min_oos_rows),
        "feature_names": expanded_feature_names(),
        "feature_count": len(expanded_feature_names()),
        "market_fields_used": False,
        "post_kickoff_features_used": False,
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
        "runtime_prediction_weight": 0,
        "model_weights_changed": False,
        "canonical_bet_logic_changed": False,
        "baseline": "INDEPENDENT_POISSON_DRAW_FROM_PREKICKOFF_HOME_AWAY_GOAL_RATES",
        "training_policy": "EXPANDING_WALK_FORWARD_BLOCKS;NO_MARKET_FIELDS;STRICTLY_PREKICKOFF_FEATURES",
    }
    if len(ordered) < int(min_train_rows) + int(min_oos_rows):
        return {
            **base,
            "status": "INSUFFICIENT_TRAINING_SAMPLE",
            "folds": [],
            "walk_forward_evaluated": 0,
            "baseline_metrics": _binary_metrics([], []),
            "challenger_metrics": _binary_metrics([], []),
            "comparison": {
                "auc_lower_95_delta": None,
                "brier_delta": None,
                "log_loss_delta": None,
                "challenger_discrimination_ready": False,
                "improves_all_primary_metrics": False,
            },
        }

    try:
        import lightgbm as lgb
    except Exception as exc:
        return {
            **base,
            "status": "LIGHTGBM_DEPENDENCY_UNAVAILABLE",
            "dependency_error": str(exc)[:200],
            "folds": [],
            "walk_forward_evaluated": 0,
            "baseline_metrics": _binary_metrics([], []),
            "challenger_metrics": _binary_metrics([], []),
            "comparison": {
                "auc_lower_95_delta": None,
                "brier_delta": None,
                "log_loss_delta": None,
                "challenger_discrimination_ready": False,
                "improves_all_primary_metrics": False,
            },
        }

    outcomes: list[int] = []
    baseline_probs: list[float] = []
    challenger_probs: list[float] = []
    fold_reports: list[dict[str, Any]] = []
    importance_sum = {name: 0.0 for name in expanded_feature_names()}
    fitted_folds = 0

    for fold_index, (start, end) in enumerate(_fold_boundaries(len(ordered), int(min_train_rows)), start=1):
        train = ordered[:start]
        test = ordered[start:end]
        y_train = [_target(row) for row in train]
        if any(value is None for value in y_train):
            continue
        positives = sum(int(value) for value in y_train if value is not None)
        negatives = len(y_train) - positives
        if positives < 10 or negatives < 10:
            continue

        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=120,
            learning_rate=0.03,
            num_leaves=7,
            max_depth=3,
            min_child_samples=20,
            reg_lambda=3.0,
            reg_alpha=0.5,
            verbosity=-1,
            random_state=42,
            n_jobs=1,
        )
        model.fit([vectorize(row) for row in train], [int(value) for value in y_train])

        x_test = [vectorize(row) for row in test]
        fold_challenger = [float(value) for value in model.predict_proba(x_test)[:, 1]]
        fold_outcomes = [int(_target(row) or 0) for row in test]
        fold_baseline = [
            poisson_draw_probability(
                float(_num(_features(row).get("team_performance.home_goal_rate_blend")) or 0.0),
                float(_num(_features(row).get("team_performance.away_goal_rate_blend")) or 0.0),
            )
            for row in test
        ]

        outcomes.extend(fold_outcomes)
        baseline_probs.extend(fold_baseline)
        challenger_probs.extend(fold_challenger)
        for name, importance in zip(expanded_feature_names(), model.feature_importances_):
            importance_sum[name] += float(importance)
        fitted_folds += 1
        fold_reports.append({
            "fold": fold_index,
            "train_rows": len(train),
            "test_rows": len(test),
            "test_draws": sum(fold_outcomes),
            "baseline": _binary_metrics(fold_baseline, fold_outcomes),
            "challenger": _binary_metrics(fold_challenger, fold_outcomes),
        })

    baseline_metrics = _binary_metrics(baseline_probs, outcomes)
    challenger_metrics = _binary_metrics(challenger_probs, outcomes)
    baseline_auc_l95 = _num((baseline_metrics.get("discrimination") or {}).get("auc_lower_95"))
    challenger_auc_l95 = _num((challenger_metrics.get("discrimination") or {}).get("auc_lower_95"))
    baseline_brier = _num(baseline_metrics.get("brier"))
    challenger_brier = _num(challenger_metrics.get("brier"))
    baseline_log = _num(baseline_metrics.get("log_loss"))
    challenger_log = _num(challenger_metrics.get("log_loss"))

    comparison = {
        "auc_lower_95_delta": (
            round(challenger_auc_l95 - baseline_auc_l95, 8)
            if challenger_auc_l95 is not None and baseline_auc_l95 is not None else None
        ),
        "brier_delta": (
            round(challenger_brier - baseline_brier, 8)
            if challenger_brier is not None and baseline_brier is not None else None
        ),
        "log_loss_delta": (
            round(challenger_log - baseline_log, 8)
            if challenger_log is not None and baseline_log is not None else None
        ),
        "challenger_discrimination_ready": bool(
            (challenger_metrics.get("discrimination") or {}).get("discrimination_ready") is True
        ),
    }
    comparison["improves_all_primary_metrics"] = bool(
        comparison["auc_lower_95_delta"] is not None
        and comparison["auc_lower_95_delta"] > 0
        and comparison["brier_delta"] is not None
        and comparison["brier_delta"] < 0
        and comparison["log_loss_delta"] is not None
        and comparison["log_loss_delta"] < 0
    )

    ranked_importance = sorted(
        (
            {"feature": name, "mean_importance": round(total / max(fitted_folds, 1), 6)}
            for name, total in importance_sum.items()
        ),
        key=lambda row: row["mean_importance"],
        reverse=True,
    )

    return {
        **base,
        "status": "RESEARCH_ONLY",
        "folds": fold_reports,
        "fitted_folds": fitted_folds,
        "walk_forward_evaluated": len(outcomes),
        "baseline_metrics": baseline_metrics,
        "challenger_metrics": challenger_metrics,
        "comparison": comparison,
        "feature_importance": ranked_importance,
        "research_followup_candidate": bool(
            len(outcomes) >= int(min_oos_rows)
            and comparison["challenger_discrimination_ready"]
            and comparison["improves_all_primary_metrics"]
        ),
        "promotion_reason": "OFFLINE_DRAW_CHALLENGER_REQUIRES_SEPARATE_VALIDATION_AND_MANUAL_REVIEW",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline Draw-specific LightGBM challenger.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-train-rows", type=int, default=MIN_TRAIN_ROWS)
    parser.add_argument("--min-oos-rows", type=int, default=MIN_OOS_ROWS)
    args = parser.parse_args()

    with open(args.dataset, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("rows") if isinstance(payload, dict) else []
    report = walk_forward(
        rows if isinstance(rows, list) else [],
        min_train_rows=args.min_train_rows,
        min_oos_rows=args.min_oos_rows,
    )
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
