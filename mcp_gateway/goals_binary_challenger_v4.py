from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from typing import Any

from mcp_gateway.oos_stage_diagnostics_v4 import _auc_discrimination

MODEL_VERSION = "LIGHTGBM_GOALS_BINARY_V4_CHALLENGER_1.0.0"
SCHEMA_VERSION = "1.0.0"
MIN_TRAIN_ROWS = 50
MIN_OOS_ROWS = 50
MIN_FOLD_ROWS = 25
MIN_CLASS_ROWS = 10
TARGETS = ("btts", "over_2_5")


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(int(value))
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _goal_rates(row: dict[str, Any]) -> tuple[float | None, float | None]:
    features = row.get("features") if isinstance(row.get("features"), dict) else {}
    home = _num(features.get("team_performance.home_goal_rate_blend"))
    away = _num(features.get("team_performance.away_goal_rate_blend"))
    if home is None or away is None or home <= 0 or away <= 0:
        return None, None
    return float(home), float(away)


def poisson_probability(home_lambda: float, away_lambda: float, target: str) -> float:
    home = max(float(home_lambda), 1e-9)
    away = max(float(away_lambda), 1e-9)
    if target == "btts":
        p = (1.0 - math.exp(-home)) * (1.0 - math.exp(-away))
    elif target == "over_2_5":
        total = home + away
        p_le_2 = math.exp(-total) * (1.0 + total + (total * total / 2.0))
        p = 1.0 - p_le_2
    else:
        raise ValueError(f"UNSUPPORTED_TARGET:{target}")
    return min(max(p, 1e-9), 1.0 - 1e-9)


def _target(row: dict[str, Any], target: str) -> int | None:
    targets = row.get("targets") if isinstance(row.get("targets"), dict) else {}
    value = targets.get(target)
    return int(value) if value in (0, 1, False, True) else None


def eligible_rows(rows: list[dict[str, Any]], target: str) -> list[dict[str, Any]]:
    if target not in TARGETS:
        raise ValueError(f"UNSUPPORTED_TARGET:{target}")
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("market_fields_included") is not False:
            continue
        if _target(row, target) is None:
            continue
        home, away = _goal_rates(row)
        if home is None or away is None:
            continue
        captured = str(row.get("feature_captured_at") or "")
        kickoff = str(row.get("kickoff") or "")
        if not captured or not kickoff:
            continue
        try:
            captured_dt = datetime.fromisoformat(captured.replace("Z", "+00:00"))
            kickoff_dt = datetime.fromisoformat(kickoff.replace("Z", "+00:00"))
        except ValueError:
            continue
        if captured_dt.tzinfo is None or kickoff_dt.tzinfo is None or captured_dt >= kickoff_dt:
            continue
        out.append(row)
    out.sort(key=lambda row: (
        str(row.get("feature_captured_at") or ""),
        int(row.get("fixture_id") or 0),
    ))
    return out


def vectorize(row: dict[str, Any], target: str) -> list[float]:
    features = row.get("features") if isinstance(row.get("features"), dict) else {}
    home, away = _goal_rates(row)
    assert home is not None and away is not None
    total_raw = _num(features.get("team_performance.total_goal_rate_blend"))
    total = float(total_raw) if total_raw is not None else home + away
    gap = abs(home - away)
    balance = min(home, away) / max(home, away, 1e-9)
    geometric = math.sqrt(max(home * away, 0.0))
    baseline = poisson_probability(home, away, target)
    values = [home, away, total, gap, balance, geometric, baseline]
    for key in (
        "availability.both_xi_confirmed",
        "availability.both_goalkeepers_confirmed",
        "availability.injury_report_count",
    ):
        value = _num(features.get(key))
        values.append(float("nan") if value is None else float(value))
        values.append(float(value is None))
    return values


def feature_names(target: str) -> list[str]:
    return [
        "home_goal_rate",
        "away_goal_rate",
        "total_goal_rate",
        "goal_rate_abs_gap",
        "goal_rate_balance_ratio",
        "goal_rate_geometric_mean",
        f"poisson_{target}_probability",
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
    obs = [{"probability": float(p), "outcome": int(y)} for p, y in zip(probabilities, outcomes)]
    n = len(obs)
    brier = sum((r["probability"] - r["outcome"]) ** 2 for r in obs) / n
    log_loss = -sum(
        r["outcome"] * math.log(max(min(r["probability"], 1.0 - 1e-15), 1e-15))
        + (1 - r["outcome"]) * math.log(max(min(1.0 - r["probability"], 1.0 - 1e-15), 1e-15))
        for r in obs
    ) / n
    positives = sum(r["outcome"] for r in obs)
    return {
        "rows": n,
        "positive_count": positives,
        "negative_count": n - positives,
        "brier": round(brier, 8),
        "log_loss": round(log_loss, 8),
        "discrimination": _auc_discrimination(obs),
    }


def _effective_train_rows(rows: list[dict[str, Any]], target: str, minimum: int) -> int | None:
    start = int(minimum)
    while start < len(rows):
        y = [_target(row, target) for row in rows[:start]]
        positives = sum(int(value) for value in y if value is not None)
        negatives = len(y) - positives
        if positives >= MIN_CLASS_ROWS and negatives >= MIN_CLASS_ROWS:
            return start
        start += 1
    return None


def _fold_boundaries(n_rows: int, start: int) -> list[tuple[int, int]]:
    remaining = max(n_rows - start, 0)
    if remaining <= 0:
        return []
    fold_size = max(MIN_FOLD_ROWS, math.ceil(remaining / 5))
    return [(i, min(i + fold_size, n_rows)) for i in range(start, n_rows, fold_size)]


def walk_forward(
    rows: list[dict[str, Any]],
    target: str,
    *,
    min_train_rows: int = MIN_TRAIN_ROWS,
    min_oos_rows: int = MIN_OOS_ROWS,
) -> dict[str, Any]:
    ordered = eligible_rows(rows, target)
    base = {
        "target": target,
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
        "baseline": f"INDEPENDENT_POISSON_{target.upper()}_FROM_PREKICKOFF_GOAL_RATES",
        "training_policy": "EXPANDING_WALK_FORWARD;FIRST_CLASS_SUPPORTED_TRAIN_POINT;NO_MARKET_FIELDS;STRICTLY_PREKICKOFF",
    }
    if len(ordered) < int(min_train_rows) + int(min_oos_rows):
        return {
            **base,
            "status": "INSUFFICIENT_TRAINING_SAMPLE",
            "walk_forward_evaluated": 0,
            "baseline_metrics": _binary_metrics([], []),
            "challenger_metrics": _binary_metrics([], []),
            "comparison": {},
            "research_followup_candidate": False,
        }

    effective = _effective_train_rows(ordered, target, int(min_train_rows))
    if effective is None:
        return {
            **base,
            "status": "INSUFFICIENT_CLASS_SUPPORT",
            "walk_forward_evaluated": 0,
            "baseline_metrics": _binary_metrics([], []),
            "challenger_metrics": _binary_metrics([], []),
            "comparison": {},
            "research_followup_candidate": False,
        }
    base["effective_training_rows"] = effective
    base["available_oos_rows"] = len(ordered) - effective

    try:
        import lightgbm as lgb
        import sklearn  # noqa: F401
    except Exception as exc:
        return {
            **base,
            "status": "LIGHTGBM_DEPENDENCY_UNAVAILABLE",
            "dependency_error": str(exc)[:200],
            "walk_forward_evaluated": 0,
            "baseline_metrics": _binary_metrics([], []),
            "challenger_metrics": _binary_metrics([], []),
            "comparison": {},
            "research_followup_candidate": False,
        }

    outcomes: list[int] = []
    baseline_probs: list[float] = []
    challenger_probs: list[float] = []
    folds: list[dict[str, Any]] = []
    importance_sum = {name: 0.0 for name in feature_names(target)}
    fitted_folds = 0

    for fold_index, (start, end) in enumerate(_fold_boundaries(len(ordered), effective), start=1):
        train = ordered[:start]
        test = ordered[start:end]
        y_train = [int(_target(row, target) or 0) for row in train]
        positives = sum(y_train)
        negatives = len(y_train) - positives
        if positives < MIN_CLASS_ROWS or negatives < MIN_CLASS_ROWS:
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
        model.fit([vectorize(row, target) for row in train], y_train)

        fold_outcomes = [int(_target(row, target) or 0) for row in test]
        fold_challenger = [float(v) for v in model.predict_proba([vectorize(row, target) for row in test])[:, 1]]
        fold_baseline = []
        for row in test:
            home, away = _goal_rates(row)
            assert home is not None and away is not None
            fold_baseline.append(poisson_probability(home, away, target))

        outcomes.extend(fold_outcomes)
        challenger_probs.extend(fold_challenger)
        baseline_probs.extend(fold_baseline)
        for name, importance in zip(feature_names(target), model.feature_importances_):
            importance_sum[name] += float(importance)
        fitted_folds += 1
        folds.append({
            "fold": fold_index,
            "train_rows": len(train),
            "test_rows": len(test),
            "test_positives": sum(fold_outcomes),
            "baseline": _binary_metrics(fold_baseline, fold_outcomes),
            "challenger": _binary_metrics(fold_challenger, fold_outcomes),
        })

    baseline_metrics = _binary_metrics(baseline_probs, outcomes)
    challenger_metrics = _binary_metrics(challenger_probs, outcomes)
    base_l95 = _num((baseline_metrics.get("discrimination") or {}).get("auc_lower_95"))
    chal_l95 = _num((challenger_metrics.get("discrimination") or {}).get("auc_lower_95"))
    base_brier = _num(baseline_metrics.get("brier"))
    chal_brier = _num(challenger_metrics.get("brier"))
    base_ll = _num(baseline_metrics.get("log_loss"))
    chal_ll = _num(challenger_metrics.get("log_loss"))

    comparison = {
        "auc_lower_95_delta": round(chal_l95 - base_l95, 8) if chal_l95 is not None and base_l95 is not None else None,
        "brier_delta": round(chal_brier - base_brier, 8) if chal_brier is not None and base_brier is not None else None,
        "log_loss_delta": round(chal_ll - base_ll, 8) if chal_ll is not None and base_ll is not None else None,
        "challenger_discrimination_ready": bool((challenger_metrics.get("discrimination") or {}).get("discrimination_ready") is True),
    }
    comparison["improves_all_primary_metrics"] = bool(
        comparison["auc_lower_95_delta"] is not None and comparison["auc_lower_95_delta"] > 0
        and comparison["brier_delta"] is not None and comparison["brier_delta"] < 0
        and comparison["log_loss_delta"] is not None and comparison["log_loss_delta"] < 0
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
        "fitted_folds": fitted_folds,
        "walk_forward_evaluated": len(outcomes),
        "folds": folds,
        "baseline_metrics": baseline_metrics,
        "challenger_metrics": challenger_metrics,
        "comparison": comparison,
        "feature_importance": ranked_importance,
        "research_followup_candidate": bool(
            len(outcomes) >= int(min_oos_rows)
            and comparison["challenger_discrimination_ready"]
            and comparison["improves_all_primary_metrics"]
        ),
    }


def build_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": "RESEARCH_ONLY",
        "targets": {target: walk_forward(rows, target) for target in TARGETS},
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
        "runtime_prediction_weight": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline BTTS and O2.5 LightGBM challengers.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with open(args.dataset, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("rows") if isinstance(payload, dict) else []
    report = build_report(rows if isinstance(rows, list) else [])
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
