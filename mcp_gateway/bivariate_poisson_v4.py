from __future__ import annotations

import math
from typing import Any

MODEL_VERSION = "BIVARIATE_POISSON_V4_BASELINE_1.0.0"
SHARED_FRACTION_GRID = tuple(round(i * 0.01, 2) for i in range(0, 41))
MAX_GOALS = 10
MIN_TRAIN_ROWS = 30


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _lambdas(row: dict[str, Any]) -> tuple[float | None, float | None]:
    features = row.get("features") if isinstance(row.get("features"), dict) else {}
    home = _num(features.get("team_performance.home_goal_rate_blend"))
    away = _num(features.get("team_performance.away_goal_rate_blend"))
    if home is None or away is None or home <= 0 or away <= 0:
        return None, None
    return home, away


def _targets(row: dict[str, Any]) -> tuple[int | None, int | None]:
    targets = row.get("targets") if isinstance(row.get("targets"), dict) else {}
    try:
        home = int(targets.get("home_goals"))
        away = int(targets.get("away_goals"))
    except (TypeError, ValueError):
        return None, None
    if home < 0 or away < 0:
        return None, None
    return home, away


def eligible_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        home_lambda, away_lambda = _lambdas(row)
        home_goals, away_goals = _targets(row)
        if home_lambda is None or away_lambda is None or home_goals is None or away_goals is None:
            continue
        out.append(row)
    out.sort(key=lambda row: (
        str(row.get("feature_captured_at") or ""),
        int(row.get("fixture_id") or 0),
    ))
    return out


def _components(home_mean: float, away_mean: float, shared_fraction: float) -> tuple[float, float, float] | None:
    if home_mean <= 0 or away_mean <= 0:
        return None
    if shared_fraction < 0 or shared_fraction >= 1:
        return None
    shared = min(home_mean, away_mean) * shared_fraction
    home_only = home_mean - shared
    away_only = away_mean - shared
    if home_only <= 0 or away_only <= 0 or shared < 0:
        return None
    return home_only, away_only, shared


def joint_pmf(
    home_goals: int,
    away_goals: int,
    home_mean: float,
    away_mean: float,
    shared_fraction: float,
) -> float:
    if home_goals < 0 or away_goals < 0:
        return 0.0
    components = _components(home_mean, away_mean, shared_fraction)
    if components is None:
        return 0.0
    home_only, away_only, shared = components
    total = 0.0
    upper = min(home_goals, away_goals)
    for k in range(upper + 1):
        total += (
            (home_only ** (home_goals - k))
            * (away_only ** (away_goals - k))
            * (shared ** k)
            / (
                math.factorial(home_goals - k)
                * math.factorial(away_goals - k)
                * math.factorial(k)
            )
        )
    return math.exp(-(home_only + away_only + shared)) * total


def probabilities(
    home_mean: float,
    away_mean: float,
    shared_fraction: float,
    max_goals: int = MAX_GOALS,
) -> dict[str, float] | None:
    if _components(home_mean, away_mean, shared_fraction) is None:
        return None

    home = draw = away = btts = over_15 = over_25 = over_35 = mass = 0.0
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = joint_pmf(h, a, home_mean, away_mean, shared_fraction)
            if p <= 0:
                continue
            mass += p
            if h > a:
                home += p
            elif h == a:
                draw += p
            else:
                away += p
            if h > 0 and a > 0:
                btts += p
            total = h + a
            if total >= 2:
                over_15 += p
            if total >= 3:
                over_25 += p
            if total >= 4:
                over_35 += p

    if mass <= 0:
        return None

    return {
        "home_win": home / mass,
        "draw": draw / mass,
        "away_win": away / mass,
        "btts": btts / mass,
        "over_1_5": over_15 / mass,
        "over_2_5": over_25 / mass,
        "over_3_5": over_35 / mass,
    }


def exact_score_nll(rows: list[dict[str, Any]], shared_fraction: float) -> float | None:
    total = 0.0
    n = 0
    for row in eligible_rows(rows):
        home_mean, away_mean = _lambdas(row)
        home_goals, away_goals = _targets(row)
        assert home_mean is not None and away_mean is not None
        assert home_goals is not None and away_goals is not None
        p = joint_pmf(home_goals, away_goals, home_mean, away_mean, shared_fraction)
        if p <= 0:
            return None
        total += -math.log(max(p, 1e-15))
        n += 1
    return total / n if n else None


def fit_shared_fraction(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates: list[tuple[float, float]] = []
    eligible = eligible_rows(rows)
    for fraction in SHARED_FRACTION_GRID:
        nll = exact_score_nll(eligible, fraction)
        if nll is not None and math.isfinite(nll):
            candidates.append((nll, fraction))

    if not candidates:
        return {
            "status": "INSUFFICIENT_DATA",
            "shared_fraction": 0.0,
            "training_rows": 0,
            "exact_score_nll": None,
        }

    candidates.sort(key=lambda item: (item[0], item[1]))
    best_nll, best_fraction = candidates[0]
    return {
        "status": "FIT",
        "shared_fraction": best_fraction,
        "training_rows": len(eligible),
        "exact_score_nll": round(best_nll, 6),
    }


def _multiclass_brier(probs: dict[str, float], home_goals: int, away_goals: int) -> float:
    actual = "home_win" if home_goals > away_goals else "away_win" if away_goals > home_goals else "draw"
    return sum((probs[key] - (1.0 if key == actual else 0.0)) ** 2 for key in ("home_win", "draw", "away_win"))


def _multiclass_log_loss(probs: dict[str, float], home_goals: int, away_goals: int) -> float:
    actual = "home_win" if home_goals > away_goals else "away_win" if away_goals > home_goals else "draw"
    return -math.log(max(probs[actual], 1e-15))


def _binary_brier(p: float, y: int) -> float:
    return (p - y) ** 2


def walk_forward(rows: list[dict[str, Any]], min_train_rows: int = MIN_TRAIN_ROWS) -> dict[str, Any]:
    ordered = eligible_rows(rows)
    evaluated: list[dict[str, Any]] = []

    for index in range(max(min_train_rows, 1), len(ordered)):
        train = ordered[:index]
        current = ordered[index]
        fit = fit_shared_fraction(train)
        fraction = float(fit.get("shared_fraction") or 0.0)

        home_mean, away_mean = _lambdas(current)
        home_goals, away_goals = _targets(current)
        assert home_mean is not None and away_mean is not None
        assert home_goals is not None and away_goals is not None

        independent = probabilities(home_mean, away_mean, 0.0)
        bivariate = probabilities(home_mean, away_mean, fraction)
        if independent is None or bivariate is None:
            continue

        total = home_goals + away_goals
        evaluated.append({
            "fixture_id": current.get("fixture_id"),
            "shared_fraction": fraction,
            "independent": independent,
            "bivariate": bivariate,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "actual_btts": int(home_goals > 0 and away_goals > 0),
            "actual_over_2_5": int(total >= 3),
        })

    def aggregate(prob_key: str) -> dict[str, Any]:
        if not evaluated:
            return {
                "n": 0,
                "brier_1x2": None,
                "log_loss_1x2": None,
                "brier_btts": None,
                "brier_over_2_5": None,
            }
        brier_1x2 = log_loss_1x2 = brier_btts = brier_o25 = 0.0
        for row in evaluated:
            probs = row[prob_key]
            brier_1x2 += _multiclass_brier(probs, row["home_goals"], row["away_goals"])
            log_loss_1x2 += _multiclass_log_loss(probs, row["home_goals"], row["away_goals"])
            brier_btts += _binary_brier(probs["btts"], row["actual_btts"])
            brier_o25 += _binary_brier(probs["over_2_5"], row["actual_over_2_5"])
        n = len(evaluated)
        return {
            "n": n,
            "brier_1x2": round(brier_1x2 / n, 6),
            "log_loss_1x2": round(log_loss_1x2 / n, 6),
            "brier_btts": round(brier_btts / n, 6),
            "brier_over_2_5": round(brier_o25 / n, 6),
        }

    independent_metrics = aggregate("independent")
    bivariate_metrics = aggregate("bivariate")
    n = len(evaluated)

    improvement = {
        "brier_1x2_delta": (
            round(bivariate_metrics["brier_1x2"] - independent_metrics["brier_1x2"], 6)
            if n else None
        ),
        "log_loss_1x2_delta": (
            round(bivariate_metrics["log_loss_1x2"] - independent_metrics["log_loss_1x2"], 6)
            if n else None
        ),
        "brier_btts_delta": (
            round(bivariate_metrics["brier_btts"] - independent_metrics["brier_btts"], 6)
            if n else None
        ),
        "brier_over_2_5_delta": (
            round(bivariate_metrics["brier_over_2_5"] - independent_metrics["brier_over_2_5"], 6)
            if n else None
        ),
    }

    latest_fit = fit_shared_fraction(ordered) if ordered else {
        "status": "INSUFFICIENT_DATA",
        "shared_fraction": 0.0,
        "training_rows": 0,
        "exact_score_nll": None,
    }

    return {
        "schema_version": "1.0.0",
        "model_version": MODEL_VERSION,
        "status": "RESEARCH_ONLY" if n else "INSUFFICIENT_OOS_SAMPLE",
        "method": "BIVARIATE_POISSON_SHARED_COMPONENT_FRACTION_ON_POINT_IN_TIME_GOAL_RATE_FEATURES",
        "eligible_rows": len(ordered),
        "minimum_training_rows": min_train_rows,
        "walk_forward_evaluated": n,
        "latest_fit": latest_fit,
        "independent_poisson": independent_metrics,
        "bivariate_poisson": bivariate_metrics,
        "improvement": improvement,
        "market_fields_used": False,
        "post_kickoff_features_used": False,
        "promotion_allowed": False,
        "promotion_reason": (
            "INSUFFICIENT_WALK_FORWARD_SAMPLE"
            if n < 100
            else "REQUIRES_FORMAL_OOS_CALIBRATION_AND_MANUAL_REVIEW"
        ),
    }
