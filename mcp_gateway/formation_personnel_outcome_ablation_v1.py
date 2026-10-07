from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from mcp_gateway import formation_matchup_fm3_oos_v1 as fm3

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "FORMATION_PERSONNEL_OUTCOME_ABLATION_V1.0.0"
MIN_TRAIN_N = 30
MIN_REVIEW_N = 100
RIDGE_ALPHA = 8.0
MAX_RESIDUAL_ABS = {
    "SHOTS": 3.0,
    "SOT": 1.25,
    "GOALS": 0.75,
}


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [list(col) for col in zip(*matrix)] if matrix else []


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float] | None:
    n = len(vector)
    if n == 0 or len(matrix) != n:
        return None
    aug = [list(matrix[i]) + [float(vector[i])] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(aug[row][col]))
        if abs(aug[pivot][col]) < 1e-10:
            return None
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        denom = aug[col][col]
        aug[col] = [value / denom for value in aug[col]]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if abs(factor) < 1e-12:
                continue
            aug[row] = [
                aug[row][j] - factor * aug[col][j]
                for j in range(n + 1)
            ]
    return [aug[i][-1] for i in range(n)]


def _standardize_training(
    records: list[tuple[list[float], float]],
    current: list[float],
) -> tuple[list[list[float]], list[float], list[float]] | None:
    if len(records) < MIN_TRAIN_N:
        return None
    width = len(current)
    if any(len(features) != width for features, _ in records):
        return None
    columns = [[features[j] for features, _ in records] for j in range(width)]
    means = [sum(col) / len(col) for col in columns]
    stds: list[float] = []
    for col, mean in zip(columns, means):
        variance = sum((value - mean) ** 2 for value in col) / max(len(col) - 1, 1)
        stds.append(math.sqrt(variance) if variance > 1e-12 else 1.0)

    x = [
        [1.0] + [(features[j] - means[j]) / stds[j] for j in range(width)]
        for features, _ in records
    ]
    y = [target for _, target in records]
    cur = [1.0] + [(current[j] - means[j]) / stds[j] for j in range(width)]
    return x, y, cur


def _ridge_predict(
    records: list[tuple[list[float], float]],
    current: list[float],
) -> float | None:
    prepared = _standardize_training(records, current)
    if prepared is None:
        return None
    x, y, cur = prepared
    xt = _transpose(x)
    width = len(xt)
    gram = [[0.0 for _ in range(width)] for _ in range(width)]
    rhs = [0.0 for _ in range(width)]
    for i in range(width):
        for j in range(width):
            gram[i][j] = sum(xt[i][k] * x[k][j] for k in range(len(x)))
        rhs[i] = sum(xt[i][k] * y[k] for k in range(len(x)))
    for i in range(1, width):
        gram[i][i] += RIDGE_ALPHA
    beta = _solve(gram, rhs)
    if beta is None:
        return None
    return sum(beta[i] * cur[i] for i in range(width))


def _personnel_features(row: dict[str, Any] | None) -> list[float] | None:
    if not isinstance(row, dict):
        return None
    home = row.get("home") if isinstance(row.get("home"), dict) else {}
    away = row.get("away") if isinstance(row.get("away"), dict) else {}

    values = [
        _num(home.get("previous_xi_overlap_rate")),
        _num(away.get("previous_xi_overlap_rate")),
        _num(home.get("last3_core_return_rate")),
        _num(away.get("last3_core_return_rate")),
        (
            1.0
            if home.get("coach_same_as_previous") is True
            else 0.0
            if home.get("coach_same_as_previous") is False
            else None
        ),
        (
            1.0
            if away.get("coach_same_as_previous") is True
            else 0.0
            if away.get("coach_same_as_previous") is False
            else None
        ),
        _num(home.get("new_starter_count_vs_previous")),
        _num(away.get("new_starter_count_vs_previous")),
    ]
    if any(value is None for value in values):
        return None
    return [float(value) for value in values]


def _metric(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    errors: list[float] = []
    signed: list[float] = []
    for row in rows:
        actual = _num(row.get("actual"))
        pred = _num(row.get(key))
        if actual is None or pred is None:
            continue
        delta = pred - actual
        signed.append(delta)
        errors.append(abs(delta))
    return {
        "n": len(errors),
        "mae": round(sum(errors) / len(errors), 6) if errors else None,
        "rmse": (
            round(math.sqrt(sum(value * value for value in signed) / len(signed)), 6)
            if signed else None
        ),
        "mean_error": round(sum(signed) / len(signed), 6) if signed else None,
    }


def _improves(base: Any, challenger: Any) -> bool:
    base_n = _num(base)
    challenger_n = _num(challenger)
    return (
        base_n is not None
        and challenger_n is not None
        and challenger_n < base_n
    )


def _delta(base: Any, challenger: Any) -> float | None:
    base_n = _num(base)
    challenger_n = _num(challenger)
    if base_n is None or challenger_n is None:
        return None
    return round(challenger_n - base_n, 6)


def build_report(
    source_rows: list[dict[str, Any]],
    personnel_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = [dict(row) for row in source_rows if isinstance(row, dict)]
    rows.sort(key=lambda row: (str(row.get("kickoff_local") or ""), int(row.get("fixture_id") or 0)))
    personnel_by_fixture = {
        int(row.get("fixture_id") or 0): row
        for row in personnel_rows
        if isinstance(row, dict) and int(row.get("fixture_id") or 0)
    }

    target_state: dict[str, dict[str, Any]] = {}
    for target in fm3.TARGETS:
        target_state[target] = {
            "global_home": [],
            "global_away": [],
            "league_home": defaultdict(list),
            "league_away": defaultdict(list),
            "home_attack": defaultdict(list),
            "home_concede": defaultdict(list),
            "away_attack": defaultdict(list),
            "away_concede": defaultdict(list),
            "training_home": [],
            "training_away": [],
            "eval_home": [],
            "eval_away": [],
            "eval_total": [],
            "eligible_fixture_ids": set(),
        }

    feature_complete_rows = 0
    evaluation_rows: list[dict[str, Any]] = []

    for row in rows:
        fixture_id = int(row.get("fixture_id") or 0)
        home_id = int(row.get("home_team_id") or 0)
        away_id = int(row.get("away_team_id") or 0)
        league = str(row.get("league_id") or row.get("league") or "UNKNOWN")
        features = _personnel_features(personnel_by_fixture.get(fixture_id))
        if features is not None:
            feature_complete_rows += 1

        for target, (home_key, away_key) in fm3.TARGETS.items():
            state = target_state[target]
            actual_home = _num(row.get(home_key))
            actual_away = _num(row.get(away_key))
            if actual_home is None or actual_away is None:
                continue

            base_home = fm3._baseline_side(
                league_values=state["league_home"][league],
                global_values=state["global_home"],
                team_attack_values=state["home_attack"][home_id],
                opponent_concession_values=state["away_concede"][away_id],
            )
            base_away = fm3._baseline_side(
                league_values=state["league_away"][league],
                global_values=state["global_away"],
                team_attack_values=state["away_attack"][away_id],
                opponent_concession_values=state["home_concede"][home_id],
            )

            if base_home is not None and base_away is not None and features is not None:
                residual_home = _ridge_predict(state["training_home"], features)
                residual_away = _ridge_predict(state["training_away"], features)
                if residual_home is not None and residual_away is not None:
                    cap = MAX_RESIDUAL_ABS[target]
                    residual_home = max(-cap, min(cap, residual_home))
                    residual_away = max(-cap, min(cap, residual_away))
                    personnel_home = max(0.0, base_home + residual_home)
                    personnel_away = max(0.0, base_away + residual_away)
                    state["eligible_fixture_ids"].add(fixture_id)
                    state["eval_home"].append(
                        {"actual": actual_home, "baseline": base_home, "personnel": personnel_home}
                    )
                    state["eval_away"].append(
                        {"actual": actual_away, "baseline": base_away, "personnel": personnel_away}
                    )
                    state["eval_total"].append(
                        {
                            "actual": actual_home + actual_away,
                            "baseline": base_home + base_away,
                            "personnel": personnel_home + personnel_away,
                        }
                    )
                    evaluation_rows.append(
                        {
                            "fixture_id": fixture_id,
                            "kickoff_local": row.get("kickoff_local"),
                            "target": target,
                            "training_rows_home": len(state["training_home"]),
                            "training_rows_away": len(state["training_away"]),
                            "baseline_home": round(base_home, 6),
                            "baseline_away": round(base_away, 6),
                            "personnel_residual_home": round(residual_home, 6),
                            "personnel_residual_away": round(residual_away, 6),
                            "personnel_home": round(personnel_home, 6),
                            "personnel_away": round(personnel_away, 6),
                            "actual_home": actual_home,
                            "actual_away": actual_away,
                        }
                    )

                # The current outcome is added only after its prediction-time features
                # and baseline have been frozen, so future fixtures may learn from it.
                state["training_home"].append((features, actual_home - base_home))
                state["training_away"].append((features, actual_away - base_away))

            state["global_home"].append(actual_home)
            state["global_away"].append(actual_away)
            state["league_home"][league].append(actual_home)
            state["league_away"][league].append(actual_away)
            state["home_attack"][home_id].append(actual_home)
            state["home_concede"][home_id].append(actual_away)
            state["away_attack"][away_id].append(actual_away)
            state["away_concede"][away_id].append(actual_home)

    targets: dict[str, Any] = {}
    ready_targets: list[str] = []
    blockers: list[str] = []

    for target, state in target_state.items():
        base_home = _metric(state["eval_home"], "baseline")
        base_away = _metric(state["eval_away"], "baseline")
        base_total = _metric(state["eval_total"], "baseline")
        personnel_home = _metric(state["eval_home"], "personnel")
        personnel_away = _metric(state["eval_away"], "personnel")
        personnel_total = _metric(state["eval_total"], "personnel")
        n = len(state["eligible_fixture_ids"])

        target_blockers: list[str] = []
        if n < MIN_REVIEW_N:
            target_blockers.append(f"PERSONNEL_{target}_ABLATION_{n}_LT_{MIN_REVIEW_N}")
        if n > 0:
            if not _improves(base_home.get("mae"), personnel_home.get("mae")):
                target_blockers.append(f"HOME_PERSONNEL_{target}_MAE_NOT_BETTER_THAN_BASELINE")
            if not _improves(base_away.get("mae"), personnel_away.get("mae")):
                target_blockers.append(f"AWAY_PERSONNEL_{target}_MAE_NOT_BETTER_THAN_BASELINE")
            if not _improves(base_total.get("mae"), personnel_total.get("mae")):
                target_blockers.append(f"TOTAL_PERSONNEL_{target}_MAE_NOT_BETTER_THAN_BASELINE")

        ready = not target_blockers
        if ready:
            ready_targets.append(target)
        blockers.extend(target_blockers)
        targets[target] = {
            "status": "OOS_REVIEW_ELIGIBLE" if ready else "RESEARCH_HOLD",
            "eligible_fixtures": n,
            "minimum_review_fixtures": MIN_REVIEW_N,
            "baseline": {
                "home": base_home,
                "away": base_away,
                "total": base_total,
            },
            "personnel_challenger": {
                "home": personnel_home,
                "away": personnel_away,
                "total": personnel_total,
            },
            "improvement": {
                "home_mae_delta": _delta(base_home.get("mae"), personnel_home.get("mae")),
                "away_mae_delta": _delta(base_away.get("mae"), personnel_away.get("mae")),
                "total_mae_delta": _delta(base_total.get("mae"), personnel_total.get("mae")),
                "home_mae_improves": _improves(base_home.get("mae"), personnel_home.get("mae")),
                "away_mae_improves": _improves(base_away.get("mae"), personnel_away.get("mae")),
                "total_mae_improves": _improves(base_total.get("mae"), personnel_total.get("mae")),
            },
            "blockers": target_blockers,
            "production_enabled": False,
            "decision_weight": 0.0,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": "OOS_REVIEW_ELIGIBLE" if ready_targets else "RESEARCH_HOLD",
        "policy": (
            "SPORT_FIRST; STRICT_PRIOR_ONLY_PERSONNEL_FEATURES; "
            "CURRENT_OUTCOME_ADDED_TO_TRAINING_ONLY_AFTER_PREDICTION; "
            "NO_MARKET; NO_INFERRED_PLAYER_ROLES; ZERO_DECISION_WEIGHT"
        ),
        "source_rows": len(rows),
        "personnel_rows": len(personnel_rows),
        "feature_complete_rows": feature_complete_rows,
        "minimum_training_rows": MIN_TRAIN_N,
        "minimum_review_fixtures": MIN_REVIEW_N,
        "feature_names": [
            "home_previous_xi_overlap_rate",
            "away_previous_xi_overlap_rate",
            "home_last3_core_return_rate",
            "away_last3_core_return_rate",
            "home_coach_same_as_previous",
            "away_coach_same_as_previous",
            "home_new_starter_count_vs_previous",
            "away_new_starter_count_vs_previous",
        ],
        "ready_targets": sorted(ready_targets),
        "goals_ready_for_fm5": "GOALS" in ready_targets,
        "targets": targets,
        "blockers": sorted(set(blockers)),
        "odds_consumed": False,
        "market_prices_consumed": False,
        "current_match_outcome_used_as_input": False,
        "inferred_player_roles_used": False,
        "production_enabled": False,
        "decision_weight": 0.0,
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
        "evaluation_rows": evaluation_rows[-1000:],
    }
