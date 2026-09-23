from __future__ import annotations

import hashlib
import json
import math
from typing import Any

MODEL_VERSION = "LIGHTGBM_GOALS_V4_CHALLENGER_1.0.0"
MIN_TRAIN_ROWS = 50
MIN_OOS_ROWS = 20
MAX_GOALS = 10

EXCLUDED_NON_NUMERIC_FEATURES = {
    "context.city",
    "context.venue",
    "context.data_tier",
    "availability.home_formation",
    "availability.away_formation",
}


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(int(value))
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def feature_columns(rows: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for row in rows:
        features = row.get("features") if isinstance(row.get("features"), dict) else {}
        for key, value in features.items():
            if key in EXCLUDED_NON_NUMERIC_FEATURES:
                continue
            if _num(value) is not None:
                names.add(str(key))
    return sorted(names)


def vectorize(
    row: dict[str, Any],
    columns: list[str],
) -> list[float]:
    features = row.get("features") if isinstance(row.get("features"), dict) else {}
    missing = row.get("feature_missing") if isinstance(row.get("feature_missing"), dict) else {}
    values: list[float] = []
    for key in columns:
        value = _num(features.get(key))
        # LightGBM handles NaN natively; preserve missingness rather than silently
        # imputing a plausible football value.
        values.append(float("nan") if value is None else value)
        values.append(float(bool(missing.get(key, value is None))))
    return values


def expanded_feature_names(columns: list[str]) -> list[str]:
    out: list[str] = []
    for key in columns:
        out.append(key)
        out.append(f"{key}__missing")
    return out


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
    out = []
    for row in rows:
        home, away = _targets(row)
        if home is None or away is None:
            continue
        if row.get("market_fields_included") is not False:
            continue
        out.append(row)
    out.sort(key=lambda row: (
        str(row.get("feature_captured_at") or ""),
        int(row.get("fixture_id") or 0),
    ))
    return out


def _poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * (lam**k) / math.factorial(k)


def probabilities(home_lambda: float, away_lambda: float) -> dict[str, float]:
    home_lambda = max(float(home_lambda), 1e-6)
    away_lambda = max(float(away_lambda), 1e-6)
    home = draw = away = btts = o15 = o25 = o35 = mass = 0.0
    for h in range(MAX_GOALS + 1):
        ph = _poisson_pmf(h, home_lambda)
        for a in range(MAX_GOALS + 1):
            p = ph * _poisson_pmf(a, away_lambda)
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
            o15 += p if total >= 2 else 0.0
            o25 += p if total >= 3 else 0.0
            o35 += p if total >= 4 else 0.0
    mass = mass or 1.0
    return {
        "home_win": home / mass,
        "draw": draw / mass,
        "away_win": away / mass,
        "btts": btts / mass,
        "over_1_5": o15 / mass,
        "over_2_5": o25 / mass,
        "over_3_5": o35 / mass,
    }


def _metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if not predictions:
        return {
            "n": 0,
            "mae_home_goals": None,
            "mae_away_goals": None,
            "mae_total_goals": None,
            "brier_1x2": None,
            "log_loss_1x2": None,
            "brier_btts": None,
            "brier_over_2_5": None,
        }

    mae_h = mae_a = mae_t = brier_1x2 = ll = brier_btts = brier_o25 = 0.0
    for row in predictions:
        hg, ag = row["actual_home"], row["actual_away"]
        ph, pa = row["pred_home"], row["pred_away"]
        probs = row["probs"]
        mae_h += abs(ph - hg)
        mae_a += abs(pa - ag)
        mae_t += abs((ph + pa) - (hg + ag))
        actual = "home_win" if hg > ag else "away_win" if ag > hg else "draw"
        brier_1x2 += sum(
            (probs[key] - (1.0 if key == actual else 0.0)) ** 2
            for key in ("home_win", "draw", "away_win")
        )
        ll += -math.log(max(probs[actual], 1e-15))
        y_btts = int(hg > 0 and ag > 0)
        y_o25 = int(hg + ag >= 3)
        brier_btts += (probs["btts"] - y_btts) ** 2
        brier_o25 += (probs["over_2_5"] - y_o25) ** 2

    n = len(predictions)
    return {
        "n": n,
        "mae_home_goals": round(mae_h / n, 6),
        "mae_away_goals": round(mae_a / n, 6),
        "mae_total_goals": round(mae_t / n, 6),
        "brier_1x2": round(brier_1x2 / n, 6),
        "log_loss_1x2": round(ll / n, 6),
        "brier_btts": round(brier_btts / n, 6),
        "brier_over_2_5": round(brier_o25 / n, 6),
    }


def walk_forward(
    rows: list[dict[str, Any]],
    *,
    min_train_rows: int = MIN_TRAIN_ROWS,
    min_oos_rows: int = MIN_OOS_ROWS,
) -> dict[str, Any]:
    ordered = eligible_rows(rows)
    columns = feature_columns(ordered)
    fingerprint = hashlib.sha256(_canonical_json(columns).encode("utf-8")).hexdigest()

    if len(ordered) < min_train_rows + min_oos_rows:
        return {
            "schema_version": "1.0.0",
            "model_version": MODEL_VERSION,
            "status": "INSUFFICIENT_TRAINING_SAMPLE",
            "eligible_rows": len(ordered),
            "minimum_training_rows": min_train_rows,
            "minimum_oos_rows": min_oos_rows,
            "walk_forward_evaluated": 0,
            "feature_columns": columns,
            "expanded_feature_count": len(columns) * 2,
            "feature_schema_fingerprint": fingerprint,
            "explicit_missingness_indicators": True,
            "silent_imputation_used": False,
            "market_fields_used": False,
            "post_kickoff_features_used": False,
            "lightgbm_imported": False,
            "promotion_allowed": False,
            "promotion_reason": "INSUFFICIENT_REPRODUCIBLE_DATASET_ROWS",
            "metrics": _metrics([]),
        }

    try:
        import lightgbm as lgb
    except Exception as exc:
        return {
            "schema_version": "1.0.0",
            "model_version": MODEL_VERSION,
            "status": "LIGHTGBM_DEPENDENCY_UNAVAILABLE",
            "eligible_rows": len(ordered),
            "minimum_training_rows": min_train_rows,
            "minimum_oos_rows": min_oos_rows,
            "walk_forward_evaluated": 0,
            "feature_columns": columns,
            "expanded_feature_count": len(columns) * 2,
            "feature_schema_fingerprint": fingerprint,
            "explicit_missingness_indicators": True,
            "silent_imputation_used": False,
            "market_fields_used": False,
            "post_kickoff_features_used": False,
            "lightgbm_imported": False,
            "dependency_error": str(exc)[:200],
            "promotion_allowed": False,
            "promotion_reason": "TRAINING_DEPENDENCY_UNAVAILABLE",
            "metrics": _metrics([]),
        }

    predictions: list[dict[str, Any]] = []
    start = max(min_train_rows, 1)
    for index in range(start, len(ordered)):
        if len(predictions) >= max(len(ordered) - start, 0):
            break
        train = ordered[:index]
        current = ordered[index]
        x_train = [vectorize(row, columns) for row in train]
        y_home = [_targets(row)[0] for row in train]
        y_away = [_targets(row)[1] for row in train]

        params = {
            "objective": "poisson",
            "n_estimators": 80,
            "learning_rate": 0.04,
            "num_leaves": 7,
            "max_depth": 3,
            "min_child_samples": 10,
            "reg_lambda": 2.0,
            "verbosity": -1,
            "random_state": 42,
            "n_jobs": 1,
        }
        home_model = lgb.LGBMRegressor(**params)
        away_model = lgb.LGBMRegressor(**params)
        home_model.fit(x_train, y_home)
        away_model.fit(x_train, y_away)

        x_current = [vectorize(current, columns)]
        pred_home = max(float(home_model.predict(x_current)[0]), 1e-6)
        pred_away = max(float(away_model.predict(x_current)[0]), 1e-6)
        actual_home, actual_away = _targets(current)
        assert actual_home is not None and actual_away is not None
        predictions.append({
            "fixture_id": current.get("fixture_id"),
            "pred_home": pred_home,
            "pred_away": pred_away,
            "actual_home": actual_home,
            "actual_away": actual_away,
            "probs": probabilities(pred_home, pred_away),
        })

    metrics = _metrics(predictions)
    return {
        "schema_version": "1.0.0",
        "model_version": MODEL_VERSION,
        "status": "RESEARCH_ONLY",
        "eligible_rows": len(ordered),
        "minimum_training_rows": min_train_rows,
        "minimum_oos_rows": min_oos_rows,
        "walk_forward_evaluated": len(predictions),
        "feature_columns": columns,
        "expanded_feature_count": len(columns) * 2,
        "feature_schema_fingerprint": fingerprint,
        "explicit_missingness_indicators": True,
        "silent_imputation_used": False,
        "market_fields_used": False,
        "post_kickoff_features_used": False,
        "lightgbm_imported": True,
        "metrics": metrics,
        "promotion_allowed": False,
        "promotion_reason": (
            "INSUFFICIENT_OOS_SAMPLE"
            if len(predictions) < 100
            else "REQUIRES_COMPARISON_VS_STATISTICAL_BASELINES_CALIBRATION_AND_MANUAL_REVIEW"
        ),
    }
