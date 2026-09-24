from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from typing import Any, Iterable

from mcp_gateway.analyze_1x2_dixon_coles import auc_discrimination

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_1X2_DRAW_SIGNAL_AUDIT_V4_1.1.0"
SAME_COHORT_WARMUP = 30
SAFE_FEATURE_MIN_COVERAGE = 0.80


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _final_outcome(result: dict[str, Any] | None) -> str | None:
    if not isinstance(result, dict):
        return None
    goals = result.get("goals") or {}
    score = result.get("score") or {}
    fulltime = score.get("fulltime") or {}
    home = goals.get("home", fulltime.get("home"))
    away = goals.get("away", fulltime.get("away"))
    try:
        home, away = int(home), int(away)
    except (TypeError, ValueError):
        return None
    return "H" if home > away else "A" if away > home else "D"


def extract_same_cohort(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    clean = [row for row in rows if isinstance(row, dict)]
    finals: dict[int, str] = {}
    for row in clean:
        try:
            fixture_id = int(row.get("fixture_id"))
        except (TypeError, ValueError):
            continue
        outcome = _final_outcome(row.get("result"))
        if outcome:
            finals[fixture_id] = outcome

    grouped: dict[int, list[tuple[datetime, dict[str, Any]]]] = {}
    for row in clean:
        try:
            fixture_id = int(row.get("fixture_id"))
        except (TypeError, ValueError):
            continue
        if fixture_id not in finals:
            continue
        generated_at = _parse_dt(row.get("generated_at_local"))
        kickoff = _parse_dt(row.get("kickoff_local"))
        if generated_at is None or kickoff is None or generated_at >= kickoff:
            continue
        grouped.setdefault(fixture_id, []).append((generated_at, row))

    latest: dict[int, dict[str, Any]] = {}
    for fixture_id, fixture_rows in grouped.items():
        fixture_rows.sort(key=lambda item: item[0])
        raw_state: dict[str, Any] | None = None
        safe_feature_state: dict[str, float] = {}
        latest_timestamp: datetime | None = None

        for generated_at, row in fixture_rows:
            latest_timestamp = generated_at
            safe_feature_state.update(_source_safe_feature_values(row))

            raw = row.get("raw_projection") if isinstance(row.get("raw_projection"), dict) else {}
            home_lambda = _num(raw.get("raw_home_goal_rate"))
            away_lambda = _num(raw.get("raw_away_goal_rate"))
            probabilities = [
                _num(raw.get("raw_home_win_prob")),
                _num(raw.get("raw_draw_prob")),
                _num(raw.get("raw_away_win_prob")),
            ]
            if (
                home_lambda is None
                or away_lambda is None
                or any(value is None or value < 0.0 for value in probabilities)
            ):
                continue
            total = sum(float(value) for value in probabilities)
            if total <= 0:
                continue
            normalized = [float(value) / total for value in probabilities]
            raw_state = {
                "home_lambda": home_lambda,
                "away_lambda": away_lambda,
                "home_probability": normalized[0],
                "draw_probability": normalized[1],
                "away_probability": normalized[2],
            }

        if raw_state is None or latest_timestamp is None:
            continue
        latest[fixture_id] = {
            "fixture_id": fixture_id,
            "timestamp": latest_timestamp.isoformat(),
            **raw_state,
            "safe_feature_values": dict(safe_feature_state),
            "actual": finals[fixture_id],
        }

    ordered = sorted(latest.values(), key=lambda row: str(row["timestamp"]))
    return ordered[SAME_COHORT_WARMUP:], len(ordered)

def _entropy(probabilities: tuple[float, float, float]) -> float:
    return -sum(value * math.log(max(value, 1e-15)) for value in probabilities)


def signal_values(row: dict[str, Any]) -> dict[str, float]:
    home_lambda = float(row["home_lambda"])
    away_lambda = float(row["away_lambda"])
    home_probability = float(row["home_probability"])
    draw_probability = float(row["draw_probability"])
    away_probability = float(row["away_probability"])
    return {
        "baseline_draw_probability": draw_probability,
        "lambda_closeness": -abs(home_lambda - away_lambda),
        "low_total_lambda": -(home_lambda + away_lambda),
        "home_away_probability_balance": -abs(home_probability - away_probability),
        "weak_favorite": -max(home_probability, away_probability),
        "one_x_two_entropy": _entropy((home_probability, draw_probability, away_probability)),
        "draw_relative_to_favorite": draw_probability - max(home_probability, away_probability),
    }


SAFE_FEATURE_EXCLUDED_TOKENS = {
    "price",
    "odds",
    "market",
    "bookmaker",
    "result",
    "outcome",
    "final",
    "stake",
    "profit",
    "roi",
}


def _flatten_numeric(prefix: str, value: Any, out: dict[str, float]) -> None:
    if isinstance(value, bool):
        return
    numeric = _num(value)
    if numeric is not None and not isinstance(value, (dict, list, tuple)):
        path_lower = prefix.lower()
        if not any(token in path_lower for token in SAFE_FEATURE_EXCLUDED_TOKENS):
            out[prefix] = numeric
        return
    if not isinstance(value, dict):
        return
    for key, nested in value.items():
        child = f"{prefix}.{key}" if prefix else str(key)
        _flatten_numeric(child, nested, out)


def _source_safe_feature_values(row: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    availability = _num(row.get("availability_confidence"))
    if availability is not None:
        out["availability_confidence"] = availability

    raw = row.get("raw_projection") if isinstance(row.get("raw_projection"), dict) else {}
    namespaces = {
        "raw_projection.sample": raw.get("sample"),
        "raw_projection.screen_scores": raw.get("screen_scores"),
        "raw_projection.relative_strength_shadow": raw.get("relative_strength_shadow"),
        "sporting_shortlist": row.get("sporting_shortlist"),
        "sporting_screen_initial": row.get("sporting_screen_initial"),
        "sporting_screen_refined": row.get("sporting_screen_refined"),
    }
    for namespace, value in namespaces.items():
        if isinstance(value, dict):
            _flatten_numeric(namespace, value, out)
    return out


def safe_persisted_feature_values(row: dict[str, Any]) -> dict[str, float]:
    values = row.get("safe_feature_values")
    if isinstance(values, dict):
        return {
            str(key): float(value)
            for key, value in values.items()
            if _num(value) is not None
        }
    return _source_safe_feature_values(row)


def _two_sided_feature_discrimination(observations: list[tuple[float, int]]) -> dict[str, Any]:
    report = auc_discrimination(observations)
    auc = _num(report.get("auc"))
    standard_error = _num(report.get("auc_standard_error"))
    lower = _num(report.get("auc_lower_95"))
    upper = (
        min(1.0, auc + 1.96 * standard_error)
        if auc is not None and standard_error is not None
        else None
    )
    if lower is not None and lower > 0.50:
        direction = "HIGHER_VALUE_MORE_DRAW"
        oriented_auc = auc
        oriented_lower = lower
    elif upper is not None and upper < 0.50:
        direction = "LOWER_VALUE_MORE_DRAW"
        oriented_auc = 1.0 - auc if auc is not None else None
        oriented_lower = 1.0 - upper
    else:
        direction = "NO_STABLE_DIRECTION"
        oriented_auc = max(auc, 1.0 - auc) if auc is not None else None
        oriented_lower = None
    return {
        **report,
        "auc_upper_95": round(upper, 8) if upper is not None else None,
        "two_sided_direction": direction,
        "ci_excludes_random": direction != "NO_STABLE_DIRECTION",
        "oriented_auc": round(oriented_auc, 8) if oriented_auc is not None else None,
        "oriented_auc_lower_95": round(oriented_lower, 8) if oriented_lower is not None else None,
    }


def _quantile_bins(observations: list[tuple[float, int]], *, bins: int = 5) -> list[dict[str, Any]]:
    if not observations:
        return []
    ordered = sorted(observations, key=lambda item: item[0])
    n = len(ordered)
    out: list[dict[str, Any]] = []
    for index in range(bins):
        start = (index * n) // bins
        end = ((index + 1) * n) // bins
        group = ordered[start:end]
        if not group:
            continue
        draws = sum(outcome for _, outcome in group)
        scores = [score for score, _ in group]
        out.append({
            "bin": index + 1,
            "n": len(group),
            "draws": draws,
            "draw_rate": round(draws / len(group), 6),
            "score_min": round(min(scores), 8),
            "score_max": round(max(scores), 8),
            "score_mean": round(sum(scores) / len(scores), 8),
        })
    return out


def build_report(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    cohort, eligible_fixtures = extract_same_cohort(rows)
    by_signal: dict[str, list[tuple[float, int]]] = {}
    by_persisted_feature: dict[str, list[tuple[float, int]]] = {}
    for row in cohort:
        outcome = 1 if row.get("actual") == "D" else 0
        for name, score in signal_values(row).items():
            by_signal.setdefault(name, []).append((float(score), outcome))
        for name, value in safe_persisted_feature_values(row).items():
            by_persisted_feature.setdefault(name, []).append((float(value), outcome))

    signals: dict[str, Any] = {}
    for name, observations in sorted(by_signal.items()):
        discrimination = auc_discrimination(observations)
        signals[name] = {
            **discrimination,
            "quintiles": _quantile_bins(observations),
        }

    nonbaseline = {
        name: report
        for name, report in signals.items()
        if name != "baseline_draw_probability"
    }
    ranked = sorted(
        nonbaseline.items(),
        key=lambda item: (
            float(item[1].get("auc_lower_95") or -1.0),
            float(item[1].get("auc") or -1.0),
        ),
        reverse=True,
    )
    strongest_name = ranked[0][0] if ranked else None
    strongest = ranked[0][1] if ranked else {}
    ready_signals = [
        name
        for name, report in ranked
        if report.get("discrimination_ready") is True
    ]

    persisted_features: dict[str, Any] = {}
    minimum_feature_rows = math.ceil(len(cohort) * SAFE_FEATURE_MIN_COVERAGE) if cohort else 0
    for name, observations in sorted(by_persisted_feature.items()):
        discrimination = _two_sided_feature_discrimination(observations)
        coverage = len(observations) / len(cohort) if cohort else 0.0
        persisted_features[name] = {
            **discrimination,
            "rows": len(observations),
            "coverage": round(coverage, 6),
            "coverage_gate": SAFE_FEATURE_MIN_COVERAGE,
            "coverage_ready": len(observations) >= minimum_feature_rows,
            "quintiles": _quantile_bins(observations),
        }

    persisted_ranked = sorted(
        persisted_features.items(),
        key=lambda item: (
            float(item[1].get("oriented_auc_lower_95") or -1.0),
            float(item[1].get("oriented_auc") or -1.0),
            float(item[1].get("coverage") or 0.0),
        ),
        reverse=True,
    )
    persisted_candidates = [
        name
        for name, report in persisted_ranked
        if report.get("coverage_ready") is True
        and report.get("ci_excludes_random") is True
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": "DRAW_SIGNAL_AUDIT_COMPLETE",
        "method": "SAME_COHORT_PREKICKOFF_FEATURE_DISCRIMINATION",
        "eligible_fixtures_before_warmup": eligible_fixtures,
        "same_cohort_warmup": SAME_COHORT_WARMUP,
        "evaluated_fixtures": len(cohort),
        "draws": sum(1 for row in cohort if row.get("actual") == "D"),
        "non_draws": sum(1 for row in cohort if row.get("actual") != "D"),
        "gate": "AUC_LOWER_95_GT_0_50",
        "signals": signals,
        "strongest_nonbaseline_signal": {
            "name": strongest_name,
            "auc": strongest.get("auc"),
            "auc_lower_95": strongest.get("auc_lower_95"),
            "discrimination_ready": strongest.get("discrimination_ready") is True,
        },
        "ready_nonbaseline_signals": ready_signals,
        "persisted_safe_feature_audit": {
            "minimum_coverage": SAFE_FEATURE_MIN_COVERAGE,
            "minimum_rows": minimum_feature_rows,
            "features": persisted_features,
            "candidate_features": persisted_candidates,
            "top_features": [
                {
                    "name": name,
                    "rows": report.get("rows"),
                    "coverage": report.get("coverage"),
                    "auc": report.get("auc"),
                    "auc_lower_95": report.get("auc_lower_95"),
                    "auc_upper_95": report.get("auc_upper_95"),
                    "two_sided_direction": report.get("two_sided_direction"),
                    "oriented_auc": report.get("oriented_auc"),
                    "oriented_auc_lower_95": report.get("oriented_auc_lower_95"),
                    "ci_excludes_random": report.get("ci_excludes_random"),
                }
                for name, report in persisted_ranked[:15]
            ],
            "exploratory_multiple_testing_warning": True,
        },
        "recommendation": (
            "BUILD_WALK_FORWARD_DRAW_CHALLENGER_FROM_PERSISTED_FEATURE_CANDIDATES"
            if persisted_candidates
            else "BUILD_DRAW_CHALLENGER_FROM_VERIFIED_SIGNAL"
            if ready_signals
            else "CURRENT_PERSISTED_SAFE_FEATURES_DO_NOT_SHOW_STABLE_UNIVARIATE_DRAW_SIGNAL"
        ),
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
        "model_weights_changed": False,
        "canonical_bet_logic_changed": False,
        "notes": [
            "Uses the same pre-kickoff fixture cohort as the Dixon-Coles audit and drops the same first 30 fixtures for matched comparison.",
            "Safe context uses point-in-time carry-forward: for each fixture, the last known approved pre-kickoff feature value is carried to the latest pre-kickoff snapshot, never across kickoff.",
            "Signal direction is authored so larger scores should indicate higher draw propensity.",
            "This audit is diagnostic only and does not fit or select a production model.",
            "Persisted feature screening is restricted to approved pre-kickoff non-market namespaces and excludes price/odds/market/result paths.",
            "Persisted feature candidates are exploratory because multiple features are screened on the same cohort; any candidate must be re-tested inside a nested walk-forward challenger before it can count as evidence.",
            "A signal clearing the conservative AUC lower-95 > 0.50 gate only justifies building a walk-forward challenger; it does not justify runtime promotion.",
        ],
    }


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit same-cohort pre-kickoff signals for Draw discrimination.")
    parser.add_argument("--ledger", default="soccer_edge_state/analysis/signal_ledger.jsonl")
    parser.add_argument("--output", default="soccer_edge_state/analysis/one_x_two_draw_signal_audit.json")
    args = parser.parse_args()

    report = build_report(_load_jsonl(args.ledger))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
