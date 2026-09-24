from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from typing import Any, Iterable

from mcp_gateway.analyze_1x2_dixon_coles import auc_discrimination

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_1X2_DRAW_SIGNAL_AUDIT_V4_1.0.0"
SAME_COHORT_WARMUP = 30


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
        fixture_id = row.get("fixture_id")
        outcome = _final_outcome(row.get("result"))
        try:
            fixture_id = int(fixture_id)
        except (TypeError, ValueError):
            continue
        if outcome:
            finals[fixture_id] = outcome

    latest: dict[int, dict[str, Any]] = {}
    for row in clean:
        try:
            fixture_id = int(row.get("fixture_id"))
        except (TypeError, ValueError):
            continue
        if fixture_id not in finals:
            continue
        raw = row.get("raw_projection") if isinstance(row.get("raw_projection"), dict) else {}
        home_lambda = _num(raw.get("raw_home_goal_rate"))
        away_lambda = _num(raw.get("raw_away_goal_rate"))
        probabilities = [
            _num(raw.get("raw_home_win_prob")),
            _num(raw.get("raw_draw_prob")),
            _num(raw.get("raw_away_win_prob")),
        ]
        generated_at = _parse_dt(row.get("generated_at_local"))
        kickoff = _parse_dt(row.get("kickoff_local"))
        if (
            home_lambda is None
            or away_lambda is None
            or any(value is None or value < 0.0 for value in probabilities)
            or generated_at is None
            or kickoff is None
            or generated_at >= kickoff
        ):
            continue
        total = sum(float(value) for value in probabilities)
        if total <= 0:
            continue
        normalized = [float(value) / total for value in probabilities]
        candidate = {
            "fixture_id": fixture_id,
            "timestamp": row.get("generated_at_local"),
            "home_lambda": home_lambda,
            "away_lambda": away_lambda,
            "home_probability": normalized[0],
            "draw_probability": normalized[1],
            "away_probability": normalized[2],
            "actual": finals[fixture_id],
        }
        prior = latest.get(fixture_id)
        if prior is None or str(candidate["timestamp"]) > str(prior["timestamp"]):
            latest[fixture_id] = candidate

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
    for row in cohort:
        outcome = 1 if row.get("actual") == "D" else 0
        for name, score in signal_values(row).items():
            by_signal.setdefault(name, []).append((float(score), outcome))

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
        "recommendation": (
            "BUILD_DRAW_CHALLENGER_FROM_VERIFIED_SIGNAL"
            if ready_signals
            else "CURRENT_LAMBDA_AND_BASE_PROBABILITY_FEATURES_DO_NOT_CLEAR_DRAW_DISCRIMINATION_GATE"
        ),
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
        "model_weights_changed": False,
        "canonical_bet_logic_changed": False,
        "notes": [
            "Uses the same latest pre-kickoff fixture cohort convention as the Dixon-Coles audit and drops the same first 30 fixtures for matched comparison.",
            "Signal direction is authored so larger scores should indicate higher draw propensity.",
            "This audit is diagnostic only and does not fit or select a production model.",
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
