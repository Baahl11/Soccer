from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from statistics import mean
from typing import Any, Iterable

from mcp_gateway import soccer_model

MODEL_VERSION = "SOCCER_DYNAMIC_STRENGTH_CHALLENGER_V4_1.0.0"
SCHEMA_VERSION = "1.0.0"
MIN_RECENT_MATCHES = 3


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _round(value: float | None, digits: int = 6) -> float | None:
    return round(value, digits) if value is not None and math.isfinite(value) else None


def _recent_strength_projection(event: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    fixture = event.get("fixture")
    sporting = event.get("sporting")
    raw = event.get("raw_projection")
    if not isinstance(fixture, dict):
        return None, "MISSING_FIXTURE"
    if not isinstance(sporting, dict) or sporting.get("sport_data") != "AVAILABLE":
        return None, "MISSING_VERIFIED_SPORTING_INPUTS"
    if not isinstance(raw, dict) or raw.get("status") != "MODELED_LIMITED":
        return None, "MISSING_BASE_RAW_PROJECTION"

    home_stats = sporting.get("home_stats") or {}
    away_stats = sporting.get("away_stats") or {}
    home_recent = sporting.get("home_recent") or []
    away_recent = sporting.get("away_recent") or []

    h_for = soccer_model._rate(home_stats, "home", "for")
    h_against = soccer_model._rate(home_stats, "home", "against")
    a_for = soccer_model._rate(away_stats, "away", "for")
    a_against = soccer_model._rate(away_stats, "away", "against")
    if any(value is None for value in (h_for, h_against, a_for, a_against)):
        return None, "MISSING_SEASON_STRENGTH_INPUTS"

    h_recent_for, h_recent_against, h_recent_n = soccer_model._recent_rates(
        home_recent,
        fixture.get("home_team_id"),
    )
    a_recent_for, a_recent_against, a_recent_n = soccer_model._recent_rates(
        away_recent,
        fixture.get("away_team_id"),
    )
    if h_recent_n < MIN_RECENT_MATCHES or a_recent_n < MIN_RECENT_MATCHES:
        return None, "INSUFFICIENT_RECENT_SAMPLE"
    if any(
        value is None
        for value in (
            h_recent_for,
            h_recent_against,
            a_recent_for,
            a_recent_against,
        )
    ):
        return None, "MISSING_RECENT_STRENGTH_INPUTS"

    base_home_lambda = _num(raw.get("raw_home_goal_rate"))
    base_away_lambda = _num(raw.get("raw_away_goal_rate"))
    if base_home_lambda is None or base_away_lambda is None:
        return None, "MISSING_BASE_GOAL_RATES"

    season_home_lambda = soccer_model._clamp((h_for + a_against) / 2.0, 0.15, 4.5)
    season_away_lambda = soccer_model._clamp((a_for + h_against) / 2.0, 0.15, 4.5)
    recent_home_lambda = soccer_model._clamp((h_recent_for + a_recent_against) / 2.0, 0.15, 4.5)
    recent_away_lambda = soccer_model._clamp((a_recent_for + h_recent_against) / 2.0, 0.15, 4.5)
    recent_probs = soccer_model._probabilities(recent_home_lambda, recent_away_lambda)

    base_probs = {
        "home_win": _num(raw.get("raw_home_win_prob")),
        "draw": _num(raw.get("raw_draw_prob")),
        "away_win": _num(raw.get("raw_away_win_prob")),
        "btts_yes": _num(raw.get("raw_btts_yes_prob")),
        "over_1_5": _num(raw.get("raw_over_1_5_prob")),
        "over_2_5": _num(raw.get("raw_over_2_5_prob")),
        "over_3_5": _num(raw.get("raw_over_3_5_prob")),
    }
    if any(value is None for value in base_probs.values()):
        return None, "MISSING_BASE_PROBABILITY_VECTOR"

    challenger_probs = {
        key: float(recent_probs[key])
        for key in ("home_win", "draw", "away_win", "btts_yes", "over_1_5", "over_2_5", "over_3_5")
    }
    delta_pp = {
        key: 100.0 * (challenger_probs[key] - float(base_probs[key]))
        for key in challenger_probs
    }

    season_goal_diff = season_home_lambda - season_away_lambda
    recent_goal_diff = recent_home_lambda - recent_away_lambda
    strength_shift = recent_goal_diff - season_goal_diff

    return {
        "fixture_id": fixture.get("fixture_id"),
        "stage": event.get("stage"),
        "status": "RESEARCH_ONLY",
        "variant": "RECENT_ONLY_COUNTERFACTUAL",
        "minimum_recent_matches_per_team": MIN_RECENT_MATCHES,
        "sample": {
            "home_recent_matches": h_recent_n,
            "away_recent_matches": a_recent_n,
        },
        "strength": {
            "season_home_goal_rate": _round(season_home_lambda, 4),
            "season_away_goal_rate": _round(season_away_lambda, 4),
            "base_blended_home_goal_rate": _round(base_home_lambda, 4),
            "base_blended_away_goal_rate": _round(base_away_lambda, 4),
            "recent_home_goal_rate": _round(recent_home_lambda, 4),
            "recent_away_goal_rate": _round(recent_away_lambda, 4),
            "season_goal_diff": _round(season_goal_diff, 4),
            "recent_goal_diff": _round(recent_goal_diff, 4),
            "dynamic_strength_shift": _round(strength_shift, 4),
        },
        "base_probabilities": {key: _round(value) for key, value in base_probs.items()},
        "challenger_probabilities": {key: _round(value) for key, value in challenger_probs.items()},
        "probability_delta_pp": {key: _round(value, 4) for key, value in delta_pp.items()},
        "provider_requests_added": 0,
        "decision_weight": 0.0,
        "production_promotion_allowed": False,
    }, None


def build_report(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    source = [event for event in events if isinstance(event, dict)]
    challengers: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    by_stage: Counter[str] = Counter()

    for event in source:
        if event.get("event_type") != "SOCCER_REFRESH":
            continue
        row, reason = _recent_strength_projection(event)
        if row is None:
            exclusions[reason or "NOT_COMPARABLE"] += 1
            continue
        challengers.append(row)
        by_stage[str(row.get("stage") or "UNKNOWN")] += 1

    probability_keys = (
        "home_win",
        "draw",
        "away_win",
        "btts_yes",
        "over_1_5",
        "over_2_5",
        "over_3_5",
    )
    probability_summary: dict[str, dict[str, Any]] = {}
    for key in probability_keys:
        deltas = [float(row["probability_delta_pp"][key]) for row in challengers]
        probability_summary[key] = {
            "rows": len(deltas),
            "mean_delta_pp": _round(mean(deltas), 4) if deltas else None,
            "mean_absolute_delta_pp": _round(mean(abs(value) for value in deltas), 4) if deltas else None,
            "positive_shift_rate": _round(sum(value > 0 for value in deltas) / len(deltas), 6) if deltas else None,
        }

    shifts = [float(row["strength"]["dynamic_strength_shift"]) for row in challengers]
    status = "RESEARCH_ONLY" if challengers else "NOT_VERIFIED_NO_COMPARABLE_ROWS"
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": status,
        "challenger_definition": (
            "RECENT_ONLY_COUNTERFACTUAL_USING_EXISTING_VERIFIED_RECENT_MATCHES_AND_THE_SAME_GOAL_RATE_POISSON_MATH"
        ),
        "source_events": len(source),
        "comparable_refresh_events": len(challengers),
        "excluded_refresh_events": dict(sorted(exclusions.items())),
        "by_stage": dict(sorted(by_stage.items())),
        "mean_dynamic_strength_shift": _round(mean(shifts), 4) if shifts else None,
        "mean_absolute_dynamic_strength_shift": _round(mean(abs(value) for value in shifts), 4) if shifts else None,
        "probability_summary": probability_summary,
        "challenger_rows": challengers,
        "provider_requests_added": 0,
        "provider_budget_changed": False,
        "decision_weight": 0.0,
        "production_promotion_allowed": False,
        "model_weights_changed": False,
        "thresholds_changed": False,
        "gates_changed": False,
        "canonical_bet_logic_changed": False,
        "strict_close_semantics_changed": False,
        "policy": (
            "DESCRIPTIVE_COUNTERFACTUAL_ONLY; REQUIRE_EXISTING_SPORTING_SNAPSHOT_AND_AT_LEAST_THREE_RECENT_MATCHES_PER_TEAM; "
            "NO_PROVIDER_CALLS; NO_THRESHOLD_OR_GATE_CHANGES; NO_PRODUCTION_DECISION_WEIGHT"
        ),
    }


def _load_source(path: str, key: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        return []
    rows = payload.get(key)
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def run(*, source_json: str, source_key: str, output_path: str) -> dict[str, Any]:
    report = build_report(_load_source(source_json, source_key))
    report["source"] = {"source_json": source_json, "source_key": source_key}
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-json", required=True)
    parser.add_argument("--source-key", default="events")
    parser.add_argument("--output", default="artifacts/v212_dynamic_strength_challenger.json")
    args = parser.parse_args()
    print(json.dumps(run(source_json=args.source_json, source_key=args.source_key, output_path=args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
