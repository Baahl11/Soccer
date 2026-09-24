from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from statistics import median
from typing import Any, Iterable

from mcp_gateway.analyze_1x2_dixon_coles import auc_discrimination

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_1X2_DRAW_HISTORY_SIGNAL_V4_1.0.0"
SAME_COHORT_WARMUP = 30
SHRINKAGE_PRIOR_GAMES = 20.0


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


def _fixture_records(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    clean = [row for row in rows if isinstance(row, dict)]
    fixtures: dict[int, dict[str, Any]] = {}

    for row in clean:
        try:
            fixture_id = int(row.get("fixture_id"))
        except (TypeError, ValueError):
            continue
        rec = fixtures.setdefault(fixture_id, {
            "fixture_id": fixture_id,
            "kickoff": None,
            "league_id": None,
            "home_team_id": None,
            "away_team_id": None,
            "actual": None,
            "baseline_draw_probability": None,
            "baseline_timestamp": None,
        })
        kickoff = _parse_dt(row.get("kickoff_local"))
        if kickoff is not None:
            rec["kickoff"] = kickoff
        for key in ("league_id", "home_team_id", "away_team_id"):
            value = row.get(key)
            if value is not None:
                try:
                    rec[key] = int(value)
                except (TypeError, ValueError):
                    pass
        outcome = _final_outcome(row.get("result"))
        if outcome:
            rec["actual"] = outcome

    for row in clean:
        try:
            fixture_id = int(row.get("fixture_id"))
        except (TypeError, ValueError):
            continue
        rec = fixtures.get(fixture_id)
        if not rec or rec.get("actual") is None:
            continue
        timestamp = _parse_dt(row.get("generated_at_local"))
        kickoff = _parse_dt(row.get("kickoff_local")) or rec.get("kickoff")
        if timestamp is None or kickoff is None or timestamp >= kickoff:
            continue
        raw = row.get("raw_projection") if isinstance(row.get("raw_projection"), dict) else {}
        draw_probability = _num(raw.get("raw_draw_prob"))
        if draw_probability is None or not 0.0 <= draw_probability <= 1.0:
            continue
        prior_timestamp = rec.get("baseline_timestamp")
        if prior_timestamp is None or timestamp > prior_timestamp:
            rec["baseline_draw_probability"] = draw_probability
            rec["baseline_timestamp"] = timestamp

    return sorted(
        [
            rec for rec in fixtures.values()
            if rec.get("kickoff") is not None and rec.get("actual") is not None
        ],
        key=lambda rec: (rec["kickoff"], rec["fixture_id"]),
    )


def _shrunk(draws: int, games: int, global_rate: float) -> float:
    return (float(draws) + SHRINKAGE_PRIOR_GAMES * global_rate) / (
        float(games) + SHRINKAGE_PRIOR_GAMES
    )


def _binary_metrics(observations: list[tuple[float, int]]) -> dict[str, Any]:
    if not observations:
        return {
            "n": 0,
            "brier": None,
            "log_loss": None,
            "mean_probability": None,
            "observed_rate": None,
        }
    brier = 0.0
    log_loss = 0.0
    mean_probability = 0.0
    positives = 0
    for probability, outcome in observations:
        p = min(max(float(probability), 1e-12), 1.0 - 1e-12)
        y = int(outcome)
        brier += (p - y) ** 2
        log_loss += -(y * math.log(p) + (1 - y) * math.log(1.0 - p))
        mean_probability += p
        positives += y
    n = len(observations)
    return {
        "n": n,
        "brier": round(brier / n, 8),
        "log_loss": round(log_loss / n, 8),
        "mean_probability": round(mean_probability / n, 8),
        "observed_rate": round(positives / n, 8),
    }


def _history_count_summary(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "min": None, "median": None, "max": None, "ge_5": 0, "ge_10": 0, "ge_20": 0}
    return {
        "n": len(values),
        "min": min(values),
        "median": float(median(values)),
        "max": max(values),
        "ge_5": sum(value >= 5 for value in values),
        "ge_10": sum(value >= 10 for value in values),
        "ge_20": sum(value >= 20 for value in values),
    }


def build_report(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    finals = _fixture_records(rows)
    evaluable = [
        row for row in finals
        if row.get("baseline_draw_probability") is not None
        and row.get("league_id") is not None
        and row.get("home_team_id") is not None
        and row.get("away_team_id") is not None
    ]
    cohort = evaluable[SAME_COHORT_WARMUP:]

    observations: dict[str, list[tuple[float, int]]] = {
        "baseline_draw_probability": [],
        "league_shrunk_draw_rate": [],
        "team_shrunk_draw_rate": [],
        "role_team_shrunk_draw_rate": [],
        "league_team_blend": [],
        "league_role_team_blend": [],
    }
    history_counts = {
        "global": [],
        "league": [],
        "home_team_all_venues": [],
        "away_team_all_venues": [],
        "home_team_home": [],
        "away_team_away": [],
    }

    for target in cohort:
        prior = [row for row in finals if row["kickoff"] < target["kickoff"]]
        if not prior:
            continue
        global_draws = sum(row["actual"] == "D" for row in prior)
        global_n = len(prior)
        global_rate = global_draws / global_n

        league_prior = [row for row in prior if row.get("league_id") == target.get("league_id")]
        home_team_prior = [
            row for row in prior
            if target["home_team_id"] in {row.get("home_team_id"), row.get("away_team_id")}
        ]
        away_team_prior = [
            row for row in prior
            if target["away_team_id"] in {row.get("home_team_id"), row.get("away_team_id")}
        ]
        home_role_prior = [
            row for row in prior if row.get("home_team_id") == target.get("home_team_id")
        ]
        away_role_prior = [
            row for row in prior if row.get("away_team_id") == target.get("away_team_id")
        ]

        league_rate = _shrunk(
            sum(row["actual"] == "D" for row in league_prior),
            len(league_prior),
            global_rate,
        )
        home_team_rate = _shrunk(
            sum(row["actual"] == "D" for row in home_team_prior),
            len(home_team_prior),
            global_rate,
        )
        away_team_rate = _shrunk(
            sum(row["actual"] == "D" for row in away_team_prior),
            len(away_team_prior),
            global_rate,
        )
        home_role_rate = _shrunk(
            sum(row["actual"] == "D" for row in home_role_prior),
            len(home_role_prior),
            global_rate,
        )
        away_role_rate = _shrunk(
            sum(row["actual"] == "D" for row in away_role_prior),
            len(away_role_prior),
            global_rate,
        )

        team_rate = (home_team_rate + away_team_rate) / 2.0
        role_team_rate = (home_role_rate + away_role_rate) / 2.0
        league_team_blend = (league_rate + team_rate) / 2.0
        league_role_team_blend = (league_rate + role_team_rate) / 2.0
        actual = 1 if target["actual"] == "D" else 0

        values = {
            "baseline_draw_probability": float(target["baseline_draw_probability"]),
            "league_shrunk_draw_rate": league_rate,
            "team_shrunk_draw_rate": team_rate,
            "role_team_shrunk_draw_rate": role_team_rate,
            "league_team_blend": league_team_blend,
            "league_role_team_blend": league_role_team_blend,
        }
        for name, value in values.items():
            observations[name].append((value, actual))

        history_counts["global"].append(global_n)
        history_counts["league"].append(len(league_prior))
        history_counts["home_team_all_venues"].append(len(home_team_prior))
        history_counts["away_team_all_venues"].append(len(away_team_prior))
        history_counts["home_team_home"].append(len(home_role_prior))
        history_counts["away_team_away"].append(len(away_role_prior))

    baseline_metrics = _binary_metrics(observations["baseline_draw_probability"])
    signals: dict[str, Any] = {}
    for name, rows_for_signal in observations.items():
        discrimination = auc_discrimination(rows_for_signal)
        metrics = _binary_metrics(rows_for_signal)
        quality_better = (
            name != "baseline_draw_probability"
            and metrics.get("brier") is not None
            and baseline_metrics.get("brier") is not None
            and float(metrics["brier"]) < float(baseline_metrics["brier"])
            and float(metrics["log_loss"]) < float(baseline_metrics["log_loss"])
        )
        signals[name] = {
            **discrimination,
            **metrics,
            "brier_delta_vs_baseline": (
                round(float(metrics["brier"]) - float(baseline_metrics["brier"]), 8)
                if metrics.get("brier") is not None and baseline_metrics.get("brier") is not None
                else None
            ),
            "log_loss_delta_vs_baseline": (
                round(float(metrics["log_loss"]) - float(baseline_metrics["log_loss"]), 8)
                if metrics.get("log_loss") is not None and baseline_metrics.get("log_loss") is not None
                else None
            ),
            "quality_better_than_baseline": quality_better,
            "formal_challenger_candidate": (
                name != "baseline_draw_probability"
                and discrimination.get("discrimination_ready") is True
                and quality_better
            ),
        }

    discrimination_candidates = [
        name for name, report in signals.items()
        if name != "baseline_draw_probability"
        and report.get("discrimination_ready") is True
    ]
    formal_candidates = [
        name for name, report in signals.items()
        if report.get("formal_challenger_candidate") is True
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": "DRAW_HISTORY_SIGNAL_AUDIT_COMPLETE",
        "method": "STRICTLY_PRIOR_FINAL_RESULTS_WITH_HIERARCHICAL_SHRINKAGE",
        "anti_leakage": "Every historical draw rate uses only fixtures with kickoff strictly earlier than the target kickoff.",
        "shrinkage_prior_games": SHRINKAGE_PRIOR_GAMES,
        "shrinkage_basis": "Roadmap directional-read sample threshold; fixed before evaluation and not tuned on the target cohort.",
        "finalized_fixtures": len(finals),
        "eligible_baseline_fixtures": len(evaluable),
        "same_cohort_warmup": SAME_COHORT_WARMUP,
        "evaluated_fixtures": len(observations["baseline_draw_probability"]),
        "draws": sum(outcome for _, outcome in observations["baseline_draw_probability"]),
        "non_draws": sum(1 - outcome for _, outcome in observations["baseline_draw_probability"]),
        "history_coverage": {
            name: _history_count_summary(values)
            for name, values in history_counts.items()
        },
        "signals": signals,
        "discrimination_candidates": discrimination_candidates,
        "formal_challenger_candidates": formal_candidates,
        "recommendation": (
            "BUILD_NESTED_WALK_FORWARD_DRAW_CHALLENGER_FROM_HISTORY_SIGNAL"
            if formal_candidates
            else "HISTORY_SIGNAL_DISCRIMINATES_BUT_REQUIRES_PROBABILITY_MODELING"
            if discrimination_candidates
            else "HISTORICAL_DRAW_PROPENSITY_DOES_NOT_CLEAR_DISCRIMINATION_GATE"
        ),
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
        "model_weights_changed": False,
        "canonical_bet_logic_changed": False,
        "notes": [
            "No odds or sportsbook prices are used.",
            "League and team rates are shrunk toward the expanding global draw rate instead of lowering a minimum same-league sample gate.",
            "The fixed 20-game prior prevents tiny league/team histories from creating extreme probabilities.",
            "This audit does not change runtime 1X2 probabilities or promote any market.",
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
    parser = argparse.ArgumentParser(description="Audit historical league/team Draw propensity without market data.")
    parser.add_argument("--ledger", default="soccer_edge_state/analysis/signal_ledger.jsonl")
    parser.add_argument("--output", default="soccer_edge_state/analysis/one_x_two_draw_history_signal.json")
    args = parser.parse_args()

    report = build_report(_load_jsonl(args.ledger))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
