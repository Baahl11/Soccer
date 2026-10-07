from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any

from mcp_gateway import analyze_formation_intelligence_v2 as formation_v2

MODEL_VERSION = "FORMATION_MATCHUP_ENGINE_V1.0.0"
MIN_STABLE_MATCHUP_N = 8


def fnum(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def observed_average(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [fnum(row.get(key)) for row in rows]
    observed = [value for value in values if value is not None]
    return {
        "n": len(observed),
        "avg": round(sum(observed) / len(observed), 4) if observed else None,
    }


def observed_rate(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    observed = [row.get(key) for row in rows if row.get(key) is not None]
    return {
        "n": len(observed),
        "rate": round(sum(int(value) for value in observed) / len(observed), 4) if observed else None,
    }


def metric_triplet(rec: dict[str, Any], metric: str) -> tuple[float | None, float | None, float | None]:
    return formation_v2.base.tactical_value(rec, metric)


def build_rows(history_dir: str) -> list[dict[str, Any]]:
    fixtures = formation_v2._enhanced_load_history(history_dir)
    rows: list[dict[str, Any]] = []

    for rec in fixtures.values():
        home_formation, away_formation = formation_v2.base.chosen_formations(rec)
        home_goals, away_goals, home_1h_goals, away_1h_goals = formation_v2.base.final_scores(
            rec.get("result")
        )
        if not home_formation or not away_formation or home_goals is None or away_goals is None:
            continue

        home_2h_goals = (
            home_goals - home_1h_goals if home_1h_goals is not None else None
        )
        away_2h_goals = (
            away_goals - away_1h_goals if away_1h_goals is not None else None
        )

        home_corners, away_corners, total_corners = metric_triplet(rec, "corners")
        home_shots, away_shots, total_shots = metric_triplet(rec, "total_shots")
        home_sot, away_sot, total_sot = metric_triplet(rec, "shots_on_goal")
        home_blocked, away_blocked, _ = metric_triplet(rec, "blocked_shots")
        home_inside, away_inside, _ = metric_triplet(rec, "shots_inside_box")
        home_outside, away_outside, _ = metric_triplet(rec, "shots_outside_box")
        home_possession, away_possession, _ = metric_triplet(rec, "possession")
        home_fouls, away_fouls, total_fouls = metric_triplet(rec, "fouls")
        home_yellow, away_yellow, total_yellow = metric_triplet(rec, "yellow_cards")

        total_goals = home_goals + away_goals
        total_1h_goals = (
            home_1h_goals + away_1h_goals
            if home_1h_goals is not None and away_1h_goals is not None
            else None
        )
        total_2h_goals = (
            home_2h_goals + away_2h_goals
            if home_2h_goals is not None and away_2h_goals is not None
            else None
        )

        rows.append(
            {
                "fixture_id": rec.get("fixture_id"),
                "kickoff_local": rec.get("kickoff_local"),
                "league_id": rec.get("league_id"),
                "league": rec.get("league"),
                "home_team_id": rec.get("home_team_id"),
                "home_team": rec.get("home_team"),
                "away_team_id": rec.get("away_team_id"),
                "away_team": rec.get("away_team"),
                "home_formation": home_formation,
                "away_formation": away_formation,
                "matchup_key": f"{home_formation} vs {away_formation}",
                "home_goals": home_goals,
                "away_goals": away_goals,
                "total_goals": total_goals,
                "home_1h_goals": home_1h_goals,
                "away_1h_goals": away_1h_goals,
                "total_1h_goals": total_1h_goals,
                "home_2h_goals": home_2h_goals,
                "away_2h_goals": away_2h_goals,
                "total_2h_goals": total_2h_goals,
                "btts": int(home_goals > 0 and away_goals > 0),
                "over_2_5": int(total_goals >= 3),
                "home_corners": home_corners,
                "away_corners": away_corners,
                "total_corners": total_corners,
                "home_shots": home_shots,
                "away_shots": away_shots,
                "total_shots": total_shots,
                "home_sot": home_sot,
                "away_sot": away_sot,
                "total_sot": total_sot,
                "home_blocked_shots": home_blocked,
                "away_blocked_shots": away_blocked,
                "home_shots_inside_box": home_inside,
                "away_shots_inside_box": away_inside,
                "home_shots_outside_box": home_outside,
                "away_shots_outside_box": away_outside,
                "home_possession": home_possession,
                "away_possession": away_possession,
                "home_fouls": home_fouls,
                "away_fouls": away_fouls,
                "total_fouls": total_fouls,
                "home_yellow_cards": home_yellow,
                "away_yellow_cards": away_yellow,
                "total_yellow_cards": total_yellow,
            }
        )

    rows.sort(key=lambda row: (row.get("kickoff_local") or "", int(row.get("fixture_id") or 0)))
    return rows


AVERAGE_METRICS = (
    "home_goals",
    "away_goals",
    "total_goals",
    "home_1h_goals",
    "away_1h_goals",
    "total_1h_goals",
    "home_2h_goals",
    "away_2h_goals",
    "total_2h_goals",
    "home_corners",
    "away_corners",
    "total_corners",
    "home_shots",
    "away_shots",
    "total_shots",
    "home_sot",
    "away_sot",
    "total_sot",
    "home_blocked_shots",
    "away_blocked_shots",
    "home_shots_inside_box",
    "away_shots_inside_box",
    "home_shots_outside_box",
    "away_shots_outside_box",
    "home_possession",
    "away_possession",
    "home_fouls",
    "away_fouls",
    "total_fouls",
    "home_yellow_cards",
    "away_yellow_cards",
    "total_yellow_cards",
)


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"n": len(rows)}
    completeness: dict[str, Any] = {}
    for key in AVERAGE_METRICS:
        observed = observed_average(rows, key)
        summary[f"avg_{key}"] = observed["avg"]
        completeness[key] = {
            "observed_n": observed["n"],
            "coverage_pct": round(observed["n"] / len(rows) * 100.0, 2) if rows else 0.0,
        }

    for key in ("btts", "over_2_5"):
        observed = observed_rate(rows, key)
        summary[f"{key}_rate"] = observed["rate"]
        completeness[key] = {
            "observed_n": observed["n"],
            "coverage_pct": round(observed["n"] / len(rows) * 100.0, 2) if rows else 0.0,
        }

    summary["metric_completeness"] = completeness
    return summary


def sample_band(n: int) -> str:
    if n >= 50:
        return "HIGH_SAMPLE_RESEARCH"
    if n >= 20:
        return "MEDIUM_SAMPLE_RESEARCH"
    if n >= MIN_STABLE_MATCHUP_N:
        return "RESEARCH_ELIGIBLE"
    return "LOW_SAMPLE"


def build_report(history_dir: str) -> dict[str, Any]:
    rows = build_rows(history_dir)
    matchup_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    home_formation_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    away_formation_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        matchup_groups[row["matchup_key"]].append(row)
        home_formation_groups[row["home_formation"]].append(row)
        away_formation_groups[row["away_formation"]].append(row)

    matchups = []
    for matchup_key, group in matchup_groups.items():
        matchups.append(
            {
                "matchup_key": matchup_key,
                "sample_band": sample_band(len(group)),
                **summarize_group(group),
            }
        )
    matchups.sort(key=lambda row: (-int(row["n"]), row["matchup_key"]))

    formations_home = [
        {"formation": formation, **summarize_group(group)}
        for formation, group in home_formation_groups.items()
    ]
    formations_home.sort(key=lambda row: (-int(row["n"]), row["formation"]))

    formations_away = [
        {"formation": formation, **summarize_group(group)}
        for formation, group in away_formation_groups.items()
    ]
    formations_away.sort(key=lambda row: (-int(row["n"]), row["formation"]))

    metric_rows = {
        metric: sum(1 for row in rows if row.get(metric) is not None)
        for metric in (
            "total_corners",
            "total_shots",
            "total_sot",
            "home_corners",
            "away_corners",
            "home_shots",
            "away_shots",
            "home_sot",
            "away_sot",
        )
    }

    return {
        "schema_version": "1.0.0",
        "model_version": MODEL_VERSION,
        "status": "RESEARCH_ONLY_FORMATION_MATCHUP_ENGINE",
        "policy": (
            "SPORT_FIRST_MARKET_SECOND; VERIFIED_PREKICKOFF_FORMATIONS_ONLY; "
            "MISSING_METRIC_IS_NOT_ZERO; DESCRIPTIVE_OUTPUT_HAS_ZERO_DECISION_WEIGHT"
        ),
        "production_enabled": False,
        "decision_weight": 0.0,
        "minimum_stable_matchup_n": MIN_STABLE_MATCHUP_N,
        "fixtures_with_verified_formation_pair_and_final": len(rows),
        "unique_matchups": len(matchups),
        "matchups_n_ge_8": sum(1 for row in matchups if int(row["n"]) >= MIN_STABLE_MATCHUP_N),
        "metric_rows": metric_rows,
        "matchups": matchups,
        "formation_role_views": {
            "home": formations_home,
            "away": formations_away,
        },
        "health": {
            "leakage_policy": "PREKICKOFF_OR_AT_KICKOFF_ONLY",
            "odds_consumed": False,
            "production_enabled": False,
            "blockers": [
                "RESEARCH_ONLY",
                "SIDE_SPECIFIC_OOS_CHALLENGER_NOT_YET_PROMOTED",
                "MARKET_VALIDATION_SEPARATE_AND_REQUIRED",
            ],
        },
        "rows": rows[-500:],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = build_report(args.history_dir)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, sort_keys=True)

    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "model_version",
                    "fixtures_with_verified_formation_pair_and_final",
                    "unique_matchups",
                    "matchups_n_ge_8",
                    "metric_rows",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
