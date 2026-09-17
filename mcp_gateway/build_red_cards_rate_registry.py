from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any

from mcp_gateway import analyze_cards_baseline as cards


def blank_match() -> dict[str, float | int]:
    return {"n": 0, "home_red_event": 0.0, "away_red_event": 0.0, "any_red_event": 0.0, "total_red": 0.0}


def add_match(row: dict[str, float | int], home_red: float, away_red: float) -> None:
    row["n"] = int(row["n"]) + 1
    row["home_red_event"] = float(row["home_red_event"]) + (1.0 if home_red > 0 else 0.0)
    row["away_red_event"] = float(row["away_red_event"]) + (1.0 if away_red > 0 else 0.0)
    row["any_red_event"] = float(row["any_red_event"]) + (1.0 if home_red + away_red > 0 else 0.0)
    row["total_red"] = float(row["total_red"]) + home_red + away_red


def blank_team() -> dict[str, float | int]:
    return {"n": 0, "own_red_event": 0.0, "opponent_red_event": 0.0}


def add_team(row: dict[str, float | int], own_red: float, opponent_red: float) -> None:
    row["n"] = int(row["n"]) + 1
    row["own_red_event"] = float(row["own_red_event"]) + (1.0 if own_red > 0 else 0.0)
    row["opponent_red_event"] = float(row["opponent_red_event"]) + (1.0 if opponent_red > 0 else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build finalized-history red-card registry for low-frequency live research.")
    ap.add_argument("--history-dir", default="soccer_edge_state/history")
    ap.add_argument("--output", default="soccer_edge_state/analysis/red_cards_rate_registry.json")
    args = ap.parse_args()

    rows = cards.load_rows(args.history_dir)
    global_row = blank_match()
    leagues: dict[str, dict[str, float | int]] = defaultdict(blank_match)
    home_teams: dict[str, dict[str, float | int]] = defaultdict(blank_team)
    away_teams: dict[str, dict[str, float | int]] = defaultdict(blank_team)
    referees: dict[str, dict[str, float | int]] = defaultdict(lambda: {"n": 0, "any_red_event": 0.0, "total_red": 0.0})

    for r in rows:
        hr = float(r.get("home_red") or 0.0)
        ar = float(r.get("away_red") or 0.0)
        add_match(global_row, hr, ar)
        if r.get("league_id") is not None:
            add_match(leagues[str(r["league_id"])], hr, ar)
        add_team(home_teams[str(r["home_team_id"])], hr, ar)
        add_team(away_teams[str(r["away_team_id"])], ar, hr)
        ref = str(r.get("referee") or "").strip()
        if ref:
            referees[ref]["n"] = int(referees[ref]["n"]) + 1
            referees[ref]["any_red_event"] = float(referees[ref]["any_red_event"]) + (1.0 if hr + ar > 0 else 0.0)
            referees[ref]["total_red"] = float(referees[ref]["total_red"]) + hr + ar

    out = {
        "schema_version": "1.0.0",
        "status": "RESEARCH_RED_CARD_RATE_REGISTRY",
        "verified_postgame_fixtures": len(rows),
        "league_pseudo_n": 80.0,
        "team_pseudo_n": 24.0,
        "referee_pseudo_n": 30.0,
        "minimum_referee_n": 20,
        "global": dict(global_row),
        "leagues": dict(leagues),
        "home_teams": dict(home_teams),
        "away_teams": dict(away_teams),
        "referees": dict(referees),
        "policy": "RED CARDS ONLY; MATCH-ANY-RED YES/NO V1; STRONG SHRINKAGE FOR LOW-FREQUENCY OUTCOME; YELLOW CARD COUNTS EXCLUDED",
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps({"status": out["status"], "fixtures": len(rows), "referees": len(referees)}, indent=2))


if __name__ == "__main__":
    main()
