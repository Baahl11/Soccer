from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Any


def final_scores(event: dict[str, Any]) -> tuple[int, int, int, int] | None:
    result = event.get("result") if isinstance(event.get("result"), dict) else {}
    score = result.get("score") if isinstance(result.get("score"), dict) else {}
    ht = score.get("halftime") if isinstance(score.get("halftime"), dict) else {}
    ft = score.get("fulltime") if isinstance(score.get("fulltime"), dict) else {}
    goals = result.get("goals") if isinstance(result.get("goals"), dict) else {}
    try:
        hh = int(ht.get("home")); ha = int(ht.get("away"))
        fh = int(ft.get("home", goals.get("home"))); fa = int(ft.get("away", goals.get("away")))
    except (TypeError, ValueError):
        return None
    sh = fh - hh; sa = fa - ha
    if min(hh, ha, sh, sa) < 0:
        return None
    return hh, ha, sh, sa


def blank() -> dict[str, float | int]:
    return {"n": 0, "home_for": 0.0, "away_for": 0.0, "home_against": 0.0, "away_against": 0.0}


def add_period(row: dict[str, float | int], home_goals: int, away_goals: int) -> None:
    row["n"] = int(row["n"]) + 1
    row["home_for"] = float(row["home_for"]) + home_goals
    row["away_for"] = float(row["away_for"]) + away_goals
    row["home_against"] = float(row["home_against"]) + away_goals
    row["away_against"] = float(row["away_against"]) + home_goals


def period_container() -> dict[str, Any]:
    return {"global": blank(), "leagues": defaultdict(blank), "home_teams": defaultdict(blank), "away_teams": defaultdict(blank)}


def materialize(container: dict[str, Any]) -> dict[str, Any]:
    return {
        "global": dict(container["global"]),
        "leagues": {str(k): dict(v) for k, v in container["leagues"].items()},
        "home_teams": {str(k): dict(v) for k, v in container["home_teams"].items()},
        "away_teams": {str(k): dict(v) for k, v in container["away_teams"].items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build compact period scoring registry for live 1H/2H research prediction.")
    ap.add_argument("--history-dir", default="soccer_edge_state/history")
    ap.add_argument("--output", default="soccer_edge_state/analysis/period_rate_registry.json")
    args = ap.parse_args()

    one_h = period_container(); two_h = period_container()
    fixtures: dict[int, dict[str, Any]] = {}
    for path in sorted(glob.glob(os.path.join(args.history_dir, "*.jsonl"))):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    tick = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    continue
                for event in tick.get("events") or []:
                    if not isinstance(event, dict) or event.get("stage") != "POSTGAME":
                        continue
                    fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
                    fid = fixture.get("fixture_id")
                    scores = final_scores(event)
                    if fid is None or scores is None:
                        continue
                    fixtures[int(fid)] = {
                        "league_id": fixture.get("league_id"),
                        "home_team_id": fixture.get("home_team_id"),
                        "away_team_id": fixture.get("away_team_id"),
                        "scores": scores,
                    }

    for rec in fixtures.values():
        lid = rec.get("league_id"); hid = rec.get("home_team_id"); aid = rec.get("away_team_id")
        hh, ha, sh, sa = rec["scores"]
        for container, hg, ag in ((one_h, hh, ha), (two_h, sh, sa)):
            add_period(container["global"], hg, ag)
            if lid is not None: add_period(container["leagues"][str(lid)], hg, ag)
            if hid is not None: add_period(container["home_teams"][str(hid)], hg, ag)
            if aid is not None: add_period(container["away_teams"][str(aid)], hg, ag)

    output = {
        "schema_version": "1.0.0",
        "status": "RESEARCH_RATE_REGISTRY",
        "finalized_fixtures": len(fixtures),
        "league_prior_games": 20.0,
        "team_prior_games": 5.0,
        "periods": {"1H": materialize(one_h), "2H": materialize(two_h)},
        "policy": "FINALIZED_FIXTURES_ONLY; LIVE PERIOD MODEL NEVER USES CURRENT FIXTURE RESULT OR MARKET TO CREATE SPORT PROJECTION",
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps({"status": output["status"], "finalized_fixtures": output["finalized_fixtures"]}, indent=2))


if __name__ == "__main__":
    main()
