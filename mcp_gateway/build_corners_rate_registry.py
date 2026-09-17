from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any

from mcp_gateway import analyze_corners_baseline as corners


def blank() -> dict[str, float | int]:
    return {"n": 0, "home_corners": 0.0, "away_corners": 0.0}


def add(row: dict[str, float | int], home: float, away: float) -> None:
    row["n"] = int(row["n"]) + 1
    row["home_corners"] = float(row["home_corners"]) + home
    row["away_corners"] = float(row["away_corners"]) + away


def materialize(values: dict[Any, dict[str, float | int]]) -> dict[str, dict[str, float | int]]:
    return {str(k): dict(v) for k, v in values.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build compact finalized-history corners registry for live research prediction.")
    ap.add_argument("--history-dir", default="soccer_edge_state/history")
    ap.add_argument("--output", default="soccer_edge_state/analysis/corners_rate_registry.json")
    args = ap.parse_args()

    rows = corners.build_rows(args.history_dir)
    global_row = blank(); leagues: dict[Any, dict[str, float | int]] = defaultdict(blank)
    home_teams: dict[Any, dict[str, float | int]] = defaultdict(blank); away_teams: dict[Any, dict[str, float | int]] = defaultdict(blank)
    for row in rows:
        h = float(row["home_corners"]); a = float(row["away_corners"])
        add(global_row, h, a)
        if row.get("league_id") is not None: add(leagues[row["league_id"]], h, a)
        if row.get("home_team_id") is not None: add(home_teams[row["home_team_id"]], h, a)
        if row.get("away_team_id") is not None: add(away_teams[row["away_team_id"]], h, a)

    output = {
        "schema_version": "1.0.0", "status": "RESEARCH_CORNERS_RATE_REGISTRY",
        "verified_postgame_fixtures": len(rows), "minimum_league_pool": 20, "team_ratio_pseudo_n": 8.0,
        "global": dict(global_row), "leagues": materialize(leagues), "home_teams": materialize(home_teams), "away_teams": materialize(away_teams),
        "policy": "FINALIZED VERIFIED POSTGAME CORNERS ONLY; LIVE MODEL NEVER USES MARKET PRICE TO CREATE CORNERS PROJECTION",
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output,"w",encoding="utf-8") as fh: json.dump(output,fh,ensure_ascii=False,indent=2,sort_keys=True); fh.write("\n")
    print(json.dumps({"status":output["status"],"verified_postgame_fixtures":len(rows)},indent=2))


if __name__ == "__main__": main()
