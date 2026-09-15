from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any


def parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def final_score(result: Any) -> tuple[int, int] | None:
    if not isinstance(result, dict):
        return None
    goals = result.get("goals") or {}
    ft = (result.get("score") or {}).get("fulltime") or {}
    try:
        return int(goals.get("home", ft.get("home"))), int(goals.get("away", ft.get("away")))
    except (TypeError, ValueError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Build research-only league/season scoring baselines from finalized Soccer Edge history.")
    ap.add_argument("--ledger", default="soccer_edge_state/analysis/signal_ledger.jsonl")
    ap.add_argument("--output", default="soccer_edge_state/analysis/league_strength_baselines.json")
    ap.add_argument("--min-matches", type=int, default=20)
    args = ap.parse_args()

    fixtures: dict[int, dict[str, Any]] = {}
    with open(args.ledger, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            fid = row.get("fixture_id")
            if not fid:
                continue
            fid = int(fid)
            rec = fixtures.setdefault(fid, {"fixture_id": fid})
            for key in ("league_id", "league", "season", "country", "kickoff_local"):
                if row.get(key) is not None:
                    rec[key] = row.get(key)
            score = final_score(row.get("result"))
            if score is not None:
                rec["score"] = score

    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for rec in fixtures.values():
        if rec.get("score") is None or rec.get("league_id") is None or rec.get("season") is None:
            continue
        if parse_dt(rec.get("kickoff_local")) is None:
            continue
        grouped[(int(rec["league_id"]), int(rec["season"]))].append(rec)

    baselines = []
    for (league_id, season), rows in grouped.items():
        rows.sort(key=lambda r: (parse_dt(r["kickoff_local"]), r["fixture_id"]))
        n = len(rows)
        hg = sum(r["score"][0] for r in rows)
        ag = sum(r["score"][1] for r in rows)
        draws = sum(1 for r in rows if r["score"][0] == r["score"][1])
        btts = sum(1 for r in rows if r["score"][0] > 0 and r["score"][1] > 0)
        over25 = sum(1 for r in rows if sum(r["score"]) >= 3)
        baselines.append({
            "league_id": league_id,
            "league": rows[-1].get("league"),
            "country": rows[-1].get("country"),
            "season": season,
            "finalized_matches": n,
            "eligible_for_research_model": n >= args.min_matches,
            "home_goals_per_match": round(hg / n, 6),
            "away_goals_per_match": round(ag / n, 6),
            "total_goals_per_match": round((hg + ag) / n, 6),
            "home_goal_share": round(hg / max(1, hg + ag), 6),
            "draw_rate": round(draws / n, 6),
            "btts_rate": round(btts / n, 6),
            "over_2_5_rate": round(over25 / n, 6),
            "first_kickoff": rows[0].get("kickoff_local"),
            "last_kickoff": rows[-1].get("kickoff_local"),
        })

    baselines.sort(key=lambda x: (-x["finalized_matches"], x["league_id"], x["season"]))
    eligible = [x for x in baselines if x["eligible_for_research_model"]]
    result = {
        "schema_version": "1.0.0",
        "status": "RESEARCH_BASELINE_REGISTRY_NOT_ACTIONABLE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "SOCCER_EDGE_FINALIZED_FIXTURE_LEDGER",
        "minimum_matches": args.min_matches,
        "leakage_policy": "This registry is descriptive. Target-fixture evaluation MUST recompute its baseline using only fixtures with kickoff strictly earlier than the target; never use the final-season aggregate for historical OOS scoring.",
        "finalized_unique_fixtures": sum(len(v) for v in grouped.values()),
        "league_seasons": len(baselines),
        "eligible_league_seasons": len(eligible),
        "baselines": baselines,
        "notes": [
            "No odds, xG, standings or inferred values are used.",
            "Separate home and away goal rates encode observed league home advantage without a hard-coded multiplier.",
            "The registry is safe for current/future research projections only when its data cutoff precedes kickoff.",
            "1X2 remains research-only and this file cannot change BET/LEAN classifications."
        ],
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "baselines"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
