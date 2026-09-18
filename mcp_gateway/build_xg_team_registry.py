from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any


def _num(value: Any) -> float | None:
    try:
        value = float(value)
        return value if value >= 0 else None
    except (TypeError, ValueError):
        return None


def _window(rows: list[dict[str, Any]], n: int) -> dict[str, Any]:
    sample = rows[-n:]
    xgf = [float(r["xg_for"]) for r in sample if _num(r.get("xg_for")) is not None]
    xga = [float(r["xg_against"]) for r in sample if _num(r.get("xg_against")) is not None]
    if not xgf or not xga:
        return {"n": 0, "avg_xg_for": None, "avg_xg_against": None, "avg_xg_diff": None}
    avg_for = sum(xgf) / len(xgf)
    avg_against = sum(xga) / len(xga)
    return {
        "n": min(len(xgf), len(xga)),
        "avg_xg_for": round(avg_for, 6),
        "avg_xg_against": round(avg_against, 6),
        "avg_xg_diff": round(avg_for - avg_against, 6),
    }


def _observations(history_dir: str) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for path in sorted(glob.glob(os.path.join(history_dir, "*.jsonl"))):
        try:
            fh = open(path, encoding="utf-8")
        except OSError:
            continue
        with fh:
            for line in fh:
                try:
                    tick = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for event in tick.get("events") or []:
                    if not isinstance(event, dict):
                        continue
                    obs = event.get("postgame_xg_observation")
                    if not isinstance(obs, dict) or obs.get("status") != "VERIFIED_API_FOOTBALL_XG":
                        continue
                    fid = obs.get("api_fixture_id")
                    home_id = obs.get("home_team_id")
                    away_id = obs.get("away_team_id")
                    home_xg = _num(obs.get("home_xg"))
                    away_xg = _num(obs.get("away_xg"))
                    try:
                        fid = int(fid)
                    except (TypeError, ValueError):
                        continue
                    if home_id is None or away_id is None or home_xg is None or away_xg is None:
                        continue
                    out[fid] = {
                        "api_fixture_id": fid,
                        "kickoff": obs.get("kickoff"),
                        "home_team_id": home_id,
                        "home_team": obs.get("home_team"),
                        "away_team_id": away_id,
                        "away_team": obs.get("away_team"),
                        "home_xg": home_xg,
                        "away_xg": away_xg,
                    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build team xG/xGA history from persisted finalized API-Football xG observations.")
    ap.add_argument("--history-dir", default="soccer_edge_state/history")
    ap.add_argument("--output", default="soccer_edge_state/analysis/xg_team_registry.json")
    args = ap.parse_args()

    fixtures = _observations(args.history_dir)
    team_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    team_names: dict[str, Any] = {}
    all_xg: list[float] = []

    for fixture in sorted(fixtures.values(), key=lambda r: str(r.get("kickoff") or "")):
        hk = str(fixture["home_team_id"])
        ak = str(fixture["away_team_id"])
        team_names[hk] = fixture.get("home_team")
        team_names[ak] = fixture.get("away_team")
        team_rows[hk].append({
            "fixture_id": fixture["api_fixture_id"],
            "kickoff": fixture.get("kickoff"),
            "venue_role": "home",
            "xg_for": fixture["home_xg"],
            "xg_against": fixture["away_xg"],
        })
        team_rows[ak].append({
            "fixture_id": fixture["api_fixture_id"],
            "kickoff": fixture.get("kickoff"),
            "venue_role": "away",
            "xg_for": fixture["away_xg"],
            "xg_against": fixture["home_xg"],
        })
        all_xg.extend([fixture["home_xg"], fixture["away_xg"]])

    profiles: dict[str, Any] = {}
    for team_id, rows in team_rows.items():
        profiles[team_id] = {
            "team_id": team_id,
            "team": team_names.get(team_id),
            "matches": len(rows),
            "windows": {
                "last_5": _window(rows, 5),
                "last_10": _window(rows, 10),
                "last_20": _window(rows, 20),
            },
        }

    count = len(fixtures)
    ready = count > 0
    report = {
        "schema_version": "1.1.0",
        "status": "RESEARCH_API_FOOTBALL_XG_TEAM_REGISTRY" if ready else "API_FOOTBALL_XG_DATA_ACCUMULATING",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "provider": "API-Football v3",
            "endpoint": "/fixtures/statistics",
            "stat_type": "expected_goals",
            "usage": "FINALIZED_FIXTURE_HISTORY_ONLY",
            "same_fixture_pregame_use_allowed": False,
        },
        "fixture_count": count,
        "team_profile_count": len(profiles),
        "global": {
            "team_observations": len(all_xg),
            "avg_team_xg": round(sum(all_xg) / len(all_xg), 6) if all_xg else None,
        },
        "profiles": profiles,
        "data_accumulating": not ready,
        "block_reason": "NO_PERSISTED_VERIFIED_POSTGAME_XG_YET" if not ready else None,
        "actionable": False,
        "decision_weight": 0.0,
        "policy": [
            "Only API-Football expected_goals captured from finalized fixture statistics is accepted as realized xG.",
            "Realized xG from a fixture may be used only as historical data for later fixtures.",
            "Internal goal lambdas, goals, shots and scorelines are never relabeled as xG.",
            "xGA equals opponent verified xG in the same finalized fixture.",
        ],
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps({
        "status": report["status"],
        "fixture_count": count,
        "team_profile_count": len(profiles),
    }, indent=2))


if __name__ == "__main__":
    main()
