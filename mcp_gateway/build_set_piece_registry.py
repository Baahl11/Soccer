from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if out >= 0 else None
    except (TypeError, ValueError):
        return None


def _stamp(value: Any) -> str:
    return str(value or "")


def _mean(rows: list[float]) -> float | None:
    return sum(rows) / len(rows) if rows else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Build research-only team set-piece component registry.")
    ap.add_argument("--history-dir", default="soccer_edge_state/history")
    ap.add_argument("--output", default="soccer_edge_state/analysis/set_piece_registry.json")
    args = ap.parse_args()

    observations: dict[int, dict[str, Any]] = {}
    for path in sorted(glob.glob(os.path.join(args.history_dir, "*.jsonl"))):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    tick = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    continue
                stamp = _stamp(tick.get("generated_at_utc") or tick.get("generated_at_local"))
                for event in tick.get("events") or []:
                    if not isinstance(event, dict):
                        continue
                    obs = event.get("postgame_set_piece_observation")
                    if not isinstance(obs, dict):
                        continue
                    fid = obs.get("api_fixture_id")
                    if fid is None:
                        continue
                    try:
                        key = int(fid)
                    except (TypeError, ValueError):
                        continue
                    old = observations.get(key)
                    if old is None or stamp >= old.get("_stamp", ""):
                        observations[key] = {**obs, "_stamp": stamp}

    by_team: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for obs in observations.values():
        home_id = obs.get("home_team_id")
        away_id = obs.get("away_team_id")
        if home_id is None or away_id is None:
            continue
        hc, ac = _num(obs.get("home_corners")), _num(obs.get("away_corners"))
        hf, af = _num(obs.get("home_free_kicks")), _num(obs.get("away_free_kicks"))
        kickoff = obs.get("kickoff")
        by_team[str(home_id)].append({
            "kickoff": kickoff,
            "team_id": home_id,
            "team": obs.get("home_team"),
            "corners_for": hc, "corners_against": ac,
            "free_kicks_for": hf, "free_kicks_against": af,
        })
        by_team[str(away_id)].append({
            "kickoff": kickoff,
            "team_id": away_id,
            "team": obs.get("away_team"),
            "corners_for": ac, "corners_against": hc,
            "free_kicks_for": af, "free_kicks_against": hf,
        })

    teams = []
    for rows in by_team.values():
        rows.sort(key=lambda r: _stamp(r.get("kickoff")))
        window = rows[-20:]
        def vals(key: str) -> list[float]:
            return [float(x[key]) for x in window if x.get(key) is not None]
        team_id = window[-1]["team_id"]
        team_name = window[-1].get("team")
        teams.append({
            "team_id": team_id,
            "team": team_name,
            "n": len(window),
            "corner_sample_n": len(vals("corners_for")),
            "free_kick_sample_n": len(vals("free_kicks_for")),
            "avg_corners_for": round(_mean(vals("corners_for")), 4) if vals("corners_for") else None,
            "avg_corners_against": round(_mean(vals("corners_against")), 4) if vals("corners_against") else None,
            "avg_free_kicks_for": round(_mean(vals("free_kicks_for")), 4) if vals("free_kicks_for") else None,
            "avg_free_kicks_against": round(_mean(vals("free_kicks_against")), 4) if vals("free_kicks_against") else None,
        })
    teams.sort(key=lambda r: (-(r.get("n") or 0), str(r.get("team") or "")))

    report = {
        "schema_version": "1.0.0",
        "status": "RESEARCH_ONLY_SET_PIECE_COMPONENT_REGISTRY",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "fixture_observations": len(observations),
        "teams": teams,
        "policy": (
            "CORNERS AND EXPLICIT FREE-KICK COUNTS ARE COMPONENT VOLUME ONLY; "
            "NO SET-PIECE GOAL OR xG CONVERSION IS INFERRED."
        ),
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps({
        "status": report["status"],
        "fixture_observations": report["fixture_observations"],
        "teams": len(teams),
    }, indent=2))


if __name__ == "__main__":
    main()
