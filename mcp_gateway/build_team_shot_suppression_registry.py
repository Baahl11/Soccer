from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Any

PRIOR_MATCHES = 8.0
MIN_TEAM_MATCHES_FOR_LIVE_FACTOR = 5
FACTOR_MIN = 0.75
FACTOR_MAX = 1.25


def _num(value: Any) -> float | None:
    try:
        if isinstance(value, str):
            value = value.replace("%", "").strip()
        return float(value)
    except (TypeError, ValueError):
        return None


def _stat_map(team_row: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in team_row.get("statistics") or []:
        if not isinstance(row, dict):
            continue
        key = str(row.get("type") or "").strip().lower()
        value = _num(row.get("value"))
        if key and value is not None:
            out[key] = value
    return out


def _fixture_shots(event: dict[str, Any]) -> list[dict[str, Any]]:
    fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
    rows = event.get("match_stats") if isinstance(event.get("match_stats"), list) else []
    parsed: list[dict[str, Any]] = []
    for team_row in rows:
        if not isinstance(team_row, dict):
            continue
        team = team_row.get("team") if isinstance(team_row.get("team"), dict) else {}
        stats = _stat_map(team_row)
        shots = stats.get("total shots")
        if shots is None:
            continue
        parsed.append({
            "team_id": team.get("id"),
            "team": team.get("name"),
            "shots": shots,
        })
    if len(parsed) != 2:
        return []
    a, b = parsed
    league_id = fixture.get("league_id")
    return [
        {
            **a,
            "opponent_id": b.get("team_id"),
            "shots_allowed": b.get("shots"),
            "fixture_id": fixture.get("fixture_id"),
            "league_id": league_id,
        },
        {
            **b,
            "opponent_id": a.get("team_id"),
            "shots_allowed": a.get("shots"),
            "fixture_id": fixture.get("fixture_id"),
            "league_id": league_id,
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build shrunk team total-shot for/allowed registry from persisted postgame stats.")
    parser.add_argument("--history-dir", default="soccer_edge_state/history")
    parser.add_argument("--output", default="soccer_edge_state/analysis/team_shot_suppression_registry.json")
    args = parser.parse_args()

    seen: dict[int, list[dict[str, Any]]] = {}
    for path in sorted(glob.glob(os.path.join(args.history_dir, "*.jsonl"))):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    tick = json.loads(line)
                except Exception:
                    continue
                for event in tick.get("events") or []:
                    if not isinstance(event, dict) or event.get("stage") != "POSTGAME":
                        continue
                    fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
                    fid = fixture.get("fixture_id")
                    if not fid:
                        continue
                    rows = _fixture_shots(event)
                    if rows:
                        seen[int(fid)] = rows

    all_rows = [row for rows in seen.values() for row in rows]
    global_shots = [float(row["shots"]) for row in all_rows if _num(row.get("shots")) is not None]
    global_allowed = [float(row["shots_allowed"]) for row in all_rows if _num(row.get("shots_allowed")) is not None]
    global_for = sum(global_shots) / len(global_shots) if global_shots else None
    global_against = sum(global_allowed) / len(global_allowed) if global_allowed else None

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        if row.get("team_id") is not None:
            grouped[str(row["team_id"])].append(row)

    teams: dict[str, dict[str, Any]] = {}
    for team_id, rows in grouped.items():
        shots_for = [float(r["shots"]) for r in rows if _num(r.get("shots")) is not None]
        shots_allowed = [float(r["shots_allowed"]) for r in rows if _num(r.get("shots_allowed")) is not None]
        n = min(len(shots_for), len(shots_allowed))
        raw_for = sum(shots_for) / len(shots_for) if shots_for else None
        raw_allowed = sum(shots_allowed) / len(shots_allowed) if shots_allowed else None
        shrunk_for = (
            (sum(shots_for) + PRIOR_MATCHES * global_for) / (len(shots_for) + PRIOR_MATCHES)
            if shots_for and global_for is not None else None
        )
        shrunk_allowed = (
            (sum(shots_allowed) + PRIOR_MATCHES * global_against) / (len(shots_allowed) + PRIOR_MATCHES)
            if shots_allowed and global_against is not None else None
        )
        factor = (
            shrunk_allowed / global_against
            if shrunk_allowed is not None and global_against not in (None, 0)
            else None
        )
        clipped = min(FACTOR_MAX, max(FACTOR_MIN, factor)) if factor is not None else None
        teams[team_id] = {
            "team_id": int(team_id) if team_id.isdigit() else team_id,
            "team": next((r.get("team") for r in reversed(rows) if r.get("team")), None),
            "matches": n,
            "raw_shots_for_per_match": round(raw_for, 4) if raw_for is not None else None,
            "raw_shots_allowed_per_match": round(raw_allowed, 4) if raw_allowed is not None else None,
            "shrunk_shots_for_per_match": round(shrunk_for, 4) if shrunk_for is not None else None,
            "shrunk_shots_allowed_per_match": round(shrunk_allowed, 4) if shrunk_allowed is not None else None,
            "opponent_shot_factor": round(clipped, 6) if clipped is not None else None,
            "factor_status": "ELIGIBLE" if n >= MIN_TEAM_MATCHES_FOR_LIVE_FACTOR else "LOW_SAMPLE_CONTEXT_ONLY",
        }

    report = {
        "schema_version": "1.0.0",
        "status": "RESEARCH_TEAM_SHOT_SUPPRESSION_REGISTRY",
        "unique_postgame_fixtures": len(seen),
        "team_rows": len(all_rows),
        "global_shots_for_per_team_match": round(global_for, 6) if global_for is not None else None,
        "global_shots_allowed_per_team_match": round(global_against, 6) if global_against is not None else None,
        "prior_matches": PRIOR_MATCHES,
        "minimum_team_matches_for_live_factor": MIN_TEAM_MATCHES_FOR_LIVE_FACTOR,
        "factor_clip": [FACTOR_MIN, FACTOR_MAX],
        "teams": teams,
        "policy": [
            "Only persisted finalized fixture statistics are used.",
            "Opponent factor is a team-level total-shots-allowed proxy, not a player-specific defensive matchup model.",
            "Low-sample team factors are context-only and must resolve to multiplier 1.0 live.",
            "Factor clipping limits overreaction while calibration sample is still small.",
            "No betting classification, tier, stake or market comparison is produced here.",
        ],
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps({
        "status": report["status"],
        "unique_postgame_fixtures": report["unique_postgame_fixtures"],
        "team_count": len(teams),
        "global_shots_allowed_per_team_match": report["global_shots_allowed_per_team_match"],
    }, indent=2))


if __name__ == "__main__":
    main()
