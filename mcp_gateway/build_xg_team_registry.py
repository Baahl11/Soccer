from __future__ import annotations

import argparse
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


def _load_rows(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows: list[dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as fh:
            if path.endswith(".jsonl"):
                for line in fh:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row, dict):
                        rows.append(row)
            else:
                payload = json.load(fh)
                source = payload if isinstance(payload, list) else payload.get("fixtures") if isinstance(payload, dict) else []
                rows.extend(row for row in (source or []) if isinstance(row, dict))
    except (OSError, json.JSONDecodeError):
        return []
    return rows


def _window(rows: list[dict[str, Any]], n: int) -> dict[str, Any]:
    sample = rows[-n:]
    if not sample:
        return {
            "n": 0,
            "avg_xg_for": None,
            "avg_xg_against": None,
            "avg_xg_diff": None,
        }
    xgf = [float(r["xg_for"]) for r in sample if _num(r.get("xg_for")) is not None]
    xga = [float(r["xg_against"]) for r in sample if _num(r.get("xg_against")) is not None]
    if not xgf or not xga:
        return {
            "n": 0,
            "avg_xg_for": None,
            "avg_xg_against": None,
            "avg_xg_diff": None,
        }
    used = min(len(xgf), len(xga))
    avg_for = sum(xgf) / len(xgf)
    avg_against = sum(xga) / len(xga)
    return {
        "n": used,
        "avg_xg_for": round(avg_for, 6),
        "avg_xg_against": round(avg_against, 6),
        "avg_xg_diff": round(avg_for - avg_against, 6),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build research-only team xG/xGA registry from verified external observations.")
    ap.add_argument("--observations", default="soccer_edge_state/external/xg_fixture_observations.jsonl")
    ap.add_argument("--output", default="soccer_edge_state/analysis/xg_team_registry.json")
    args = ap.parse_args()

    raw = _load_rows(args.observations)
    fixture_rows: dict[int, dict[str, Any]] = {}
    rejected = 0
    for row in raw:
        fid = row.get("api_fixture_id")
        home_id = row.get("home_team_id")
        away_id = row.get("away_team_id")
        home_xg = _num(row.get("home_xg"))
        away_xg = _num(row.get("away_xg"))
        if fid is None or home_id is None or away_id is None or home_xg is None or away_xg is None:
            rejected += 1
            continue
        try:
            fid_int = int(fid)
        except (TypeError, ValueError):
            rejected += 1
            continue
        fixture_rows[fid_int] = {
            "api_fixture_id": fid_int,
            "kickoff": row.get("kickoff"),
            "home_team_id": home_id,
            "home_team": row.get("home_team"),
            "away_team_id": away_id,
            "away_team": row.get("away_team"),
            "home_xg": home_xg,
            "away_xg": away_xg,
            "source_fixture_id": row.get("source_fixture_id"),
            "source": row.get("source") or "Sportmonks",
            "metric_type_id": int(row.get("metric_type_id") or 5304),
        }

    team_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    team_names: dict[str, Any] = {}
    all_team_xg: list[float] = []
    for fixture in sorted(fixture_rows.values(), key=lambda r: str(r.get("kickoff") or "")):
        home_key = str(fixture["home_team_id"])
        away_key = str(fixture["away_team_id"])
        team_names[home_key] = fixture.get("home_team")
        team_names[away_key] = fixture.get("away_team")
        team_rows[home_key].append({
            "fixture_id": fixture["api_fixture_id"],
            "kickoff": fixture.get("kickoff"),
            "venue_role": "home",
            "xg_for": fixture["home_xg"],
            "xg_against": fixture["away_xg"],
        })
        team_rows[away_key].append({
            "fixture_id": fixture["api_fixture_id"],
            "kickoff": fixture.get("kickoff"),
            "venue_role": "away",
            "xg_for": fixture["away_xg"],
            "xg_against": fixture["home_xg"],
        })
        all_team_xg.extend([fixture["home_xg"], fixture["away_xg"]])

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

    fixture_count = len(fixture_rows)
    ready = fixture_count > 0
    report = {
        "schema_version": "1.0.0",
        "status": "RESEARCH_XG_TEAM_REGISTRY" if ready else "EXTERNAL_XG_DATA_BLOCKED",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "provider": "Sportmonks",
            "metric": "Expected Goals (xG)",
            "metric_type_id": 5304,
            "definition": "pre-shot chance-quality expected goals supplied by external provider",
            "source_fixture_mapping": "explicit API-Football fixture/team IDs in normalized observation rows",
        },
        "fixture_count": fixture_count,
        "team_profile_count": len(profiles),
        "rejected_observations": rejected,
        "global": {
            "team_observations": len(all_team_xg),
            "avg_team_xg": round(sum(all_team_xg) / len(all_team_xg), 6) if all_team_xg else None,
        },
        "profiles": profiles,
        "external_data_blocked": not ready,
        "block_reason": "NO_VERIFIED_EXTERNAL_XG_OBSERVATIONS_IMPORTED" if not ready else None,
        "actionable": False,
        "decision_weight": 0.0,
        "policy": [
            "Only externally supplied xG type 5304 observations mapped to explicit API-Football fixture/team IDs are accepted.",
            "Internal goal lambdas, final goals, shots or scorelines are never relabeled as xG.",
            "xGA is the opponent xG conceded in the same verified fixture, not a separately invented metric.",
            "Registry windows are descriptive research inputs only until walk-forward lift and calibration are demonstrated.",
        ],
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps({
        "status": report["status"],
        "fixture_count": fixture_count,
        "team_profile_count": len(profiles),
        "rejected_observations": rejected,
    }, indent=2))


if __name__ == "__main__":
    main()
