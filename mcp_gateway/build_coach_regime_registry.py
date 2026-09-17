from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Any


def parse_dt(value: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")) if value else None
    except ValueError:
        return None


def final_goals(event: dict[str, Any]) -> tuple[int, int] | None:
    result = event.get("result") if isinstance(event.get("result"), dict) else {}
    goals = result.get("goals") if isinstance(result.get("goals"), dict) else {}
    score = result.get("score") if isinstance(result.get("score"), dict) else {}
    ft = score.get("fulltime") if isinstance(score.get("fulltime"), dict) else {}
    fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
    fg = fixture.get("goals") if isinstance(fixture.get("goals"), dict) else {}
    h = goals.get("home", ft.get("home", fg.get("home")))
    a = goals.get("away", ft.get("away", fg.get("away")))
    try:
        return int(h), int(a)
    except (TypeError, ValueError):
        return None


def load(history_dir: str) -> list[dict[str, Any]]:
    fixtures: dict[int, dict[str, Any]] = {}
    for path in sorted(glob.glob(os.path.join(history_dir, "*.jsonl"))):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    tick = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    continue
                stamp = parse_dt(tick.get("generated_at_utc") or tick.get("generated_at_local"))
                for event in tick.get("events") or []:
                    if not isinstance(event, dict):
                        continue
                    fx = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
                    fid = fx.get("fixture_id")
                    ko = parse_dt(fx.get("kickoff"))
                    if fid is None or ko is None:
                        continue
                    rec = fixtures.setdefault(int(fid), {
                        "fixture_id": int(fid), "kickoff": ko,
                        "league_id": fx.get("league_id"), "league": fx.get("league"),
                        "home_team_id": fx.get("home_team_id"), "home_team": fx.get("home_team"),
                        "away_team_id": fx.get("away_team_id"), "away_team": fx.get("away_team"),
                        "coach_obs": [], "result": None,
                    })
                    lineups = event.get("lineups") if isinstance(event.get("lineups"), dict) else {}
                    teams = lineups.get("teams") or []
                    if teams:
                        mapping = {}
                        for row in teams:
                            if not isinstance(row, dict) or row.get("team_id") is None:
                                continue
                            cid = row.get("coach_id")
                            name = str(row.get("coach") or "").strip() or None
                            if cid is not None or name:
                                mapping[int(row["team_id"])] = {"coach_id": cid, "coach": name}
                        if mapping:
                            rec["coach_obs"].append((stamp, mapping))
                    if event.get("stage") == "POSTGAME":
                        fg = final_goals(event)
                        if fg is not None:
                            rec["result"] = fg

    rows = []
    for rec in fixtures.values():
        if rec.get("result") is None or not rec.get("coach_obs"):
            continue
        obs = sorted(rec["coach_obs"], key=lambda x: x[0] or datetime.min)
        mapping = obs[-1][1]
        hid = rec.get("home_team_id"); aid = rec.get("away_team_id")
        if hid is None or aid is None or int(hid) not in mapping or int(aid) not in mapping:
            continue
        rec = dict(rec)
        rec["home_coach"] = mapping[int(hid)]
        rec["away_coach"] = mapping[int(aid)]
        rows.append(rec)
    return sorted(rows, key=lambda r: (r["kickoff"], r["fixture_id"]))


def regime_summary(segment: list[dict[str, Any]], team_id: int, coach: dict[str, Any]) -> dict[str, Any]:
    gf = ga = over25 = btts = home_n = away_n = 0
    for row in segment:
        h, a = row["result"]
        is_home = int(row["home_team_id"]) == team_id
        tgf, tga = (h, a) if is_home else (a, h)
        gf += tgf; ga += tga
        over25 += int(h + a >= 3)
        btts += int(h > 0 and a > 0)
        home_n += int(is_home); away_n += int(not is_home)
    n = len(segment)
    first = segment[0]["kickoff"]; last = segment[-1]["kickoff"]
    return {
        "coach_id": coach.get("coach_id"), "coach": coach.get("coach"),
        "matches": n, "first_match": first.isoformat(), "last_match": last.isoformat(),
        "home_matches": home_n, "away_matches": away_n,
        "goals_for_per_match": round(gf / n, 4) if n else None,
        "goals_against_per_match": round(ga / n, 4) if n else None,
        "over_2_5_rate": round(over25 / n, 4) if n else None,
        "btts_rate": round(btts / n, 4) if n else None,
    }


def same_coach(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if a.get("coach_id") is not None and b.get("coach_id") is not None:
        return str(a.get("coach_id")) == str(b.get("coach_id"))
    return str(a.get("coach") or "").strip().lower() == str(b.get("coach") or "").strip().lower()


def main() -> None:
    ap = argparse.ArgumentParser(description="Build descriptive coach-regime registry from verified lineup coach identities and final results.")
    ap.add_argument("--history-dir", default="soccer_edge_state/history")
    ap.add_argument("--output", default="soccer_edge_state/analysis/coach_regime_registry.json")
    args = ap.parse_args()
    fixtures = load(args.history_dir)
    by_team: dict[int, list[dict[str, Any]]] = defaultdict(list)
    team_names: dict[int, str | None] = {}
    for row in fixtures:
        hid = int(row["home_team_id"]); aid = int(row["away_team_id"])
        by_team[hid].append({**row, "team_coach": row["home_coach"]}); team_names[hid] = row.get("home_team")
        by_team[aid].append({**row, "team_coach": row["away_coach"]}); team_names[aid] = row.get("away_team")

    teams = {}
    for team_id, rows in by_team.items():
        rows.sort(key=lambda r: (r["kickoff"], r["fixture_id"]))
        segments: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        for row in rows:
            if not current or same_coach(current[-1]["team_coach"], row["team_coach"]):
                current.append(row)
            else:
                segments.append(current); current = [row]
        if current:
            segments.append(current)
        summaries = [regime_summary(seg, team_id, seg[-1]["team_coach"]) for seg in segments]
        current_summary = summaries[-1] if summaries else None
        previous_summary = summaries[-2] if len(summaries) >= 2 else None
        delta = None
        if current_summary and previous_summary and current_summary["matches"] >= 5 and previous_summary["matches"] >= 5:
            delta = {
                "goals_for_per_match": round(current_summary["goals_for_per_match"] - previous_summary["goals_for_per_match"], 4),
                "goals_against_per_match": round(current_summary["goals_against_per_match"] - previous_summary["goals_against_per_match"], 4),
                "over_2_5_rate": round(current_summary["over_2_5_rate"] - previous_summary["over_2_5_rate"], 4),
                "btts_rate": round(current_summary["btts_rate"] - previous_summary["btts_rate"], 4),
                "interpretation": "DESCRIPTIVE_DIFFERENCE_NOT_CAUSAL_EFFECT",
            }
        teams[str(team_id)] = {
            "team_id": team_id, "team": team_names.get(team_id),
            "current_regime": current_summary, "previous_regime": previous_summary,
            "current_vs_previous_descriptive_delta": delta,
            "regime_count": len(summaries), "recent_regimes": summaries[-4:],
        }

    out = {
        "schema_version": "1.0.0", "status": "RESEARCH_COACH_REGIME_REGISTRY",
        "fixtures_with_verified_coach_pair_and_final": len(fixtures), "teams": teams,
        "policy": "COACH REGIME METRICS ARE DESCRIPTIVE CONTEXT ONLY; BEFORE/AFTER DIFFERENCES ARE NOT CAUSAL; NO BET UPGRADE WITHOUT OOS FEATURE LIFT",
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2, sort_keys=True); fh.write("\n")
    print(json.dumps({"status": out["status"], "fixtures": len(fixtures), "teams": len(teams)}, indent=2))


if __name__ == "__main__":
    main()
