from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any


def _is_postgame_phase(value: Any) -> bool:
    return str(value or "").upper().startswith("POSTGAME")


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _avg(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [_num(r.get(key)) for r in rows]
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def _sum(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [_num(r.get(key)) for r in rows]
    vals = [v for v in vals if v is not None]
    return round(sum(vals), 3) if vals else None


def _per90(total: float | None, minutes: float | None) -> float | None:
    if total is None or minutes is None or minutes <= 0:
        return None
    return round(total * 90.0 / minutes, 6)


def _hit(rows: list[dict[str, Any]], key: str, threshold: float) -> dict[str, Any]:
    vals = [_num(r.get(key)) for r in rows]
    vals = [v for v in vals if v is not None]
    hits = sum(v >= threshold for v in vals)
    return {"n": len(vals), "hits": hits, "rate": round(hits / len(vals), 4) if vals else None}


def _save_result_proxy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    saves = conceded = 0.0
    n = 0
    for row in rows:
        s = _num(row.get("saves"))
        c = _num(row.get("goals_conceded"))
        if s is None or c is None:
            continue
        denom = s + c
        if denom <= 0:
            continue
        saves += s
        conceded += c
        n += 1
    denom = saves + conceded
    return {
        "n": n,
        "saves": round(saves, 3),
        "goals_conceded": round(conceded, 3),
        "save_result_proxy": round(saves / denom, 6) if denom > 0 else None,
        "definition": "saves/(saves+goals_conceded); NOT PSxG, NOT shot-quality adjusted, NOT guaranteed complete SOT faced",
    }


def _role_block(played: list[dict[str, Any]]) -> dict[str, Any]:
    observed = [r for r in played if isinstance(r.get("substitute"), bool)]
    starters = [r for r in observed if r.get("substitute") is False]
    subs = [r for r in observed if r.get("substitute") is True]
    return {
        "substitute_flag_observed_n": len(observed),
        "starts": len(starters),
        "substitute_appearances": len(subs),
        "start_rate_conditional_on_observed_appearance": round(len(starters) / len(observed), 4) if observed else None,
        "starter_avg_minutes": _avg(starters, "minutes"),
        "substitute_avg_minutes": _avg(subs, "minutes"),
        "minutes_45plus": _hit(played, "minutes", 45),
        "minutes_60plus": _hit(played, "minutes", 60),
        "minutes_75plus": _hit(played, "minutes", 75),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-dir", default="soccer_edge_state/history")
    parser.add_argument("--output", default="soccer_edge_state/analysis/player_trends.json")
    parser.add_argument(
        "--registry-backfill",
        default="soccer_edge_state/analysis/player_trend_registry_backfill.json",
    )
    args = parser.parse_args()

    seen: dict[int, dict[str, Any]] = {}
    for path in sorted(glob.glob(os.path.join(args.history_dir, "*.jsonl"))):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    tick = json.loads(line)
                except Exception:
                    continue
                for event in tick.get("events") or []:
                    if not isinstance(event, dict):
                        continue
                    fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
                    fid = fixture.get("fixture_id")
                    candidates = [
                        event.get("postgame_player_stats"),
                        (event.get("result") or {}).get("player_stats") if isinstance(event.get("result"), dict) else None,
                        event.get("player_trends_research"),
                    ]
                    capture = next(
                        (x for x in candidates if isinstance(x, dict) and x.get("status") == "RESEARCH_ONLY_PLAYER_FIXTURE_STATS"),
                        None,
                    )
                    if not capture or not fid:
                        continue
                    phase = capture.get("capture_phase") or "PREGAME"
                    current = seen.get(int(fid))
                    if current is not None and _is_postgame_phase(current.get("capture_phase")) and not _is_postgame_phase(phase):
                        continue
                    seen[int(fid)] = {
                        "fixture_id": int(fid),
                        "kickoff": fixture.get("kickoff"),
                        "teams": capture.get("teams") or [],
                        "capture_phase": phase,
                    }

    # Registry-only postgame backfills are deliberately kept separate from
    # historical prediction events. They can improve future player profiles,
    # but they must never create retroactive pregame signals, markets, or CLV.
    supplement = _load(args.registry_backfill)
    for capture in supplement.get("captures") or []:
        if not isinstance(capture, dict) or capture.get("fixture_id") is None:
            continue
        teams = capture.get("teams") if isinstance(capture.get("teams"), list) else []
        if not teams:
            continue
        fid = int(capture["fixture_id"])
        phase = capture.get("capture_phase") or "POSTGAME_REGISTRY_BACKFILL"
        current = seen.get(fid)
        if current is not None and _is_postgame_phase(current.get("capture_phase")) and not _is_postgame_phase(phase):
            continue
        seen[fid] = {
            "fixture_id": fid,
            "kickoff": capture.get("kickoff"),
            "teams": teams,
            "capture_phase": phase,
        }

    rows: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    names: dict[Any, Any] = {}
    positions: dict[Any, Any] = {}
    team_ids: dict[Any, set[str]] = defaultdict(set)
    for fixture in sorted(seen.values(), key=lambda x: str(x.get("kickoff") or "")):
        for team in fixture["teams"]:
            if not isinstance(team, dict):
                continue
            for player in team.get("players") or []:
                if not isinstance(player, dict):
                    continue
                pid = player.get("player_id")
                if not pid:
                    continue
                names[pid] = player.get("name")
                if player.get("position"):
                    positions[pid] = player.get("position")
                if team.get("team_id") is not None:
                    team_ids[pid].add(str(team.get("team_id")))
                row = dict(player)
                row["fixture_id"] = fixture["fixture_id"]
                row["kickoff"] = fixture.get("kickoff")
                row["team_id"] = team.get("team_id")
                row["capture_phase"] = fixture.get("capture_phase")
                rows[pid].append(row)

    output_players = []
    for pid, player_rows in rows.items():
        item: dict[str, Any] = {
            "player_id": pid,
            "name": names.get(pid),
            "last_known_position": positions.get(pid),
            "team_ids": sorted(team_ids.get(pid) or []),
            "matches_captured": len(player_rows),
            "postgame_matches_captured": sum(_is_postgame_phase(r.get("capture_phase")) for r in player_rows),
            "windows": {},
        }
        for n in (5, 10, 20):
            sample = player_rows[-n:]
            played = [r for r in sample if (_num(r.get("minutes")) or 0) > 0]
            gk_rows = [
                r for r in played
                if str(r.get("position") or "").upper() in {"G", "GK", "GOALKEEPER"}
                or r.get("saves") is not None
                or r.get("goals_conceded") is not None
            ]
            total_minutes = _sum(played, "minutes")
            shots_total = _sum(played, "shots")
            sot_total = _sum(played, "shots_on_target")
            goals_total = _sum(played, "goals")
            assists_total = _sum(played, "assists")
            yellow_cards_total = _sum(played, "yellow_cards")
            red_cards_total = _sum(played, "red_cards")
            item["windows"][f"last_{n}"] = {
                "n": len(sample),
                "played_n": len(played),
                "avg_minutes": _avg(played, "minutes"),
                "total_minutes": total_minutes,
                "avg_rating": _avg(played, "rating"),
                "avg_shots": _avg(played, "shots"),
                "avg_sot": _avg(played, "shots_on_target"),
                "shots_total": shots_total,
                "sot_total": sot_total,
                "goals_total": goals_total,
                "assists_total": assists_total,
                "yellow_cards_total": yellow_cards_total,
                "red_cards_total": red_cards_total,
                "goals": goals_total if goals_total is not None else 0.0,
                "assists": assists_total if assists_total is not None else 0.0,
                "rates_per90": {
                    "shots": _per90(shots_total, total_minutes),
                    "shots_on_target": _per90(sot_total, total_minutes),
                    "goals": _per90(goals_total, total_minutes),
                    "assists": _per90(assists_total, total_minutes),
                    "yellow_cards": _per90(yellow_cards_total, total_minutes),
                    "red_cards": _per90(red_cards_total, total_minutes),
                },
                "sot_1plus": _hit(played, "shots_on_target", 1),
                "sot_2plus": _hit(played, "shots_on_target", 2),
                "shots_2plus": _hit(played, "shots", 2),
                "yellow_card_1plus": _hit(played, "yellow_cards", 1),
                "role": _role_block(played),
                "goalkeeper": {
                    "gk_matches": len(gk_rows),
                    "avg_saves": _avg(gk_rows, "saves"),
                    "avg_goals_conceded": _avg(gk_rows, "goals_conceded"),
                    "save_result_proxy": _save_result_proxy(gk_rows),
                },
            }
        output_players.append(item)

    report = {
        "schema_version": "1.5.0",
        "status": "RESEARCH_ONLY_PLAYER_TRENDS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision_weight": 0.0,
        "unique_fixtures_captured": len(seen),
        "postgame_fixtures_captured": sum(_is_postgame_phase(x.get("capture_phase")) for x in seen.values()),
        "unique_players": len(output_players),
        "players_with_5plus_matches": sum(x["matches_captured"] >= 5 for x in output_players),
        "goalkeepers_with_5plus_gk_matches": sum(
            int((((x.get("windows") or {}).get("last_20") or {}).get("goalkeeper") or {}).get("gk_matches") or 0) >= 5
            for x in output_players
        ),
        "policy": [
            "Only persisted verified fixture/player captures are used.",
            "Postgame capture is preferred over pregame capture for a duplicate fixture.",
            "Fixture IDs are deduplicated to avoid repeated scheduler snapshots.",
            "Last-5/10/20 and role/minutes rates are descriptive inputs until an explicit probability registry applies shrinkage and OOS validation.",
            "Player discipline retains yellow-card and red-card counts separately; player-card modeling targets yellow cards only.",
            "Goalkeeper save_result_proxy is saves/(saves+goals_conceded); it is not PSxG and not shot-quality adjusted.",
            "No player or goalkeeper trend can change classification, tier, stake, or bet eligibility.",
        ],
        "players": sorted(output_players, key=lambda x: (-x["matches_captured"], str(x["name"]))),
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "players"}, indent=2))


if __name__ == "__main__":
    main()
