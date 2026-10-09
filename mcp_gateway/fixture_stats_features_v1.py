"""Flatten *observed* API-Football fixture statistics into provenance-preserving features.

Missing provider fields do not become zero. Do not reuse these match facts as
pre-match features, xG, or market/BET inputs.
"""
from __future__ import annotations

import math
import re
from typing import Any, Callable

TEAM_METRICS = {
    "Corner Kicks": ("corners", "corners"),
    "Yellow Cards": ("cards", "yellow_cards"),
    "Red Cards": ("cards", "red_cards"),
    "Offsides": ("stats", "offsides"),
    "Fouls": ("stats", "fouls"),
    "Total Shots": ("stats", "total_shots"),
    "Shots on Goal": ("stats", "shots_on_goal"),
    "Shots off Goal": ("stats", "shots_off_goal"),
    "Blocked Shots": ("stats", "blocked_shots"),
    "Shots insidebox": ("stats", "shots_inside_box"),
    "Shots outsidebox": ("stats", "shots_outside_box"),
    "Ball Possession": ("stats", "possession_pct"),
    "Goalkeeper Saves": ("stats", "keeper_saves"),
    "Total passes": ("stats", "passes"),
    "Passes accurate": ("stats", "passes_accurate"),
    "Passes %": ("stats", "pass_accuracy_pct"),
}

PLAYER_FIELDS = {
    "minutes": ("games", "minutes"),
    "position": ("games", "position"),
    "rating": ("games", "rating"),
    "shots": ("shots", "total"),
    "shots_on": ("shots", "on"),
    "goals": ("goals", "total"),
    "assists": ("goals", "assists"),
    "yellow_cards": ("cards", "yellow"),
    "red_cards": ("cards", "red"),
    "key_passes": ("passes", "key"),
    "tackles": ("tackles", "total"),
    "fouls": ("fouls", "committed"),
}

def _numeric(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (float, int)) and math.isfinite(value) and value >= 0:
        return value
    if isinstance(value, str) and re.fullmatch(r"\d+(?:\.\d+)?%?", value.strip()):
        number = float(value.strip().rstrip("%"))
        return int(number) if number.is_integer() else number
    return None


def _side(raw: Any, home_id: Any, away_id: Any) -> str | None:
    if str(raw) == str(home_id): return "home"
    if str(raw) == str(away_id): return "away"
    return None


def as_features(sporting: dict[str, Any], fx: dict[str, Any],
                captured_at: str, feature: Callable[..., dict[str, Any]]) -> dict[str, dict[str, Any]]:
    fields: dict[str, dict[str, Any]] = {}
    home_id, away_id = fx.get("home_team_id"), fx.get("away_team_id")
    if not home_id or not away_id:
        return fields

    def add(key: str, value: Any, source: str):
        if value is None: return
        fields[key] = feature(
            value, source=source, captured_at=captured_at, sample_n=1,
            freshness="AS_CAPTURED_FIXTURE_OBSERVATION"
        )

    for team in (sporting.get("fixture_statistics") or [])[:2]:
        if not isinstance(team, dict): continue
        side = _side((team.get("team") or {}).get("id"), home_id, away_id)
        if not side: continue
        for item in team.get("statistics") or []:
            if not isinstance(item, dict): continue
            mapping = TEAM_METRICS.get(item.get("type"))
            if not mapping: continue
            number = _numeric(item.get("value"))
            if number is None: continue
            prefix, metric = mapping
            add(f"{prefix}.{side}_{metric}", number, "API_FOOTBALL_FIXTURE_STATS")

    for team in (sporting.get("fixture_players") or [])[:2]:
        if not isinstance(team, dict): continue
        side = _side((team.get("team") or {}).get("id"), home_id, away_id)
        if not side: continue
        for row in (team.get("players") or [])[:60]:
            if not isinstance(row, dict): continue
            player = row.get("player") or {}
            if not isinstance(player, dict): continue
            pid = player.get("id")
            if not isinstance(pid, int) or pid <= 0: continue
            prefix = f"players.{side}_{pid}"
            name = player.get("name")
            if isinstance(name, str) and name.strip():
                add(f"{prefix}_name", name[:100], "API_FOOTBALL_FIXTURE_PLAYERS")
            record = next((v for v in row.get("statistics",[]) if isinstance(v,dict)),None)
            if not isinstance(record, dict): continue
            for field, (section, key) in PLAYER_FIELDS.items():
                data = record.get(section)
                value = data.get(key) if isinstance(data,dict) else None
                if field == "position" and isinstance(value,str) and value.strip():
                    add(f"{prefix}_{field}",value[:45],"API_FOOTBALL_FIXTURE_PLAYERS")
                else:
                    number = _numeric(value)
                    if number is not None:
                        add(f"{prefix}_{field}",number,"API_FOOTBALL_FIXTURE_PLAYERS")

    for team in (sporting.get("fixture_lineups") or [])[:2]:
        if not isinstance(team, dict): continue
        side = _side((team.get("team") or {}).get("id"), home_id, away_id)
        if not side: continue
        formation = team.get("formation")
        if isinstance(formation, str) and re.fullmatch(r"[0-9](?:-[0-9]){2,4}",formation):
            add(f"stats.{side}_formation",formation,"API_FOOTBALL_FIXTURE_LINEUPS")
        for starters, rows in ((True,team.get("startXI") or []),
                               (False,team.get("substitutes") or [])):
            for player_row in rows[:30]:
                player = player_row.get("player") if isinstance(player_row,dict) else None
                if not isinstance(player,dict):continue
                pid=player.get("id")
                if not isinstance(pid,int) or pid <= 0:continue
                prefix = f"players.{side}_{pid}"
                name = player.get("name")
                if f"{prefix}_name" not in fields and isinstance(name,str) and name.strip():
                    add(f"{prefix}_name",name[:100],"API_FOOTBALL_FIXTURE_LINEUPS")
                add(f"{prefix}_starter",starters,"API_FOOTBALL_FIXTURE_LINEUPS")

    # Provider fixture events can exist even in leagues with no aggregate match
    # statistics, for example reserve competitions. Do NOT infer card totals.
    for index, row in enumerate((sporting.get("fixture_events") or [])[:80]):
        if not isinstance(row,dict): continue
        side = _side((row.get("team") or {}).get("id"),home_id,away_id)
        if not side:continue
        prefix=f"events.{side}_{index:03d}"
        typ=row.get("type")
        detail=row.get("detail")
        if isinstance(typ,str) and typ.strip():
            add(f"{prefix}_type",typ[:60],"API_FOOTBALL_FIXTURE_EVENTS")
        if isinstance(detail,str) and detail.strip():
            add(f"{prefix}_detail",detail[:110],"API_FOOTBALL_FIXTURE_EVENTS")
        tm=row.get("time") or {}
        elapsed=_numeric(tm.get("elapsed")) if isinstance(tm,dict) else None
        if elapsed is not None: add(f"{prefix}_minute",elapsed,"API_FOOTBALL_FIXTURE_EVENTS")
        player=row.get("player") or {}
        if isinstance(player,dict):
            name=player.get("name")
            if isinstance(name,str) and name.strip():
                add(f"{prefix}_player",name[:100],"API_FOOTBALL_FIXTURE_EVENTS")
    return fields
