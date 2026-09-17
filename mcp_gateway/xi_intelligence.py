from __future__ import annotations

from datetime import datetime, timedelta, timezone as dt_timezone
from typing import Any

from mcp_gateway import automation as base

SCHEMA_VERSION = "1.0.0"
CACHE_TTL = timedelta(hours=12)


def _team_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    starters = [p for p in (row.get("starters") or []) if isinstance(p, dict)]
    starter_ids = sorted(str(p.get("id")) for p in starters if p.get("id") is not None)
    gks = [p for p in (row.get("goalkeepers") or []) if isinstance(p, dict)]
    gk_ids = sorted(str(p.get("id")) for p in gks if p.get("id") is not None)
    return {
        "team_id": row.get("team_id"),
        "team": row.get("team"),
        "formation": row.get("formation"),
        "coach_id": row.get("coach_id"),
        "coach": row.get("coach"),
        "starter_ids": starter_ids,
        "starter_count": len(starter_ids),
        "goalkeeper_ids": gk_ids,
    }


def _snapshot(lineups: dict[str, Any]) -> dict[str, Any]:
    teams = [_team_snapshot(row) for row in (lineups.get("teams") or []) if isinstance(row, dict)]
    teams.sort(key=lambda r: str(r.get("team_id")))
    fingerprint_parts = []
    for row in teams:
        fingerprint_parts.append(
            f"{row.get('team_id')}|{row.get('formation')}|{row.get('coach_id')}|"
            f"{','.join(row.get('starter_ids') or [])}|{','.join(row.get('goalkeeper_ids') or [])}"
        )
    return {
        "teams": teams,
        "fingerprint": "||".join(fingerprint_parts),
        "both_xi_confirmed": bool(lineups.get("both_xi_confirmed")),
        "both_goalkeepers_confirmed": bool(lineups.get("both_goalkeepers_confirmed")),
        "lineup_state": lineups.get("lineup_state"),
    }


def _diff(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    prev_by = {str(r.get("team_id")): r for r in previous.get("teams") or [] if isinstance(r, dict)}
    cur_by = {str(r.get("team_id")): r for r in current.get("teams") or [] if isinstance(r, dict)}
    changes = []
    for team_id in sorted(set(prev_by) | set(cur_by)):
        p = prev_by.get(team_id, {}); c = cur_by.get(team_id, {})
        pstar = set(p.get("starter_ids") or []); cstar = set(c.get("starter_ids") or [])
        added = sorted(cstar - pstar); removed = sorted(pstar - cstar)
        formation_changed = p.get("formation") != c.get("formation")
        goalkeeper_changed = set(p.get("goalkeeper_ids") or []) != set(c.get("goalkeeper_ids") or [])
        coach_changed = p.get("coach_id") != c.get("coach_id") or p.get("coach") != c.get("coach")
        if added or removed or formation_changed or goalkeeper_changed or coach_changed:
            changes.append({
                "team_id": c.get("team_id", p.get("team_id")),
                "team": c.get("team", p.get("team")),
                "starters_added": added,
                "starters_removed": removed,
                "formation_changed": formation_changed,
                "previous_formation": p.get("formation"),
                "current_formation": c.get("formation"),
                "goalkeeper_changed": goalkeeper_changed,
                "previous_goalkeeper_ids": p.get("goalkeeper_ids") or [],
                "current_goalkeeper_ids": c.get("goalkeeper_ids") or [],
                "coach_changed": coach_changed,
            })
    return {"changed": bool(changes), "changes": changes}


def build(event: dict[str, Any], now: datetime) -> dict[str, Any]:
    fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
    lineups = event.get("lineups") if isinstance(event.get("lineups"), dict) else None
    fid = fixture.get("fixture_id")
    if fid is None or not isinstance(lineups, dict):
        return {
            "schema_version": SCHEMA_VERSION,
            "fixture_id": fid,
            "status": "NOT_VERIFIED",
            "reason": "LINEUP_PAYLOAD_NOT_AVAILABLE",
            "actionable": False,
        }
    current = _snapshot(lineups)
    key = str(fid)
    previous = base._cache_get("xi_intelligence", key, CACHE_TTL, now)
    comparison = _diff(previous, current) if isinstance(previous, dict) else {"changed": False, "changes": []}
    record = {**current, "captured_at_utc": now.isoformat(), "stage": event.get("stage")}
    base._cache_set("xi_intelligence", key, record, now)
    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": fid,
        "status": "CONFIRMED" if current["both_xi_confirmed"] else "PENDING",
        "source": "API_FIXTURE_LINEUPS",
        "current": current,
        "previous_capture_available": isinstance(previous, dict),
        "lineup_changed_since_previous_capture": comparison["changed"],
        "changes": comparison["changes"],
        "material_change": any(
            bool(change.get("goalkeeper_changed") or change.get("formation_changed") or change.get("starters_added") or change.get("starters_removed"))
            for change in comparison["changes"]
        ),
        "canonical_availability_gate_changed": False,
        "policy": "OBSERVE/PERSIST XI CHANGES ONLY; DO NOT INVENT PLAYER IMPACT; MATERIAL CHANGES REQUIRE RECHECK BUT DO NOT ALTER CANONICAL PROBABILITY WITHOUT PLAYER-IMPACT MODEL",
    }


def attach(payload: dict[str, Any]) -> dict[str, int]:
    now = datetime.now(dt_timezone.utc)
    observed = confirmed = changed = material = 0
    for event in payload.get("events") or []:
        if not isinstance(event, dict) or event.get("event_type") != "SOCCER_REFRESH" or event.get("stage") in {"POSTGAME", "HT"}:
            continue
        intel = build(event, now)
        event["xi_intelligence"] = intel
        if intel.get("status") != "NOT_VERIFIED": observed += 1
        if intel.get("status") == "CONFIRMED": confirmed += 1
        if intel.get("lineup_changed_since_previous_capture"): changed += 1
        if intel.get("material_change"): material += 1
        mi = event.get("match_intelligence")
        if isinstance(mi, dict) and isinstance(mi.get("areas"), dict): mi["areas"]["xi_change_intelligence"] = intel
    return {"observed_lineup_events": observed, "confirmed_xi_events": confirmed, "lineup_change_events": changed, "material_lineup_change_events": material, "provider_requests_added": 0}
