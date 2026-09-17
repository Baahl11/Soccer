from __future__ import annotations

from datetime import datetime, timedelta, timezone as dt_timezone
from typing import Any

import httpx

from mcp_gateway import automation as base

SCHEMA_VERSION = "1.0.0"
REGISTRY_URL = "https://raw.githubusercontent.com/Baahl11/Soccer/soccer-edge-state/soccer_edge_state/analysis/coach_regime_registry.json"
CACHE_TTL = timedelta(hours=6)


def _parse_dt(value: Any) -> datetime | None:
    try:
        out = datetime.fromisoformat(str(value).replace("Z", "+00:00")) if value else None
    except ValueError:
        return None
    if out is not None and out.tzinfo is None:
        out = out.replace(tzinfo=dt_timezone.utc)
    return out


def load_registry() -> dict[str, Any] | None:
    now = datetime.now(dt_timezone.utc)
    cached = base._cache_get("coach_regime_registry", "latest", CACHE_TTL, now)
    if isinstance(cached, dict) and cached.get("schema_version"):
        return cached
    try:
        response = httpx.get(REGISTRY_URL, timeout=5.0, follow_redirects=True)
        if response.status_code != 200:
            return None
        payload = response.json()
    except Exception:
        return None
    if not isinstance(payload, dict) or payload.get("status") != "RESEARCH_COACH_REGIME_REGISTRY":
        return None
    base._cache_set("coach_regime_registry", "latest", payload, now)
    return payload


def _same_coach(live: dict[str, Any], hist: dict[str, Any] | None) -> bool:
    if not isinstance(hist, dict):
        return False
    if live.get("coach_id") is not None and hist.get("coach_id") is not None:
        return str(live.get("coach_id")) == str(hist.get("coach_id"))
    return str(live.get("coach") or "").strip().lower() == str(hist.get("coach") or "").strip().lower()


def _live_coaches(event: dict[str, Any]) -> dict[int, dict[str, Any]]:
    lineups = event.get("lineups") if isinstance(event.get("lineups"), dict) else {}
    out: dict[int, dict[str, Any]] = {}
    for row in lineups.get("teams") or []:
        if not isinstance(row, dict) or row.get("team_id") is None:
            continue
        if row.get("coach_id") is None and not str(row.get("coach") or "").strip():
            continue
        out[int(row["team_id"])] = {"coach_id": row.get("coach_id"), "coach": row.get("coach")}
    return out


def _team_profile(team_id: int, team_name: Any, live: dict[str, Any], registry: dict[str, Any] | None, kickoff: Any) -> dict[str, Any]:
    teams = registry.get("teams") if isinstance(registry, dict) and isinstance(registry.get("teams"), dict) else {}
    row = teams.get(str(team_id)) if isinstance(teams.get(str(team_id)), dict) else None
    current = row.get("current_regime") if isinstance(row, dict) and isinstance(row.get("current_regime"), dict) else None
    previous = row.get("previous_regime") if isinstance(row, dict) and isinstance(row.get("previous_regime"), dict) else None
    matches = int((current or {}).get("matches") or 0)
    same = _same_coach(live, current)
    first = _parse_dt((current or {}).get("first_match")) if same else None
    ko = _parse_dt(kickoff)
    tenure_days = (ko - first).days if first is not None and ko is not None else None
    if not same:
        status = "NEW_OR_UNSEEN_CURRENT_COACH"
        current_out = None
        delta = None
        matches = 0
    else:
        status = "MATCHED_CURRENT_REGIME"
        current_out = current
        delta = row.get("current_vs_previous_descriptive_delta") if isinstance(row, dict) else None
    return {
        "team_id": team_id, "team": team_name,
        "live_coach_id": live.get("coach_id"), "live_coach": live.get("coach"),
        "status": status,
        "regime_matches": matches,
        "regime_tenure_days": tenure_days,
        "sample_band": "HIGH" if matches >= 15 else "MEDIUM" if matches >= 6 else "LOW",
        "current_regime_descriptive": current_out,
        "previous_regime_descriptive": previous,
        "current_vs_previous_descriptive_delta": delta,
        "recent_change_flag": bool(not same or matches < 5 or (tenure_days is not None and tenure_days < 45)),
        "causal_effect_claimed": False,
    }


def build(event: dict[str, Any], registry: dict[str, Any] | None) -> dict[str, Any]:
    fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
    lineups = event.get("lineups") if isinstance(event.get("lineups"), dict) else {}
    if not lineups.get("both_xi_confirmed"):
        return {"schema_version": SCHEMA_VERSION, "fixture_id": fixture.get("fixture_id"), "status": "NOT_VERIFIED", "reason": "CONFIRMED_BOTH_XI_REQUIRED_FOR_COACH_CONTEXT", "actionable": False, "decision_weight": 0.0}
    live = _live_coaches(event)
    hid = fixture.get("home_team_id"); aid = fixture.get("away_team_id")
    if hid is None or aid is None or int(hid) not in live or int(aid) not in live:
        return {"schema_version": SCHEMA_VERSION, "fixture_id": fixture.get("fixture_id"), "status": "NOT_VERIFIED", "reason": "BOTH_CURRENT_COACHES_NOT_AVAILABLE", "actionable": False, "decision_weight": 0.0}
    home = _team_profile(int(hid), fixture.get("home_team"), live[int(hid)], registry, fixture.get("kickoff"))
    away = _team_profile(int(aid), fixture.get("away_team"), live[int(aid)], registry, fixture.get("kickoff"))
    return {
        "schema_version": SCHEMA_VERSION, "fixture_id": fixture.get("fixture_id"), "status": "LIVE_RESEARCH_PROFILE",
        "home": home, "away": away,
        "any_recent_regime_change": bool(home["recent_change_flag"] or away["recent_change_flag"]),
        "actionable": False, "decision_weight": 0.0,
        "production_status": "LIVE_RESEARCH_CONTEXT_ONLY",
        "missing": ["SUBSTITUTION_BEHAVIOR_MODEL_PENDING", "ROTATION_BEHAVIOR_MODEL_PENDING", "OOS_COACH_FEATURE_LIFT_NOT_ESTABLISHED"],
        "policy": "COACH REGIME CONTEXT IS DESCRIPTIVE; BEFORE/AFTER DIFFERENCES ARE NOT CAUSAL; ZERO DECISION WEIGHT UNTIL OOS FEATURE LIFT",
    }


def attach(payload: dict[str, Any]) -> dict[str, int | bool]:
    registry = load_registry(); profiled = recent_change = 0
    for event in payload.get("events") or []:
        if not isinstance(event, dict) or event.get("event_type") != "SOCCER_REFRESH" or event.get("stage") in {"POSTGAME", "HT"}:
            continue
        intel = build(event, registry); event["coach_intelligence"] = intel
        if intel.get("status") == "LIVE_RESEARCH_PROFILE": profiled += 1
        if intel.get("any_recent_regime_change"): recent_change += 1
        mi = event.get("match_intelligence")
        if isinstance(mi, dict) and isinstance(mi.get("areas"), dict): mi["areas"]["coaches"] = intel
    return {"coach_registry_loaded": bool(registry), "profiled_events": profiled, "events_with_recent_regime_change": recent_change, "provider_requests_added": 0}
