from __future__ import annotations

from datetime import datetime, timedelta, timezone as dt_timezone
from typing import Any

import httpx

from mcp_gateway import automation as base

SCHEMA_VERSION = "1.0.0"
REGISTRY_URL = "https://raw.githubusercontent.com/Baahl11/Soccer/soccer-edge-state/soccer_edge_state/analysis/goalkeeper_profiles.json"
CACHE_TTL = timedelta(hours=6)


def load_registry() -> dict[str, Any] | None:
    now = datetime.now(dt_timezone.utc)
    cached = base._cache_get("goalkeeper_profile_registry", "latest", CACHE_TTL, now)
    if isinstance(cached, dict) and cached.get("schema_version"):
        return cached
    try:
        response = httpx.get(REGISTRY_URL, timeout=5.0, follow_redirects=True)
        if response.status_code != 200:
            return None
        payload = response.json()
    except Exception:
        return None
    if not isinstance(payload, dict) or payload.get("status") != "RESEARCH_GOALKEEPER_PROFILE_REGISTRY":
        return None
    base._cache_set("goalkeeper_profile_registry", "latest", payload, now)
    return payload


def _starter_gks(event: dict[str, Any]) -> list[dict[str, Any]]:
    lineups = event.get("lineups") if isinstance(event.get("lineups"), dict) else {}
    out: list[dict[str, Any]] = []
    for team in lineups.get("teams") or []:
        if not isinstance(team, dict):
            continue
        for gk in team.get("goalkeepers") or []:
            if not isinstance(gk, dict):
                continue
            out.append({
                "team_id": team.get("team_id"),
                "team": team.get("team"),
                "player_id": gk.get("id"),
                "player": gk.get("name"),
            })
    return out


def build(event: dict[str, Any], registry: dict[str, Any] | None) -> dict[str, Any]:
    fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
    lineups = event.get("lineups") if isinstance(event.get("lineups"), dict) else {}
    if not lineups.get("both_goalkeepers_confirmed"):
        return {
            "schema_version": SCHEMA_VERSION,
            "fixture_id": fixture.get("fixture_id"),
            "status": "NOT_VERIFIED",
            "reason": "BOTH_STARTING_GOALKEEPERS_NOT_CONFIRMED",
            "actionable": False,
            "decision_weight": 0.0,
        }

    profiles = registry.get("goalkeepers") if isinstance(registry, dict) and isinstance(registry.get("goalkeepers"), dict) else {}
    gks = _starter_gks(event)
    if len(gks) < 2:
        return {
            "schema_version": SCHEMA_VERSION,
            "fixture_id": fixture.get("fixture_id"),
            "status": "NOT_VERIFIED",
            "reason": "STARTING_GOALKEEPER_IDENTITIES_NOT_AVAILABLE",
            "actionable": False,
            "decision_weight": 0.0,
        }

    attached = []
    for gk in gks:
        pid = gk.get("player_id")
        profile = profiles.get(str(pid)) if pid is not None and isinstance(profiles.get(str(pid)), dict) else None
        attached.append({
            **gk,
            "profile_status": "MATCHED" if profile else "PROFILE_NOT_AVAILABLE",
            "profile": profile,
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": fixture.get("fixture_id"),
        "status": "LIVE_RESEARCH_PROFILE",
        "starting_goalkeepers_confirmed": True,
        "goalkeepers": attached,
        "profiles_matched": sum(x["profile_status"] == "MATCHED" for x in attached),
        "impact_model_available": False,
        "canonical_goal_lambda_adjustment": 0.0,
        "actionable": False,
        "decision_weight": 0.0,
        "production_status": "LIVE_RESEARCH_CONTEXT_ONLY",
        "missing": [
            "SHOT_QUALITY_ADJUSTED_GOALKEEPER_METRIC_NOT_AVAILABLE",
            "PSXG_NOT_AVAILABLE",
            "OOS_GOALKEEPER_FEATURE_LIFT_NOT_ESTABLISHED",
            "FINALIZED_GK_SAMPLE_MAY_BE_SPARSE_UNTIL_POSTGAME_PLAYER_CAPTURE_ACCUMULATES",
        ],
        "policy": "CONFIRMED STARTER IDENTITY + DESCRIPTIVE SAVES/CONCEDED PROFILE ONLY; NO CANONICAL LAMBDA CHANGE; SAVE_RESULT_PROXY IS NOT PSxG",
    }


def attach(payload: dict[str, Any]) -> dict[str, int | bool]:
    registry = load_registry()
    confirmed = profiled = matched = 0
    for event in payload.get("events") or []:
        if not isinstance(event, dict) or event.get("event_type") != "SOCCER_REFRESH" or event.get("stage") in {"POSTGAME", "HT"}:
            continue
        intel = build(event, registry)
        event["goalkeeper_intelligence"] = intel
        if intel.get("starting_goalkeepers_confirmed"):
            confirmed += 1
        if intel.get("status") == "LIVE_RESEARCH_PROFILE":
            profiled += 1
        matched += int(intel.get("profiles_matched") or 0)
        mi = event.get("match_intelligence")
        if isinstance(mi, dict) and isinstance(mi.get("areas"), dict):
            mi["areas"]["goalkeeper"] = intel
    return {
        "goalkeeper_registry_loaded": bool(registry),
        "events_with_confirmed_starting_goalkeepers": confirmed,
        "profiled_events": profiled,
        "matched_goalkeeper_profiles": matched,
        "provider_requests_added": 0,
    }
