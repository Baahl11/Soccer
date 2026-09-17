from __future__ import annotations

from datetime import datetime, timedelta, timezone as dt_timezone
from typing import Any

import httpx

from mcp_gateway import automation as base

SCHEMA_VERSION = "1.0.0"
REGISTRY_URL = "https://raw.githubusercontent.com/Baahl11/Soccer/soccer-edge-state/soccer_edge_state/analysis/player_trend_model_registry.json"
CACHE_TTL = timedelta(hours=6)


def load_registry() -> dict[str, Any] | None:
    now = datetime.now(dt_timezone.utc)
    cached = base._cache_get("player_trend_model_registry", "latest", CACHE_TTL, now)
    if isinstance(cached, dict) and cached.get("schema_version"):
        return cached
    try:
        response = httpx.get(REGISTRY_URL, timeout=5.0, follow_redirects=True)
        if response.status_code != 200:
            return None
        payload = response.json()
    except Exception:
        return None
    if not isinstance(payload, dict) or payload.get("status") != "RESEARCH_PLAYER_TREND_MODEL_REGISTRY":
        return None
    base._cache_set("player_trend_model_registry", "latest", payload, now)
    return payload


def build(event: dict[str, Any], registry: dict[str, Any] | None) -> dict[str, Any]:
    fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
    lineups = event.get("lineups") if isinstance(event.get("lineups"), dict) else {}
    if not lineups.get("both_xi_confirmed"):
        return {
            "schema_version": SCHEMA_VERSION,
            "fixture_id": fixture.get("fixture_id"),
            "status": "NOT_VERIFIED",
            "reason": "BOTH_STARTING_XI_NOT_CONFIRMED",
            "actionable": False,
            "decision_weight": 0.0,
        }

    profiles = registry.get("profiles") if isinstance(registry, dict) and isinstance(registry.get("profiles"), dict) else {}
    rows: list[dict[str, Any]] = []
    matched = 0
    modeled = 0

    for team in lineups.get("teams") or []:
        if not isinstance(team, dict):
            continue
        for starter in team.get("starters") or []:
            if not isinstance(starter, dict):
                continue
            pid = starter.get("id")
            profile = profiles.get(str(pid)) if pid is not None and isinstance(profiles.get(str(pid)), dict) else None
            if profile:
                matched += 1
            role = profile.get("role_model") if isinstance(profile, dict) and isinstance(profile.get("role_model"), dict) else {}
            windows = profile.get("windows") if isinstance(profile, dict) and isinstance(profile.get("windows"), dict) else {}
            l20 = windows.get("last_20") if isinstance(windows.get("last_20"), dict) else {}
            metrics = l20.get("metrics") if isinstance(l20.get("metrics"), dict) else {}
            modeled_metrics = sum(
                isinstance(value, dict) and value.get("status") == "MODELED"
                for value in metrics.values()
            )
            if modeled_metrics:
                modeled += 1
            rows.append({
                "team_id": team.get("team_id"),
                "team": team.get("team"),
                "player_id": pid,
                "player": starter.get("name"),
                "position": starter.get("pos"),
                "confirmed_starter": True,
                "p_start_live": 1.0,
                "historical_p_start_smoothed": role.get("historical_p_start_smoothed"),
                "p_60plus_smoothed": role.get("p_60plus_smoothed"),
                "expected_minutes_if_confirmed_starter": role.get("expected_minutes_if_confirmed_starter"),
                "sample_band": profile.get("sample_band") if profile else "NONE",
                "profile_status": "MATCHED" if profile else "PROFILE_NOT_AVAILABLE",
                "last20_modeled_metric_count": modeled_metrics,
                "last20_count_rate_models": metrics if profile else {},
                "expected_counts_if_confirmed_starter": profile.get("expected_counts_if_confirmed_starter") if profile else {},
                "market_line_probabilities_created": False,
            })

    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": fixture.get("fixture_id"),
        "status": "LIVE_RESEARCH_MODELED",
        "model": "ROLE_MINUTES_PLUS_GAMMA_POISSON_COUNT_RATES_v0.1",
        "confirmed_starters": rows,
        "confirmed_starter_count": len(rows),
        "matched_profiles": matched,
        "players_with_modeled_rates": modeled,
        "profile_match_rate": round(matched / len(rows), 4) if rows else None,
        "actionable": False,
        "decision_weight": 0.0,
        "production_status": "LIVE_RESEARCH_NOT_ACTIONABLE",
        "market_line_probabilities_created": False,
        "calibration_gate": {
            "minimum_oos_player_games_for_rate_review": 500,
            "minimum_oos_player_games_before_prop_dependency": 1500,
            "requires": [
                "walk-forward calibration by competition and position",
                "confirmed XI identity at prediction time",
                "minutes MAE review",
                "count-rate dispersion review",
            ],
        },
        "policy": "CONFIRMED XI + SHRUNK ROLE/MINUTES + SHRUNK COUNT RATES ONLY; NO PLAYER PROP LINE PROBABILITY; NO CANONICAL BET PROMOTION",
    }


def attach(payload: dict[str, Any]) -> dict[str, int | bool]:
    registry = load_registry()
    modeled_events = starters = matched = modeled_players = 0
    for event in payload.get("events") or []:
        if not isinstance(event, dict) or event.get("event_type") != "SOCCER_REFRESH" or event.get("stage") in {"POSTGAME", "HT", "CLOSE"}:
            continue
        intel = build(event, registry)
        event["player_trend_intelligence"] = intel
        if intel.get("status") == "LIVE_RESEARCH_MODELED":
            modeled_events += 1
        starters += int(intel.get("confirmed_starter_count") or 0)
        matched += int(intel.get("matched_profiles") or 0)
        modeled_players += int(intel.get("players_with_modeled_rates") or 0)
        mi = event.get("match_intelligence")
        if isinstance(mi, dict) and isinstance(mi.get("areas"), dict):
            mi["areas"]["player_trends"] = intel
    return {
        "player_trend_registry_loaded": bool(registry),
        "modeled_events": modeled_events,
        "confirmed_starters_seen": starters,
        "matched_profiles": matched,
        "players_with_modeled_rates": modeled_players,
        "provider_requests_added": 0,
    }
