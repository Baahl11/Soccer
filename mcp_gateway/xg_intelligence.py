from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone as dt_timezone
from typing import Any

import httpx

from mcp_gateway import automation as base

SCHEMA_VERSION = "1.2.0"
REGISTRY_URL = "https://raw.githubusercontent.com/Baahl11/Soccer/soccer-edge-state/soccer_edge_state/analysis/xg_team_registry.json"
CACHE_TTL = timedelta(hours=6)
PRIOR_MATCHES = 10.0
FACTOR_CLIP = (0.65, 1.45)
MIN_TEAM_MATCHES = 5


def _num(value: Any) -> float | None:
    try:
        value = float(value)
        return value if value >= 0 else None
    except (TypeError, ValueError):
        return None


def _norm_stat_type(value: Any) -> str:
    return "_".join(str(value or "").strip().lower().replace("-", " ").split())


def _fixture(event: dict[str, Any]) -> dict[str, Any]:
    row = event.get("fixture")
    return row if isinstance(row, dict) else {}


def _extract_team_xg(match_stats: Any, team_id: Any) -> float | None:
    if not isinstance(match_stats, list):
        return None
    for row in match_stats:
        if not isinstance(row, dict):
            continue
        team = row.get("team") if isinstance(row.get("team"), dict) else {}
        if str(team.get("id")) != str(team_id):
            continue
        for stat in row.get("statistics") or []:
            if not isinstance(stat, dict):
                continue
            if _norm_stat_type(stat.get("type")) in {"expected_goals", "xg"}:
                value = _num(stat.get("value"))
                if value is not None:
                    return value
    return None


def build_postgame_observation(event: dict[str, Any]) -> dict[str, Any]:
    fixture = _fixture(event)
    fid = fixture.get("fixture_id")
    home_id = fixture.get("home_team_id")
    away_id = fixture.get("away_team_id")
    home_xg = _extract_team_xg(event.get("match_stats"), home_id)
    away_xg = _extract_team_xg(event.get("match_stats"), away_id)
    verified = home_xg is not None and away_xg is not None
    return {
        "schema_version": "1.0.0",
        "status": "VERIFIED_API_FOOTBALL_XG" if verified else "API_FOOTBALL_XG_NOT_AVAILABLE_FOR_FIXTURE",
        "api_fixture_id": fid,
        "kickoff": fixture.get("kickoff"),
        "home_team_id": home_id,
        "home_team": fixture.get("home_team"),
        "away_team_id": away_id,
        "away_team": fixture.get("away_team"),
        "home_xg": round(home_xg, 6) if home_xg is not None else None,
        "away_xg": round(away_xg, 6) if away_xg is not None else None,
        "source": "API-Football v3",
        "source_endpoint": "/fixtures/statistics",
        "source_stat_type": "expected_goals",
        "same_fixture_pregame_use_allowed": False,
        "historical_feature_use_only": True,
        "provider_requests_added": 0,
    }


def load_registry() -> dict[str, Any] | None:
    now = datetime.now(dt_timezone.utc)
    cached = base._cache_get("xg_team_registry", "latest", CACHE_TTL, now)
    if isinstance(cached, dict) and cached.get("status") in {
        "RESEARCH_API_FOOTBALL_XG_TEAM_REGISTRY",
        "API_FOOTBALL_XG_DATA_ACCUMULATING",
    }:
        return cached
    try:
        response = httpx.get(REGISTRY_URL, timeout=5.0, follow_redirects=True)
        if response.status_code != 200:
            return None
        payload = response.json()
    except Exception:
        return None
    if not isinstance(payload, dict) or payload.get("status") not in {
        "RESEARCH_API_FOOTBALL_XG_TEAM_REGISTRY",
        "API_FOOTBALL_XG_DATA_ACCUMULATING",
    }:
        return None
    base._cache_set("xg_team_registry", "latest", payload, now)
    return payload


def _profile(registry: dict[str, Any], team_id: Any) -> dict[str, Any]:
    profiles = registry.get("profiles") if isinstance(registry.get("profiles"), dict) else {}
    row = profiles.get(str(team_id))
    return row if isinstance(row, dict) else {}


def _l10(profile: dict[str, Any]) -> dict[str, Any]:
    windows = profile.get("windows") if isinstance(profile.get("windows"), dict) else {}
    row = windows.get("last_10")
    return row if isinstance(row, dict) else {}


def _shrunk_ratio(value: float, baseline: float, n: int) -> float:
    raw = value / baseline
    ratio = (n * raw + PRIOR_MATCHES) / (n + PRIOR_MATCHES)
    return max(FACTOR_CLIP[0], min(FACTOR_CLIP[1], ratio))


def _side_projection(
    own_profile: dict[str, Any],
    opponent_profile: dict[str, Any],
    global_xg: float,
) -> dict[str, Any]:
    own = _l10(own_profile)
    opp = _l10(opponent_profile)
    own_n = int(own.get("n") or 0)
    opp_n = int(opp.get("n") or 0)
    own_xgf = _num(own.get("avg_xg_for"))
    opp_xga = _num(opp.get("avg_xg_against"))
    if (
        own_n < MIN_TEAM_MATCHES
        or opp_n < MIN_TEAM_MATCHES
        or own_xgf is None
        or opp_xga is None
        or global_xg <= 0
    ):
        return {
            "status": "INSUFFICIENT_HISTORICAL_XG",
            "projected_xg_research": None,
            "own_sample_n": own_n,
            "opponent_sample_n": opp_n,
            "numeric_modifier_applied_to_canonical_model": False,
        }
    attack = _shrunk_ratio(own_xgf, global_xg, own_n)
    defense = _shrunk_ratio(opp_xga, global_xg, opp_n)
    factor = math.sqrt(attack * defense)
    projected = global_xg * factor
    return {
        "status": "RESEARCH_XG_PROJECTION_AVAILABLE",
        "projected_xg_research": round(projected, 6),
        "global_avg_team_xg": round(global_xg, 6),
        "own_last10_avg_xg_for": round(own_xgf, 6),
        "opponent_last10_avg_xg_against": round(opp_xga, 6),
        "own_attack_factor_shrunk": round(attack, 6),
        "opponent_xga_factor_shrunk": round(defense, 6),
        "combined_factor": round(factor, 6),
        "prior_matches_at_neutral": PRIOR_MATCHES,
        "factor_clip": list(FACTOR_CLIP),
        "numeric_modifier_applied_to_canonical_model": False,
    }


def build_pregame(event: dict[str, Any], registry: dict[str, Any] | None) -> dict[str, Any]:
    fixture = _fixture(event)
    fid = fixture.get("fixture_id")
    if not isinstance(registry, dict):
        return {
            "schema_version": SCHEMA_VERSION,
            "fixture_id": fid,
            "status": "XG_REGISTRY_NOT_YET_PERSISTED",
            "source": "API-Football v3 /fixtures/statistics via finalized historical fixtures",
            "actionable": False,
            "decision_weight": 0.0,
        }
    if registry.get("status") != "RESEARCH_API_FOOTBALL_XG_TEAM_REGISTRY":
        return {
            "schema_version": SCHEMA_VERSION,
            "fixture_id": fid,
            "status": "API_FOOTBALL_XG_DATA_ACCUMULATING",
            "fixture_count": int(registry.get("fixture_count") or 0),
            "team_profile_count": int(registry.get("team_profile_count") or 0),
            "block_reason": registry.get("block_reason"),
            "actionable": False,
            "decision_weight": 0.0,
        }

    global_row = registry.get("global") if isinstance(registry.get("global"), dict) else {}
    global_xg = _num(global_row.get("avg_team_xg"))
    if global_xg is None or global_xg <= 0:
        return {
            "schema_version": SCHEMA_VERSION,
            "fixture_id": fid,
            "status": "XG_GLOBAL_BASELINE_UNAVAILABLE",
            "actionable": False,
            "decision_weight": 0.0,
        }

    home_id = fixture.get("home_team_id")
    away_id = fixture.get("away_team_id")
    home_profile = _profile(registry, home_id)
    away_profile = _profile(registry, away_id)
    home = _side_projection(home_profile, away_profile, global_xg)
    away = _side_projection(away_profile, home_profile, global_xg)
    ready = (
        home.get("status") == "RESEARCH_XG_PROJECTION_AVAILABLE"
        and away.get("status") == "RESEARCH_XG_PROJECTION_AVAILABLE"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": fid,
        "status": "LIVE_RESEARCH_HISTORICAL_XG_XGA" if ready else "INSUFFICIENT_TEAM_XG_HISTORY",
        "source": registry.get("source"),
        "registry_fixture_count": int(registry.get("fixture_count") or 0),
        "home": {
            "team_id": home_id,
            "team": fixture.get("home_team"),
            "history": _l10(home_profile),
            **home,
        },
        "away": {
            "team_id": away_id,
            "team": fixture.get("away_team"),
            "history": _l10(away_profile),
            **away,
        },
        "xg_xga_definition": (
            "Historical xG for = API-Football expected_goals from finalized fixtures; "
            "xGA = opponent verified xG in the same finalized fixture."
        ),
        "same_fixture_realized_xg_leakage_allowed": False,
        "canonical_goal_lambda_adjustment": 0.0,
        "canonical_model_weights_changed": False,
        "actionable": False,
        "decision_weight": 0.0,
        "calibration_gate": {
            "minimum_oos_fixtures_for_feature_review": 500,
            "minimum_oos_fixtures_for_lambda_challenger": 1000,
            "minimum_oos_fixtures_for_production_review": 2000,
        },
        "policy": (
            "API-FOOTBALL REALIZED xG MAY ENTER ONLY AFTER FIXTURE FINALIZATION AND ONLY AS HISTORY. "
            "NEVER USE SAME-FIXTURE REALIZED xG PREGAME; NEVER RELABEL INTERNAL GOAL LAMBDAS AS xG."
        ),
    }


def attach(payload: dict[str, Any]) -> dict[str, int | bool | str]:
    registry = load_registry()
    live = blocked = captured = unavailable = 0
    for event in payload.get("events") or []:
        if not isinstance(event, dict) or event.get("event_type") != "SOCCER_REFRESH":
            continue
        stage = event.get("stage")
        if stage == "POSTGAME":
            obs = build_postgame_observation(event)
            event["postgame_xg_observation"] = obs
            if obs.get("status") == "VERIFIED_API_FOOTBALL_XG":
                captured += 1
            else:
                unavailable += 1
            continue
        if stage in {"HT", "CLOSE"}:
            continue
        intel = build_pregame(event, registry)
        event["xg_xga_intelligence"] = intel
        if intel.get("status") == "LIVE_RESEARCH_HISTORICAL_XG_XGA":
            live += 1
        else:
            blocked += 1
        mi = event.get("match_intelligence")
        if isinstance(mi, dict) and isinstance(mi.get("areas"), dict):
            mi["areas"]["xg_xga"] = intel
    return {
        "registry_loaded": bool(registry),
        "registry_status": str((registry or {}).get("status") or "UNAVAILABLE"),
        "live_research_events": live,
        "data_accumulating_events": blocked,
        "postgame_xg_captured": captured,
        "postgame_xg_unavailable": unavailable,
        "provider_requests_added": 0,
    }
