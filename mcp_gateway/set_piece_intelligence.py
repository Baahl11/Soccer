from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import httpx

SCHEMA_VERSION = "1.0.0"
REGISTRY_URL = (
    "https://raw.githubusercontent.com/Baahl11/Soccer/"
    "soccer-edge-state/soccer_edge_state/analysis/set_piece_registry.json"
)
_REGISTRY_CACHE: dict[str, Any] = {}
_REGISTRY_AT: datetime | None = None
REGISTRY_TTL_SECONDS = 1800

_CORNER_TYPES = {"corner_kicks", "corners"}
_FREE_KICK_TYPES = {"free_kicks", "freekicks", "free_kick"}


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if out >= 0 else None
    except (TypeError, ValueError):
        return None


def _norm(value: Any) -> str:
    return "_".join(str(value or "").strip().lower().replace("-", " ").split())


def _fixture(event: dict[str, Any]) -> dict[str, Any]:
    row = event.get("fixture")
    return row if isinstance(row, dict) else {}


def _extract(match_stats: Any, team_id: Any, accepted: set[str]) -> tuple[float | None, str | None]:
    if not isinstance(match_stats, list):
        return None, None
    for row in match_stats:
        if not isinstance(row, dict):
            continue
        team = row.get("team") if isinstance(row.get("team"), dict) else {}
        if str(team.get("id")) != str(team_id):
            continue
        for stat in row.get("statistics") or []:
            if not isinstance(stat, dict):
                continue
            stat_type = _norm(stat.get("type"))
            if stat_type not in accepted:
                continue
            value = _num(stat.get("value"))
            if value is not None:
                return value, stat_type
    return None, None


def build_postgame_observation(event: dict[str, Any]) -> dict[str, Any]:
    fixture = _fixture(event)
    stats = event.get("match_stats")
    hc, hc_type = _extract(stats, fixture.get("home_team_id"), _CORNER_TYPES)
    ac, ac_type = _extract(stats, fixture.get("away_team_id"), _CORNER_TYPES)
    hf, hf_type = _extract(stats, fixture.get("home_team_id"), _FREE_KICK_TYPES)
    af, af_type = _extract(stats, fixture.get("away_team_id"), _FREE_KICK_TYPES)

    corners_verified = hc is not None and ac is not None
    free_kicks_verified = hf is not None and af is not None
    any_verified = corners_verified or free_kicks_verified

    return {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "VERIFIED_API_FOOTBALL_SET_PIECE_COMPONENTS"
            if any_verified else
            "API_FOOTBALL_SET_PIECE_COMPONENTS_NOT_AVAILABLE"
        ),
        "api_fixture_id": fixture.get("fixture_id"),
        "kickoff": fixture.get("kickoff"),
        "home_team_id": fixture.get("home_team_id"),
        "home_team": fixture.get("home_team"),
        "away_team_id": fixture.get("away_team_id"),
        "away_team": fixture.get("away_team"),
        "home_corners": round(hc, 4) if hc is not None else None,
        "away_corners": round(ac, 4) if ac is not None else None,
        "home_free_kicks": round(hf, 4) if hf is not None else None,
        "away_free_kicks": round(af, 4) if af is not None else None,
        "corners_verified": corners_verified,
        "free_kicks_verified": free_kicks_verified,
        "corner_stat_type": hc_type if corners_verified and hc_type == ac_type else None,
        "free_kick_stat_type": hf_type if free_kicks_verified and hf_type == af_type else None,
        "set_piece_goals": "NOT_VERIFIED",
        "set_piece_xg": "NOT_VERIFIED",
        "aerial_duel_mismatch": "NOT_MODELED",
        "source": "API-Football v3 /fixtures/statistics",
        "provider_requests_added": 0,
        "actionable": False,
        "decision_weight": 0.0,
        "policy": (
            "EXPLICIT PROVIDER COMPONENTS ONLY; CORNERS/FREE KICKS ARE VOLUME CONTEXT, "
            "NOT SET-PIECE GOALS OR SET-PIECE xG"
        ),
    }


def _load_registry() -> dict[str, Any]:
    global _REGISTRY_CACHE, _REGISTRY_AT
    now = datetime.now(timezone.utc)
    if _REGISTRY_AT and (now - _REGISTRY_AT).total_seconds() < REGISTRY_TTL_SECONDS:
        return _REGISTRY_CACHE
    try:
        response = httpx.get(REGISTRY_URL, timeout=4.0, follow_redirects=True)
        data = response.json() if response.status_code == 200 else {}
        _REGISTRY_CACHE = data if isinstance(data, dict) else {}
    except Exception:
        _REGISTRY_CACHE = {}
    _REGISTRY_AT = now
    return _REGISTRY_CACHE


def _team_profile(registry: dict[str, Any], team_id: Any) -> dict[str, Any] | None:
    for row in registry.get("teams") or []:
        if isinstance(row, dict) and str(row.get("team_id")) == str(team_id):
            return row
    return None


def _avg(*values: Any) -> float | None:
    nums = [_num(x) for x in values]
    vals = [x for x in nums if x is not None]
    return sum(vals) / len(vals) if vals else None


def build_pregame(event: dict[str, Any]) -> dict[str, Any]:
    fixture = _fixture(event)
    registry = _load_registry()
    home = _team_profile(registry, fixture.get("home_team_id"))
    away = _team_profile(registry, fixture.get("away_team_id"))

    if not home or not away:
        return {
            "schema_version": SCHEMA_VERSION,
            "fixture_id": fixture.get("fixture_id"),
            "status": "SET_PIECE_HISTORY_INSUFFICIENT",
            "source_registry_status": registry.get("status") or "NOT_AVAILABLE",
            "set_piece_goal_probability": None,
            "set_piece_xg": "NOT_VERIFIED",
            "actionable": False,
            "decision_weight": 0.0,
            "block_reason": "PERSISTED_TEAM_SET_PIECE_COMPONENT_HISTORY_NOT_SUFFICIENT",
        }

    h_corner = _avg(home.get("avg_corners_for"), away.get("avg_corners_against"))
    a_corner = _avg(away.get("avg_corners_for"), home.get("avg_corners_against"))
    h_fk = _avg(home.get("avg_free_kicks_for"), away.get("avg_free_kicks_against"))
    a_fk = _avg(away.get("avg_free_kicks_for"), home.get("avg_free_kicks_against"))

    corner_total = (h_corner + a_corner) if h_corner is not None and a_corner is not None else None
    fk_total = (h_fk + a_fk) if h_fk is not None and a_fk is not None else None
    sample_min = min(int(home.get("n") or 0), int(away.get("n") or 0))

    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": fixture.get("fixture_id"),
        "status": "LIVE_RESEARCH_SET_PIECE_VOLUME_CONTEXT",
        "source_registry_generated_at_utc": registry.get("generated_at_utc"),
        "minimum_team_sample": sample_min,
        "home": {
            "team_id": fixture.get("home_team_id"),
            "team": fixture.get("home_team"),
            "expected_corner_component": round(h_corner, 3) if h_corner is not None else None,
            "expected_free_kick_component": round(h_fk, 3) if h_fk is not None else None,
        },
        "away": {
            "team_id": fixture.get("away_team_id"),
            "team": fixture.get("away_team"),
            "expected_corner_component": round(a_corner, 3) if a_corner is not None else None,
            "expected_free_kick_component": round(a_fk, 3) if a_fk is not None else None,
        },
        "expected_corner_component_total": round(corner_total, 3) if corner_total is not None else None,
        "expected_free_kick_component_total": round(fk_total, 3) if fk_total is not None else None,
        "set_piece_goal_probability": None,
        "set_piece_xg": "NOT_VERIFIED",
        "set_piece_conversion_rate": "NOT_MODELED",
        "aerial_duel_mismatch": "NOT_MODELED",
        "actionable": False,
        "decision_weight": 0.0,
        "production_status": "CONTEXT_ONLY_NOT_ACTIONABLE",
        "promotion_gate": {
            "requires_verified_set_piece_goal_or_set_piece_xg_target": True,
            "requires_oos_feature_lift": True,
            "requires_market_clv_before_betting_use": True,
        },
        "policy": (
            "VOLUME CONTEXT ONLY. CORNERS/FREE KICKS MAY DESCRIBE SET-PIECE OPPORTUNITY VOLUME "
            "BUT MUST NEVER BE RELABELED AS SET-PIECE GOALS, SET-PIECE xG OR BET PROBABILITY."
        ),
    }


def attach(payload: dict[str, Any]) -> dict[str, int | str]:
    postgame_captured = postgame_unavailable = pregame_modeled = pregame_blocked = 0
    for event in payload.get("events") or []:
        if not isinstance(event, dict) or event.get("event_type") != "SOCCER_REFRESH":
            continue
        stage = event.get("stage")
        if stage == "POSTGAME":
            obs = build_postgame_observation(event)
            event["postgame_set_piece_observation"] = obs
            if obs.get("status") == "VERIFIED_API_FOOTBALL_SET_PIECE_COMPONENTS":
                postgame_captured += 1
            else:
                postgame_unavailable += 1
            continue
        if stage in {"HT", "CLOSE"}:
            continue
        intel = build_pregame(event)
        event["set_piece_intelligence"] = intel
        if intel.get("status") == "LIVE_RESEARCH_SET_PIECE_VOLUME_CONTEXT":
            pregame_modeled += 1
        else:
            pregame_blocked += 1
        mi = event.get("match_intelligence")
        if isinstance(mi, dict) and isinstance(mi.get("areas"), dict):
            mi["areas"]["set_pieces"] = intel
    return {
        "postgame_component_observations_captured": postgame_captured,
        "postgame_component_observations_unavailable": postgame_unavailable,
        "pregame_context_events": pregame_modeled,
        "pregame_history_blocked_events": pregame_blocked,
        "provider_requests_added": 0,
        "production_status": "CONTEXT_ONLY_NOT_ACTIONABLE",
    }
