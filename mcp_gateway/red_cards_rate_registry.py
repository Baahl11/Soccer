from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone as dt_timezone
from typing import Any

import httpx

from mcp_gateway import automation as base

REGISTRY_URL = "https://raw.githubusercontent.com/Baahl11/Soccer/soccer-edge-state/soccer_edge_state/analysis/red_cards_rate_registry.json"
CACHE_TTL = timedelta(hours=6)


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def load_registry() -> dict[str, Any] | None:
    now = datetime.now(dt_timezone.utc)
    cached = base._cache_get("red_cards_rate_registry", "latest", CACHE_TTL, now)
    if isinstance(cached, dict) and cached.get("schema_version"):
        return cached
    try:
        response = httpx.get(REGISTRY_URL, timeout=5.0, follow_redirects=True)
        if response.status_code != 200:
            return None
        payload = response.json()
    except Exception:
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("global"), dict):
        return None
    base._cache_set("red_cards_rate_registry", "latest", payload, now)
    return payload


def _event_rate(row: dict[str, Any] | None, key: str, prior: float, pseudo_n: float) -> tuple[float, int]:
    if not isinstance(row, dict):
        return prior, 0
    n = int(row.get("n") or 0)
    events = _num(row.get(key)) or 0.0
    return (events + pseudo_n * prior) / (n + pseudo_n), n


def _raw_rate(row: dict[str, Any] | None, key: str) -> tuple[float, int] | None:
    if not isinstance(row, dict):
        return None
    n = int(row.get("n") or 0)
    events = _num(row.get(key))
    if n <= 0 or events is None:
        return None
    return events / n, n


def model_fixture(fixture: dict[str, Any], registry: dict[str, Any] | None = None) -> dict[str, Any] | None:
    registry = registry or load_registry()
    if not isinstance(registry, dict):
        return None
    global_row = registry.get("global") if isinstance(registry.get("global"), dict) else {}
    gh = _raw_rate(global_row, "home_red_event")
    ga = _raw_rate(global_row, "away_red_event")
    gany = _raw_rate(global_row, "any_red_event")
    if gh is None or ga is None or gany is None:
        return None

    lid = str(fixture.get("league_id")) if fixture.get("league_id") is not None else None
    hid = str(fixture.get("home_team_id")) if fixture.get("home_team_id") is not None else None
    aid = str(fixture.get("away_team_id")) if fixture.get("away_team_id") is not None else None
    leagues = registry.get("leagues") if isinstance(registry.get("leagues"), dict) else {}
    home_teams = registry.get("home_teams") if isinstance(registry.get("home_teams"), dict) else {}
    away_teams = registry.get("away_teams") if isinstance(registry.get("away_teams"), dict) else {}
    refs = registry.get("referees") if isinstance(registry.get("referees"), dict) else {}

    lp = float(registry.get("league_pseudo_n") or 80.0)
    tp = float(registry.get("team_pseudo_n") or 24.0)
    rp = float(registry.get("referee_pseudo_n") or 30.0)
    minimum_ref_n = int(registry.get("minimum_referee_n") or 20)
    league_row = leagues.get(lid) if lid else None
    league_home, league_n = _event_rate(league_row, "home_red_event", gh[0], lp)
    league_away, _ = _event_rate(league_row, "away_red_event", ga[0], lp)
    league_any, _ = _event_rate(league_row, "any_red_event", gany[0], lp)

    hrow = home_teams.get(hid) if hid else None
    arow = away_teams.get(aid) if aid else None
    home_own, home_n = _event_rate(hrow, "own_red_event", league_home, tp)
    home_draws_away, _ = _event_rate(hrow, "opponent_red_event", league_away, tp)
    away_own, away_n = _event_rate(arow, "own_red_event", league_away, tp)
    away_draws_home, _ = _event_rate(arow, "opponent_red_event", league_home, tp)

    p_home_red = _clamp(math.sqrt(max(1e-9, home_own * away_draws_home)), 0.002, 0.30)
    p_away_red = _clamp(math.sqrt(max(1e-9, away_own * home_draws_away)), 0.002, 0.30)
    base_any = 1.0 - (1.0 - p_home_red) * (1.0 - p_away_red)

    referee = str(fixture.get("referee") or "").strip()
    refrow = refs.get(referee) if referee else None
    ref_n = int((refrow or {}).get("n") or 0)
    referee_scale = 1.0
    referee_rate = None
    if ref_n >= minimum_ref_n:
        referee_rate, _ = _event_rate(refrow, "any_red_event", league_any, rp)
        referee_scale = _clamp(referee_rate / max(league_any, 1e-9), 0.60, 1.80)
    p_any = _clamp(base_any * referee_scale, 0.003, 0.60)

    return {
        "model": "EMPIRICAL_BAYES_ANY_RED_CARD_MATCH_v0.1",
        "p_home_red": p_home_red,
        "p_away_red": p_away_red,
        "p_any_red": p_any,
        "p_no_red": 1.0 - p_any,
        "base_any_red_before_referee": base_any,
        "global_any_red_rate": gany[0],
        "league_any_red_rate": league_any,
        "referee": referee or None,
        "referee_prior_n": ref_n,
        "referee_rate_shrunk": referee_rate,
        "referee_scale": referee_scale,
        "prior_league_n": league_n,
        "prior_home_home_n": home_n,
        "prior_away_away_n": away_n,
        "registry_verified_postgame_fixtures": int(registry.get("verified_postgame_fixtures") or 0),
        "scope": "ANY_RED_CARD_IN_MATCH_YES_NO_ONLY",
    }
