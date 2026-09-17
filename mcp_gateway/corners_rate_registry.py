from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone as dt_timezone
from typing import Any

import httpx

from mcp_gateway import automation as base

REGISTRY_URL = "https://raw.githubusercontent.com/Baahl11/Soccer/soccer-edge-state/soccer_edge_state/analysis/corners_rate_registry.json"
CACHE_TTL = timedelta(hours=6)


def _num(value: Any) -> float | None:
    try: return float(value)
    except (TypeError, ValueError): return None


def load_registry() -> dict[str, Any] | None:
    now = datetime.now(dt_timezone.utc)
    cached = base._cache_get("corners_rate_registry", "latest", CACHE_TTL, now)
    if isinstance(cached, dict) and cached.get("schema_version"): return cached
    try:
        response = httpx.get(REGISTRY_URL, timeout=5.0, follow_redirects=True)
        if response.status_code != 200: return None
        payload = response.json()
    except Exception:
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("global"), dict): return None
    base._cache_set("corners_rate_registry", "latest", payload, now)
    return payload


def _avg(row: dict[str, Any] | None, key: str) -> tuple[float, int] | None:
    if not isinstance(row, dict): return None
    n = int(row.get("n") or 0); total = _num(row.get(key))
    if n <= 0 or total is None: return None
    return total / n, n


def _shrink_ratio(total: float, baseline: float, n: int, pseudo_n: float) -> float:
    if n <= 0 or baseline <= 0: return 1.0
    raw = total / (baseline * n)
    weight = n / (n + pseudo_n)
    return 1.0 + weight * (raw - 1.0)


def model_fixture(fixture: dict[str, Any], registry: dict[str, Any] | None = None) -> dict[str, Any] | None:
    registry = registry or load_registry()
    if not isinstance(registry, dict): return None
    global_row = registry.get("global") if isinstance(registry.get("global"), dict) else {}
    ghome = _avg(global_row, "home_corners"); gaway = _avg(global_row, "away_corners")
    if ghome is None or gaway is None: return None
    lid = str(fixture.get("league_id")) if fixture.get("league_id") is not None else None
    hid = str(fixture.get("home_team_id")) if fixture.get("home_team_id") is not None else None
    aid = str(fixture.get("away_team_id")) if fixture.get("away_team_id") is not None else None
    leagues = registry.get("leagues") if isinstance(registry.get("leagues"), dict) else {}
    home_teams = registry.get("home_teams") if isinstance(registry.get("home_teams"), dict) else {}
    away_teams = registry.get("away_teams") if isinstance(registry.get("away_teams"), dict) else {}
    league_row = leagues.get(lid) if lid else None
    league_n = int((league_row or {}).get("n") or 0)
    pool = league_row if isinstance(league_row, dict) and league_n >= int(registry.get("minimum_league_pool") or 20) else global_row
    league_home = _avg(pool, "home_corners"); league_away = _avg(pool, "away_corners")
    if league_home is None or league_away is None: return None
    lh, la = league_home[0], league_away[0]
    hrow = home_teams.get(hid) if hid else None; arow = away_teams.get(aid) if aid else None
    hn = int((hrow or {}).get("n") or 0); an = int((arow or {}).get("n") or 0)
    pseudo = float(registry.get("team_ratio_pseudo_n") or 8.0)
    h_own = _num((hrow or {}).get("home_corners")) or 0.0
    h_concede = _num((hrow or {}).get("away_corners")) or 0.0
    a_own = _num((arow or {}).get("away_corners")) or 0.0
    a_concede = _num((arow or {}).get("home_corners")) or 0.0
    h_att = _shrink_ratio(h_own, lh, hn, pseudo)
    a_att = _shrink_ratio(a_own, la, an, pseudo)
    a_def_weak = _shrink_ratio(a_concede, lh, an, pseudo)
    h_def_weak = _shrink_ratio(h_concede, la, hn, pseudo)
    home_lam = max(1.0, min(10.0, lh * math.sqrt(max(0.25, h_att * a_def_weak))))
    away_lam = max(1.0, min(10.0, la * math.sqrt(max(0.25, a_att * h_def_weak))))
    return {
        "model": "LEAGUE_TEAM_CORNERS_POISSON_LIVE_v0.1", "home_lambda": home_lam, "away_lambda": away_lam,
        "total_lambda": home_lam + away_lam, "prior_pool_n": int(pool.get("n") or 0),
        "prior_home_home_matches": hn, "prior_away_away_matches": an,
        "formation_adjustment_applied": False, "formation_adjustment_status": "OFFLINE_CHALLENGER_NOT_LIVE_WEIGHTED",
        "registry_verified_postgame_fixtures": int(registry.get("verified_postgame_fixtures") or 0),
    }
