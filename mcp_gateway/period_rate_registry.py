from __future__ import annotations

from datetime import datetime, timedelta, timezone as dt_timezone
from typing import Any

import httpx

from mcp_gateway import automation as base

REGISTRY_URL = (
    "https://raw.githubusercontent.com/Baahl11/Soccer/"
    "soccer-edge-state/soccer_edge_state/analysis/period_rate_registry.json"
)
CACHE_TTL = timedelta(hours=6)


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rate(row: dict[str, Any] | None, key: str) -> tuple[float, int] | None:
    if not isinstance(row, dict):
        return None
    n = int(row.get("n") or 0)
    total = _num(row.get(key))
    if n <= 0 or total is None:
        return None
    return total / n, n


def _shrunk(total: float, n: int, prior_rate: float, prior_games: float) -> float:
    return (total + prior_games * prior_rate) / (n + prior_games)


def _shrunk_from_row(row: dict[str, Any] | None, key: str, prior_rate: float, prior_games: float) -> tuple[float, int]:
    if not isinstance(row, dict):
        return prior_rate, 0
    n = int(row.get("n") or 0)
    total = _num(row.get(key)) or 0.0
    return _shrunk(total, n, prior_rate, prior_games), n


def load_registry() -> dict[str, Any] | None:
    now = datetime.now(dt_timezone.utc)
    cached = base._cache_get("period_rate_registry", "latest", CACHE_TTL, now)
    if isinstance(cached, dict) and cached.get("schema_version"):
        return cached
    try:
        response = httpx.get(REGISTRY_URL, timeout=5.0, follow_redirects=True)
        if response.status_code != 200:
            return None
        payload = response.json()
    except Exception:
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("periods"), dict):
        return None
    base._cache_set("period_rate_registry", "latest", payload, now)
    return payload


def model_fixture(fixture: dict[str, Any], period: str, registry: dict[str, Any] | None = None) -> dict[str, Any] | None:
    registry = registry or load_registry()
    if not isinstance(registry, dict):
        return None
    periods = registry.get("periods") if isinstance(registry.get("periods"), dict) else {}
    data = periods.get(period) if isinstance(periods.get(period), dict) else None
    if not isinstance(data, dict):
        return None

    global_row = data.get("global") if isinstance(data.get("global"), dict) else {}
    global_home = _rate(global_row, "home_for")
    global_away = _rate(global_row, "away_for")
    if global_home is None or global_away is None or global_home[0] <= 0 or global_away[0] <= 0:
        return None

    league_prior = float(registry.get("league_prior_games") or 20.0)
    team_prior = float(registry.get("team_prior_games") or 5.0)
    lid = str(fixture.get("league_id")) if fixture.get("league_id") is not None else None
    hid = str(fixture.get("home_team_id")) if fixture.get("home_team_id") is not None else None
    aid = str(fixture.get("away_team_id")) if fixture.get("away_team_id") is not None else None

    leagues = data.get("leagues") if isinstance(data.get("leagues"), dict) else {}
    home_teams = data.get("home_teams") if isinstance(data.get("home_teams"), dict) else {}
    away_teams = data.get("away_teams") if isinstance(data.get("away_teams"), dict) else {}
    league_row = leagues.get(lid) if lid else None
    home_row = home_teams.get(hid) if hid else None
    away_row = away_teams.get(aid) if aid else None

    league_home, league_n = _shrunk_from_row(league_row, "home_for", global_home[0], league_prior)
    league_away, _ = _shrunk_from_row(league_row, "away_for", global_away[0], league_prior)
    home_for, home_n = _shrunk_from_row(home_row, "home_for", league_home, team_prior)
    home_against, _ = _shrunk_from_row(home_row, "home_against", league_away, team_prior)
    away_for, away_n = _shrunk_from_row(away_row, "away_for", league_away, team_prior)
    away_against, _ = _shrunk_from_row(away_row, "away_against", league_home, team_prior)

    home_attack = home_for / max(league_home, 1e-9)
    away_def_weak = away_against / max(league_home, 1e-9)
    away_attack = away_for / max(league_away, 1e-9)
    home_def_weak = home_against / max(league_away, 1e-9)
    home_lambda = max(0.05, min(3.5, league_home * home_attack * away_def_weak))
    away_lambda = max(0.05, min(3.5, league_away * away_attack * home_def_weak))

    return {
        "period": period,
        "model": f"HIERARCHICAL_{period}_POISSON_STRENGTH_LIVE_v0.1",
        "home_lambda": home_lambda,
        "away_lambda": away_lambda,
        "total_lambda": home_lambda + away_lambda,
        "global_home_rate": global_home[0],
        "global_away_rate": global_away[0],
        "league_home_rate": league_home,
        "league_away_rate": league_away,
        "registry_finalized_fixtures": int(registry.get("finalized_fixtures") or 0),
        "prior_league_matches": league_n,
        "prior_home_home_matches": home_n,
        "prior_away_away_matches": away_n,
        "registry_status": registry.get("status"),
    }
