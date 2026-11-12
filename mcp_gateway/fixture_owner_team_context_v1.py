"""Authenticated OWNER/ADMIN-only sport context from season statistics.

Never use this context as a raw sport projection or an actionable market signal.
Source licensing/OOS not verified for public redisplay. Bounded in-memory cache
and rate quota protect the existing scheduler. No odds endpoint is reachable.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from threading import Lock
from time import monotonic
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE_URL = "https://v3.football.api-sports.io"
TTL_SECONDS = 6 * 3600
HOURLY_FIXTURE_CAP = 5
DAILY_FIXTURE_CAP = 18
MIN_PROVIDER_DAILY_REMAINING = 1200
_lock = Lock()
_cache: dict[tuple[int, int, int, int, int], tuple[float, dict[str, Any]]] = {}
_hour_begin = 0.0
_hour_count = 0
_day_begin = 0.0
_day_count = 0


def _posint(v: Any) -> int | None:
    return v if type(v) is int and 0 <= v <= 999 else None


def _goal_avg(v: Any) -> float | None:
    if type(v) not in (int, float, str) or isinstance(v, bool):
        return None
    try:
        n = float(v)
        return round(n, 2) if n == n and 0 <= n <= 20 else None
    except (TypeError, ValueError):
        return None


def _get(team: int, league: int, season: int, key: str) -> dict[str, Any]:
    query = urlencode({"team":team,"league":league,"season":season})
    req = Request(
        BASE_URL+"/teams/statistics?"+query,
        headers={"x-apisports-key":key,"Accept":"application/json"},
        method="GET",
    )
    with urlopen(req, timeout=5.0) as response:
        if response.status != 200:
            raise RuntimeError("PROVIDER_TEAM_STATS_HTTP_FAILURE")
        raw = response.read(192*1024+1)
        if len(raw) > 192*1024:
            raise RuntimeError("SOURCE_RESPONSE_OVERSIZE")
    return json.loads(raw)


def _parse(response: dict[str, Any], team: int, league: int, season: int,
           venue: str) -> dict[str, Any] | None:
    data = response.get("response")
    if not isinstance(data, dict):
        return None
    if (data.get("team") or {}).get("id") != team:
        return None
    l = data.get("league") or {}
    if l.get("id") != league or l.get("season") != season:
        return None
    quota = response.get("quota") or {}
    remaining = quota.get("daily_remaining")
    if remaining is not None:
        try:
            if int(remaining) < MIN_PROVIDER_DAILY_REMAINING:
                return None
        except (TypeError, ValueError):
            return None
    fixtures = data.get("fixtures") or {}
    played = fixtures.get("played") or {}
    gf = (data.get("goals") or {}).get("for") or {}
    ga = (data.get("goals") or {}).get("against") or {}
    wins,draws,loses = (fixtures.get("wins") or {},
                        fixtures.get("draws") or {},
                        fixtures.get("loses") or {})
    total = _posint(played.get("total"))
    split = _posint(played.get(venue))
    if total is None or total == 0 or split is None or split == 0:
        return None
    out = {
        "team_id":team,
        "team_name":str((data.get("team") or {}).get("name") or "")[:90],
        "competition":str(l.get("name") or "")[:120],
        "season":season,
        "venue":venue,
        "played":total,
        "played_venue":split,
        "wins":_posint(wins.get("total")),
        "draws":_posint(draws.get("total")),
        "losses":_posint(loses.get("total")),
        "goals_for":_posint((gf.get("total") or {}).get("total")),
        "goals_against":_posint((ga.get("total") or {}).get("total")),
        "goals_for_venue":_posint((gf.get("total") or {}).get(venue)),
        "goals_against_venue":_posint((ga.get("total") or {}).get(venue)),
        "goals_for_per_match_venue":_goal_avg((gf.get("average") or {}).get(venue)),
        "goals_against_per_match_venue":_goal_avg((ga.get("average") or {}).get(venue)),
        "raw_xg":None,  # goals averages cannot be represented as expected goals
    }
    if (out["wins"] is None or out["draws"] is None or out["losses"] is None
            or out["goals_for"] is None or out["goals_against"] is None):
        return None
    if out["wins"]+out["draws"]+out["losses"] != total:
        return None
    return out


def read_for_owner(
    fixture_id: Any, league_id: Any, season: Any,
    home_team_id: Any, away_team_id: Any, *,
    fetcher: Any = None, clock: Any = None,
) -> dict[str, Any]:
    """Must be invoked after owner/admin auth on a known registry fixture."""
    ints=(fixture_id,league_id,season,home_team_id,away_team_id)
    if any(type(v) is not int or v <= 0 for v in ints):
        return {"status":"NOT_VERIFIED","provider_requests_added":0}
    if not 2000 <= season <= 2100:
        return {"status":"NOT_VERIFIED","provider_requests_added":0}
    if home_team_id == away_team_id:
        return {"status":"NOT_VERIFIED","provider_requests_added":0}
    key=tuple(ints)
    now=float(clock() if clock is not None else monotonic())
    global _hour_begin,_hour_count,_day_begin,_day_count
    with _lock:
        found=_cache.get(key)
        if found and now-found[0] < TTL_SECONDS:
            return {**found[1],"provider_requests_added":0,"from_cache":True}
        if now-_hour_begin >= 3600:
            _hour_begin,_hour_count=now,0
        if now-_day_begin >= 86400:
            _day_begin,_day_count=now,0
        if _hour_count >= HOURLY_FIXTURE_CAP or _day_count >= DAILY_FIXTURE_CAP:
            return {"status":"NOT_VERIFIED","reason":"PRIVATE_RESEARCH_CALL_BUDGET_EXHAUSTED",
                    "provider_requests_added":0}
        api_key=os.getenv("API_FOOTBALL_KEY","").strip()
        if not api_key and fetcher is None:
            return {"status":"NOT_VERIFIED","reason":"PROVIDER_NOT_CONFIGURED",
                    "provider_requests_added":0}
        _hour_count+=1
        _day_count+=1
    rows=[]
    called=0
    for tid,venue in ((home_team_id,"home"),(away_team_id,"away")):
        try:
            response=(fetcher(tid,league_id,season,api_key) if fetcher else
                      _get(tid,league_id,season,api_key))
            called+=1
            parsed=_parse(response,tid,league_id,season,venue)
            if parsed:
                rows.append(parsed)
        except Exception:
            called+=1  # count attempted requests, even on provider failure
    payload={
        "status":"SPORTING_SEASON_CONTEXT_ONLY" if rows else "NOT_VERIFIED",
        "teams":rows,
        "fixture_id":fixture_id,
        "competition_id":league_id,
        "season":season,
        "fetched_at_utc":datetime.now(timezone.utc).isoformat(),
        "source":"API_FOOTBALL_TEAM_SEASON_STATISTICS",
        "provider_requests_added":called,
        "source_redisplay_rights_verified":False,
        "owner_admin_only":True,
        "bet_eligible":False,
        "commercial_display_allowed":False,
        "projection":None,
        "oos_validated":False,
        "status_note":"Muestra histórica, no modelo comparativo ni probabilidades.",
    }
    with _lock:
        _cache[key]=(now,payload)
    return payload
