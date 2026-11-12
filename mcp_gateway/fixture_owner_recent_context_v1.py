"""Authenticated owner-only recent football form from completed official fixtures.

Intended for descriptive sporting research when competition season splits are
empty or too small. No market inputs, OOS calibration, expected goals, or BET.
Do not use cross-competition form as calibrated/raw 1X2 without new validation.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
import os
from threading import Lock
from time import monotonic
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE_URL = "https://v3.football.api-sports.io"
SOURCE = "API_FOOTBALL_RECENT_COMPLETED_FIXTURES"
LAST_FIXTURES = 12
MAX_RESULTS_PER_TEAM = 12
MAX_LOOKBACK = timedelta(days=730)
TTL_SECONDS = 6 * 3600
HOURLY_FIXTURE_CAP = 6
DAILY_FIXTURE_CAP = 24
MIN_PROVIDER_DAILY_REMAINING = 1200
FINISHED = frozenset(("FT","AET","PEN"))
_lock = Lock()
_cache: dict[tuple[int,int,int,str], tuple[float,dict[str,Any]]] = {}
_hour_begin = 0.0
_hour_count = 0
_day_begin = 0.0
_day_count = 0


def _utc(value: Any) -> datetime | None:
    if not isinstance(value,str):
        return None
    try:
        dt=datetime.fromisoformat(value.replace("Z","+00:00"))
        return dt.astimezone(timezone.utc) if dt.tzinfo else None
    except (ValueError,TypeError,OverflowError):
        return None


def _get(team_id: int, token: str) -> dict[str, Any]:
    """Only /fixtures?team=...&last=12; no odds or predictions routes."""
    url=BASE_URL+"/fixtures?"+urlencode({
        "team":team_id,"last":LAST_FIXTURES,"timezone":"UTC",
    })
    request=Request(url,headers={
        "x-apisports-key":token,"Accept":"application/json",
    },method="GET")
    with urlopen(request,timeout=5.5) as response:
        if response.status!=200:
            raise RuntimeError("RECENT_FIXTURES_HTTP_FAILURE")
        raw=response.read(256*1024+1)
        if len(raw)>256*1024:
            raise RuntimeError("RECENT_FIXTURES_UNBOUNDED_BODY")
        remaining=response.headers.get("x-ratelimit-requests-remaining")
    payload=json.loads(raw)
    if not isinstance(payload,dict):
        raise ValueError("RECENT_FIXTURES_INVALID_BODY")
    # Same daily quota guard as other official lookups, if provider reports it.
    if remaining is not None and str(remaining).isdigit():
        payload["quota"]={"daily_remaining":int(remaining)}
    return payload


def _parse(payload: Any, team: int, cutoff: datetime) -> list[dict[str, Any]]:
    if not isinstance(payload,dict):
        return []
    q=payload.get("quota")
    if isinstance(q,dict) and q.get("daily_remaining") is not None:
        try:
            if int(q["daily_remaining"])<MIN_PROVIDER_DAILY_REMAINING:
                return []
        except (TypeError,ValueError):
            return []
    entries=payload.get("response")
    if not isinstance(entries,list) or len(entries)>MAX_RESULTS_PER_TEAM:
        return []
    observations=[]
    ids=set()
    for row in entries:
        if not isinstance(row,dict):
            continue
        fixture=row.get("fixture") or {}
        teams=row.get("teams") or {}
        goals=row.get("goals") or {}
        league=row.get("league") or {}
        home=teams.get("home") or {}
        away=teams.get("away") or {}
        fid=fixture.get("id")
        if type(fid) is not int or fid<=0 or fid in ids:
            continue
        ids.add(fid)
        homeid,awayid=home.get("id"),away.get("id")
        if not (type(homeid) is int and type(awayid) is int
                and team in (homeid,awayid) and homeid!=awayid):
            continue
        if (fixture.get("status") or {}).get("short") not in FINISHED:
            continue
        date=_utc(fixture.get("date"))
        if date is None or date>=cutoff or date<cutoff-MAX_LOOKBACK:
            continue
        home_goals,away_goals=goals.get("home"),goals.get("away")
        if (type(home_goals) is not int or type(away_goals) is not int
                or not (0<=home_goals<=30 and 0<=away_goals<=30)):
            continue
        is_home=team==homeid
        gf,ga=(home_goals,away_goals) if is_home else (away_goals,home_goals)
        outcome="W" if gf>ga else "D" if gf==ga else "L"
        opponent=away if is_home else home
        lid=league.get("id")
        season=league.get("season")
        observations.append({
            "fixture_id":fid,
            "kickoff_utc":date.isoformat(),
            "competition":str(league.get("name") or "Competición sin verificar")[:100],
            "competition_id":lid if type(lid) is int and lid>0 else None,
            "season":season if type(season) is int else None,
            "opponent_id":opponent.get("id"),
            "opponent":str(opponent.get("name") or "Rival no identificado")[:100],
            "venue":"LOCAL" if is_home else "VISITANTE",
            "goals_for":gf,"goals_against":ga,"result":outcome,
            "source_fixture_status":(fixture.get("status") or {}).get("short"),
        })
    return sorted(observations,key=lambda x:x["kickoff_utc"],reverse=True)[:LAST_FIXTURES]


def _team_view(team_id: int,name: str,rows: list[dict[str,Any]]) -> dict[str, Any]:
    sample=rows[:LAST_FIXTURES]
    wins=sum(x["result"]=="W" for x in sample)
    draws=sum(x["result"]=="D" for x in sample)
    losses=sum(x["result"]=="L" for x in sample)
    goals_for=sum(x["goals_for"] for x in sample)
    goals_against=sum(x["goals_against"] for x in sample)
    leagues=sorted({x["competition"] for x in sample})
    home_rows=[x for x in sample if x["venue"]=="LOCAL"]
    away_rows=[x for x in sample if x["venue"]=="VISITANTE"]
    return {
        "team_id":team_id,"team_name":name,
        "sample":len(sample),
        "wins":wins,"draws":draws,"losses":losses,
        "goals_for":goals_for,"goals_against":goals_against,
        "home_sample":len(home_rows),"away_sample":len(away_rows),
        "competitions":leagues[:8],
        "multi_competition":len(leagues)>1,
        "recent_fixtures":sample,
    }


def describe(team: dict[str, Any]) -> list[str]:
    if not isinstance(team,dict) or type(team.get("sample")) is not int or team["sample"]<1:
        return []
    name=team.get("team_name") or "Equipo"
    facts=[
        f"{name}: últimos {team['sample']} encuentros terminados verificados, "
        f"{team['wins']} ganados, {team['draws']} empatados, {team['losses']} perdidos; "
        f"{team['goals_for']} goles marcados, {team['goals_against']} recibidos.",
        f"En la muestra: {team['home_sample']} como local, "
        f"{team['away_sample']} como visitante.",
    ]
    if team.get("multi_competition"):
        facts.append(
            "El historial mezcla competiciones; sus rivales y dificultades "
            "NO son equivalentes ni sirven para una probabilidad 1X2 calibrada."
        )
    if team["sample"]<5:
        facts.append(
            "Muestra reciente inferior a cinco partidos; "
            "lectura descriptiva únicamente."
        )
    return facts


def read_for_owner(
    fixture_id: Any, home_team_id: Any, away_team_id: Any,
    home_name: str, away_name: str, kickoff: Any, *,
    fetcher: Any = None, clock: Any = None, utcnow: Any = None,
) -> dict[str, Any]:
    """Owner/admin authorization must be checked BEFORE calling this."""
    ints=(fixture_id,home_team_id,away_team_id)
    blank={
        "status":"NOT_VERIFIED","source":"NOT_VERIFIED",
        "teams":[],"facts":[],"provider_requests_added":0,
        "model_probabilities":None,"bet_eligible":False,
        "market_queried":False,"commercial_display_allowed":False,
        "source_redisplay_rights_verified":False,
        "owner_admin_only":True,
    }
    if any(type(x) is not int or x<=0 for x in ints) or home_team_id==away_team_id:
        return blank
    kick=_utc(kickoff)
    nowutc=utcnow() if utcnow is not None else datetime.now(timezone.utc)
    if not isinstance(nowutc,datetime) or nowutc.tzinfo is None:
        return blank
    # Stats cannot be used as pre-match evidence after kickoff when originally
    # captured later. This endpoint is historical description, never backtest.
    if kick is None:
        return blank
    cutoff=min(kick,nowutc.astimezone(timezone.utc))
    now=float(clock() if clock is not None else monotonic())
    cache_key=(fixture_id,home_team_id,away_team_id,kick.isoformat())
    global _hour_begin,_hour_count,_day_begin,_day_count
    with _lock:
        cached=_cache.get(cache_key)
        if cached and now-cached[0]<TTL_SECONDS:
            return {**cached[1],"provider_requests_added":0,"from_cache":True}
        if now-_hour_begin>=3600:
            _hour_begin,_hour_count=now,0
        if now-_day_begin>=86400:
            _day_begin,_day_count=now,0
        if _hour_count>=HOURLY_FIXTURE_CAP or _day_count>=DAILY_FIXTURE_CAP:
            return {**blank,"reason":"PRIVATE_RECENT_HISTORY_BUDGET_EXHAUSTED"}
        token=os.getenv("API_FOOTBALL_KEY","").strip()
        if fetcher is None and not token:
            return {**blank,"reason":"SOURCE_NOT_CONFIGURED"}
        _hour_count+=1
        _day_count+=1
    def request(team:int)->dict[str,Any]:
        return fetcher(team,token) if fetcher is not None else _get(team,token)
    teams=[]
    called=0
    # Independent team lookups in parallel, at most two official calls.
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures=[(home_team_id,home_name,executor.submit(request,home_team_id)),
                 (away_team_id,away_name,executor.submit(request,away_team_id))]
        for tid,name,future in futures:
            called+=1
            try:
                rows=_parse(future.result(timeout=6.0),tid,cutoff)
                if rows:
                    teams.append(_team_view(tid,str(name or "")[:90],rows))
            except Exception:
                continue
    facts=[]
    for team in teams:
        facts.extend(describe(team))
    result={
        **blank,
        "status":("VERIFIED_RECENT_FORM" if len(teams)==2 else
                  "PARTIAL_RECENT_FORM" if teams else "NOT_VERIFIED"),
        "source":SOURCE if teams else "NOT_VERIFIED",
        "teams":teams,
        "facts":facts[:12],
        "provider_requests_added":called,
        "captured_at_utc":nowutc.isoformat(),
        "fixture_cutoff_utc":cutoff.isoformat(),
        "pre_kickoff":nowutc<kick,
        "source_type":"MULTI_COMPETITION_PRIOR_MATCHES",
        "market_queried":False,
        "model_probabilities":None,
        "bet_eligible":False,
        "commercial_display_allowed":False,
        "source_redisplay_rights_verified":False,
    }
    with _lock:
        _cache[cache_key]=(now,result)
    return result
