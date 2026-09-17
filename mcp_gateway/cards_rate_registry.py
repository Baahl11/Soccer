from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone as dt_timezone
from typing import Any

import httpx

from mcp_gateway import automation as base

REGISTRY_URL="https://raw.githubusercontent.com/Baahl11/Soccer/soccer-edge-state/soccer_edge_state/analysis/cards_rate_registry.json"
CACHE_TTL=timedelta(hours=6)


def _num(v:Any)->float|None:
    try:return float(v)
    except (TypeError,ValueError):return None


def load_registry()->dict[str,Any]|None:
    now=datetime.now(dt_timezone.utc); cached=base._cache_get("cards_rate_registry","latest",CACHE_TTL,now)
    if isinstance(cached,dict) and cached.get("schema_version"):return cached
    try:
        r=httpx.get(REGISTRY_URL,timeout=5.0,follow_redirects=True)
        if r.status_code!=200:return None
        payload=r.json()
    except Exception:return None
    if not isinstance(payload,dict) or not isinstance(payload.get("global"),dict):return None
    base._cache_set("cards_rate_registry","latest",payload,now);return payload


def _rate(row:dict[str,Any]|None,key:str)->tuple[float,int]|None:
    if not isinstance(row,dict):return None
    n=int(row.get("n") or 0);total=_num(row.get(key))
    if n<=0 or total is None:return None
    return total/n,n


def _shrunk_rate(row:dict[str,Any]|None,key:str,prior:float,pseudo:float)->tuple[float,int]:
    if not isinstance(row,dict):return prior,0
    n=int(row.get("n") or 0);total=_num(row.get(key)) or 0.0
    return (total+pseudo*prior)/(n+pseudo),n


def _ratio(total:float,expected_rate:float,n:int,pseudo:float)->float:
    if n<=0 or expected_rate<=0:return 1.0
    raw=total/(expected_rate*n);w=n/(n+pseudo);return 1.0+w*(raw-1.0)


def model_fixture(fixture:dict[str,Any],registry:dict[str,Any]|None=None)->dict[str,Any]|None:
    registry=registry or load_registry()
    if not isinstance(registry,dict):return None
    global_row=registry.get("global") if isinstance(registry.get("global"),dict) else {};gh=_rate(global_row,"home_yellow");ga=_rate(global_row,"away_yellow")
    if gh is None or ga is None:return None
    lid=str(fixture.get("league_id")) if fixture.get("league_id") is not None else None;hid=str(fixture.get("home_team_id")) if fixture.get("home_team_id") is not None else None;aid=str(fixture.get("away_team_id")) if fixture.get("away_team_id") is not None else None
    leagues=registry.get("leagues") if isinstance(registry.get("leagues"),dict) else {};home_teams=registry.get("home_teams") if isinstance(registry.get("home_teams"),dict) else {};away_teams=registry.get("away_teams") if isinstance(registry.get("away_teams"),dict) else {}
    lp=float(registry.get("league_pseudo_n") or 25.0);tp=float(registry.get("team_ratio_pseudo_n") or 8.0);rp=float(registry.get("referee_pseudo_n") or 12.0)
    league_row=leagues.get(lid) if lid else None;lh,league_n=_shrunk_rate(league_row,"home_yellow",gh[0],lp);la,_=_shrunk_rate(league_row,"away_yellow",ga[0],lp)
    hrow=home_teams.get(hid) if hid else None;arow=away_teams.get(aid) if aid else None;hn=int((hrow or {}).get("n") or 0);an=int((arow or {}).get("n") or 0)
    hown=_ratio(_num((hrow or {}).get("home_yellow")) or 0.0,lh,hn,tp);aown=_ratio(_num((arow or {}).get("away_yellow")) or 0.0,la,an,tp)
    away_draws_home=_ratio(_num((arow or {}).get("home_yellow")) or 0.0,lh,an,tp);home_draws_away=_ratio(_num((hrow or {}).get("away_yellow")) or 0.0,la,hn,tp)
    home_lam=max(0.25,min(5.5,lh*math.sqrt(max(0.20,hown*away_draws_home))));away_lam=max(0.25,min(5.5,la*math.sqrt(max(0.20,aown*home_draws_away))));base_total=home_lam+away_lam
    referee=str(fixture.get("referee") or "").strip();refs=registry.get("referees") if isinstance(registry.get("referees"),dict) else {};refrow=refs.get(referee) if referee else None;ref_n=int((refrow or {}).get("n") or 0);ref_scale=1.0
    if ref_n>=8:
        ref_avg=(_num((refrow or {}).get("total_yellow")) or 0.0)/ref_n;raw=ref_avg/max(lh+la,1e-9);ref_scale=(ref_n*raw+rp)/(ref_n+rp);ref_scale=max(0.75,min(1.35,ref_scale))
    return {"model":"TEAM_DISCIPLINE_PLUS_OPTIONAL_REFEREE_YELLOW_CARDS_LIVE_v0.1","home_yellow_lambda":home_lam,"away_yellow_lambda":away_lam,"base_total_yellow_lambda":base_total,"total_yellow_lambda":base_total*ref_scale,"referee":referee or None,"referee_prior_n":ref_n,"referee_scale":ref_scale,"prior_league_n":league_n,"prior_home_home_n":hn,"prior_away_away_n":an,"registry_verified_postgame_fixtures":int(registry.get("verified_postgame_fixtures") or 0),"red_cards_included":False}
