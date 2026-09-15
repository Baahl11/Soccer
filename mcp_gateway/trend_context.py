from __future__ import annotations
import json
from datetime import datetime, timezone
from typing import Any
import httpx

TREND_URL='https://raw.githubusercontent.com/Baahl11/Soccer/soccer-edge-state/soccer_edge_state/analysis/trend_intelligence.json'
_cache: dict[str,Any]={}
_cache_at: datetime|None=None
TTL=1800

def _load()->dict[str,Any]:
    global _cache,_cache_at
    now=datetime.now(timezone.utc)
    if _cache_at and (now-_cache_at).total_seconds()<TTL:return _cache
    try:
        r=httpx.get(TREND_URL,timeout=4.0,follow_redirects=True)
        if r.status_code==200 and r.text.strip():
            x=r.json();_cache=x if isinstance(x,dict) else {}
        else:_cache={}
    except Exception:_cache={}
    _cache_at=now
    return _cache

def _window(team:dict[str,Any],venue:str)->dict[str,Any]:
    w=(team.get('windows') or {}).get('last_10') or {}
    vw=(team.get('venue_windows') or {}).get(f'{venue}_last_10') or {}
    return {'overall':w,'venue':vw}
def _avg(a,b):
    try:return round((float(a)+float(b))/2,4)
    except (TypeError,ValueError):return None

def build(fx:dict[str,Any],raw:dict[str,Any]|None=None)->dict[str,Any]:
    data=_load(); teams={str(x.get('team_id')):x for x in data.get('teams') or [] if isinstance(x,dict)}
    h=teams.get(str(fx.get('home_team_id')));a=teams.get(str(fx.get('away_team_id')))
    if not h or not a:return {'status':'NOT_ENOUGH_PERSISTED_TREND_DATA','actionable':False,'decision_weight':0.0}
    hw=_window(h,'home');aw=_window(a,'away');ho=hw['overall'];ao=aw['overall'];hv=hw['venue'];av=aw['venue']
    signals=[];dis=[]
    h_o=ho.get('over_2_5_rate');a_o=ao.get('over_2_5_rate');h_b=ho.get('btts_rate');a_b=ao.get('btts_rate')
    if None not in (h_o,a_o):
        if h_o>=.6 and a_o>=.6:signals.append('GOALS_OVER_CONVERGENCE')
        elif h_o<=.4 and a_o<=.4:signals.append('GOALS_UNDER_CONVERGENCE')
    if None not in (h_b,a_b) and h_b>=.6 and a_b>=.6:signals.append('BTTS_YES_CONVERGENCE')
    hcf=ho.get('avg_team_corners');aca=ao.get('avg_opponent_corners');acf=ao.get('avg_team_corners');hca=ho.get('avg_opponent_corners')
    corner_env=None
    if None not in (hcf,aca,acf,hca):
        corner_env=round(((hcf+aca)/2)+((acf+hca)/2),2)
        if corner_env>=10:signals.append('CORNERS_HIGH_CONVERGENCE')
        elif corner_env<=8:signals.append('CORNERS_LOW_CONVERGENCE')
    raw=raw or {};po=raw.get('raw_over_2_5_prob');pb=raw.get('raw_btts_yes_prob')
    if po is not None and None not in (h_o,a_o):
        if h_o>=.6 and a_o>=.6 and po<=.45:dis.append('MODEL_UNDER_VS_TREND_OVER')
        if h_o<=.4 and a_o<=.4 and po>=.55:dis.append('MODEL_OVER_VS_TREND_UNDER')
    if pb is not None and None not in (h_b,a_b) and h_b>=.6 and a_b>=.6 and pb<=.45:dis.append('MODEL_NO_BTTS_VS_TREND_BTTS')
    n=min(int(ho.get('n') or 0),int(ao.get('n') or 0));reliability=round(min(1,n/10),2)
    return {'status':'RESEARCH_ONLY_TREND_CONTEXT','actionable':False,'decision_weight':0.0,'source_generated_at_utc':data.get('generated_at_utc'),'window':'last_10','reliability':reliability,'signals':signals,'model_disagreements':dis,'deep_dive_flag':bool(dis),'home':{'team':fx.get('home_team'),'over25_rate':h_o,'btts_rate':h_b,'scored_rate':ho.get('scored_rate'),'avg_goals_for':ho.get('avg_goals_for'),'avg_goals_against':ho.get('avg_goals_against'),'avg_corners_for':hcf,'venue_over25_rate':hv.get('over_2_5_rate')},'away':{'team':fx.get('away_team'),'over25_rate':a_o,'btts_rate':a_b,'scored_rate':ao.get('scored_rate'),'avg_goals_for':ao.get('avg_goals_for'),'avg_goals_against':ao.get('avg_goals_against'),'avg_corners_for':acf,'venue_over25_rate':av.get('over_2_5_rate')},'expected_corner_environment':corner_env,'policy':'CONTEXT_ONLY; MAY FLAG DEEP_DIVE; NEVER CHANGES CLASSIFICATION,TIER,STAKE OR BET_ELIGIBILITY'}
