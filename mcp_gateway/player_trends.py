from __future__ import annotations
from typing import Any
from mcp_gateway import automation_v4 as v4

MAX_PLAYER_FIXTURES_PER_TICK=2
_ALLOWED_STAGES={'T-40','T-20','T-10'}

def _num(v):
    try:return float(v)
    except (TypeError,ValueError):return None

def _compact(payload:dict[str,Any])->dict[str,Any]:
    teams=[]
    for team in payload.get('response') or []:
        if not isinstance(team,dict):continue
        t=team.get('team') or {}; players=[]
        grouped=team.get('players')
        # Defensive fallback for a flat player row. Grouped-by-team is the
        # normal shape used by the current compact path, but valid player data
        # must never be discarded solely because provider shape differs.
        source_rows=grouped if isinstance(grouped,list) else [team]
        for row in source_rows or []:
            if not isinstance(row,dict):continue
            p=row.get('player') or {}; stats=(row.get('statistics') or [{}]);s=stats[0] if stats and isinstance(stats[0],dict) else {}
            games=s.get('games') or {};shots=s.get('shots') or {};goals=s.get('goals') or {};passes=s.get('passes') or {};tackles=s.get('tackles') or {};duels=s.get('duels') or {};cards=s.get('cards') or {}
            players.append({'player_id':p.get('id'),'name':p.get('name'),'minutes':_num(games.get('minutes')),'position':games.get('position'),'rating':_num(games.get('rating')),'captain':games.get('captain'),'substitute':games.get('substitute'),'shots':_num(shots.get('total')),'shots_on_target':_num(shots.get('on')),'goals':_num(goals.get('total')),'assists':_num(goals.get('assists')),'yellow_cards':_num(cards.get('yellow')),'red_cards':_num(cards.get('red')),'goals_conceded':_num(goals.get('conceded')),'saves':_num(goals.get('saves')),'passes':_num(passes.get('total')),'key_passes':_num(passes.get('key')),'tackles':_num(tackles.get('total')),'duels':_num(duels.get('total')),'duels_won':_num(duels.get('won'))})
        teams.append({'team_id':t.get('id'),'team':t.get('name'),'players':players})
    return {'status':'RESEARCH_ONLY_PLAYER_FIXTURE_STATS','actionable':False,'decision_weight':0.0,'teams':teams,'goalkeeper_fields_retained':['saves','goals_conceded'],'player_card_fields_retained':['yellow_cards','red_cards'],'policy':'OBSERVATION_ONLY; VERIFIED FIXTURE PLAYER STATS; GOALKEEPER SAVES/CONCEDED AND PLAYER YELLOW/RED CARDS RETAINED WHEN SUPPLIED; NEVER CHANGES CLASSIFICATION,TIER,STAKE OR BET_ELIGIBILITY'}

def eligible(event:dict[str,Any])->bool:
    if event.get('stage') not in _ALLOWED_STAGES:return False
    if event.get('event_type') in {'POSTGAME','DAILY_DISCOVERY'}:return False
    fx=event.get('fixture') or {}
    if not fx.get('fixture_id'):return False
    # Only spend a call on fixtures already selected for material pregame work.
    return bool(event.get('raw_projection') or event.get('market') or event.get('classification') in {'BET','LEAN','WATCH'})

async def capture(event:dict[str,Any])->dict[str,Any]:
    fid=(event.get('fixture') or {}).get('fixture_id')
    if not fid:return {'status':'NOT_ELIGIBLE','actionable':False,'decision_weight':0.0}
    try:
        payload=await v4._paced_api_get('fixtures/players',{'fixture':int(fid)})
        out=_compact(payload);out['fixture_id']=int(fid);return out
    except Exception as exc:
        return {'status':'UNAVAILABLE','fixture_id':fid,'error':str(exc)[:180],'actionable':False,'decision_weight':0.0}
