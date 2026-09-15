from __future__ import annotations
from typing import Any
from mcp_gateway import automation_v24 as v24
from mcp_gateway import player_trends

MODEL_VERSION=v24.MODEL_VERSION
AUTOMATION_VERSION='3.2.0'
MAX_POSTGAME_PLAYER_CALLS_PER_TICK=2

async def run_tick()->dict[str,Any]:
    payload=await v24.run_tick();attempted=0;captured=0;players=0;seen=set()
    for event in payload.get('events') or []:
        if attempted>=MAX_POSTGAME_PLAYER_CALLS_PER_TICK:break
        if not isinstance(event,dict) or event.get('stage')!='POSTGAME':continue
        fx=event.get('fixture') or {};fid=fx.get('fixture_id')
        if not fid or fid in seen:continue
        seen.add(fid);attempted+=1
        research=await player_trends.capture(event);research['capture_phase']='POSTGAME'
        event['postgame_player_stats']=research
        result=event.get('result')
        if not isinstance(result,dict):result={};event['result']=result
        result['player_stats']=research
        if research.get('status')=='RESEARCH_ONLY_PLAYER_FIXTURE_STATS':
            captured+=1;players+=sum(len(t.get('players') or []) for t in research.get('teams') or [])
    payload['postgame_player_fixture_calls_attempted_this_tick']=attempted
    payload['postgame_player_fixtures_captured_this_tick']=captured
    payload['postgame_players_captured_this_tick']=players
    payload['postgame_player_max_fixture_calls_per_tick']=MAX_POSTGAME_PLAYER_CALLS_PER_TICK
    payload['postgame_player_policy']='TRACKED_POSTGAME_EVENTS_ONLY; DEDUPE_WITHIN_TICK; MAX_2_FIXTURES_PER_TICK; RESEARCH_ONLY; ZERO_DECISION_WEIGHT'
    payload['version']=AUTOMATION_VERSION;payload['model_version']=MODEL_VERSION
    return payload
