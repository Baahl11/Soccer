from __future__ import annotations
import os
from typing import Any
from mcp_gateway import automation_v24 as v24
from mcp_gateway import player_trends

MODEL_VERSION=v24.MODEL_VERSION
AUTOMATION_VERSION='3.2.0'
MAX_POSTGAME_PLAYER_CALLS_PER_TICK=max(2,min(int(os.getenv('SOCCER_POSTGAME_PLAYER_MAX_FIXTURES_PER_TICK','6')),10))

def _postgame_candidates(events:list[Any])->list[dict[str,Any]]:
    candidates=[];seen=set()
    for event in events or []:
        if not isinstance(event,dict) or event.get('stage')!='POSTGAME':continue
        fx=event.get('fixture') or {};fid=fx.get('fixture_id')
        if not fid or fid in seen:continue
        seen.add(fid);candidates.append(event)
    candidates.sort(key=lambda event:(
        0 if bool((event.get('coverage') or {}).get('statistics_players')) else 1,
        str((event.get('fixture') or {}).get('kickoff') or ''),
        int((event.get('fixture') or {}).get('fixture_id') or 0),
    ))
    return candidates

async def run_tick()->dict[str,Any]:
    payload=await v24.run_tick();attempted=0;captured=0;players=0
    for event in _postgame_candidates(payload.get('events') or []):
        if attempted>=MAX_POSTGAME_PLAYER_CALLS_PER_TICK:break
        fx=event.get('fixture') or {};fid=fx.get('fixture_id')
        attempted+=1
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
    payload['postgame_player_policy']=f'TRACKED_POSTGAME_EVENTS_ONLY; PLAYER_STATS_COVERAGE_FIRST; DEDUPE_WITHIN_TICK; MAX_{MAX_POSTGAME_PLAYER_CALLS_PER_TICK}_FIXTURES_PER_TICK; RESEARCH_ONLY; ZERO_DECISION_WEIGHT'
    payload['version']=AUTOMATION_VERSION;payload['model_version']=MODEL_VERSION
    return payload
