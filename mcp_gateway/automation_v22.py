from __future__ import annotations
from typing import Any
from mcp_gateway import automation_v21 as v21
from mcp_gateway import player_trends

MODEL_VERSION=v21.MODEL_VERSION
AUTOMATION_VERSION='3.0.0'

async def run_tick()->dict[str,Any]:
    payload=await v21.run_tick();captured=0;calls_attempted=0;players=0
    seen=set()
    for event in payload.get('events') or []:
        if calls_attempted>=player_trends.MAX_PLAYER_FIXTURES_PER_TICK:break
        if not isinstance(event,dict) or not player_trends.eligible(event):continue
        fid=(event.get('fixture') or {}).get('fixture_id')
        if fid in seen:continue
        seen.add(fid);calls_attempted+=1
        research=await player_trends.capture(event);event['player_trends_research']=research
        if research.get('status')=='RESEARCH_ONLY_PLAYER_FIXTURE_STATS':
            captured+=1;players+=sum(len(t.get('players') or []) for t in research.get('teams') or [])
    payload['player_trend_fixture_calls_attempted_this_tick']=calls_attempted
    payload['player_trend_fixtures_captured_this_tick']=captured
    payload['player_trend_players_captured_this_tick']=players
    payload['player_trend_max_fixture_calls_per_tick']=player_trends.MAX_PLAYER_FIXTURES_PER_TICK
    payload['player_trend_policy']=f'SELECTIVE_T40_T30_T20_T10_ONLY; MAX_{player_trends.MAX_PLAYER_FIXTURES_PER_TICK}_FIXTURES_PER_TICK; RESEARCH_ONLY; ZERO_DECISION_WEIGHT; NO_BET_UPGRADE'
    payload['version']=AUTOMATION_VERSION;payload['model_version']=MODEL_VERSION
    contract=payload.get('presentation_contract')
    if isinstance(contract,dict):contract['player_trends']='RESEARCH_ONLY; SHOW VERIFIED PLAYER FIXTURE STATS WHEN CAPTURED; NEVER USE AS BET JUSTIFICATION UNTIL OOS MODEL GATE PASSES'
    return payload
