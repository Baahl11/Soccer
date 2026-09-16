from __future__ import annotations
from typing import Any

from mcp_gateway import automation as base
from mcp_gateway import automation_v28 as v28

MODEL_VERSION=v28.MODEL_VERSION
AUTOMATION_VERSION='3.5.0'
RESEARCH_HORIZON_MINUTES=720.0
NORMAL_STAGE_HORIZON_MINUTES=95.0
_ORIGINAL_STAGE_FOR=base._stage_for


def _research_first_stage(minutes_to:float,status:str)->str|None:
    stage=_ORIGINAL_STAGE_FOR(minutes_to,status)
    if stage:
        return stage
    # T-90 is intentionally reused as a SPORT-FIRST research stage. It does not
    # request a betting market in the current engine. This makes the slate earn a
    # sporting projection hours before kickoff instead of waiting for T-40.
    if NORMAL_STAGE_HORIZON_MINUTES < minutes_to <= RESEARCH_HORIZON_MINUTES:
        return 'T-90'
    return None


async def run_tick()->dict[str,Any]:
    previous=base._stage_for
    base._stage_for=_research_first_stage
    try:
        payload=await v28.run_tick()
    finally:
        base._stage_for=previous

    events=payload.get('events') or []
    early=[e for e in events if isinstance(e,dict) and e.get('stage')=='T-90' and e.get('event_type')=='SOCCER_REFRESH']
    researched=[]
    for e in early:
        raw=e.get('raw_projection') or {}
        screen=e.get('sporting_screen_refined') or e.get('sporting_screen_initial') or e.get('sporting_shortlist') or {}
        researched.append({
            'fixture':e.get('fixture'),
            'classification':e.get('classification'),
            'tier':e.get('tier') or ((e.get('coverage') or {}).get('data_tier') if isinstance(e.get('coverage'),dict) else None),
            'sporting_screen':screen,
            'raw_projection':raw,
            'notes':e.get('notes') or [],
        })
    payload['daily_research']={
        'status':'ACTIVE',
        'horizon_hours':12,
        'stage':'T-90_RESEARCH_ONLY',
        'fixtures_researched_this_tick':len(researched),
        'rows':researched,
        'policy':'SPORTING_RESEARCH_UP_TO_12H_PREKICKOFF; MARKET_VALIDATION_REMAINS_T40_T20_T10_CLOSE; NO_FORCED_PICKS',
    }
    payload['daily_research_count_this_tick']=len(researched)
    payload['daily_research_horizon_hours']=12
    payload['version']=AUTOMATION_VERSION
    payload['model_version']=MODEL_VERSION
    return payload
