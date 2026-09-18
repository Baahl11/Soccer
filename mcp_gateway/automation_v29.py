from __future__ import annotations
from typing import Any

from mcp_gateway import automation as base
from mcp_gateway import automation_v2 as v2
from mcp_gateway import automation_v28 as v28

MODEL_VERSION=v28.MODEL_VERSION
AUTOMATION_VERSION='3.5.0'
RESEARCH_HORIZON_MINUTES=720.0
NORMAL_STAGE_HORIZON_MINUTES=95.0
EARLY_RESEARCH_STAGE='EARLY_RESEARCH'
_ORIGINAL_STAGE_FOR=base._stage_for


def _research_first_stage(minutes_to:float,status:str)->str|None:
    stage=_ORIGINAL_STAGE_FOR(minutes_to,status)
    if stage:
        return stage
    # EARLY_RESEARCH is deliberately distinct from the real T-90 lifecycle
    # checkpoint so early discovery cannot consume/dedupe the true T-90 refresh.
    if NORMAL_STAGE_HORIZON_MINUTES < minutes_to <= RESEARCH_HORIZON_MINUTES:
        return EARLY_RESEARCH_STAGE
    return None


async def run_tick()->dict[str,Any]:
    previous=base._stage_for
    prior_sporting_stages=set(v2.SPORTING_STAGES)
    prior_market_stages=set(v2.MARKET_STAGES)
    base._stage_for=_research_first_stage
    # Early/T-90 price snapshots are allowed only after the existing SPORT-FIRST
    # shortlist survives. They are research snapshots; canonical evaluation still
    # begins at T-40 and all current model/availability gates remain intact.
    v2.SPORTING_STAGES.add(EARLY_RESEARCH_STAGE)
    v2.MARKET_STAGES.update({EARLY_RESEARCH_STAGE,'T-90'})
    try:
        payload=await v28.run_tick()
    finally:
        base._stage_for=previous
        v2.SPORTING_STAGES.clear()
        v2.SPORTING_STAGES.update(prior_sporting_stages)
        v2.MARKET_STAGES.clear()
        v2.MARKET_STAGES.update(prior_market_stages)

    events=payload.get('events') or []
    early=[e for e in events if isinstance(e,dict) and e.get('stage')==EARLY_RESEARCH_STAGE and e.get('event_type')=='SOCCER_REFRESH']
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
        'stage':EARLY_RESEARCH_STAGE,
        'fixtures_researched_this_tick':len(researched),
        'rows':researched,
        'policy':'SPORTING_RESEARCH_UP_TO_12H_PREKICKOFF; EARLY_RESEARCH_AND_TRUE_T90_ARE_DISTINCT; PRICE_SNAPSHOT_ONLY_AFTER_SPORT_FIRST_SHORTLIST; CANONICAL_MARKET_EVALUATION_REMAINS_T40_T20_T10; NO_FORCED_PICKS',
    }
    payload['daily_research_count_this_tick']=len(researched)
    payload['daily_research_horizon_hours']=12
    payload['early_research_stage_distinct_from_t90']=True
    payload['early_research_market_snapshot_after_shortlist']=True
    payload['true_t90_market_snapshot_after_shortlist']=True
    payload['version']=AUTOMATION_VERSION
    payload['model_version']=MODEL_VERSION
    return payload
