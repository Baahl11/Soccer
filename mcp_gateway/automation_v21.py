from __future__ import annotations
from typing import Any
from mcp_gateway import automation_v20 as v20
from mcp_gateway.trend_context import build as build_trend_context

MODEL_VERSION=v20.MODEL_VERSION
AUTOMATION_VERSION='2.9.0'

def _attach(payload:dict[str,Any])->None:
    contexts=0; disagreements=0; signals=0
    by_fixture={}
    for event in payload.get('events') or []:
        if not isinstance(event,dict):continue
        fx=event.get('fixture')
        if not isinstance(fx,dict) or event.get('event_type') in {'DAILY_DISCOVERY','POSTGAME'}:continue
        fid=fx.get('fixture_id');raw=event.get('raw_projection') if isinstance(event.get('raw_projection'),dict) else {}
        ctx=build_trend_context(fx,raw)
        event['trend_context']=ctx
        if ctx.get('status')=='RESEARCH_ONLY_TREND_CONTEXT':
            contexts+=1;signals+=len(ctx.get('signals') or []);disagreements+=int(bool(ctx.get('model_disagreements')))
            by_fixture[fid]=ctx
    # Presentation only: expose compact diagnostics without touching classification.
    for row in payload.get('match_table_rows') or []:
        if not isinstance(row,dict):continue
        ctx=by_fixture.get(row.get('fixture_id'))
        if not ctx:continue
        row['trend_context']={'reliability':ctx.get('reliability'),'signals':ctx.get('signals') or [],'model_disagreements':ctx.get('model_disagreements') or [],'deep_dive_flag':ctx.get('deep_dive_flag'),'expected_corner_environment':ctx.get('expected_corner_environment')}
    payload['trend_context_count_this_tick']=contexts
    payload['trend_signal_count_this_tick']=signals
    payload['trend_model_disagreement_count_this_tick']=disagreements
    payload['trend_context_policy']='RESEARCH_ONLY_CONTEXT; ZERO_DECISION_WEIGHT; DISAGREEMENT_FLAGS_DEEP_DIVE_ONLY; NO_CLASSIFICATION_OR_STAKE_CHANGE'

async def run_tick()->dict[str,Any]:
    payload=await v20.run_tick();_attach(payload);payload['version']=AUTOMATION_VERSION;payload['model_version']=MODEL_VERSION
    contract=payload.get('presentation_contract')
    if isinstance(contract,dict):
        contract['trend_context']='SHOW_CONVERGENCE_AND_MODEL_DISAGREEMENT_WHEN_PRESENT; RESEARCH_ONLY; NEVER PRESENT AS BET JUSTIFICATION'
    return payload
