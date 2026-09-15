from __future__ import annotations
from collections import Counter
from typing import Any

from mcp_gateway import automation_v27 as v27

MODEL_VERSION=v27.MODEL_VERSION
AUTOMATION_VERSION='3.4.0'


def _num(v):
    try:return round(float(v),4)
    except (TypeError,ValueError):return None


def _research_row(event:dict[str,Any])->dict[str,Any]:
    fx=event.get('fixture') or {}; raw=event.get('raw_projection') or {}; trend=event.get('trend_context') or {}
    screen=event.get('sporting_screen_refined') or event.get('sporting_screen_initial') or event.get('sporting_shortlist') or {}
    market=event.get('market_decision') or {}; best=event.get('best_market') or {}
    classification=event.get('classification') or 'UNCLASSIFIED'
    has_projection=isinstance(raw,dict) and any(raw.get(k) is not None for k in ('raw_total_goals','raw_over_2_5_prob','raw_btts_yes_prob','raw_home_win_prob'))
    signals=list(trend.get('signals') or []) if isinstance(trend,dict) else []
    disagreements=list(trend.get('model_disagreements') or []) if isinstance(trend,dict) else []
    if classification in ('BET','LEAN'): research_status='ACTIONABLE_MARKET_SIGNAL'
    elif classification=='WATCH': research_status='RESEARCHED_WATCH'
    elif has_projection or signals or screen: research_status='RESEARCHED_NO_ACTIONABLE_EDGE'
    else: research_status='INSUFFICIENT_DATA'
    return {
      'fixture_id':fx.get('fixture_id'),'kickoff':fx.get('kickoff'),'league_id':fx.get('league_id'),'league':fx.get('league'),
      'home':fx.get('home_name') or fx.get('home_team'),'away':fx.get('away_name') or fx.get('away_team'),
      'stage':event.get('stage'),'tier':event.get('tier') or (event.get('coverage') or {}).get('data_tier'),
      'research_status':research_status,'classification':classification,'bet_eligible':bool(event.get('bet_eligible')),
      'goals':{'lambda_total':_num(raw.get('raw_total_goals')),'over_2_5_prob':_num(raw.get('raw_over_2_5_prob'))},
      'btts_yes_prob':_num(raw.get('raw_btts_yes_prob')),
      'one_x_two_research':{'home':_num(raw.get('raw_home_win_prob')),'draw':_num(raw.get('raw_draw_prob')),'away':_num(raw.get('raw_away_win_prob')),'actionable':False},
      'trend_reliability':_num(trend.get('reliability')) if isinstance(trend,dict) else None,
      'trend_signals':signals,'model_trend_disagreements':disagreements,'deep_dive_flag':bool(trend.get('deep_dive_flag')) if isinstance(trend,dict) else False,
      'expected_corner_environment':_num(trend.get('expected_corner_environment')) if isinstance(trend,dict) else None,
      'sporting_screen':screen if isinstance(screen,dict) else None,
      'market_status':market.get('status') if isinstance(market,dict) else None,'market_reason':market.get('reason') if isinstance(market,dict) else None,
      'best_market':best if isinstance(best,dict) else None,
      'reason':event.get('reason') or market.get('reason') if isinstance(market,dict) else event.get('reason'),
    }


def _low_data_rows(payload:dict[str,Any])->list[dict[str,Any]]:
    rows=[]
    for e in payload.get('events') or []:
        if not isinstance(e,dict) or e.get('event_type')!='LOW_DATA_SCREEN_SUMMARY':continue
        for item in e.get('fixtures') or e.get('screened_fixtures') or e.get('rows') or []:
            if not isinstance(item,dict):continue
            fx=item.get('fixture') or item
            rows.append({'fixture_id':fx.get('fixture_id'),'kickoff':fx.get('kickoff'),'league_id':fx.get('league_id'),'league':fx.get('league'),'home':fx.get('home_name') or fx.get('home_team'),'away':fx.get('away_name') or fx.get('away_team'),'stage':item.get('stage'),'tier':item.get('tier') or (item.get('coverage') or {}).get('data_tier'),'research_status':'INSUFFICIENT_DATA','classification':'PASS','reason':item.get('reason') or 'LOW_DATA_PRE_DEEP_DIVE'})
    return rows


async def run_tick()->dict[str,Any]:
    payload=await v27.run_tick(); rows=[]; seen=set()
    for e in payload.get('events') or []:
        if not isinstance(e,dict) or e.get('event_type') in ('DAILY_DISCOVERY','LOW_DATA_SCREEN_SUMMARY') or e.get('stage')=='POSTGAME':continue
        row=_research_row(e); key=(row.get('fixture_id'),row.get('stage'))
        if key in seen:continue
        seen.add(key);rows.append(row)
    for row in _low_data_rows(payload):
        key=(row.get('fixture_id'),row.get('stage'))
        if key not in seen:seen.add(key);rows.append(row)
    counts=Counter(r['research_status'] for r in rows)
    payload['research_report']={
      'status':'RESEARCH_FIRST_SLATE_REPORT','decision_policy':'RESEARCH_IS_NOT_A_BET; NO_BET_IS_VALID; NO_RESEARCH_IS_NOT',
      'rows':rows,'counts':dict(counts),'researched_count':sum(v for k,v in counts.items() if k!='INSUFFICIENT_DATA'),
      'insufficient_data_count':counts.get('INSUFFICIENT_DATA',0),'actionable_signal_count':counts.get('ACTIONABLE_MARKET_SIGNAL',0),
      'watch_count':counts.get('RESEARCHED_WATCH',0),'no_actionable_edge_count':counts.get('RESEARCHED_NO_ACTIONABLE_EDGE',0),
      'scope':['FT_GOALS','BTTS_RESEARCH','1X2_RESEARCH_ONLY','TREND_CONTEXT','CORNERS_CONTEXT','MARKET_AFTER_SPORT'],
    }
    payload['research_report_count_this_tick']=len(rows);payload['research_completed_count_this_tick']=payload['research_report']['researched_count']
    payload['research_insufficient_data_count_this_tick']=payload['research_report']['insufficient_data_count']
    payload['research_policy']='EVERY_DUE_EVENT_MUST_END_AS_RESEARCHED_OR_EXPLICIT_INSUFFICIENT_DATA; ZERO_FORCED_PICKS; MARKET_REMAINS_AFTER_SPORT'
    payload['version']=AUTOMATION_VERSION;payload['model_version']=MODEL_VERSION
    return payload
