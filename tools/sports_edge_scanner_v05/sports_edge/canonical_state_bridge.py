"""Canonical Soccer state -> read-only market-specific research comparisons.

Never promotes a research calibration, interprets third-party odds as a local
sportsbook offer, or generates BET. Uses independently screened slate identity.
"""
from __future__ import annotations
from datetime import datetime, timezone
from collections import Counter
from pathlib import Path
from math import isfinite
from typing import Any
import json

def _dt(x):
    try:
        d=datetime.fromisoformat(str(x).replace('Z','+00:00'))
        return d.astimezone(timezone.utc) if d.tzinfo else None
    except (ValueError,TypeError,OverflowError): return None

def _num(x,lo=0,hi=1):
    try:
        n=float(x)
        return n if isfinite(n) and lo<n<hi else None
    except (ValueError,TypeError,OverflowError): return None

def research_probability(row):
    """Over 2.5 calibrator is NOT an Under 2.5 probability: complement it."""
    if row.get('phase16_calibration_status')!='RESEARCH_CALIBRATION_APPLIED':
        return None,'CALIBRATION_NOT_APPLIED'
    p=_num(row.get('p_model_calibrated'))
    art=row.get('phase16_calibrator_artifact') or {}
    c=(art.get('exact_calibrator_payload') or {}).get('calibrator') or {}
    if p is None or art.get('source_model_version_matches') is not True:
        return None,'CALIBRATION_PROBABILITY_OR_VERSION_UNVERIFIED'
    if c.get('status',c.get('calibrator_status'))!='RESEARCH_CALIBRATOR_FITTED' or c.get('market_fields_used') is not False:
        return None,'CALIBRATOR_LINEAGE_NOT_VERIFIED'
    sel=str(row.get('selection') or '').strip().casefold()
    fam=str(row.get('market_family') or '').upper()
    name=str(row.get('market') or '').strip().casefold()
    origin=str(row.get('phase16_calibration_source') or '').upper()
    try: line=float(row.get('line'))
    except (TypeError,ValueError): line=None
    if fam in ('FT_TOTALS','FT_TOTALS_RESEARCH') and name=='goals over/under' and line==2.5 and origin.endswith(':OVER_2_5'):
        if sel=='over':return p,'DIRECT_OVER_2_5_RESEARCH'
        if sel=='under':return 1-p,'COMPLEMENT_OF_OVER_2_5_RESEARCH'
    if fam=='BTTS' and name in ('both teams score','both teams to score') and origin.endswith(':BTTS'):
        if sel in ('yes','sí','si'):return p,'DIRECT_BTTS_YES_RESEARCH'
        if sel=='no':return 1-p,'COMPLEMENT_OF_BTTS_YES_RESEARCH'
    if fam=='1X2' and name=='match winner' and origin.endswith(':1X2'):
        if row.get('phase16_1x2_class_discrimination_ready') is not True:
            return None,'ONE_X_TWO_CLASS_DISCRIMINATION_NOT_READY'
        if sel in ('home','draw','away'):return p,'RESEARCH_1X2_SELECTION'
    return None,'CALIBRATION_TARGET_MARKET_MISMATCH'

def canonical_research_comparisons(state,slate,*,now=None,max_age_hours=6,quote_hours=4):
    now=(now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    out={'schema_version':'SOCCER_CANONICAL_RESEARCH_BRIDGE_v1.1',
         'source':'soccer-edge-state:soccer_edge_state/latest.json',
         'generated_at_utc':now.isoformat(),'status':'NOT_VERIFIED',
         'canonical_generated_at_utc':None,'total_matching_rows':0,
         'selection_calibration_research_rows':0,'reference_price_rows':0,
         'production_approved_models':0,'authorized_bookmaker_offers':0,
         'actionable_bets':0,'actionable_parlays':0,'rows':[],'blocker_counts':{},
         'note':'Research p and reference breakeven are NOT calibrated betting EV or bank risk.'}
    if not isinstance(state,dict) or not isinstance(state.get('match_table_rows'),list):
        out['status']='CANONICAL_SOURCE_NOT_FOUND';return out
    out['canonical_generated_at_utc']=state.get('generated_at_utc')
    t=_dt(state.get('generated_at_utc'))
    if t is None or not -60<=(now-t).total_seconds()<=max_age_hours*3600:
        out['status']='STALE_CANONICAL_SOURCE';return out
    if not slate.get('eligible_slate_verified') or not (slate.get('provider_world_slate_audit') or {}).get('complete_slate_verified'):
        out['status']='SLATE_NOT_VERIFIED';return out
    fixtures={}
    for f in slate.get('eligible_fixtures') or []:
        ko=_dt(f.get('kickoff'))
        if f.get('status')=='NS' and ko and ko>now:
            fixtures[str(f.get('fixture_id'))]=(f,ko)
    seen=set();counts=Counter()
    for row in state['match_table_rows'][:1000]:
        if not isinstance(row,dict):continue
        fid=str(row.get('fixture_id') or '')
        if fid not in fixtures or row.get('status')!='NS':continue
        f,ko=fixtures[fid]
        rt=_dt(row.get('kickoff'))
        if not rt or abs((rt-ko).total_seconds())>60:
            counts['FIXTURE_KICKOFF_MISMATCH']+=1;continue
        if (str(row.get('home') or '').strip().casefold()!=str(f.get('home_team') or '').strip().casefold()
          or str(row.get('away') or '').strip().casefold()!=str(f.get('away_team') or '').strip().casefold()):
            counts['FIXTURE_TEAM_MISMATCH']+=1;continue
        key=(fid,str(row.get('market')),str(row.get('selection')),str(row.get('line')))
        if key in seen:continue
        seen.add(key)
        p,mapping=research_probability(row)
        price=_num(row.get('price'),1,100)
        pub=_dt(row.get('provider_update') or row.get('price_resolution_provider_update'))
        age=(now-pub).total_seconds()/3600 if pub else None
        price_ok=(price is not None and age is not None and -1/60<=age<=quote_hours
                  and row.get('price_resolution_source')=='API_FOOTBALL_ODDS_V3'
                  and bool(row.get('bookmaker')))
        quote=price if price_ok else None
        blockers=[]
        if p is None:blockers.append(mapping)
        if not quote:blockers.append('REFERENCE_PRICE_NOT_FRESH_OR_NOT_VERIFIED')
        if row.get('production_promotion_allowed') is not True:
            blockers.append('RESEARCH_MODEL_NOT_APPROVED_FOR_PRODUCTION')
        if row.get('p_shrunk') is None or row.get('p_lower') is None:
            blockers.append('PRODUCTION_SHRUNK_LOWER_BOUND_MISSING')
        blockers.append('EXACT_CALIENTE_PLAYDOIT_OFFER_MISSING')
        counts.update(blockers)
        out['rows'].append({
            'fixture_id':fid,'match':str(f.get('home_team'))+' vs '+str(f.get('away_team')),
            'kickoff_utc':ko.isoformat(),'market':row.get('market'),'market_family':row.get('market_family'),
            'selection':row.get('selection'),'line':row.get('line'),
            'raw_sport_probability':_num(row.get('p_raw')),
            'research_probability_for_selection':round(p,8) if p is not None else None,
            'calibration_mapping':mapping,
            'oos_research_sample':(row.get('phase16_binary_calibration_diagnostics') or {}).get('rows'),
            'indicative_price':quote,'reference_bookmaker':row.get('bookmaker'),
            'reference_price_published_at':pub.isoformat() if pub else None,
            'reference_breakeven':round(1/quote,8) if quote else None,
            'research_gap_pp_NOT_EV':round(100*(p-1/quote),2) if p is not None and quote else None,
            'source_model_version':state.get('model_version'),
            'status':'WATCH','model_production_approved':False,
            'bookmaker_offer_verified':False,'calibrated_lower_probability':None,
            'blockers':blockers})
    out['rows'].sort(key=lambda x:(x['kickoff_utc'],x['match'],str(x['market'])))
    out.update(status='CURRENT_RESEARCH_ONLY',total_matching_rows=len(out['rows']),
               selection_calibration_research_rows=sum(x['research_probability_for_selection'] is not None for x in out['rows']),
               reference_price_rows=sum(x['indicative_price'] is not None for x in out['rows']),
               blocker_counts=dict(counts))
    return out

def load_canonical_research_file(path,slate,*,now=None):
    try:
        p=Path(path)
        if not p.is_file() or p.stat().st_size>5_000_000:raise ValueError('missing')
        data=json.loads(p.read_text(encoding='utf-8'))
    except (OSError,ValueError,TypeError,json.JSONDecodeError):data=None
    return canonical_research_comparisons(data,slate,now=now)
