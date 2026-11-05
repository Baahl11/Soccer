"""Full Bet365 reference-market cross-check AFTER independent sport-first slate.

Use only exact modeled match/market/selection/line probabilities. Derive extra
half-integer FT-goal probabilities from the existing model's RAW Poisson total
only; NEVER infer first-half, second-half, corners, fouls or player probabilities.
All results are RESEARCH / WATCH, never BET, OOS calibration or ticket EV.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from datetime import datetime,timezone
from itertools import combinations
from math import exp,factorial,isfinite
from pathlib import Path
from typing import Any
import json

from .parlay_radar import _valid_quote
from .parlay_risk import _raw_probability
from .market_history_evidence import past_goals_market_evidence

VERSION='SOCCER_BET365_SPORT_FIRST_MARKET_MATRIX_v1.0'

def _number(value,low=None,high=None):
    if value is None or isinstance(value,bool):return None
    try:
        n=float(value)
        if not isfinite(n):return None
        if low is not None and n<low:return None
        if high is not None and n>high:return None
        return n
    except (ValueError,TypeError,OverflowError):return None

def raw_exact_market_probability(leg,raw):
    """No bookmaker inputs in the sporting projection."""
    p,why=_raw_probability(leg,raw)
    if p is not None:return p,why
    if raw.get('status')!='MODELED_LIMITED':return None,'RAW_MODEL_NOT_AVAILABLE'
    if leg.get('family')!='FULL_TIME_GOALS':return None,why
    selection=str(leg.get('selection') or '').casefold()
    line=_number(leg.get('point'),.5,6.5)
    lam=_number(raw.get('raw_total_goals'),0,12)
    if selection not in ('over','under') or line is None or lam is None:
        return None,why
    if abs(line%1-.5)>1e-9:return None,'REFUND_LINE_NOT_BINARY'
    if abs(lam)>1e-12:
        under=exp(-lam)*sum(lam**k/factorial(k) for k in range(int(line)+1))
    else:under=1.
    p=1-under if selection=='over' else under
    return round(max(0.,min(1.,p)),6),'RAW_TOTAL_POISSON_DERIVED_HALF_LINE_UNCALIBRATED'

def _fair_market_lookup(quotes):
    """Only same book, exact named market, threshold and timestamp."""
    groups=defaultdict(dict)
    for q in quotes:
        key=(q['event_id'],q['bookmaker'].casefold(),q['market'].casefold(),
             str(q['point']),q['source_updated_at'],q['family'])
        sel=str(q['selection']).casefold().strip()
        # Duplicate same selection/market: refuse no-vig rather than arbitrary choice.
        if sel in groups[key]:groups[key][sel]=None
        else:groups[key][sel]=q['price_decimal']
    fair={}
    for key,prices in groups.items():
        family=key[-1]
        expected=({'home','draw','away'} if family=='1X2' else
                  {'yes','no'} if family=='BTTS' else
                  {'over','under'} if family in ('FULL_TIME_GOALS','FIRST_HALF_GOALS',
                      'SECOND_HALF_GOALS','CORNERS','FOULS','CARDS') else set())
        if not expected or set(prices)!=expected or any(
            v is None or v<=1 for v in prices.values()):continue
        margin=sum(1/v for v in prices.values())
        if margin<=0 or not 0.90<=margin<=1.40:continue
        for sel,price in prices.items():
            fair[key+(sel,)]=(1/price)/margin
    return fair

def _quotes_from_snapshots(snapshots):
    quotes=[]
    failed=[]
    for x in snapshots:
        try:
            p=Path(str(x['file']))
            if not p.is_file() or p.stat().st_size>20_000_000:raise ValueError('Snapshot missing/large')
            d=json.loads(p.read_text(encoding='utf-8'))
            if str(d.get('fixture_id'))!=str(x['fixture_id']):
                raise ValueError('Snapshot fixture mismatch')
            for q in d.get('quotes',[]):
                if str(q.get('event_id'))!=str(x['fixture_id']):
                    raise ValueError('Cross-fixture quote')
                quotes.append(q)
        except (OSError,ValueError,KeyError,TypeError):
            failed.append(str(x.get('fixture_id')))
    return quotes,failed

def examine_bet365_market_matrix(snapshots, sport_screen, market_registry, *,
                                 now=None,max_age_minutes=120,top_n=25):
    now=(now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    rows=sport_screen.get('screened_fixtures') or []
    fixtures={str(r.get('fixture_id')):r for r in rows if isinstance(r,dict)}
    families={f.get('family'):f for f in market_registry.get('families',[])
              if isinstance(f,dict) and f.get('family')}
    raw_quotes,failed=_quotes_from_snapshots(snapshots)
    accepted=[];reject=Counter();seen=set()
    for q in raw_quotes:
        leg,reason=_valid_quote(q,now,max_age_minutes,1.001,100.0)
        if not leg:
            reject[reason]+=1;continue
        if leg['bookmaker'].casefold()!='bet365':
            reject['NON_BET365_QUOTE']+=1;continue
        if leg['event_id'] not in fixtures:
            reject['NO_SPORT_FIRST_FIXTURE']+=1;continue
        key=(leg['event_id'],leg['bookmaker'].casefold(),leg['market'].casefold(),
             leg['selection'].casefold(),leg['point'],leg['source_updated_at'])
        if key in seen:
            reject['DUPLICATE_QUOTE']+=1;continue
        seen.add(key);accepted.append(leg)
    fair=_fair_market_lookup(accepted)
    diagnostics=[];modeled=0;raw_above=0
    for leg in accepted:
        fx=fixtures[leg['event_id']]
        raw=fx.get('raw_sport_projection') or {}
        p,why=raw_exact_market_probability(leg,raw)
        be=1/leg['price_decimal']
        market_key=(leg['event_id'],leg['bookmaker'].casefold(),leg['market'].casefold(),
                    str(leg['point']),leg['source_updated_at'],leg['family'],
                    leg['selection'].casefold().strip())
        m=fair.get(market_key)
        sample=_number((raw.get('sample') or {}).get('minimum_split_sample'),0)
        family_map={'FULL_TIME_GOALS':'FT_TOTALS','FIRST_HALF_GOALS':'FIRST_HALF',
                    'SECOND_HALF_GOALS':'SECOND_HALF','CORNERS':'FT_CORNERS'}
        family=family_map.get(leg['family'],leg['family'])
        gate=families.get(family,{})
        gap=(p-be)*100 if p is not None else None
        history_evidence=past_goals_market_evidence(leg,fx)
        # A positive RAW gap is NOT calibrated edge or EV.
        status=('RAW_ABOVE_BREAKEVEN_RESEARCH' if gap is not None and gap>0
                else 'RAW_NOT_ABOVE_BREAKEVEN' if gap is not None
                else 'MARKET_NO_EXACT_SPORT_MODEL')
        modeled+=int(p is not None)
        raw_above+=int(gap is not None and gap>0)
        diagnostics.append({**leg,'sport_first_fixture_verified':True,
              'raw_sport_probability':p,'raw_probability_source':why,
              'all_competition_past_goal_evidence':history_evidence,
              'venue_sample_min':int(sample) if sample is not None else None,
              'sample_sufficient_for_bet':False if sample is None or sample<8 else None,
              'breakeven_probability':round(be,6),
              'market_fair_no_vig':round(m,6) if m is not None else None,
              'raw_minus_breakeven_pp_NOT_EV':round(gap,2) if gap is not None else None,
              'raw_minus_fair_pp_NOT_EDGE':round((p-m)*100,2) if p is not None and m is not None else None,
              'market_validation':gate.get('research_status','NOT_VERIFIED'),
              'market_production_approved':bool(gate.get('production_promotion_allowed') is True
                    and gate.get('bet_eligible') is True),
              'classification':'WATCH','research_signal':status,
              'calibrated_probability':None,'conservative_probability_lower':None,
              'mexico_book_offer_verified':False,'expected_value':None,
              'notes':'Research raw Poisson, not OOS calibrated; Bet365 price observed through API-Football; reconfirm on Bet365 before staking.'})
    preliminary=[x for x in diagnostics if x['raw_sport_probability'] is not None]
    preliminary.sort(key=lambda x:(
        -(x['venue_sample_min'] or 0),
        -(x['raw_minus_breakeven_pp_NOT_EV'] or -100),
        x['event_id'],x['market'],x['selection']))
    top_markets=preliminary[:top_n]
    # Look for cross-match tickets near +100 where both legs at least pass RAW
    # breakeven sanity, rather than price proximity alone.
    # Bet365 can label the identical FT total as "Goal Line" and
    # "Goals Over/Under". Same event, family, outcome and half-integer
    # threshold is ONE sporting proposition, never two independent legs.
    candidates=[x for x in diagnostics if x['raw_sport_probability'] is not None
                and (x['raw_minus_breakeven_pp_NOT_EV'] or 0)>0
                and 1.28<=x['price_decimal']<=1.55]
    unique_legs={}
    for x in candidates:
        k=(x['event_id'],x['family'],x['selection'].casefold(),x['point'])
        prior=unique_legs.get(k)
        # Only choose the more favorable same-book identical-SETTLEMENT
        # reference line, without inventing an executable offered ticket.
        if prior is None or x['price_decimal']>prior['price_decimal']:
            unique_legs[k]=x
    priced_legs=list(unique_legs.values())
    pairs=[]
    for a,b in combinations(priced_legs,2):
        if a['event_id']==b['event_id']:continue
        combined=a['price_decimal']*b['price_decimal']
        if not 1.9<=combined<=2.1:continue
        min_sample=min(a['venue_sample_min'] or 0,b['venue_sample_min'] or 0)
        worst_gap=min(a['raw_minus_breakeven_pp_NOT_EV'],b['raw_minus_breakeven_pp_NOT_EV'])
        history_a=a.get('all_competition_past_goal_evidence') or {}
        history_b=b.get('all_competition_past_goal_evidence') or {}
        history_min=min(history_a.get('sample_minimum') or 0,history_b.get('sample_minimum') or 0)
        history_concordance=(history_a.get('both_team_directional_support') is True
                             and history_b.get('both_team_directional_support') is True
                             and history_min>=6)
        pairs.append({'classification':'WATCH','legs':[
            {k:x[k] for k in ('event_id','event_name','market','selection','point',
                              'price_decimal','raw_sport_probability',
                              'raw_probability_source','venue_sample_min',
                              'raw_minus_breakeven_pp_NOT_EV','source_updated_at',
                              'all_competition_past_goal_evidence')}
            for x in (a,b)],
            'bookmaker':'Bet365','theoretical_combined_decimal':round(combined,4),
            'combined_breakeven':round(1/combined,6),
            'minimum_raw_margin_pp_NOT_EV':round(worst_gap,2),
            'minimum_venue_sample':min_sample,
            'minimum_all_competitions_prior_sample':history_min,
            'all_competitions_descriptive_directional_agreement':history_concordance,
            'all_competitions_event_evidence_is_calibrated':False,
            'research_sample_status':('VERY_LOW_LEAGUE_VENUE_SAMPLE'
                                      if min_sample<5 else
                                      'SAMPLE_STILL_NOT_OOS_CERTIFIED'),
            'joint_probability':None,'joint_lower_probability':None,
            'joint_ev':None,'independence_verified':False,
            'mexico_offer_verified':False,
            'priority':'SPORTING_RESEARCH_REVIEW_NOT_RISK_RANK',
            'reason':'RAW_ONLY;OOS_CALIBRATION_AND_AVAILABILITY_AND_JOINT_DEPENDENCE_UNVERIFIED'})
    pairs.sort(key=lambda p:(-int(p['all_competitions_descriptive_directional_agreement']),
        -p['minimum_all_competitions_prior_sample'],
        -p['minimum_venue_sample'],
        -p['minimum_raw_margin_pp_NOT_EV'],
        abs(p['theoretical_combined_decimal']-2.0),
        p['legs'][0]['event_id'],p['legs'][1]['event_id']))
    diverse=[];event_pairs=Counter()
    for p in pairs:
        key=tuple(sorted(x['event_id'] for x in p['legs']))
        if event_pairs[key]>=4:continue
        event_pairs[key]+=1
        diverse.append(p)
        if len(diverse)>=12:break
    return {'version':VERSION,'generated_at_utc':now.isoformat(),
            'source':'API-Football Bet365 bookmaker ID 8',
            'scope':'PRICE_RESEARCH_ONLY_NOT_BETTING',
            'bookmaker':'Bet365','total_normalized_quotes':len(raw_quotes),
            'fresh_supported_market_quotes':len(accepted),
            'fresh_quoted_fixtures':len({x['event_id'] for x in accepted}),
            'bookmaker_snapshots_fetched':len(snapshots),
            'reject_reason_counts':dict(reject),
            'snapshot_read_failures':failed,
            'exact_or_poisson_derived_raw_quotes':modeled,
            'raw_above_breakeven_research_quotes':raw_above,
            'two_way_or_three_way_novig_quotes':len(fair),
            'all_quotes':diagnostics,
            'top_model_matched_markets':top_markets,
            'sporting_screened_plus100_pairs_found':len(pairs),
            'sporting_screened_plus100_pairs':diverse,
            'sporting_screened_plus100_pairs_min_sample_5':sum(
                p['minimum_venue_sample']>=5 for p in pairs),
            'sporting_screened_plus100_pairs_with_6plus_history':sum(
                p['minimum_all_competitions_prior_sample']>=6 for p in pairs),
            'sporting_screened_plus100_pairs_history_agreement':sum(
                p['all_competitions_descriptive_directional_agreement'] for p in pairs),
            'same_outcome_alias_quotes_collapsed':len(candidates)-len(priced_legs),
            'actionable_bets':0,'actionable_parlays':0,
            'caveats':['No model weights changed or fitted to bookmaker lines.',
                'Raw probability vs breakeven is NOT EV; quote may be stale outside the window.',
                'Independent events do not establish betting-outcome independence.',
                'Historical form/lineups, market family OOS and book offer remain material gates.',
                'Bet365 reference quote is not a directly confirmed Bet365 betslip.']}