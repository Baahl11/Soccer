"""SPORTS EDGE ENGINE - auditable, sport-isolated price and candidate screening.

Never manufactures sporting probabilities from bookmaker prices.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from typing import Any, Iterable
import math

# Thresholds are provisional and sourced from sport-specific project masters.
# Important: they are thresholds against fair probability if an opposite quote exists;
# otherwise against breakeven. They are NOT a claim of model calibration.
EDGE_THRESHOLDS_PP = {
    'soccer': {'side': {'B': 3.0, 'A': 5.0, 'S': 7.5},
               'goals': {'B': 3.5, 'A': 5.5, 'S': 8.0},
               'derivative': {'B': 4.0, 'A': 6.0, 'S': 9.0}},
    'nfl': {'side': {'B': 3.0, 'A': 5.0, 'S': 8.0},
            'goals': {'B': 3.0, 'A': 5.0, 'S': 8.0},
            'derivative': {'B': 4.0, 'A': 6.0, 'S': 9.0}},
    'ncaaf': {'side': {'B': 3.0, 'A': 5.0, 'S': 8.0},
              'goals': {'B': 3.0, 'A': 5.0, 'S': 8.0},
              'derivative': {'B': 4.0, 'A': 6.0, 'S': 9.0}},
}

def dec_to_american(odds: float) -> int:
    if odds <= 1 or not math.isfinite(odds):
        raise ValueError('Invalid decimal odds')
    return round(100 * (odds - 1)) if odds >= 2 else round(-100 / (odds - 1))

def breakeven(odds_decimal: float) -> float:
    if odds_decimal <= 1 or not math.isfinite(odds_decimal):
        raise ValueError('Odds decimal must be finite and >1')
    return 1 / odds_decimal

def ev_per_unit(p: float, odds_decimal: float) -> float:
    return p * odds_decimal - 1

def parse_iso(ts: str) -> datetime:
    d = datetime.fromisoformat(ts.replace('Z', '+00:00'))
    if d.tzinfo is None:
        raise ValueError('Timestamps must have timezone')
    return d.astimezone(timezone.utc)

def fresh(quote: dict[str, Any], now: datetime, minutes: int) -> bool:
    try:
        # Provider publication timestamp takes precedence over retrieval timestamp.
        # Cached odds must not be relabeled as fresh merely because we fetched them.
        if not quote.get('source_updated_at'):
            return False
        age = (now - parse_iso(quote['source_updated_at'])).total_seconds()
        return -60 <= age <= minutes * 60
    except (ValueError, KeyError, TypeError):
        return False

def market_family(sport: str, market: str) -> str:
    key = market.lower()
    if sport not in EDGE_THRESHOLDS_PP:
        return 'unsupported'
    if key in ('h2h','moneyline','1x2','match winner','draw no bet',
               'draw_no_bet','asian handicap','asian_handicap','spreads'):
        return 'side'
    if key in ('totals', 'btts', 'over_under','goals over/under','both teams score'):
        return 'goals'
    return 'derivative'

def same_quote(q: dict[str, Any], p: dict[str, Any]) -> bool:
    return all(str(q.get(k,'')) == str(p.get(k,'')) for k in
               ('sport','event_id','market','selection','point'))

def validate_projection(p: dict[str, Any]) -> None:
    for key in ('p_raw','p_shrunk','p_lower'):
        value = float(p.get(key, -1))
        if not 0 < value < 1 or not math.isfinite(value):
            raise ValueError(f'Projection missing/invalid {key}')
    if float(p['p_lower']) > float(p['p_shrunk']):
        raise ValueError('p_lower must be <= p_shrunk')
    if p.get('sport') not in EDGE_THRESHOLDS_PP:
        raise ValueError('Unsupported sport without its own calibrated rules')

def evaluate(quote: dict[str,Any], projection: dict[str,Any] | None,
             now: datetime | None = None, max_age_minutes: int = 180) -> dict[str,Any]:
    now = now or datetime.now(timezone.utc)
    row = {k:quote.get(k) for k in ('sport','event_id','event_name','starts_at_utc',
        'market','selection','point','bookmaker','source','captured_at_utc','source_updated_at')}
    row.update({'status':'WATCH','reason':'MODEL_NOT_VERIFIED','price_decimal':quote.get('price_decimal'),
                'stake_units':0,'p_model':None,'p_lower':None,'p_breakeven':None,
                'prob_edge_pp':None,'conservative_ev':None,'tier':'NONE'})
    try:
        price = float(quote['price_decimal']); be = breakeven(price)
        row['p_breakeven'] = round(be,6)
    except (KeyError,ValueError,TypeError):
        row.update(reason='PRICE_NOT_VERIFIED'); return row
    if not quote.get('source') or not quote.get('bookmaker') or not fresh(quote,now,max_age_minutes):
        row.update(reason='SOURCE_OR_TIMESTAMP_NOT_VERIFIED'); return row
    # A missing kickoff cannot be assumed to be a future event.
    if not quote.get('starts_at_utc'):
        row.update(reason='KICKOFF_NOT_VERIFIED'); return row
    try:
        if parse_iso(quote['starts_at_utc']) <= now:
            row.update(reason='EVENT_ALREADY_STARTED'); return row
    except (ValueError, TypeError):
        row.update(reason='KICKOFF_NOT_VERIFIED'); return row
    if not projection or not same_quote(quote,projection): return row
    try: validate_projection(projection)
    except (ValueError, TypeError):
        row.update(reason='MODEL_INPUT_INVALID'); return row
    p = float(projection['p_shrunk']); lower = float(projection['p_lower'])
    fair = quote.get('fair_market_probability')
    if fair is not None:
        try:
            fair = float(fair)
            if not (0 < fair < 1): raise ValueError()
        except (TypeError,ValueError):
            row.update(reason='FAIR_PRICE_INVALID'); return row
    reference = fair if fair is not None else be
    edge = 100*(p-reference)
    row.update(p_model=round(p,6),p_lower=round(lower,6),
               raw_projection=projection['p_raw'],model_version=projection.get('model_version'),
               fair_market_probability=fair,prob_edge_pp=round(edge,2),
               conservative_ev=round(ev_per_unit(lower,price),4),
               availability_confidence=projection.get('availability_confidence'))
    if not projection.get('calibrated',False):
        row.update(reason='PROVISIONAL_UNCALIBRATED_MODEL');return row
    if not projection.get('material_availability_verified',False) or not (0.5 <= float(projection.get('availability_confidence',0)) <=1):
        row.update(reason='AVAILABILITY_NOT_VERIFIED');return row
    family=market_family(projection['sport'],projection['market'])
    if family == 'unsupported':
        row.update(reason='SPORT_NOT_SUPPORTED'); return row
    th=EDGE_THRESHOLDS_PP[projection['sport']][family]
    tier='NONE'
    for candidate in ('S','A','B'):
        if edge >= th[candidate]: tier=candidate;break
    row['tier']=tier
    if tier=='NONE' or ev_per_unit(lower,price)<=0:
        row.update(status='PASS',reason='INSUFFICIENT_VALUE');return row
    if tier=='B':
        row.update(status='LEAN',reason='BELOW_PARLAY_MINIMUM_TIER_A');return row
    row.update(status='BET',reason='VERIFIED_MODEL_AND_PRICE')
    return row

def generate_parlays(evals: list[dict[str,Any]], min_decimal:float=1.9,
                     max_decimal:float=2.2, require_independence:bool=True,
                     evidence:dict[str,str] | None=None) -> list[dict[str,Any]]:
    """Two-leg cross-event only; refuse unverified dependency and no implied SGP prices.
    Each p_lower is an explicitly supplied, calibrated conservative lower estimate.
    Independence evidence is an input from joint-risk validation (not inferred from names).
    """
    approved=[r for r in evals if r['status']=='BET' and r['tier'] in ('S','A')]
    found=[]
    for a,b in combinations(approved,2):
        if a['event_id']==b['event_id'] or a['bookmaker']!=b['bookmaker']:
            continue
        pair_key='|'.join(sorted([str(a['event_id']),str(b['event_id'])]))
        check=(evidence or {}).get(pair_key,'NOT_VERIFIED')
        if require_independence and check != 'VERIFIED': continue
        oddsa=float(a['price_decimal']);oddsb=float(b['price_decimal'])
        combined=oddsa*oddsb
        if not min_decimal <= combined <= max_decimal: continue
        # Only multiply if supplied independence evidence is verified.
        if check != 'VERIFIED':continue
        p_lower=float(a['p_lower'])*float(b['p_lower'])
        if p_lower <= 1/combined: continue
        found.append({
            'status':'WATCH', # Additional portfolio review and offered parlay price required.
            'reason':'PARLAY_PRICE_AND_PORTFOLIO_REVIEW_REQUIRED',
            'event_ids':[a['event_id'],b['event_id']],
            'selections':[a['selection'],b['selection']],
            'bookmaker':a['bookmaker'],'theoretical_price_decimal':round(combined,4),
            'theoretical_american':dec_to_american(combined),
            'conservative_joint_p':round(p_lower,6),
            'theoretical_ev':round(ev_per_unit(p_lower,combined),4),
            'dependence_evidence':check,
        })
    return sorted(found,key=lambda x:x['theoretical_ev'],reverse=True)


def attach_market_fair_probabilities(quotes: list[dict[str,Any]]) -> list[dict[str,Any]]:
    """Devig only mutually exclusive/exhaustive quotes from same market/book/snapshot.

    No devig for overlapping double chance, derivatives, uncertain pairs, or SGP.
    Never de-vig the spread number itself.
    """
    from collections import defaultdict
    groups=defaultdict(list)
    for q in quotes:
        market=str(q.get('market','')).lower()
        if market not in ('match winner','1x2','h2h','moneyline','draw no bet','draw_no_bet'):
            continue
        key=tuple(str(q.get(k,'')) for k in
                  ('sport','event_id','bookmaker','market','point','source_updated_at'))
        groups[key].append(q)
    out=[dict(q) for q in quotes]
    for positions in groups.values():
        names=[str(x.get('selection','')).lower() for x in positions]
        if len(positions) not in (2,3) or len(set(names))!=len(names):continue
        # Require exhaustive event outcome set for 3-way football and two-way moneylines.
        m=str(positions[0].get('market','')).lower()
        if m in ('match winner','1x2') and len(positions)!=3:continue
        if m in ('h2h','moneyline','draw no bet','draw_no_bet') and len(positions)!=2 and len(positions)!=3:continue
        try:
            implied=[breakeven(float(q['price_decimal'])) for q in positions]
        except (KeyError,TypeError,ValueError):continue
        total=sum(implied)
        if total<=1 or total>1.5:continue  # validate actual bookmaker overround
        for original,p in zip(positions,implied):
            # Keep matching original quote identity, not cross-book or cross-time pairing.
            ix=next((i for i,v in enumerate(out) if v.get('event_id')==original.get('event_id') and
                v.get('bookmaker')==original.get('bookmaker') and v.get('market')==original.get('market') and
                v.get('selection')==original.get('selection') and v.get('source_updated_at')==original.get('source_updated_at')),None)
            if ix is not None:out[ix]['fair_market_probability']=round(p/total,9)
    return out