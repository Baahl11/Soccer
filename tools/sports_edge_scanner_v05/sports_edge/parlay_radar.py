"""Price-only two-leg parlay *discovery* across researched Soccer fixtures.

SPORT FIRST: consumes existing sports shortlist and saved bookmaker snapshots;
never requests new odds, invents model probabilities, or recommends wagers.
Any result is WATCH until an independent validated model, book-specific offered
combined price, pregame availability and dependence checks all pass elsewhere.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from math import isfinite
from pathlib import Path
from typing import Any
import json

from .core import dec_to_american, parse_iso


_SUPPORTED_FAMILIES = {
    'match winner': '1X2',
    '1x2': '1X2',
    'both teams to score': 'BTTS',
    'both teams score': 'BTTS',
    'double chance': 'DOUBLE_CHANCE',
}


def _market_family(market: str, selection: str, point: str) -> str | None:
    m = (market or '').strip().lower()
    s = (selection or '').strip().lower()
    p = (point or '').strip()
    if m in ('match winner', '1x2'):
        return '1X2' if s in {'home','draw','away','1','x','2'} and not p else None
    if m in ('both teams to score','both teams score'):
        return 'BTTS' if s in {'yes','no','sí','si'} and not p else None
    if m == 'double chance':
        return 'DOUBLE_CHANCE' if s in {'1x','12','x2','home or draw','draw or away','home or away'} and not p else None
    if s not in {'over','under'}:
        return None
    # Exact line required. A selection such as "Over" without a point is ambiguous.
    try:
        line = float(p)
        if not isfinite(line) or line < 0 or line > 100 or not p:
            return None
    except (TypeError, ValueError):
        return None
    if 'corner' in m:
        return 'CORNERS'
    if 'foul' in m:
        return 'FOULS'
    if 'card' in m or 'booking' in m:
        return 'CARDS'
    if 'first half' in m or '1st half' in m or '1h ' in m:
        return 'FIRST_HALF_GOALS'
    if 'second half' in m or '2nd half' in m or '2h ' in m:
        return 'SECOND_HALF_GOALS'
    if 'goal' in m or m in {'over/under','totals'}:
        return 'FULL_TIME_GOALS'
    return None


def _valid_quote(quote: dict[str, Any], now: datetime, max_age_minutes: int,
                 min_leg_decimal: float, max_leg_decimal: float) -> tuple[dict[str, Any] | None, str]:
    if quote.get('sport') != 'soccer':
        return None, 'OTHER_SPORT'
    event = str(quote.get('event_id') or '').strip()
    book = str(quote.get('bookmaker') or '').strip()
    if not event or not book or not quote.get('source'):
        return None, 'UNVERIFIED_EVENT_BOOK_SOURCE'
    try:
        price = float(quote['price_decimal'])
        if not isfinite(price) or not min_leg_decimal <= price <= max_leg_decimal:
            return None, 'LEG_PRICE_OUTSIDE_RESEARCH_WINDOW'
        kickoff = parse_iso(str(quote['starts_at_utc']))
        published = parse_iso(str(quote['source_updated_at']))
    except (ValueError, KeyError, TypeError):
        return None, 'UNVERIFIED_KICKOFF_OR_QUOTE_TIMESTAMP'
    if kickoff <= now:
        return None, 'EVENT_ALREADY_STARTED'
    age = (now - published).total_seconds() / 60
    if age < -1 or age > max_age_minutes:
        return None, 'QUOTE_STALE_OR_FUTURE'
    market = str(quote.get('market') or '')
    sel = str(quote.get('selection') or '')
    point = str(quote.get('point') or '')
    family = _market_family(market, sel, point)
    if not family:
        return None, 'UNSUPPORTED_OR_AMBIGUOUS_MARKET'
    return {
        'event_id': event,
        'event_name': str(quote.get('event_name') or ''),
        'bookmaker': book,
        'source': str(quote.get('source') or ''),
        'market': market,
        'family': family,
        'selection': sel,
        'point': point,
        'price_decimal': round(price, 5),
        'price_american': dec_to_american(price),
        'source_updated_at': published.isoformat(),
        'quote_age_minutes': round(age, 1),
        'starts_at_utc': kickoff.isoformat(),
    }, 'OK'


def discover_two_leg_price_pairs(
        quotes: list[dict[str, Any]], *,
        now: datetime | None = None,
        target_min: float = 1.90,
        target_max: float = 2.10,
        min_leg_decimal: float = 1.24,
        max_leg_decimal: float = 1.55,
        max_age_minutes: int = 120,
        top_n: int = 15,
        ) -> dict[str, Any]:
    """Expose *priced research pairs*, not edge or risk estimates.

    Different games and the same actual named bookmaker are necessary but do
    NOT prove statistical independence; every result is WATCH only.
    """
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if not (1 < target_min < target_max < 4) or not (1 < min_leg_decimal < max_leg_decimal < 4):
        raise ValueError('Invalid decimal odds bounds')
    if not (1 <= max_age_minutes <= 1440) or not (1 <= top_n <= 100):
        raise ValueError('Invalid scan limits')
    accepted, exclusions = [], Counter()
    seen = set()
    for q in quotes:
        leg, error = _valid_quote(q, now, max_age_minutes, min_leg_decimal, max_leg_decimal)
        if leg is None:
            exclusions[error] += 1
            continue
        # Duplicate provider pages may repeat an identical book/event/selection.
        key = (leg['event_id'], leg['bookmaker'].casefold(), leg['family'],
               leg['market'].casefold(), leg['selection'].casefold(), leg['point'],
               leg['price_decimal'], leg['source_updated_at'])
        if key not in seen:
            accepted.append(leg)
            seen.add(key)
    books = defaultdict(list)
    for leg in accepted:
        books[(leg['source'], leg['bookmaker'].casefold())].append(leg)
    results = []
    for _, legs in books.items():
        for a,b in combinations(legs, 2):
            if a['event_id'] == b['event_id']:
                continue
            decimal = a['price_decimal']*b['price_decimal']
            if not target_min <= decimal <= target_max:
                continue
            legs_pair = sorted([a,b],key=lambda x:(x['event_id'],x['market'],x['selection'],x['point']))
            results.append({
                'status':'WATCH',
                'priority':'PRICE_TARGET_ONLY_NOT_SPORTING_EDGE',
                'legs':legs_pair,
                'bookmaker':a['bookmaker'],
                'source':a['source'],
                'theoretical_combined_decimal':round(decimal,4),
                'theoretical_combined_american':dec_to_american(decimal),
                'joint_breakeven':round(1/decimal,6),
                'difference_from_decimal_2':round(abs(decimal-2),4),
                'joint_model_probability':None,
                'estimated_ev':None,
                'calibrated_both_legs':False,
                'dependence_evidence':'NOT_VERIFIED',
                'offered_parlay_price':'NOT_VERIFIED',
                'caliente_playdoit_price':'NOT_VERIFIED',
                'availability_confidence':'NOT_VERIFIED',
                'reason':'TWO_CALIBRATED_PROJECTIONS_AND_INDEPENDENCE_AND_BOOK_OFFER_REQUIRED',
            })
    # Proximity to +100 is a sorting convenience; it does NOT mean lower risk.
    results.sort(key=lambda x:(x['difference_from_decimal_2'],
                               max(y['quote_age_minutes'] for y in x['legs']),
                               x['legs'][0]['event_id'],x['legs'][1]['event_id']))
    return {
        'status':'PRICE_ONLY_WATCH',
        'generated_at_utc':now.isoformat(),
        'scope':'TWO_LEG_DIFFERENT_FIXTURES_SAME_NAMED_BOOK',
        'target_decimal_range':[target_min,target_max],
        'accepted_legs':len(accepted),
        'rejected_quote_reasons':dict(exclusions),
        'pairs_found_before_display_cap':len(results),
        'pairs':results[:top_n],
        'actionable_parlays':0,
        'wagers_placed':0,
        'notes':[
            'No price-only candidate is a pick, model-validated probability or positive-EV wager.',
            'Different events do not by themselves prove outcome independence.',
            'API-Football third-party odds are NOT verified available at Caliente/Playdoit.',
            'The combined price is arithmetic multiplication; the actual parlay offer is unverified.',
            'No selection enters BET without independent sport-first projections, OOS calibration, '
            'verified availability, lower-confidence-bound EV, dependence review and exact sportsbook price.',
        ],
    }


def discover_saved_snapshot_pairs(snapshots: list[dict[str, Any]], *, now: datetime | None = None,
                                  max_age_minutes: int = 120) -> dict[str, Any]:
    """Read only saved price observations. Never makes additional API calls."""
    quotes: list[dict[str,Any]] = []
    skipped = []
    for s in snapshots:
        if not isinstance(s,dict) or not s.get('file'):
            skipped.append('MISSING_SNAPSHOT_PATH');continue
        p=Path(str(s['file']))
        if not p.is_file():
            skipped.append('SNAPSHOT_NOT_FOUND');continue
        try:
            payload=json.loads(p.read_text(encoding='utf-8'))
            if str(payload.get('fixture_id')) != str(s.get('fixture_id')):
                raise ValueError('Fixture mismatch')
            for q in payload.get('quotes') or []:
                if str(q.get('event_id')) != str(s.get('fixture_id')):
                    raise ValueError('Quote fixture mismatch')
                quotes.append(q)
        except (ValueError,OSError,TypeError):
            skipped.append('SNAPSHOT_INVALID')
    result=discover_two_leg_price_pairs(quotes,now=now,max_age_minutes=max_age_minutes)
    result['snapshots_read']=len(snapshots)-len(skipped)
    result['snapshot_read_failures']=skipped
    return result