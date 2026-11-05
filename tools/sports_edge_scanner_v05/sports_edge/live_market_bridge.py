"""Bridge real, observed sport-first Soccer fixtures to real reference market rows.

This is *not* a sportsbook execution feed: API-Football Bet365 prices are
reference observations, and MODELED_LIMITED probabilities are not calibrated.
The exact same final decision gate is executed on each match, preserving its
explicit blockers. Authorised signed feeds remain the only BET promotion path.

No new network requests, probability fitting, or changes in model weights.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from math import isfinite
from typing import Any

from .core import parse_iso
from .decision_gate import evaluate_verified_leg

VERSION = 'SOCCER_BET365_LIVE_DECISION_BRIDGE_v1.5.0'


def _finite_probability(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        x = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return x if isfinite(x) and 0.0 < x < 1.0 else None


def _future_timestamp(value: Any, now: datetime) -> bool:
    try:
        return parse_iso(str(value)) > now
    except (ValueError, TypeError, OverflowError):
        return False


def _fresh_timestamp(value: Any, now: datetime, max_age_minutes: int) -> bool:
    try:
        age = (now - parse_iso(str(value))).total_seconds() / 60.0
        return -1 <= age <= max_age_minutes
    except (ValueError, TypeError, OverflowError):
        return False


def _line(row: dict[str, Any]) -> str:
    # Preserve point identity exactly as the source market: 2.5 != 3.5.
    return str(row.get('point') or '').strip()


def _key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (str(row.get('event_id') or '').strip(),
            str(row.get('market') or '').strip().casefold(),
            str(row.get('selection') or '').strip().casefold(),
            _line(row))


def _shadow_model(quote: dict[str, Any], sport: dict[str, Any]) -> dict[str, Any]:
    raw = sport.get('raw_sport_projection') or {}
    # Intentionally NOT promoted: the original v1.0 Poisson is uncalibrated.
    return {
        'sport': 'soccer', 'event_id': str(quote['event_id']),
        'market': quote['market'], 'selection': quote['selection'],
        'point': _line(quote), 'model_version': raw.get('model_version'),
        'model_artifact_id': None,
        'promotion_status': 'RESEARCH_ONLY',
        'production_promotion_allowed': False,
        'calibrated': False, 'calibration_out_of_sample': False,
        'p_raw': quote.get('raw_sport_probability'),
        'p_shrunk': None, 'p_lower': None,
        'raw_sport_built_without_market_input': True,
        'fixture_status': sport.get('fixture_status') or 'NOT_VERIFIED',
        'fixture_status_verified': False,
        'xi_and_goalkeepers_verified': False,
        'material_availability_verified': False,
        'unresolved_material_conflict': None,
    }


def bridge_real_reference_markets(
    matrix: dict[str, Any], sport_first: dict[str, Any],
    *, now: datetime | None = None, max_age_minutes: int = 120,
    max_rows: int = 600,
) -> dict[str, Any]:
    """Run the original decision gate against *actual* Bet365 market evidence.

    The output must never be treated as a BET/EV feed. Raw gaps are
    descriptive only and cannot be used as calibrated risk rankings.
    """
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if not 1 <= max_rows <= 2000 or not 1 <= max_age_minutes <= 240:
        raise ValueError('Invalid bounds')
    report: dict[str, Any] = {
        'version': VERSION, 'generated_at_utc': now.isoformat(),
        'source': 'API_FOOTBALL_BET365_REFERENCE+EXISTING_RAW_SOCCER_MODEL',
        'status': 'RESEARCH_ONLY', 'feed_kind': 'NON_ACTIONABLE_REFERENCE',
        'screened_slate_complete': bool(sport_first.get('complete_sport_screen')),
        'quotes_examined': 0, 'quotes_matched_to_sport': 0,
        'quotes_exact_raw_available': 0, 'quotes_sent_to_final_decision_gate': 0,
        'raw_above_breakeven': 0, 'raw_at_or_below_breakeven': 0,
        'final_bets': 0, 'final_parlay_bets': 0,
        'rejection_reasons': {}, 'decision_blocker_counts': {},
        'decisions': [], 'parlay_pair_reviews': [],
        'note': 'Original decision gate evaluates real quoted markets. Research raw estimates and Bet365 reference odds NEVER become actionable offers or model approvals.',
    }
    if not report['screened_slate_complete']:
        report['status'] = 'INCOMPLETE_SPORT_FIRST_SLATE'
        report['final_parlay_bets'] = sum(x['classification'] == 'BET' for x in report['parlay_pair_reviews'])
    return report
    sport_rows = sport_first.get('screened_fixtures') or []
    sport_index = {str(s.get('fixture_id')): s for s in sport_rows if isinstance(s, dict)}
    matrix_rows = matrix.get('all_quotes') or []
    rejections: Counter[str] = Counter()
    blockers: Counter[str] = Counter()
    keys: set[tuple[str, str, str, str, str]] = set()
    decisions: list[dict[str, Any]] = []
    by_quote_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in matrix_rows[:max_rows]:
        report['quotes_examined'] += 1
        if not isinstance(row, dict):
            rejections['NOT_A_QUOTE'] += 1
            continue
        if str(row.get('bookmaker') or '').casefold() != 'bet365' or row.get('sport_first_fixture_verified') is not True:
            rejections['NOT_SAME_REFERENCE_BOOK_OR_SPORT'] += 1
            continue
        fid = str(row.get('event_id') or '').strip()
        sport = sport_index.get(fid)
        if not sport:
            rejections['FIXTURE_NOT_IN_SPORT_FIRST_SLATE'] += 1
            continue
        if not _future_timestamp(row.get('starts_at_utc'), now):
            rejections['MATCH_STARTED_OR_KICKOFF_NOT_VERIFIED'] += 1
            continue
        if not _fresh_timestamp(row.get('source_updated_at'), now, max_age_minutes):
            rejections['REFERENCE_PRICE_STALE'] += 1
            continue
        if str(row.get('market') or '').strip() == '' or str(row.get('selection') or '').strip() == '':
            rejections['MARKET_IDENTITY_MISSING'] += 1
            continue
        try:
            price = float(row.get('price_decimal'))
            if not isfinite(price) or not 1.0 < price < 100.0:
                raise ValueError('invalid price')
        except (TypeError, ValueError, OverflowError):
            rejections['PRICE_NOT_VERIFIED'] += 1
            continue
        event_key = _key(row)
        unique_key = event_key + (str(row.get('source_updated_at')),)
        if unique_key in keys:
            rejections['EXACT_DUPLICATE_PRICE_QUOTE'] += 1
            continue
        keys.add(unique_key)
        report['quotes_matched_to_sport'] += 1
        p_raw = _finite_probability(row.get('raw_sport_probability'))
        if p_raw is not None:
            report['quotes_exact_raw_available'] += 1
        # Reject the common mistake of using model probabilities on incompatible
        # markets. Matrix computes only an exactly matched or valid derived RAW.
        quote = {
            'sport': 'soccer', 'event_id': fid,
            'event_name': row.get('event_name'),
            'market': row.get('market'), 'selection': row.get('selection'),
            'point': _line(row), 'bookmaker': 'Bet365',
            'price_decimal': price, 'source': row.get('source'),
            'starts_at_utc': row.get('starts_at_utc'),
            'source_updated_at': row.get('source_updated_at'),
            # Verified observation from the selected API-Football Bet365 ID 8.
            # This asserts the provenance of the quote, NOT the account betslip.
            'bookmaker_id': 8,
            'price_observation_verified': row.get('source') == 'API-Football',
            'verified_offer_source': 'API_FOOTBALL_BET365_ID8',
            'verified_offer': False,
            'offer_reference_id': None,
        }
        model = _shadow_model({**quote, 'raw_sport_probability': p_raw}, sport) if p_raw is not None else None
        decision = evaluate_verified_leg(quote, model, now=now)
        # If these ever become BET, fail loudly: the reference path must NEVER
        # inherit production approval from synthetic or unsigned research data.
        if decision['status'] == 'BET' and (
                not model or model.get('production_promotion_allowed') is not True):
            raise AssertionError('Unapproved sport model illegally promoted to BET')
        report['quotes_sent_to_final_decision_gate'] += 1
        blockers.update(decision.get('blockers') or [])
        breakeven = 1.0 / price
        gap = (p_raw - breakeven) * 100 if p_raw is not None else None
        if gap is not None and gap > 0:
            report['raw_above_breakeven'] += 1
        elif gap is not None:
            report['raw_at_or_below_breakeven'] += 1
        try:
            sample = int(row.get('venue_sample_min'))
        except (ValueError, TypeError, OverflowError):
            sample = 0
        # These labels evaluate available evidence, NOT bets, odds, or joint risk.
        stage = ('NO_EXACT_SPORT_PROBABILITY' if p_raw is None else
                 'INSUFFICIENT_VENUE_SPLIT_SAMPLE' if sample < 8 else
                 'RAW_ABOVE_PRICE_RESEARCH' if gap is not None and gap > 0 else
                 'RAW_NOT_ABOVE_PRICE_RESEARCH')
        item = {
            'fixture_id': fid, 'event_name': row.get('event_name'),
            'market': row.get('market'), 'family': row.get('family'),
            'selection': row.get('selection'), 'point': _line(row),
            'bookmaker': 'Bet365', 'price_decimal': price,
            'source_updated_at': row.get('source_updated_at'),
            'raw_sport_probability_UNCALIBRATED': p_raw,
            'breakeven_probability': round(breakeven, 6),
            'raw_gap_pp_NOT_EV': round(gap, 3) if gap is not None else None,
            'minimum_venue_split_sample': sample,
            'sporting_evidence_stage': stage,
            'model_version': (sport.get('raw_sport_projection') or {}).get('model_version'),
            'market_production_approved': False,
            'sportsbook_offer_verified': False,
            'decision': decision['status'],
            'decision_reason': decision.get('reason'),
            'decision_blockers': decision.get('blockers', []),
            'conservative_joint_probability': None,
            'expected_value': None,
        }
        decisions.append(item)
        by_quote_key.setdefault(event_key, item)
    report['decisions'] = sorted(decisions, key=lambda x:(
        -x['minimum_venue_split_sample'],
        -float(x['raw_gap_pp_NOT_EV'] if x['raw_gap_pp_NOT_EV'] is not None else -1000),
        x['fixture_id'], str(x['market']), str(x['selection']), str(x['point'])))
    report['final_bets'] = sum(x['decision'] == 'BET' for x in decisions)
    report['rejection_reasons'] = dict(rejections)
    report['decision_blocker_counts'] = dict(blockers)
    # Full price-only parlays are audited using the SAME actual leg decisions,
    # never multiplied probabilities or inferred independence.
    for pair in matrix.get('sporting_screened_plus100_pairs') or []:
        leg_reviews = []
        for leg in pair.get('legs') or []:
            record = by_quote_key.get(_key(leg))
            leg_reviews.append({
                'event_id': str(leg.get('event_id') or ''),
                'market': leg.get('market'),
                'selection': leg.get('selection'),
                'point': _line(leg),
                'observed': record is not None,
                'decision': record.get('decision') if record else 'NOT_VERIFIED',
                'sporting_evidence_stage': record.get('sporting_evidence_stage') if record else 'NOT_VERIFIED',
                'blockers': record.get('decision_blockers') if record else ['LEG_NOT_MATCHED_IN_REFERENCE_SNAPSHOT'],
            })
        report['parlay_pair_reviews'].append({
            'classification': 'WATCH', 'bookmaker': 'Bet365',
            'theoretical_combined_decimal': pair.get('theoretical_combined_decimal'),
            'legs': leg_reviews,
            'offered_ticket_verified': False,
            'joint_p_lower': None, 'joint_ev': None,
            'reason': 'EXACT_OFFER_AND_INDEPENDENT_JOINT_RISK_MODEL_NOT_VERIFIED',
        })
    return report