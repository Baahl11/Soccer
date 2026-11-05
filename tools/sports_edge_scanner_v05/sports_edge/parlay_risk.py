"""Auditable sport-first evidence overlay for price-only Soccer parlay research.

Never fabricates calibration, bookmaker offers, joint independence, stake or EV.
Raw Poisson estimates are displayed as research diagnostics, not risk rankings.
"""
from __future__ import annotations

from copy import deepcopy
from math import isfinite
from typing import Any

VERSION = 'SOCCER_PARLAY_LEG_EVIDENCE_V0.9.0'


def _prob(value: Any) -> float | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        x = float(value)
        if isfinite(x) and 0 <= x <= 1:
            return x
    except (TypeError, ValueError):
        pass
    return None


def _raw_probability(leg: dict[str, Any], raw: dict[str, Any]) -> tuple[float | None, str]:
    """Use existing raw model's exact available market keys, never interpolate lines."""
    if raw.get('status') != 'MODELED_LIMITED':
        return None, 'RAW_SPORT_MODEL_NOT_AVAILABLE'
    family = leg.get('family')
    selection = str(leg.get('selection') or '').strip().casefold()
    point = str(leg.get('point') or '').strip()
    field = None
    invert = False
    if family == '1X2' and not point:
        field = {
            '1': 'raw_home_win_prob', 'home': 'raw_home_win_prob',
            'x': 'raw_draw_prob', 'draw': 'raw_draw_prob',
            '2': 'raw_away_win_prob', 'away': 'raw_away_win_prob',
        }.get(selection)
    elif family == 'BTTS' and not point:
        if selection in {'yes', 'si', 'sí'}:
            field = 'raw_btts_yes_prob'
        elif selection == 'no':
            field = 'raw_btts_yes_prob'
            invert = True
    elif family == 'FULL_TIME_GOALS' and selection in {'over', 'under'}:
        # ONLY lines actually supplied by the original raw soccer model.
        field = {'1.5': 'raw_over_1_5_prob',
                 '2.5': 'raw_over_2_5_prob',
                 '3.5': 'raw_over_3_5_prob'}.get(point)
        invert = selection == 'under'
    elif family == 'DOUBLE_CHANCE' and not point:
        keys = {
            '1x': ('raw_home_win_prob', 'raw_draw_prob'),
            '12': ('raw_home_win_prob', 'raw_away_win_prob'),
            'x2': ('raw_draw_prob', 'raw_away_win_prob'),
            'home or draw': ('raw_home_win_prob', 'raw_draw_prob'),
            'home or away': ('raw_home_win_prob', 'raw_away_win_prob'),
            'draw or away': ('raw_draw_prob', 'raw_away_win_prob'),
        }.get(selection)
        if keys:
            p1, p2 = _prob(raw.get(keys[0])), _prob(raw.get(keys[1]))
            if p1 is not None and p2 is not None and p1 + p2 <= 1.000001:
                return round(p1 + p2, 6), 'RAW_SUM_OF_DISJOINT_1X2_OUTCOMES_UNCALIBRATED'
            return None, 'INVALID_RAW_1X2_COMPONENTS'
    if field is None:
        return None, 'NO_SPORT_PROBABILITY_FOR_EXACT_MARKET_AND_LINE'
    value = _prob(raw.get(field))
    if value is None:
        return None, 'EXACT_RAW_PROBABILITY_MISSING_OR_INVALID'
    return round(1 - value if invert else value, 6), 'RAW_POISSON_UNCALIBRATED'


def _family_gate(leg: dict[str, Any]) -> str:
    f = str(leg.get('family') or '')
    if f == 'FULL_TIME_GOALS':
        return 'FT_TOTALS'
    if f == 'FIRST_HALF_GOALS':
        return 'FIRST_HALF'
    if f == 'SECOND_HALF_GOALS':
        return 'SECOND_HALF'
    if f == 'CORNERS':
        m = str(leg.get('market') or '').casefold()
        in_first_half = any(token in m for token in ('1st half', 'first half', '1h ', '(1h)'))
        team = 'home corner' in m or 'away corner' in m or 'team corner' in m
        if in_first_half:
            return 'FIRST_HALF_TEAM_CORNERS' if team else 'FIRST_HALF_CORNERS'
        return 'TEAM_CORNERS' if team else 'FT_CORNERS'
    return {'1X2': '1X2', 'BTTS': 'BTTS',
            'DOUBLE_CHANCE': 'DOUBLE_CHANCE', 'FOULS': 'FOULS',
            'CARDS': 'CARDS'}.get(f, f or 'NOT_VERIFIED')


def frechet_joint_lower(p1_lower: float, p2_lower: float) -> float:
    """Distribution-free lower bound; only meaningful for *validated* marginal bounds.

    For any events A and B: P(A and B) >= max(0, P(A)+P(B)-1).
    Never feed raw uncalibrated probabilities into this function as "lower" inputs.
    """
    a, b = _prob(p1_lower), _prob(p2_lower)
    if a is None or b is None:
        raise ValueError('Both lower probabilities must be finite numbers in [0,1]')
    return max(0.0, a + b - 1.0)


def annotate_parlay_radar(
    radar: dict[str, Any], sport_screen: dict[str, Any], registry: dict[str, Any]
) -> dict[str, Any]:
    """Annotate existing pairs with original sport-model evidence; always WATCH.

    This is intentionally not a calibrator. The reports for a market family do
    not constitute fixture-specific OOS lower probabilities, and the provided
    book prices are not verified at the user's bookmaker.
    """
    result = deepcopy(radar)
    fixtures = {
        str(r['fixture_id']): r for r in sport_screen.get('screened_fixtures', [])
        if isinstance(r, dict) and r.get('fixture_id') is not None
    }
    families = {
        str(f['family']): f for f in registry.get('families', [])
        if isinstance(f, dict) and f.get('family')
    }
    rejected_raw = 0
    unmodeled = 0
    annotated_pairs = []
    for pair in result.get('pairs', []):
        if not isinstance(pair, dict) or len(pair.get('legs') or []) != 2:
            continue
        annotated = []
        blockers = ['JOINT_DEPENDENCE_NOT_VERIFIED',
                    'CALIENTE_PLAYDOIT_EXACT_OFFER_NOT_VERIFIED',
                    'BOTH_LEGS_OOS_LOWER_PROBABILITIES_NOT_VERIFIED',
                    'PORTFOLIO_AND_STAKE_NOT_VERIFIED']
        for leg in pair['legs']:
            fx = fixtures.get(str(leg.get('event_id')))
            raw = (fx or {}).get('raw_sport_projection') or {}
            p, status = _raw_probability(leg, raw)
            sample = (raw.get('sample') or {}).get('minimum_split_sample')
            try:
                n = int(sample) if sample is not None and not isinstance(sample, bool) else None
            except (ValueError, TypeError):
                n = None
            family = _family_gate(leg)
            gate = families.get(family, {})
            try:
                price = float(leg['price_decimal'])
                price_break_even = 1 / price if isfinite(price) and price > 1 else None
            except (TypeError, KeyError, ValueError):
                price_break_even = None
            comparison = None
            if p is not None and price_break_even is not None:
                comparison = round(100.0 * (p - price_break_even), 2)
                if comparison <= 0:
                    rejected_raw += 1
            else:
                unmodeled += 1
            leg_blockers = []
            if fx is None:
                leg_blockers.append('FIXTURE_NOT_IN_COMPLETE_SPORT_SCREEN')
            if p is None:
                leg_blockers.append(status)
            if n is None or n < 8:
                leg_blockers.append('VENUE_SPLIT_SAMPLE_BELOW_8_EXPLORATORY_THRESHOLD')
            if (fx or {}).get('availability_verified') is not True:
                leg_blockers.append('XI_AND_MATERIAL_AVAILABILITY_NOT_VERIFIED')
            if gate.get('production_promotion_allowed') is not True or gate.get('bet_eligible') is not True:
                leg_blockers.append('MARKET_FAMILY_NOT_APPROVED_FOR_BET')
            if comparison is not None and comparison <= 0:
                leg_blockers.append('RAW_BELOW_LEG_PRICE_BREAKEVEN')
            annotated.append({
                **leg,
                'raw_sport_probability': p,
                'raw_probability_status': status,
                'raw_minus_individual_breakeven_pp': comparison,
                'individual_price_breakeven': round(price_break_even, 6) if price_break_even is not None else None,
                'raw_model_version': raw.get('model_version') or None,
                'raw_sample_venue_split_minimum': n,
                'market_family_gate': family,
                'market_validation_status': gate.get('research_status') or 'NOT_VERIFIED',
                'market_evidence_model_version': gate.get('model_version'),
                'market_evidence_blockers': list(gate.get('blockers') or [])[:5],
                'oos_calibrated_probability': None,
                'conservative_individual_lower_probability': None,
                'availability_confidence': None,
                'leg_blockers': leg_blockers,
            })
            blockers.extend(leg_blockers)
        joint_raw_sanity = 'NOT_VERIFIED'
        valid_comparisons = [leg['raw_minus_individual_breakeven_pp'] for leg in annotated]
        if any(v is not None and v <= 0 for v in valid_comparisons):
            joint_raw_sanity = 'AT_LEAST_ONE_LEG_RAW_BELOW_ITS_BREAK_EVEN'
        elif all(v is not None for v in valid_comparisons):
            joint_raw_sanity = 'BOTH_LEGS_RAW_ABOVE_BREAK_EVEN_NOT_CALIBRATED'
        pair = {
            **pair, 'status': 'WATCH',
            'legs': annotated,
            'joint_raw_sanity_status': joint_raw_sanity,
            'risk_rank': None,
            'joint_lower_probability': None,
            'joint_calibrated_probability': None,
            'conservative_joint_ev': None,
            'calibration_provenance': 'NOT_VERIFIED',
            'independence_status': 'NOT_VERIFIED',
            'offered_parlay_price': 'NOT_VERIFIED',
            'approval': 'NOT_ELIGIBLE_FOR_BET',
            'blockers': sorted(set(blockers)),
        }
        annotated_pairs.append(pair)
    result.update({
        'evidence_model_version': VERSION,
        'risk_ranking_available': False,
        'pairs': annotated_pairs,
        'pairs_with_negative_raw_leg': sum(
            p['joint_raw_sanity_status'] == 'AT_LEAST_ONE_LEG_RAW_BELOW_ITS_BREAK_EVEN'
            for p in annotated_pairs
        ),
        'legs_without_exact_raw_model': unmodeled,
        'raw_legs_below_price_breakeven': rejected_raw,
        'joint_lower_probability': None,
        'estimated_ev': None,
        'actionable_parlays': 0,
        'wagers_placed': 0,
        'reason': ('NO_OOS_CALIBRATED_LOWER_PROBABILITIES_OR_BOOK_OFFER; '
                   'PRICE_PROXIMITY_DOES_NOT_ESTABLISH_LOW_RISK'),
    })
    return result