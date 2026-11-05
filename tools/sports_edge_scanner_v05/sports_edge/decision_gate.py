"""SOCCER EDGE v1.0 decision bridge: real classifications, no hardcoded zero.

The canonical model owns sport-first inputs, OOS calibrations and promotion.
A sportsbook/authorized adapter owns exact offers. This module only adjudicates
*matched, explicit* evidence. It never promotes a research calibrator to BET.

Caution: input assertions are not provenance proof by themselves; trusted
production adapters must verify their own upstream source authenticity.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from math import isfinite
from typing import Any

from .core import evaluate, ev_per_unit, parse_iso, dec_to_american, market_family

VERSION = 'SOCCER_DECISION_GATE_v1.5.0'
# Single bookmaker reference: API-Football Bet365, bookmaker ID 8.
# A current provider quote is not the same thing as a confirmed account betslip.
BOOK_SOURCES = {'Bet365': 'API_FOOTBALL_BET365_ID8'}
MIN_OOS_ROWS = {'side': 200, 'goals': 200, 'derivative': 100}


def _num(v: Any) -> float | None:
    try:
        val = float(v)
        return val if isfinite(val) else None
    except (ValueError, TypeError):
        return None


def _identity(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (str(row.get('sport', '')).lower(), str(row.get('event_id', '')),
            str(row.get('market', '')).strip().casefold(),
            str(row.get('selection', '')).strip().casefold(),
            str(row.get('point', '') or '').strip())


def _utc(t: Any) -> datetime | None:
    try:
        return parse_iso(str(t))
    except (TypeError, ValueError):
        return None


def _age_ok(timestamp: Any, now: datetime, minutes: int) -> bool:
    dt = _utc(timestamp)
    return dt is not None and -60 <= (now-dt).total_seconds() <= minutes*60


def _model_checks(model: dict[str, Any] | None, quote: dict[str, Any],
                  now: datetime) -> list[str]:
    if not model:
        return ['EXACT_MARKET_MODEL_MISSING']
    blockers = []
    if _identity(model) != _identity(quote):
        blockers.append('MODEL_MARKET_OR_FIXTURE_MISMATCH')
    if model.get('model_version') in ('', None) or not model.get('model_artifact_id'):
        blockers.append('MODEL_LINEAGE_NOT_VERIFIED')
    # An OOS research calibrator does NOT grant trading approval. The existing
    # production model's promotion gate is authoritative, not a price signal.
    if (model.get('promotion_status') != 'PRODUCTION_APPROVED'
            or model.get('production_promotion_allowed') is not True
            or model.get('calibrated') is not True):
        blockers.append('MODEL_NOT_APPROVED_FOR_PRODUCTION')
    if model.get('calibration_out_of_sample') is not True:
        blockers.append('OOS_CALIBRATION_NOT_VERIFIED')
    family = market_family('soccer', str(model.get('market') or ''))
    if family == 'unsupported':
        blockers.append('MARKET_FAMILY_UNSUPPORTED')
    if (not isinstance(model.get('oos_rows'), int)
            or model['oos_rows'] < MIN_OOS_ROWS.get(family, 200)):
        blockers.append('MARKET_SPECIFIC_OOS_SAMPLE_INSUFFICIENT')
    if model.get('raw_sport_built_without_market_input') is not True:
        blockers.append('RAW_SPORT_INDEPENDENCE_NOT_VERIFIED')
    if model.get('data_tier') not in ('A', 'B'):
        blockers.append('DATA_TIER_A_B_NOT_VERIFIED')
    if model.get('fixture_status') != 'NS' or model.get('fixture_status_verified') is not True:
        blockers.append('MATCH_STATUS_NOT_VERIFIED')
    if (model.get('xi_and_goalkeepers_verified') is not True
            or model.get('material_availability_verified') is not True
            or model.get('availability_source') not in ('OFFICIAL_TEAM', 'OFFICIAL_LEAGUE', 'VERIFIED_PROVIDER_XI')):
        blockers.append('XI_GK_AVAILABILITY_NOT_VERIFIED')
    conf = _num(model.get('availability_confidence'))
    if conf is None or conf < 0.90 or conf > 1:
        blockers.append('AVAILABILITY_CONFIDENCE_BELOW_PARLAY_STANDARD')
    if model.get('unresolved_material_conflict') is not False:
        blockers.append('MATERIAL_CONFLICT_NOT_CLEARED')
    vals = [_num(model.get(x)) for x in ('p_raw', 'p_shrunk', 'p_lower')]
    if (any(x is None or x <= 0 or x >= 1 for x in vals)
            or (vals[2] is not None and vals[1] is not None and vals[2] > vals[1])):
        blockers.append('VALID_RAW_SHRUNK_LOWER_PROBABILITIES_MISSING')
    # The production model must have been generated in this pregame cycle.
    if not _age_ok(model.get('model_generated_at_utc'), now, 360):
        blockers.append('MODEL_RUN_STALE_OR_UNDATED')
    return blockers


def _offer_checks(quote: dict[str, Any], now: datetime) -> list[str]:
    blockers = []
    book = str(quote.get('bookmaker') or '')
    if (book != 'Bet365' or quote.get('bookmaker_id') != 8
            or quote.get('source') != 'API-Football'
            or quote.get('verified_offer_source') != BOOK_SOURCES['Bet365']
            or quote.get('price_observation_verified') is not True):
        blockers.append('BET365_PROVIDER_PRICE_SOURCE_NOT_VERIFIED')
    if (quote.get('fair_market_probability') is not None
            and quote.get('fair_market_probability_verified') is not True):
        blockers.append('MARKET_FAIR_PROBABILITY_ORIGIN_UNVERIFIED')
    if not _age_ok(quote.get('source_updated_at'), now, 90):
        blockers.append('BOOKMAKER_PRICE_NOT_FRESH')
    if (not (1 < (_num(quote.get('price_decimal')) or 0) <= 20)
            or not _utc(quote.get('starts_at_utc'))
            or _utc(quote.get('starts_at_utc')) <= now):
        blockers.append('PRICE_KICKOFF_OR_MARKET_NOT_VERIFIED')
    if (quote.get('sport') != 'soccer' or not all(str(quote.get(x) or '').strip() for x in ('event_id','market','selection'))):
        blockers.append('INVALID_MARKET_IDENTITY')
    return blockers


def evaluate_verified_leg(quote: dict[str, Any], model: dict[str, Any] | None,
                          *, now: datetime | None = None) -> dict[str, Any]:
    """Return BET/LEAN/PASS/WATCH based on inputs; never hard-code BET to zero.

    Bet365 bookmaker-8 observations are a timestamped price source, not a
    directly confirmed Bet365 betslip. Reconfirm price before placing a stake.
    """
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    blockers = list(dict.fromkeys(_offer_checks(quote, now) + _model_checks(model, quote, now)))
    result = {'sport':'soccer', 'event_id':str(quote.get('event_id','')),
              'event_name':quote.get('event_name'), 'market':quote.get('market'),
              'selection':quote.get('selection'), 'point':quote.get('point'),
              'bookmaker':quote.get('bookmaker'), 'price_decimal':quote.get('price_decimal'),
              'status':'WATCH', 'tier':'NONE', 'reason':'VERIFICATION_INCOMPLETE',
              'blockers':blockers, 'p_raw':None, 'p_shrunk':None, 'p_lower':None,
              'conservative_ev':None, 'stake_units':0, 'model_version':None}
    if model and _identity(model) == _identity(quote):
        result['model_version'] = model.get('model_version')
        # Never display research calibration as final EV.
        if not blockers:
            result['p_raw'] = model.get('p_raw')
            result['p_shrunk'] = model.get('p_shrunk')
            result['p_lower'] = model.get('p_lower')
    if blockers:
        return result
    eval_row = evaluate(quote, model, now=now, max_age_minutes=90)
    result.update({k:eval_row.get(k) for k in ('status','tier','reason','p_model','p_lower',
                   'conservative_ev','prob_edge_pp','p_breakeven')})
    result['p_shrunk'] = model['p_shrunk']
    result['p_raw'] = model['p_raw']
    result['blockers'] = []
    result['stake_units'] = 0  # No staking unless portfolio risk/ticket are approved.
    if result['status'] == 'BET':
        result['reason'] = 'SINGLE_BET365_REFERENCE_ELIGIBLE_RECHECK_BEFORE_STAKE'
    return result


def _leg_ticket_key(row: dict[str, Any]) -> tuple[str,str,str,str,str]:
    return _identity(row)


def evaluate_verified_parlay(leg_a: dict[str, Any], leg_b: dict[str, Any],
                             offered_ticket: dict[str, Any] | None,
                             joint_evidence: dict[str, Any] | None,
                             *, now: datetime | None = None,
                             min_joint_lower: float = 0.55) -> dict[str, Any]:
    """Evaluate a Bet365-only two-leg +100 candidate without inventing a betslip.

    Price = mathematical product of two individually published Bet365 bookmaker-8
    quotes. For joint risk, audited OOS joint data is preferred. Otherwise the
    Frechet/Boole bound max(0,p_lower_1+p_lower_2-1) is valid under ANY
    dependence, provided both marginal lower bounds are independently approved.
    A BET is a price-dependent recommendation, NOT a placed wager; recheck the
    displayed combined price in Bet365 and approve portfolio exposure first.
    """
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    blockers: list[str] = []
    legs = (leg_a, leg_b)
    if any(x.get('status') != 'BET' or x.get('tier') not in ('A', 'S') for x in legs):
        blockers.append('BOTH_LEGS_VALIDATED_TIER_A_S_REQUIRED')
    if not leg_a.get('event_id') or not leg_b.get('event_id') or str(leg_a['event_id']) == str(leg_b['event_id']):
        blockers.append('MUST_BE_DIFFERENT_FIXTURES')
    if any(x.get('bookmaker') != 'Bet365' for x in legs):
        blockers.append('SAME_BET365_MARKET_SOURCE_REQUIRED')

    q = offered_ticket or {}
    offered = _num(q.get('price_decimal'))
    a_price = _num(leg_a.get('price_decimal'))
    b_price = _num(leg_b.get('price_decimal'))
    product = a_price * b_price if a_price and b_price else None
    keys = q.get('leg_keys') or []
    quoted_keys = {tuple(x) for x in keys if isinstance(x, (list, tuple)) and len(x) == 5}
    expected_keys = {_identity(x) for x in legs}
    if (offered is None or product is None
            or not 1.90 <= offered <= 2.10
            or abs(offered-product) > 0.001
            or q.get('bookmaker') != 'Bet365'
            or q.get('bookmaker_id') != 8
            or q.get('verified_offer_source') != BOOK_SOURCES['Bet365']
            or q.get('theoretical_from_two_quotes') is not True
            or q.get('individual_quotes_verified') is not True
            or not _age_ok(q.get('source_updated_at'), now, 90)
            or quoted_keys != expected_keys):
        blockers.append('BET365_TWO_QUOTE_PRICE_NOT_VERIFIED')

    p_a = _num(leg_a.get('p_lower'))
    p_b = _num(leg_b.get('p_lower'))
    joint_lower = None
    joint_method = None
    if joint_evidence is None:
        joint_method = 'FRECHET_ANY_DEPENDENCE'
        if p_a is None or p_b is None or not (0 < p_a < 1 and 0 < p_b < 1):
            blockers.append('MARGINAL_OOS_LOWER_BOUNDS_MISSING')
        else:
            joint_lower = max(0.0, p_a + p_b - 1.0)
    else:
        j = joint_evidence
        joint_method = 'JOINT_OUTCOME_OOS'
        joint_lower = _num(j.get('joint_p_lower'))
        if (j.get('status') != 'VERIFIED'
                or j.get('method') != 'JOINT_OUTCOME_OOS'
                or j.get('approved_for_decisions') is not True
                or not isinstance(j.get('oos_joint_rows'), int)
                or j.get('oos_joint_rows') < 200
                or set(map(str, j.get('event_ids') or [])) !=
                    {str(leg_a.get('event_id')), str(leg_b.get('event_id'))}
                or joint_lower is None or not 0 < joint_lower < 1
                or p_a is None or p_b is None
                or joint_lower > min(p_a, p_b)):
            blockers.append('JOINT_OOS_EVIDENCE_INVALID')
    # No stake recommendation unless portfolio constraints are explicitly met.
    if q.get('portfolio_exposure_approved') is not True:
        blockers.append('PORTFOLIO_EXPOSURE_NOT_APPROVED')
    out = {
        'status': 'WATCH', 'reason': 'MISSING_REQUIRED_EVIDENCE',
        'bookmaker': 'Bet365', 'events': [leg_a.get('event_id'), leg_b.get('event_id')],
        'leg_keys': [list(_identity(x)) for x in legs],
        'ticket_price_decimal': offered, 'ticket_reference_id': None,
        'theoretical_price_only': True, 'bet365_betslip_recheck_required': True,
        'joint_probability_method': joint_method,
        'joint_p_lower': joint_lower if not blockers else None,
        'conservative_joint_ev': None, 'blockers': list(dict.fromkeys(blockers)),
        'stake_units': 0,
    }
    if blockers:
        return out
    if joint_lower is None or offered is None:
        return out
    out['p_breakeven'] = round(1/offered, 6)
    out['conservative_joint_ev'] = round(ev_per_unit(joint_lower, offered), 4)
    if joint_lower < max(min_joint_lower, 1/offered) or out['conservative_joint_ev'] <= 0:
        out['status'] = 'PASS'
        out['reason'] = 'CONSERVATIVE_JOINT_RISK_OR_EV_INSUFFICIENT'
    else:
        out['status'] = 'BET'
        out['reason'] = 'BET365_VERIFIED_TWO_QUOTE_MODEL_EDGE_RECHECK_BETSLIP'
        out['ticket_price_american'] = dec_to_american(offered)
    return out


def adjudicate_offers(quotes: list[dict[str, Any]], models: list[dict[str, Any]],
                     parlay_offers: list[dict[str, Any]] | None = None,
                     joint_reports: list[dict[str, Any]] | None = None,
                     *, now: datetime | None = None, slate_complete: bool = True) -> dict[str, Any]:
    """Live path: map exact sportsbook offers to separately built soccer models.

    All input lists are real by contract. Callers must NEVER pass synthetic
    fixtures/quotes to a live invocation. Tests use synthetic fixtures only.
    """
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    models_by_id = {_identity(x): x for x in models}
    single = []
    for q in quotes:
        result = evaluate_verified_leg(q, models_by_id.get(_identity(q)), now=now)
        if not slate_complete:
            result.update(status='WATCH',reason='INCOMPLETE_SPORTING_SLATE',
                          blockers=['INCOMPLETE_SPORTING_SLATE'],stake_units=0)
        single.append(result)
    # A Bet result here does not place a wager. It means all individual
    # material model/market gates passed; actual staking is separate.
    tickets = []
    for ticket in parlay_offers or []:
        keys=ticket.get('leg_keys') or []
        if len(keys)!=2:
            continue
        matches=[]
        for key in keys:
            matches.append(next((v for v in single if list(_leg_ticket_key(v))==key),None))
        if any(x is None for x in matches):
            tickets.append({'status':'WATCH','reason':'TICKET_LEG_NOT_MATCHED',
                            'blockers':['EXACT_LEG_MATCH_REQUIRED'],'stake_units':0})
            continue
        report=next((x for x in joint_reports or [] if
                     set(map(str,x.get('event_ids') or []))==set(map(str,[matches[0]['event_id'],matches[1]['event_id']]))),None)
        out=evaluate_verified_parlay(matches[0],matches[1],ticket,report,now=now)
        if not slate_complete:
            out.update(status='WATCH',reason='INCOMPLETE_SPORTING_SLATE',
                       blockers=['INCOMPLETE_SPORTING_SLATE'],stake_units=0)
        tickets.append(out)
    counts=dict(Counter(x['status'] for x in single))
    pc=dict(Counter(x['status'] for x in tickets))
    return {'schema_version':VERSION,'generated_at_utc':now.isoformat(),
            'sport':'soccer','slate_complete_verified':bool(slate_complete),
            'singles':single,'parlays':tickets,'singles_by_status':counts,
            'parlays_by_status':pc,
            'actionable_bets':counts.get('BET',0),
            'actionable_parlays':pc.get('BET',0),'wagers_placed':0}


def inspect_canonical_research(canonical: dict[str, Any], *, max_rows: int=30) -> dict[str, Any]:
    """Actual deployed rows are RESEARCH unless certified; no fake BET or prices.

    This dashboard diagnostic is useful even with zero authenticated sportsbook
    offers, and never changes an original candidate's canonical classification.
    """
    rows=(canonical.get('matching_fixture_rows') or [])[:max_rows]
    out=[]
    for row in rows:
        blockers=[]
        if not canonical.get('is_fresh'):
            blockers.append('CANONICAL_PIPELINE_STALE')
        if row.get('production_promotion_allowed') is not True or row.get('bet_eligible') is not True:
            blockers.append('MODEL_RESEARCH_ONLY_NOT_PROMOTED')
        if row.get('p_shrunk') is None or row.get('p_lower') is None:
            blockers.append('SHRUNK_LOWER_BOUND_NOT_EXPORTED')
        if (row.get('price') is None
                or str(row.get('bookmaker') or '').casefold() != 'bet365'
                or not (row.get('provider_update') or row.get('price_resolution_provider_update'))):
            blockers.append('BET365_REFERENCE_PRICE_NOT_VERIFIED')
        if row.get('status') not in ('NS',):
            blockers.append('FIXTURE_STATUS_NOT_PREGAME')
        out.append({'fixture_id':row.get('fixture_id'),'home':row.get('home'),
                    'away':row.get('away'),'market':row.get('market'),
                    'selection':row.get('selection'),'line':row.get('line'),
                    'canonical_classification':row.get('classification'),
                    'stage':row.get('stage'),'status':'WATCH',
                    'reason':'UNPROMOTED_CANONICAL_EVIDENCE',
                    'blockers':blockers})
    return {'source':'CURRENT_CANONICAL_RESEARCH_VIEWS',
            'canonical_status':canonical.get('status','NOT_VERIFIED'),
            'rows_examined':len(out),'rows':out,
            'blocker_counts':dict(Counter(k for r in out for k in r['blockers'])),
            'note':'This diagnostic does not substitute for a bookmaker-offer decision feed.'}