"""Complete-slate Soccer research and auditable paper-only market observations.

The sports-screen prioritization is a *triage heuristic*, not a model-predicted
probability, a market edge, a recommendation, or an estimate of expected goals.
No wager is ever submitted. The three sport-specific discovery tracks remain
separate: side, goal environment, and corners (NOT VERIFIED without corner data).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
import json
import math

from .providers import ProviderFailure, api_football_odds, normalize_api_football_odds
import os
from .pro_pipeline import sporting_research


def _float(v: Any) -> float | None:
    try:
        x = float(v)
        return x if math.isfinite(x) and x >= 0 else None
    except (ValueError, TypeError):
        return None


def preliminary_signals(screen: dict[str, Any]) -> dict[str, Any]:
    """Only sample-based research sorting; NEVER probabilities or xG estimates."""
    home, away = screen.get('home_team') or {}, screen.get('away_team') or {}
    if not screen.get('min_sample_met'):
        return {'side_signal': None, 'goals_signal': None, 'corners_signal': None,
                'signal_quality': 'NOT_VERIFIED_INSUFFICIENT_SAMPLE',
                'probability_generated': False}
    hf, ha = _float(home.get('gf_avg')), _float(home.get('ga_avg'))
    af, aa = _float(away.get('gf_avg')), _float(away.get('ga_avg'))
    if None in (hf, ha, af, aa):
        return {'side_signal': None, 'goals_signal': None, 'corners_signal': None,
                'signal_quality': 'NOT_VERIFIED_MISSING_SCORES',
                'probability_generated': False}
    # Simple descriptive interaction of own scoring rate and opponent concession.
    # These are NOT calibrated, opponent-adjusted model outputs or projected goals.
    home_pressure = (hf + aa) / 2
    away_pressure = (af + ha) / 2
    return {'side_signal': round(abs(home_pressure - away_pressure), 4),
            'side_direction': 'home' if home_pressure > away_pressure else
                              'away' if away_pressure > home_pressure else 'balanced',
            'goals_signal': round(home_pressure + away_pressure, 4),
            'corners_signal': None, 'signal_quality': 'DESCRIPTIVE_ONLY_UNCALIBRATED',
            'probability_generated': False,
            'source': 'current-season team GF/GA per match (not opponent-adjusted)'}


def _shortlist(screened: list[dict[str, Any]], side_limit=12, goals_limit=12) -> dict[str, Any]:
    viable = [s for s in screened if s.get('status') == 'WATCH'
              and s.get('min_sample_met') and s.get('screening_signals', {}).get('side_signal') is not None]
    sides = sorted(viable, key=lambda s: (-s['screening_signals']['side_signal'], s['fixture_id']))
    goals = sorted(viable, key=lambda s: (-s['screening_signals']['goals_signal'], s['fixture_id']))
    side_ids = [s['fixture_id'] for s in sides[:side_limit]]
    goals_ids = [s['fixture_id'] for s in goals[:goals_limit]]
    # Do not fabricate corner signals from team match goals.
    return {'side_fixture_ids': side_ids,
            'goals_fixture_ids': goals_ids,
            'corners_fixture_ids': [],
            'corners_status': 'NOT_VERIFIED_NO_VALIDATED_CORNERS_INPUT',
            'merged_fixture_ids': list(dict.fromkeys(side_ids + goals_ids)),
            'shortlists_are': 'RESEARCH_PRIORITY_ONLY_NOT_BETTING_EDGE',
            'deep_dive_required': True}


def screen_complete_slate(slate: dict[str, Any],
                          research_fn: Callable[[dict[str, Any]], dict[str, Any]] = sporting_research,
                          side_limit: int = 12, goals_limit: int = 12) -> dict[str, Any]:
    """Screen EVERY configured, not-started fixture; no odds queried here.

    Failed sports-data reads are recorded per fixture as WATCH; a partial stage
    can never be presented as a complete sporting screen.
    """
    if not slate.get('eligible_slate_verified') or not slate.get('provider_world_slate_audit', {}).get('complete_slate_verified'):
        raise ProviderFailure('Global fixture slate is incomplete: refuse full-slate sporting screen')
    rows = slate.get('eligible_fixtures')
    if not isinstance(rows, list) or len(rows) != slate.get('eligible_count'):
        raise ProviderFailure('Eligible fixture count inconsistent with provider audit')
    if len({x.get('fixture_id') for x in rows}) != len(rows):
        raise ProviderFailure('Duplicate event IDs in eligible slate')
    out, failed = [], []
    for fixture in rows:
        fid = fixture.get('fixture_id')
        try:
            screen = research_fn(fixture)
            if screen.get('fixture_id') != fid or screen.get('stage') != 'SPORT_FIRST_RESEARCH':
                raise ValueError('Mismatched or missing sport-first research identity')
            if screen.get('status') != 'WATCH' or screen.get('calibrated') is not False:
                raise ValueError('Sport-first screen must not claim a validated BET')
            screen = dict(screen)
            screen['screening_signals'] = preliminary_signals(screen)
            out.append(screen)
        except (ProviderFailure, ValueError, KeyError, TypeError, OSError) as exc:
            failed.append(fid)
            out.append({'fixture_id': fid, 'match': f"{fixture.get('home_team', '')} vs {fixture.get('away_team', '')}",
                        'stage': 'SPORT_FIRST_RESEARCH', 'status': 'WATCH',
                        'reason': 'SPORTING_DATA_NOT_VERIFIED',
                        'failure_detail': str(exc)[:350],
                        'calibrated': False, 'min_sample_met': False,
                        'raw_model_probability': None, 'market_shrunk_probability': None,
                        'screening_signals': {'signal_quality': 'NOT_VERIFIED', 'probability_generated': False}})
    shortlists = _shortlist(out, side_limit, goals_limit)
    return {'generated_at_utc': datetime.now(timezone.utc).isoformat(), 'sport': 'soccer',
            'match_date': slate['match_date'], 'timezone': slate['timezone'],
            'source_slate_complete_verified': True,
            'eligible_count': len(rows), 'all_fixtures_attempted': len(out) == len(rows),
            'complete_sport_screen': len(out) == len(rows) and not failed,
            'not_verified_fixture_ids': failed,
            'screened_fixtures': out,
            'shortlists': shortlists,
            'calibrated_probabilities_produced': 0,
            'raw_existing_engine_projections': sum(
                1 for row in out
                if (row.get('raw_sport_projection') or {}).get('status') == 'MODELED_LIMITED'),
            'classification': 'WATCH', 'wagers_placed': 0,
            'notes': ['Preliminary side/goals research indicators are not calibrated probability forecasts.',
                      'Corners NOT VERIFIED until independent historical/team corner inputs are modeled.',
                      'Do not use this report to justify bets or generate parlay EV.',
                      'Prices have not been consulted during sporting screen.']}


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic output: do not leave a half-written report if a process is interrupted.
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def fetch_research_reference_odds(batch: dict[str, Any], directory: str | Path,
                                  max_fixtures: int = 8,
                                  odds_fn: Callable[[int], dict[str, Any]] = api_football_odds) -> dict[str, Any]:
    """Price observations for sports shortlists ONLY; no actionable wagers.

    Snapshots are immutable, independently timestamped and bookstore-specific.
    A fresh download does not refresh an earlier bookmaker source_updated_at.
    """
    if not batch.get('source_slate_complete_verified') or not batch.get('all_fixtures_attempted'):
        raise ProviderFailure('No verified sport-first full-slate attempt; refuse market stage')
    if not isinstance(max_fixtures, int) or max_fixtures < 0 or max_fixtures > 25:
        raise ValueError('max_fixtures must be between 0 and 25')
    screened = {s['fixture_id']: s for s in batch.get('screened_fixtures', [])}
    selected = []
    for fid in batch.get('shortlists', {}).get('merged_fixture_ids', []):
        if fid in selected:
            continue
        s = screened.get(fid)
        if not s or not s.get('min_sample_met') or s.get('status') != 'WATCH':
            continue
        selected.append(fid)
        if len(selected) >= max_fixtures:
            break
    if max_fixtures == 0:
        selected = []
    # One book is the price proxy; do not silently replace missing Bet365
    # observations with another provider's quote.
    proxy_name = os.getenv('SOCCER_PROXY_BOOKMAKER_NAME', '').strip()
    proxy_id_text = os.getenv('SOCCER_PROXY_BOOKMAKER_ID', '').strip()
    if bool(proxy_name) != bool(proxy_id_text):
        raise ProviderFailure('Proxy bookmaker name and ID must be configured together')
    proxy_id = None
    if proxy_id_text:
        if not proxy_id_text.isdecimal() or int(proxy_id_text) <= 0:
            raise ProviderFailure('Invalid proxy bookmaker ID')
        proxy_id = int(proxy_id_text)
    root = Path(directory)
    snapshots, failures = [], []
    proxy_quote_count, proxy_quote_fixtures = 0, 0
    for fid in selected:
        try:
            fetched = (api_football_odds(int(fid), bookmaker_id=proxy_id)
                       if proxy_id is not None and odds_fn is api_football_odds
                       else odds_fn(int(fid)))
            normalized_all = normalize_api_football_odds(
                fetched, {'fixture_id': fid, 'event_name': screened[fid].get('match', '')})
            normalized = ([q for q in normalized_all
                           if q.get('bookmaker', '').strip().casefold() == proxy_name.casefold()]
                          if proxy_name else normalized_all)
            if proxy_name and not normalized:
                failures.append({'fixture_id': fid,
                                 'reason': 'NO_' + proxy_name.upper() + '_REFERENCE_QUOTES_FOR_FIXTURE'})
                continue
            proxy_quote_count += len(normalized)
            proxy_quote_fixtures += int(bool(normalized))
            # Exclude any data whose fixture IDs disagree with our selected event.
            if any(str(q['event_id']) != str(fid) for q in normalized):
                raise ProviderFailure('Odds payload contains mismatched event IDs')
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            destination = root / 'reference_odds' / f'{stamp}_{fid}.json'
            payload = {'source': 'API-Football', 'fixture_id': fid,
                       'proxy_bookmaker': proxy_name or 'ALL_REFERENCE_BOOKS',
                       'proxy_bookmaker_id': proxy_id,
                       'mexico_offer_verified': False,
                       'checked_at_utc': datetime.now(timezone.utc).isoformat(),
                       'actionable_at_caliente_or_playdoit': False,
                       'classification': 'WATCH', 'quote_count': len(normalized),
                       'quotes': normalized,
                       'warning': 'Third-party bookmaker prices; stale publication and coverage require verification.'}
            _write_json(destination, payload)
            snapshots.append({'fixture_id': fid, 'file': str(destination), 'quotes': len(normalized),
                              'status': 'WATCH'})
        except (ProviderFailure, ValueError, KeyError, TypeError, OSError) as exc:
            failures.append({'fixture_id': fid, 'reason': str(exc)[:350]})
    return {'generated_at_utc': datetime.now(timezone.utc).isoformat(),
            'source': 'API-Football', 'purpose': 'PRICE_DISCOVERY_ONLY',
            'reference_bookmaker': proxy_name or 'ALL_REFERENCE_BOOKS',
            'reference_bookmaker_id': proxy_id,
            'proxy_quote_count': proxy_quote_count,
            'proxy_quote_fixtures': proxy_quote_fixtures,
            'proxy_prices_actionable_in_mexico': False,
            'selected_count': len(selected), 'captured_count': len(snapshots),
            'snapshots': snapshots, 'failures': failures,
            'actionable_bets': 0, 'classification': 'WATCH',
            'reasons': ['NO_CALIBRATED_MODEL_PROBABILITIES',
                        'ACTIONABLE_CALIENTE_PLAYDOIT_PRICE_NOT_VERIFIED',
                        'AVAILABILITY_NOT_VERIFIED']}


def run_daily(date: str, league_ids: set[int], directory: str | Path,
              fixture_source_fn: Callable[[str], dict[str, Any]],
              research_fn: Callable[[dict[str, Any]], dict[str, Any]] = sporting_research,
              odds_limit: int = 0,
              odds_fn: Callable[[int], dict[str, Any]] = api_football_odds) -> dict[str, Any]:
    """Daily pre-final run, with raw provider evidence; not a background daemon."""
    from .pro_pipeline import build_pro_slate
    target = Path(directory)
    raw = fixture_source_fn(date)
    _write_json(target / f'raw_fixtures_{date}.json', raw)
    slate = build_pro_slate(raw, date, league_ids)
    _write_json(target / f'slate_{date}.json', slate)
    if not slate.get('eligible_slate_verified'):
        result = {'generated_at_utc': datetime.now(timezone.utc).isoformat(),
                  'match_date': date, 'status': 'BLOCKED_INCOMPLETE_SLATE',
                  'source_slate_complete_verified': False,
                  'actionable_bets': 0, 'wagers_placed': 0,
                  'missing_minimum': slate['provider_world_slate_audit']['missing_minimum'],
                  'next_step': 'RECHECK_AUTHORIZED_DIRECT_PROVIDER_FULL_SLATE'}
        _write_json(target / f'daily_summary_{date}.json', result)
        return result
    batch = screen_complete_slate(slate, research_fn=research_fn)
    _write_json(target / f'sport_first_{date}.json', batch)
    from .market_context import collect_historical_market_context
    from .pro_pipeline import api_pro_get
    import os
    historical_market = ({'status':'NOT_ENABLED','reports':[],'selected_fixture_count':0}
                         if os.getenv('SOCCER_ENABLE_HISTORICAL_MARKETS','0') != '1'
                         else collect_historical_market_context(batch, api_pro_get,
                                max_fixtures=int(os.getenv('SOCCER_ADVANCED_DEEP_DIVES','3'))))
    _write_json(target / f'historical_market_context_{date}.json', historical_market)
    # Pre-kickoff safety check is limited to the already screened shortlist.
    from .refresh_batch import refresh_due_shortlist
    pregame = refresh_due_shortlist(slate, batch, target, max_fixtures=5)
    _write_json(target / f'pregame_availability_{date}.json', pregame)
    odds = fetch_research_reference_odds(batch, target, odds_limit, odds_fn=odds_fn)
    from .market_registry import load_market_registry
    from .canonical_snapshot import fetch_canonical_market_views, match_canonical_fixtures
    import os
    market_registry = load_market_registry(os.getenv('SOCCER_MARKET_EVIDENCE_DIR', target / 'market_evidence'))
    canonical = ({'status': 'NOT_ENABLED', 'is_fresh': False, 'matching_fixture_rows': []}
                 if os.getenv('SOCCER_ENABLE_CANONICAL_VIEWS', '0') != '1'
                 else match_canonical_fixtures(fetch_canonical_market_views(),
                        {r['fixture_id'] for r in batch['screened_fixtures']}))
    _write_json(target / f'market_registry_{date}.json', market_registry)
    _write_json(target / f'canonical_market_views_{date}.json', canonical)
    # Existing canonical GitHub state is read-only research, not a BET certificate.
    from .canonical_state_bridge import load_canonical_research_file
    research_state = load_canonical_research_file(
        os.getenv('SOCCER_CANONICAL_STATE_PATH', target / 'missing_canonical.json'), slate)
    _write_json(target / f'canonical_research_compared_{date}.json', research_state)
    from .bookmaker_catalogue import fetch_bookmaker_directory
    bookmaker_directory = ({'status':'NOT_ENABLED','actionable_offers':0}
        if os.getenv('SOCCER_BOOKMAKER_DIRECTORY_ENABLED','0') != '1'
        else fetch_bookmaker_directory(api_pro_get))
    _write_json(target / f'bookmaker_coverage_{date}.json', bookmaker_directory)
    # Dynamic BET/LEAN/WATCH/PASS path. No research calibration promotion.
    from .decision_bridge import adjudicate_daily
    decision = adjudicate_daily(canonical, slate_complete=batch['complete_sport_screen'])
    _write_json(target / f'decision_gate_{date}.json', decision)
    result = {'generated_at_utc': datetime.now(timezone.utc).isoformat(),
              'match_date': date, 'status': 'PREFINAL_WATCH_RESEARCH',
              'source_slate_complete_verified': True, 'eligible_games': slate['eligible_count'],
              'sports_screened': len(batch['screened_fixtures']),
              'complete_sport_screen': batch['complete_sport_screen'],
              'not_verified_games': len(batch['not_verified_fixture_ids']),
              'preliminary_side_shortlist': batch['shortlists']['side_fixture_ids'],
              'preliminary_goals_shortlist': batch['shortlists']['goals_fixture_ids'],
              'corners_track': batch['shortlists']['corners_status'],
              'raw_existing_engine_projections': batch.get('raw_existing_engine_projections', 0),
              'historical_market_deep_dives': historical_market.get('selected_fixture_count',0),
              'market_families_audited': len(market_registry['families']),
              'market_evidence_available_families': sum(x['report_loaded'] for x in market_registry['families']),
              'canonical_market_status': canonical.get('status'),
              'canonical_same_fixture_rows': len(canonical.get('matching_fixture_rows', [])),
              'canonical_state_bridge_status': research_state['status'],
              'canonical_state_matches': research_state['total_matching_rows'],
              'canonical_selection_research_models': research_state['selection_calibration_research_rows'],
              'canonical_indicative_quotes': research_state['reference_price_rows'],
              'bookmaker_catalogue_status': bookmaker_directory.get('status'),
              'authorized_sportsbook_offers': bookmaker_directory.get('actionable_offers',0),
              'history_multi_competition_verified': sum(
                  1 for x in batch['screened_fixtures']
                  if x.get('multi_competition_history', {}).get('status') == 'VERIFIED_PAST_MATCHES'),
              'pregame_candidates_in_window': pregame['eligible_candidates_in_window'],
              'pregame_refresh_attempted': pregame['attempted'],
              'pregame_refresh_completed': pregame['completed'],
              'pregame_official_lineups_verified': 0,
              'pregame_status': pregame['status'],
              'odds_research_selected': odds['selected_count'],
              'odds_snapshots_captured': odds['captured_count'],
              'proxy_reference_bookmaker': odds.get('reference_bookmaker'),
              'proxy_reference_bookmaker_id': odds.get('reference_bookmaker_id'),
              'proxy_quote_count': odds.get('proxy_quote_count',0),
              'proxy_quote_fixtures': odds.get('proxy_quote_fixtures',0),
              'proxy_price_available_in_mexico_verified': False,
              'price_research_failures': odds['failures'],
              'actionable_bets': decision['actionable_bets'], 'wagers_placed': 0,
              'decision_feed_status': decision.get('feed_status'),
              'decision_singles_evaluated': len(decision['singles']),
              'decision_parlays_evaluated': len(decision['parlays']),
              'canonical_unpromoted_rows': decision['canonical_research']['rows_examined'],
              'blockers': ['RAW_SPORT_MODEL_NEEDS_VALIDATED_CALIBRATION_AND_AVAILABILITY',
                           'PREMATCH_AVAILABILITY_NOT_VERIFIED',
                           'ACTIONABLE_BOOK_PRICES_NOT_VERIFIED'],
              'next_step': 'OUT_OF_SAMPLE_CALIBRATION_AND_T-40_REFRESH'}
    # PRICE TARGET SECOND, after sport-first analysis and saved provider quotes.
    # This does not derive probabilities from odds or declare bets.
    from .parlay_radar import discover_saved_snapshot_pairs
    parlay_radar = discover_saved_snapshot_pairs(odds.get('snapshots', []))
    from .parlay_risk import annotate_parlay_radar
    parlay_radar = annotate_parlay_radar(parlay_radar, batch, market_registry)
    result['parlay_price_pairs_watch'] = len(parlay_radar['pairs'])
    result['parlay_price_pairs_found'] = parlay_radar['pairs_found_before_display_cap']
    result['parlay_actionable_bets'] = decision['actionable_parlays']
    result['parlay_raw_negative_pairs'] = parlay_radar.get('pairs_with_negative_raw_leg', 0)
    result['parlay_legs_without_exact_raw_model'] = parlay_radar.get('legs_without_exact_raw_model', 0)
    result['parlay_risk_ranking_available'] = bool(decision['actionable_parlays'])
    if result['actionable_bets'] or result['parlay_actionable_bets']:
        result['blockers'] = []
        result['next_step'] = 'RECONFIRM_PRICE_AND_PORTFOLIO_BEFORE_STAKE'
    else:
        result['blockers'].append('CERTIFIED_MODEL_AND_BOOKMAKER_DECISION_FEED_NOT_READY')
    _write_json(target / f'parlay_radar_{date}.json', parlay_radar)
    _write_json(target / f'parlay_leg_evidence_{date}.json', parlay_radar)
    _write_json(target / f'odds_research_{date}.json', odds)
    _write_json(target / f'daily_summary_{date}.json', result)
    from .board import write_board
    write_board(batch, result, target / f'board_{date}.html',
                market_registry=market_registry, canonical=canonical, advanced_context=historical_market,
                parlay_radar=parlay_radar, decision_report=decision,
                canonical_research=research_state, bookmaker_catalogue=bookmaker_directory)
    return result