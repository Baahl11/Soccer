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
    root = Path(directory)
    snapshots, failures = [], []
    for fid in selected:
        try:
            fetched = odds_fn(int(fid))
            normalized = normalize_api_football_odds(fetched, {'fixture_id': fid, 'event_name': screened[fid].get('match', '')})
            # Exclude any data whose fixture IDs disagree with our selected event.
            if any(str(q['event_id']) != str(fid) for q in normalized):
                raise ProviderFailure('Odds payload contains mismatched event IDs')
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            destination = root / 'reference_odds' / f'{stamp}_{fid}.json'
            payload = {'source': 'API-Football', 'fixture_id': fid,
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
              'history_multi_competition_verified': sum(
                  1 for x in batch['screened_fixtures']
                  if x.get('multi_competition_history', {}).get('status') == 'VERIFIED_PAST_MATCHES'),
              'odds_research_selected': odds['selected_count'],
              'odds_snapshots_captured': odds['captured_count'],
              'price_research_failures': odds['failures'],
              'actionable_bets': 0, 'wagers_placed': 0,
              'blockers': ['RAW_SPORT_MODEL_NEEDS_VALIDATED_CALIBRATION_AND_AVAILABILITY',
                           'PREMATCH_AVAILABILITY_NOT_VERIFIED',
                           'ACTIONABLE_BOOK_PRICES_NOT_VERIFIED'],
              'next_step': 'OUT_OF_SAMPLE_CALIBRATION_AND_T-40_REFRESH'}
    _write_json(target / f'odds_research_{date}.json', odds)
    _write_json(target / f'daily_summary_{date}.json', result)
    from .board import write_board
    write_board(batch, result, target / f'board_{date}.html',
                market_registry=market_registry, canonical=canonical, advanced_context=historical_market)
    return result