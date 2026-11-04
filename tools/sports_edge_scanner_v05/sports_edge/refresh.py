"""Pre-game Soccer availability and price refresh, research only; no BET output."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .core import parse_iso
from .daily import _write_json
from .pro_pipeline import api_pro_get
from .providers import ProviderFailure, api_football_odds, normalize_api_football_odds


def _lineup_check(reply: dict[str, Any], fixture: dict[str, Any]) -> dict[str, Any]:
    rows = reply.get('response', [])
    expected = {fixture.get('home_team_id'), fixture.get('away_team_id')}
    if (not isinstance(rows, list) or len(rows) != 2 or None in expected
            or len(expected) != 2):
        return {'provider_lineups_complete': False, 'team_reports': len(rows) if isinstance(rows,list) else 0,
                'reason': 'TWO_TEAM_STARTING_XI_NOT_AVAILABLE'}
    ids = set()
    for row in rows:
        if not isinstance(row, dict):
            return {'provider_lineups_complete':False, 'reason':'INVALID_LINEUP_RECORD'}
        team_id = (row.get('team') or {}).get('id')
        players = row.get('startXI') or []
        ids.add(team_id)
        if len(players) != 11:
            return {'provider_lineups_complete':False, 'reason':'STARTING_XI_INCOMPLETE'}
        if not any(str((player.get('player') or {}).get('pos','')).upper() in ('G','GK') for player in players):
            return {'provider_lineups_complete':False,'reason':'GOALKEEPER_NOT_IDENTIFIED'}
    if ids != expected:
        return {'provider_lineups_complete':False,'reason':'TEAM_IDS_MISMATCH'}
    return {'provider_lineups_complete': True, 'team_reports':2,
            'reason':'PROVIDER_XI_COMPLETE_INDEPENDENT_OFFICIAL_VERIFICATION_STILL_REQUIRED'}


def pregame_refresh(batch: dict[str, Any], fixture_id: int, directory: str | Path,
                    *, now: datetime | None = None, reference_odds: bool = False,
                    request_fn: Callable[[str,dict[str,Any]],dict[str,Any]] = api_pro_get,
                    odds_fn: Callable[[int],dict[str,Any]] = api_football_odds) -> dict[str, Any]:
    """T-90/T-40/T-20 research; no bookmaker authorization or stake eligibility.

    Requires a previously shortlisted fixture from a full sporting screening.
    Prevents market-only discovery from bypassing SPORT FIRST.
    """
    if (not batch.get('source_slate_complete_verified') or
        not batch.get('all_fixtures_attempted')):
        raise ProviderFailure('No full-slate sporting screen; cannot refresh a candidate')
    shortlisted = set(batch.get('shortlists',{}).get('merged_fixture_ids',[]))
    if fixture_id not in shortlisted:
        raise ProviderFailure('Fixture not in sporting shortlist')
    records = [r for r in batch.get('screened_fixtures',[]) if r.get('fixture_id') == fixture_id]
    if len(records) != 1 or not records[0].get('min_sample_met'):
        raise ProviderFailure('No verified minimum sporting data for candidate')
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError('now must carry a UTC offset')
    root = Path(directory)
    details = request_fn('fixtures',{'id':fixture_id})
    games = details.get('response') or []
    if len(games) != 1 or not isinstance(games[0],dict):
        raise ProviderFailure('Fixture details NOT VERIFIED')
    fixture = games[0].get('fixture') or {}
    if fixture.get('id') != fixture_id:
        raise ProviderFailure('Fixture identity mismatch')
    start = parse_iso(fixture['date'])
    status = str((fixture.get('status') or {}).get('short','UNKNOWN'))
    remaining = round((start - now.astimezone(timezone.utc)).total_seconds() / 60, 1)
    report = {'generated_at_utc':now.astimezone(timezone.utc).isoformat(),
              'fixture_id': fixture_id, 'kickoff_utc': start.isoformat(),
              'minutes_until_kickoff': remaining,'provider_status':status,
              'classification':'WATCH','actionable_bet':False,'wagers_placed':0,
              'official_lineup_verified': False,
              'price_actionable_at_caliente_playdoit': False,
              'refresh_stage':'NOT_IN_WINDOW','provider_lineups':None,
              'provider_injuries':None,'reference_odds_count':0,'notes':[]}
    if status != 'NS' or remaining <= 0:
        report['reason'] = 'MATCH_STARTED_OR_NOT_ELIGIBLE'
    elif remaining > 100:
        report['reason'] = 'PREGAME_WINDOW_NOT_REACHED'
    elif remaining < 10:
        report['reason'] = 'T20_FINAL_RETRY_WINDOW_PASSED'
    else:
        report['refresh_stage'] = ('T-90' if remaining > 55 else 'T-40' if remaining > 25 else 'T-20')
        for label,endpoint in [('provider_lineups','fixtures/lineups'),('provider_injuries','injuries')]:
            try:
                reply = request_fn(endpoint,{'fixture':fixture_id})
                if label=='provider_lineups':
                    report[label] = _lineup_check(reply, {'home_team_id':(games[0].get('teams') or {}).get('home',{}).get('id'), 'away_team_id':(games[0].get('teams') or {}).get('away',{}).get('id')})
                else:
                    obj = reply.get('response')
                    report[label] = {'provider_records':len(obj) if isinstance(obj,list) else 0,
                                     'injuries_cleared':False,
                                     'interpretation':'NO_RECORDS_DOES_NOT_VERIFY_FULL_AVAILABILITY'}
            except (ProviderFailure, ValueError, KeyError, TypeError, OSError) as exc:
                report[label] = {'verified':False,'reason':str(exc)[:350]}
        report['reason'] = 'OFFICIAL_AVAILABILITY_AND_CALIBRATED_MODEL_NOT_VERIFIED'
        if reference_odds:
            try:
                odds = normalize_api_football_odds(odds_fn(fixture_id),
                    {'fixture_id':fixture_id,'event_name':records[0].get('match','')})
                if any(str(q.get('event_id')) != str(fixture_id) for q in odds):
                    raise ProviderFailure('Provider returned odds for another fixture')
                stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                path=root/'reference_odds'/f'pregame_{stamp}_{fixture_id}.json'
                _write_json(path,{'fixture_id':fixture_id,'checked_at_utc':datetime.now(timezone.utc).isoformat(),
                                  'quotes':odds,'classification':'WATCH',
                                  'source':'API-Football','actionable_book_prices_verified':False})
                report['reference_odds_count']=len(odds)
                report['reference_odds_file']=str(path)
            except (ProviderFailure, ValueError, KeyError, TypeError, OSError) as exc:
                report['notes'].append('REFERENCE_PRICE_NOT_VERIFIED: '+str(exc)[:300])
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path=root/'refresh'/f'{stamp}_{fixture_id}.json'
    _write_json(path,report)
    report['report_file']=str(path)
    return report