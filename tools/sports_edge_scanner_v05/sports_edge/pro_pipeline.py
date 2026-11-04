"""Pro-tier soccer discovery with full-slate and sport-first fail-safe gates.

All probabilities remain NOT VERIFIED: a statistical screen is not a wagering model.
The API-Football secret is read from the process environment; never persisted.
"""
from __future__ import annotations
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Any

from .coverage import audit_fixture_response
from .providers import ProviderFailure, get_json, now_iso, _reserve_api_football_research
import os

BASE = 'https://v3.football.api-sports.io'


def api_pro_get(endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
    """Authorized direct call; one quota reservation per HTTP request.

    Allows only fixed endpoint names. Never accesses a sportsbook website.
    """
    allowed = {'fixtures', 'leagues', 'teams/statistics', 'injuries', 'fixtures/lineups', 'odds'}
    if endpoint not in allowed:
        raise ProviderFailure(f'Endpoint not allowed by research adapter: {endpoint}')
    key = os.environ.get('API_FOOTBALL_KEY', '')
    if not key:
        raise ProviderFailure('API_FOOTBALL_KEY not configured locally (never paste it in chat)')
    _reserve_api_football_research()
    result = get_json(f'{BASE}/{endpoint}', params, key, header='x-apisports-key', retries=1)
    if not isinstance(result,dict):
        raise ProviderFailure('API-Football expected JSON object')
    if result.get('errors'):
        raise ProviderFailure(f'API-Football provider error: {result["errors"]}')
    return result


def _normalize_direct_fixture(record: dict[str, Any]) -> dict[str, Any]:
    fix=record.get('fixture') or {}
    league=record.get('league') or {}
    teams=record.get('teams') or {}
    home=teams.get('home') or {}; away=teams.get('away') or {}
    status=fix.get('status') or {}
    return {
        'fixture_id':fix.get('id'), 'kickoff':fix.get('date'),
        'status':status.get('short'), 'league_id':league.get('id'),
        'league':league.get('name'), 'season':league.get('season'),
        'home_team_id':home.get('id'), 'home_team':home.get('name'),
        'away_team_id':away.get('id'), 'away_team':away.get('name'),
        'venue':(fix.get('venue') or {}).get('name'),
    }


def build_pro_slate(raw: dict[str,Any], match_date: str, league_ids: set[int],
                    timezone_name: str='America/Mexico_City', source_kind: str='direct') -> dict[str,Any]:
    """Complete fixture audit precedes filtering. No manual reconstruction.

    Each fixture must be on requested local calendar date, otherwise reject the
    source as untrustworthy for an all-games daily screen.
    """
    if not league_ids or not all(isinstance(v,int) and v>0 for v in league_ids):
        raise ValueError('Configured positive league IDs required: no all-world indiscriminate scan')
    date=datetime.fromisoformat(match_date).date().isoformat()
    tz=ZoneInfo(timezone_name)
    if source_kind=='direct':
        # Input may be a raw API reply OR the earlier CLI provider_payload wrapper.
        payload=raw.get('provider_payload',raw)
        audit=audit_fixture_response(payload,source_kind='direct')
        rows=[_normalize_direct_fixture(r) for r in payload['response']]
    elif source_kind=='connected':
        audit=audit_fixture_response(raw,source_kind='connected')
        rows=list(raw['fixtures'])
    else:
        raise ValueError('source_kind must be direct or connected')
    audit.update({'match_date':date,'timezone':timezone_name})
    anomalies=[];seen=set()
    for item in rows:
        fid=item.get('fixture_id')
        if fid is None or fid in seen:
            anomalies.append('DUPLICATE_OR_MISSING_FIXTURE_ID');continue
        seen.add(fid)
        try:
            start=datetime.fromisoformat(str(item['kickoff']).replace('Z','+00:00'))
            local=start.astimezone(tz).date().isoformat() if start.tzinfo else None
            if local!=date:anomalies.append('FIXTURE_OUTSIDE_REQUESTED_LOCAL_DATE')
        except (ValueError,TypeError,KeyError):
            anomalies.append('INVALID_KICKOFF')
    if anomalies:
        audit['complete_slate_verified']=False
        audit['coverage_status']='DATA_ANOMALIES'
    requested=[r for r in rows if r.get('league_id') in league_ids]
    eligible=[r for r in requested if r.get('status')=='NS']
    excluded=[r for r in requested if r.get('status')!='NS']
    # Missing coverage cannot be repaired by filtering a truncated world slate.
    return {
        'generated_at_utc':now_iso(), 'sport':'soccer', 'match_date':date,
        'timezone':timezone_name, 'requested_league_ids':sorted(league_ids),
        'provider_world_slate_audit':audit,
        'eligible_slate_verified':bool(audit['complete_slate_verified']),
        'all_received_fixture_count':len(rows),
        'requested_leagues_observed_count':len(requested),
        'eligible_count':len(eligible), 'excluded_count':len(excluded),
        'eligible_fixtures':eligible, 'excluded_fixtures':excluded,
        'warnings':([] if audit['complete_slate_verified'] else ['INCOMPLETE_GLOBAL_SLATE_BLOCKS_BET']),
        'next_step':'SPORTING_SCREEN_FIRST_NO_ODDS_YET',
    }


def get_cached_league_coverage(league_id:int,season:int) -> dict[str,Any]:
    """Weekly league-season metadata cache; do not spend one request per fixture.

    The API's coverage flags are only a provider capability hint, not evidence
    of confirmed lineups/injuries for an actual match.
    """
    base=Path(os.getenv('API_FOOTBALL_CACHE_DIR','data/api-football-cache'))
    file=base/f'league_{league_id}_{season}.json'
    if file.exists():
        try:
            record=json.loads(file.read_text(encoding='utf-8'))
            checked=datetime.fromisoformat(record['cached_at_utc'])
            if checked.tzinfo and datetime.now(timezone.utc)-checked < timedelta(days=7):
                data=record['provider_payload']
                if isinstance(data,dict) and isinstance(data.get('response'),list) and data['response']:
                    return data
        except (KeyError,ValueError,TypeError,OSError,json.JSONDecodeError):
            pass
    data=api_pro_get('leagues',{'id':league_id,'season':season})
    if isinstance(data.get('response'),list) and data['response']:
        base.mkdir(parents=True,exist_ok=True)
        file.write_text(json.dumps({'cached_at_utc':now_iso(),'provider_payload':data},ensure_ascii=False),encoding='utf-8')
    return data



def get_cached_team_stats(team_id: int, league_id: int, season: int, ttl_minutes: int = 90) -> dict[str, Any]:
    """Reuse recent *provider* team-season statistics between fixtures/refreshed runs.

    This caches source data, not a sporting projection. A cache hit never proves
    current lineup/injuries or means the raw sporting model is calibrated.
    """
    if not all(isinstance(n, int) and n > 0 for n in (team_id, league_id, season)):
        raise ValueError('Invalid team/league/season identity')
    if not 1 <= ttl_minutes <= 1440:
        raise ValueError('Invalid team statistics cache TTL')
    root = Path(os.getenv('API_FOOTBALL_CACHE_DIR', 'data/api-football-cache'))
    file = root / f'team_{league_id}_{season}_{team_id}.json'
    if file.exists():
        try:
            cached = json.loads(file.read_text(encoding='utf-8'))
            checked = datetime.fromisoformat(cached['cached_at_utc'])
            age = datetime.now(timezone.utc) - checked
            reply = cached['provider_payload']
            if (checked.tzinfo and timedelta(0) <= age <= timedelta(minutes=ttl_minutes)
                    and isinstance(reply, dict) and isinstance(reply.get('response'), dict)
                    and reply['response']):
                return reply
        except (KeyError, ValueError, TypeError, OSError, json.JSONDecodeError):
            pass
    reply = api_pro_get('teams/statistics', {'team': team_id, 'league': league_id, 'season': season})
    if isinstance(reply.get('response'), dict) and reply['response']:
        root.mkdir(parents=True, exist_ok=True)
        record = {'cached_at_utc': now_iso(), 'provider_payload': reply}
        tmp = file.with_suffix('.tmp')
        tmp.write_text(json.dumps(record, ensure_ascii=False), encoding='utf-8')
        tmp.replace(file)
    return reply

def _snapshot_status(result:dict[str, Any]) -> dict[str,Any]:
    obj=result.get('response',None)
    count=len(obj) if isinstance(obj,list) else (1 if isinstance(obj,dict) and obj else 0)
    return {'data_present':bool(count), 'result_count':count,
            'source':'API-Football', 'checked_at_utc':now_iso(),
            'quota':result.get('quota',{})}


def sporting_research(fixture:dict[str,Any]) -> dict[str,Any]:
    """Gather league coverage + both team aggregate statistics BEFORE market prices.

    Averages/records are evidence, not calibrated P(win), not an EV estimate.
    Calls are bounded at 3; availability remains unverified until close to KO.
    """
    for field in ('fixture_id','league_id','season','home_team_id','away_team_id'):
        if not isinstance(fixture.get(field),int):
            raise ValueError(f'Missing reliable fixture field: {field}')
    lid=int(fixture['league_id']);season=int(fixture['season'])
    coverage=get_cached_league_coverage(lid,season)
    stats=[]
    for team_id in (int(fixture['home_team_id']),int(fixture['away_team_id'])):
        result=get_cached_team_stats(team_id,lid,season)
        obj=result.get('response')
        if not isinstance(obj,dict) or not obj:
            stats.append({'team_id':team_id,'verified':False,'played':None,'gf_avg':None,'ga_avg':None})
            continue
        played=(obj.get('fixtures') or {}).get('played') or {}
        goals=obj.get('goals') or {}
        gf=goals.get('for') or {};ga=goals.get('against') or {}
        def to_float(v):
            try:return float(v) if v is not None else None
            except (ValueError,TypeError):return None
        stats.append({'team_id':team_id,'verified':True,'played':played.get('total'),
                      'gf_avg':to_float((gf.get('average') or {}).get('total')),
                      'ga_avg':to_float((ga.get('average') or {}).get('total')),
                      'wins':(obj.get('fixtures') or {}).get('wins',{}).get('total'),
                      'losses':(obj.get('fixtures') or {}).get('loses',{}).get('total')})
    coverage_rows=coverage.get('response') or []
    coverage_flags=((coverage_rows[0].get('seasons') or [{}])[0].get('coverage') or {}) if coverage_rows else {}
    enough=all(r['verified'] and isinstance(r['played'],int) and r['played']>=5
               and r['gf_avg'] is not None and r['ga_avg'] is not None for r in stats)
    return {
        'generated_at_utc':now_iso(), 'stage':'SPORT_FIRST_RESEARCH',
        'fixture_id':fixture['fixture_id'], 'match':f"{fixture.get('home_team','')} vs {fixture.get('away_team','')}",
        'league_id':lid,'season':season,'coverage_flags':coverage_flags,
        'home_team':stats[0],'away_team':stats[1],
        'min_sample_met':enough,
        'raw_model_probability':None,'market_shrunk_probability':None,
        'calibrated':False,'availability_verified':False,
        'status':'WATCH','reason':('SPORTING_SAMPLE_AVAILABLE_MODEL_STILL_REQUIRED' if enough
                                    else 'INADEQUATE_TEAM_SEASON_SAMPLE'),
        'next_step':'MODEL_STYLE_AND_AVAILABILITY_VERIFICATION_BEFORE_ODDS',
    }