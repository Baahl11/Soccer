"""Historical descriptive evidence for advanced Soccer markets; strictly pre-fixture.

This is NOT a calibrated corners/cards/1H/fouls pricing engine. It gathers
real per-match measurements and verified past formations for deep-dive candidates.
Research is performed before the sportsbook market stage.
"""
from __future__ import annotations
from typing import Any, Callable
from datetime import datetime, timezone
import statistics

from .providers import ProviderFailure

STAT_KEYS = {
    'Corner Kicks':'corners', 'Fouls':'fouls', 'Yellow Cards':'yellow_cards',
    'Red Cards':'red_cards', 'Shots on Goal':'shots_on_target',
    'Total Shots':'shots', 'Offsides':'offsides',
}


def _int_or_none(v: Any) -> int | None:
    try:
        if isinstance(v, bool):
            return None
        if isinstance(v, int) and 0 <= v <= 500:
            return v
        if isinstance(v, str) and v.isdigit() and 0 <= int(v) <= 500:
            return int(v)
    except (TypeError, ValueError):
        pass
    return None


def parse_historical_stats(payload: dict[str, Any], team_id: int) -> dict[str, int | None]:
    """Require precise provider team ID; absence/null never means zero."""
    response = payload.get('response') if isinstance(payload, dict) else None
    if not isinstance(response, list):
        raise ProviderFailure('HISTORICAL_STATS_MISSING_RESPONSE')
    for item in response:
        if (item.get('team') or {}).get('id') != team_id:
            continue
        vals = item.get('statistics') or []
        if not isinstance(vals, list):
            raise ProviderFailure('HISTORICAL_STATS_SCHEMA_MISMATCH')
        found = {}
        for row in vals:
            if isinstance(row, dict) and row.get('type') in STAT_KEYS:
                found[STAT_KEYS[row['type']]] = _int_or_none(row.get('value'))
        return {k: found.get(k) for k in STAT_KEYS.values()}
    raise ProviderFailure('HISTORICAL_STATS_TEAM_ID_MISMATCH')


def parse_historical_formation(payload: dict[str, Any], team_id: int) -> str | None:
    """Historical starting formation is credible only with exactly 11 documented starters."""
    response = payload.get('response') if isinstance(payload, dict) else None
    if not isinstance(response, list):
        return None
    for item in response:
        if (item.get('team') or {}).get('id') != team_id:
            continue
        form = item.get('formation')
        starters = item.get('startXI')
        if isinstance(form,str) and len(form) <= 14 and isinstance(starters,list) and len(starters) == 11:
            return form
    return None


def _mean(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    x = [r['stats'][key] for r in rows if isinstance(r.get('stats',{}).get(key), int)]
    return {'sample_count':len(x), 'average':round(statistics.mean(x),3) if x else None}


def collect_historical_market_context(batch: dict[str, Any],
                                      api_get: Callable[[str, dict[str, Any]], dict[str, Any]],
                                      *, max_fixtures: int = 3,
                                      games_per_team: int = 4,
                                      formation_per_team: int = 2) -> dict[str, Any]:
    if not 0 <= max_fixtures <= 6 or not 1 <= games_per_team <= 5 or not 0 <= formation_per_team <= 3:
        raise ValueError('Unsafe historical deep-dive limits')
    screened = {r['fixture_id']:r for r in batch.get('screened_fixtures',[])}
    priority = batch.get('shortlists',{}).get('merged_fixture_ids',[])
    selected = list(dict.fromkeys(x for x in priority if x in screened))[:max_fixtures]
    stats_cache, lineups_cache = {}, {}
    reports = []
    for fid in selected:
        source = screened[fid]
        hist = source.get('multi_competition_history') or {}
        sides = {}
        for label, team_key in [('home','home_team'),('away','away_team')]:
            team_id = (source.get(team_key) or {}).get('team_id')
            past = list(hist.get(label+'_fixtures') or [])[:games_per_team]
            rows, errors = [], []
            for j, game in enumerate(past):
                prior_id = game.get('fixture_id')
                if not isinstance(prior_id,int) or not isinstance(team_id,int):
                    continue
                try:
                    if prior_id not in stats_cache:
                        stats_cache[prior_id] = api_get('fixtures/statistics', {'fixture':prior_id, 'half':'true'})
                    observations = parse_historical_stats(stats_cache[prior_id],team_id)
                except (ProviderFailure, ValueError, TypeError, OSError) as exc:
                    errors.append({'fixture_id':prior_id,'section':'statistics','reason':str(exc)[:150]})
                    observations = {k:None for k in STAT_KEYS.values()}
                formation = None
                if j < formation_per_team:
                    try:
                        if prior_id not in lineups_cache:
                            lineups_cache[prior_id] = api_get('fixtures/lineups',{'fixture':prior_id})
                        formation = parse_historical_formation(lineups_cache[prior_id],team_id)
                    except (ProviderFailure,ValueError,TypeError,OSError) as exc:
                        errors.append({'fixture_id':prior_id,'section':'historical_lineup','reason':str(exc)[:150]})
                rows.append({'fixture_id':prior_id, 'kickoff':game.get('kickoff'),
                             'league_id':game.get('league_id'),
                             'opponent_team_id':game.get('away_team_id') if game.get('home_team_id')==team_id else game.get('home_team_id'),
                             'stats':observations,'verified_historical_formation':formation})
            sides[label] = {
                'team_id':team_id, 'matches_examined':len(rows),
                'past_formation_samples':sum(bool(x['verified_historical_formation']) for x in rows),
                'observed_averages':{k:_mean(rows,k) for k in STAT_KEYS.values()},
                'history':rows, 'errors':errors,
                'availability_verified_for_upcoming_match':False,
            }
        reports.append({'fixture_id':fid,'match':source.get('match'),
                        'home':sides['home'],'away':sides['away'],
                        'status':'DESCRIPTIVE_ONLY_NO_CALIBRATED_PROBABILITIES',
                        'current_formation_vs_formation':'NOT_VERIFIED_PREMATCH_XI_REQUIRED',
                        'first_half_and_second_half_goals':'REQUIRES_SEPARATE_VALIDATED_PERIOD_MODEL',
                        'fouls_model':'RESEARCH_ONLY_NO_CALIBRATED_PRICING',
                        'market_prices_used':False,'bet_eligible':False})
    return {'status':'SPORT_FIRST_HISTORICAL_MARKET_RESEARCH',
            'selected_fixture_count':len(reports),'reports':reports,
            'historical_stats_requests':len(stats_cache),
            'historical_lineups_requests':len(lineups_cache),
            'source':'API_FOOTBALL_VERIFIED_PRIOR_FIXTURE_STATISTICS_AND_LINEUPS',
            'note':'Historical description; not adjusted for opponent strength or calibrated for market pricing.',
            'wagers_placed':0,'bet_eligible':False}