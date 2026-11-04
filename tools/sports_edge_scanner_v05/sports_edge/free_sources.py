"""Zero-cost input adapters: The Odds API free tier, and saved Soccer Edge API replies.

No provider key is embedded in code or stored on disk. No paid plan assumed.
No Caliente/Playdoit scraper. All odds require original bookmaker timestamps.
"""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from urllib import parse, request, error
from .providers import ProviderFailure, now_iso
import json, os

SPORT_KEYS = {
    'nfl': 'americanfootball_nfl',
    'ncaaf': 'americanfootball_ncaaf',
    'mlb': 'baseball_mlb',
    'nba': 'basketball_nba',
    'soccer_epl': 'soccer_epl',
    'soccer_uefa_champs_league': 'soccer_uefa_champs_league',
}

def utc_now():
    return datetime.now(timezone.utc)

def _read_ledger(path):
    p=Path(path)
    if p.exists():
        v=json.loads(p.read_text(encoding='utf-8'))
        if not isinstance(v,dict):raise ProviderFailure('Invalid local quota ledger')
        return v
    return {}

def _write_ledger(path,data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(data,sort_keys=True,indent=2),encoding='utf-8')
    tmp.replace(p)

def reserve_quota(path,unit_cost=1,month_limit=500,month_reserve=100,daily_cap=10,day=None):
    """Local conservative budget. Count a reservation BEFORE a request.
    Failed provider calls may cost a quota unit, so no automatic refund.
    Concurrent workers must use a database/lock instead of this one-process ledger.
    """
    if not 0 < unit_cost <= daily_cap:raise ProviderFailure('Invalid request cost / daily cap')
    today=day or utc_now().strftime('%Y-%m-%d');month=today[:7]
    data=_read_ledger(path)
    if data.get('month') != month: data={'month':month,'used':0,'days':{}}
    used=int(data.get('used',0));today_used=int(data.get('days',{}).get(today,0))
    if used+unit_cost > month_limit-month_reserve:raise ProviderFailure('Free monthly allowance budget protected; request not made')
    if today_used+unit_cost > daily_cap:raise ProviderFailure('Free daily allowance budget protected; request not made')
    data['used']=used+unit_cost;data.setdefault('days',{})[today]=today_used+unit_cost
    _write_ledger(path,data)
    return {'month':month,'month_used_reserved':data['used'],'day_reserved':data['days'][today]}

def the_odds_api(sport_key,markets='h2h',regions='us',quota_file='data/free_quota.json',max_credits_per_request=2):
    """Publicly documented free-tier request, restricted to small cost.

    Market/region combinations incur quota; caller opts into one region and h2h
    by default. Always use decimal; downstream probability checks handle vig.
    """
    key=os.environ.get('THE_ODDS_API_KEY','')
    if not key:raise ProviderFailure('Free The Odds API key not configured (THE_ODDS_API_KEY); $0 signup required')
    if sport_key not in SPORT_KEYS.values():raise ProviderFailure('Unknown sport key')
    m=[x.strip() for x in markets.split(',') if x.strip()]
    r=[x.strip() for x in regions.split(',') if x.strip()]
    if not m or not r or len(m)*len(r)>max_credits_per_request:
        raise ProviderFailure('Request would consume too many free credits')
    if any(x not in ('h2h','spreads','totals') for x in m):
        raise ProviderFailure('Free scanner initially supports only core markets')
    if any(x not in ('us','uk','eu','au') for x in r):
        raise ProviderFailure('Region must be US/UK/EU/AU')
    reserve_quota(quota_file,len(m)*len(r))
    url=f'https://api.the-odds-api.com/v4/sports/{parse.quote(sport_key)}/odds/'
    params={'apiKey':key,'regions':','.join(r),'markets':','.join(m),'oddsFormat':'decimal'}
    req=request.Request(url+'?'+parse.urlencode(params),headers={'Accept':'application/json','User-Agent':'SportsEdgeResearch/0.2'})
    try:
        with request.urlopen(req,timeout=25) as res:
            data=json.loads(res.read().decode('utf-8'))
            if not isinstance(data,list):raise ProviderFailure('Unexpected The Odds API JSON shape')
            return {'response':data,'retrieved_at_utc':now_iso(), 'source':'The Odds API',
                    'sport_key':sport_key, 'markets':m, 'regions':r,
                    'quota_remaining_reported':res.headers.get('x-requests-remaining')}
    except error.HTTPError as exc:
        # Not retrying authentication, billing or throttling errors.
        raise ProviderFailure(f'The Odds API HTTP {exc.code}; check key/quota, no automatic retry') from exc
    except (error.URLError,TimeoutError,ValueError) as exc:
        raise ProviderFailure('The Odds API unavailable or invalid JSON') from exc

def normalize_the_odds_api(data,sport):
    if sport not in ('soccer','nfl','ncaaf','mlb','nba'):
        raise ProviderFailure('Unsupported sport label')
    rows=[];captured=data.get('retrieved_at_utc') or now_iso()
    for event in data.get('response',[]):
        teams=[event.get('home_team',''),event.get('away_team','')]
        for book in event.get('bookmakers',[]):
            for market in book.get('markets',[]):
                for outcome in market.get('outcomes',[]):
                    try:
                        price=float(outcome['price'])
                        if price<=1:continue
                    except (KeyError,TypeError,ValueError):continue
                    # Trust bookmaker/market last_update, never the download timestamp.
                    rows.append({'sport':sport,'event_id':str(event.get('id','')),
                        'event_name':' vs '.join(teams),
                        'starts_at_utc':event.get('commence_time'),
                        'market':str(market.get('key','')),
                        'selection':str(outcome.get('name','')),
                        'point':str(outcome.get('point','')) if outcome.get('point') is not None else '',
                        'price_decimal':price,
                        'bookmaker':str(book.get('title') or book.get('key') or ''),
                        'source':'The Odds API',
                        'captured_at_utc':captured,
                        'source_updated_at':market.get('last_update') or book.get('last_update')})
    return rows

def normalize_connected_soccer_odds(raw, fixture_names=None):
    """Convert saved mcp Soccer_Edge_API.get_odds JSON into scanner quotes.
    This input needs no new external provider account and costs no new plan.
    """
    from .providers import normalize_api_football_odds
    if raw.get('endpoint')!='/odds' or not isinstance(raw.get('response'),list):
        raise ProviderFailure('Input is not a connected Soccer Edge API odds reply')
    data={'response':raw['response'],'retrieved_at_utc':raw.get('retrieved_at_utc') or now_iso()}
    result=[]
    for fix in data['response']:
        fid=str(fix.get('fixture',{}).get('id',''))
        meta={'fixture_id':fid,'event_name':(fixture_names or {}).get(fid,'')}
        # Per-fixture metadata: normalized downstream with exact selection line.
        result.extend(normalize_api_football_odds({'response':[fix], 'retrieved_at_utc':data['retrieved_at_utc']},meta))
    return result

def summarize_connected_soccer_fixtures(raw):
    """Audit full-slate integrity; never treat a 100-row connector sample as complete."""
    from .coverage import audit_fixture_response
    a = audit_fixture_response(raw, source_kind='connected')
    return {
        **a,
        'total_reported_by_api': a['total_reported'],
        'fixtures_received': a['received'],
        'coverage_warning': None if a['complete_slate_verified'] else 'INCOMPLETE_SLATE',
        'league_counts_in_received_sample': a['observed_league_counts'],
        'fixtures':raw['fixtures'],
    }