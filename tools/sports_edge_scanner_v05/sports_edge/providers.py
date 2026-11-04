"""Documented, API-key authenticated odds connectors. No sportsbook page scraping.

Provider coverage must be checked; we never assume Caliente/Playdoit support.
"""
from __future__ import annotations
import json, os, time, re
from urllib import request,parse,error
from datetime import datetime,timezone
from typing import Any

class ProviderFailure(RuntimeError):pass

def now_iso():return datetime.now(timezone.utc).isoformat()

def get_json(base:str,params:dict[str,Any],key:str,header:str='X-API-Key',retries:int=2)->dict[str,Any]:
    if not key: raise ProviderFailure('API key not configured')
    url=base+'?'+parse.urlencode(params)
    last=None
    for attempt in range(retries+1):
        try:
            req=request.Request(url,headers={header:key,'Accept':'application/json','User-Agent':'SportsEdgeResearch/0.1'})
            with request.urlopen(req,timeout=25) as res:
                data=json.loads(res.read().decode('utf-8'))
                if not isinstance(data,(dict,list)):raise ProviderFailure('Invalid JSON structure')
                return data
        except error.HTTPError as exc:
            if exc.code in (401,403):raise ProviderFailure('API access denied; stop, do not bypass') from exc
            if exc.code==429:raise ProviderFailure('Rate limit reached; no automatic retry') from exc
            if 500<=exc.code<600:
                last=exc
                if attempt<retries:time.sleep(2**attempt);continue
            raise ProviderFailure(f'API HTTP {exc.code}') from exc
        except (error.URLError,TimeoutError,ValueError) as exc:
            last=exc
            if attempt<retries:time.sleep(2**attempt);continue
    raise ProviderFailure(f'API unavailable: {last}')

def _reserve_api_football_research():
    # A deliberately conservative LOCAL limit, NOT the provider's live balance.
    # Calls via the ChatGPT connector are counted only in the provider's quota.
    # The default Pro research cap is 300/day despite a confirmed 7,500/day plan.
    from .free_sources import reserve_quota
    daily_cap=int(os.getenv('API_FOOTBALL_RESEARCH_DAILY_CAP','300'))
    monthly_cap=int(os.getenv('API_FOOTBALL_RESEARCH_MONTHLY_CAP','9300'))
    if not 1 <= daily_cap <= 7500 or monthly_cap < daily_cap:
        raise ProviderFailure('Invalid API-Football research quota configuration')
    return reserve_quota(os.getenv('API_FOOTBALL_QUOTA_FILE','data/api_football_pro_research.json'),
                         unit_cost=1,month_limit=monthly_cap,month_reserve=0,daily_cap=daily_cap)


def api_football_fixtures(day:str, timezone_name:str='America/Mexico_City'):
    key=os.environ.get('API_FOOTBALL_KEY','')
    if not key:raise ProviderFailure('API_FOOTBALL_KEY not configured; direct HTTP unavailable')
    _reserve_api_football_research()
    data=get_json('https://v3.football.api-sports.io/fixtures',
                  {'date':day,'timezone':timezone_name},key,header='x-apisports-key',retries=0)
    if data.get('errors'):raise ProviderFailure(f'Provider errors: {data["errors"]}')
    from .coverage import audit_fixture_response
    coverage = audit_fixture_response(data, source_kind='direct')
    coverage['match_date'] = day
    coverage['timezone'] = timezone_name
    # Include original rows; the CLI saves both direct source and independent audit.
    return {'provider_payload':data, 'coverage_audit':coverage}

def api_football_odds(fixture_id:int):
    key=os.environ.get('API_FOOTBALL_KEY','')
    if not key:raise ProviderFailure('API_FOOTBALL_KEY not configured; cannot call direct odds API')
    allrows=[];page=1
    while True:
        _reserve_api_football_research()  # each response page costs a request
        data=get_json('https://v3.football.api-sports.io/odds',
                      {'fixture':fixture_id,'page':page},key,header='x-apisports-key',retries=0)
        if data.get('errors'):raise ProviderFailure(f'Provider errors: {data["errors"]}')
        allrows.extend(data.get('response',[]))
        if page>=int(data.get('paging',{}).get('total',1)):break
        page+=1
    return {'response':allrows,'retrieved_at_utc':now_iso()}

def normalize_api_football_odds(data:dict[str,Any], fixture_meta:dict[str,Any]|None=None):
    meta=fixture_meta or {};captured=data.get('retrieved_at_utc') or now_iso();rows=[]
    for fix in data.get('response',[]):
        fixture=fix.get('fixture',{});fid=str(fixture.get('id',meta.get('fixture_id','')))
        kickoff=fixture.get('date',meta.get('kickoff_utc'))
        for book in fix.get('bookmakers',[]):
            for bet in book.get('bets',[]):
                market=str(bet.get('name',''))
                for sel in bet.get('values',[]):
                    try:price=float(sel['odd'])
                    except (KeyError,TypeError,ValueError):continue
                    if price<=1:continue
                    raw_selection=str(sel.get('value',''))
                    m=re.match(r'^(Over|Under)\s+(\d+(?:\.\d+)?)$',raw_selection,re.I)
                    selection=m.group(1).capitalize() if m else raw_selection
                    point=m.group(2) if m else ''
                    rows.append(dict(sport='soccer',event_id=fid,event_name=meta.get('event_name',''),
                        starts_at_utc=kickoff,market=market,selection=selection,
                        point=point,price_decimal=price,bookmaker=str(book.get('name','')), 
                        source='API-Football',captured_at_utc=captured,source_updated_at=fix.get('update')))
    return rows

def propline_odds(sport_key:str,markets:str='h2h,spreads,totals',bookmakers:str|None=None):
    key=os.environ.get('PROPLINE_API_KEY','');params={'markets':markets}
    if bookmakers:params['bookmakers']=bookmakers
    return get_json(f'https://api.prop-line.com/v1/sports/{parse.quote(sport_key)}/odds',params,key)

def normalize_propline_odds(data:Any,sport:str,source='PropLine'):
    captured=now_iso();rows=[]
    games=data if isinstance(data,list) else data.get('events',data.get('data',[]))
    if not isinstance(games,list):raise ProviderFailure('Unexpected provider response: events not found')
    for game in games:
        eid=str(game.get('id',''))
        name=' vs '.join([str(game.get('home_team','')),str(game.get('away_team',''))])
        for book in game.get('bookmakers',[]):
            bookname=str(book.get('key') or book.get('title') or '')
            for market in book.get('markets',[]):
                for sel in market.get('outcomes',[]):
                    try:price=float(sel['price'])
                    except (KeyError,ValueError,TypeError):continue
                    if price<=1:continue
                    rows.append(dict(sport=sport,event_id=eid,event_name=name,
                        starts_at_utc=game.get('commence_time'),market=str(market.get('key','')),
                        selection=str(sel.get('name','')),point=str(sel.get('point','')) if sel.get('point') is not None else '',
                        price_decimal=price,bookmaker=bookname,source=source,
                        captured_at_utc=captured,source_updated_at=book.get('last_update') or market.get('last_update')))
    return rows