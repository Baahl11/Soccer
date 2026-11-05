"""API-Football bookmaker DIRECTORY only: not authorized sportsbook odds."""
from __future__ import annotations
from datetime import datetime, timezone

def inspect_bookmaker_directory(response):
    out={'source':'API_FOOTBALL_ODDS_BOOKMAKERS_DIRECTORY','status':'UNVERIFIED',
         'bookmakers_seen':0,'directory_complete':False,'caliente_matches':[],
         'playdoit_matches':[],'authenticated_sportsbook_price':False,
         'actionable_offers':0}
    if not isinstance(response,dict) or not isinstance(response.get('response'),list):
        return out
    if response.get('errors') not in ({},[],None):
        out['status']='PROVIDER_ERRORS';return out
    paging=response.get('paging') or {}
    try:
        current=int(paging.get('current',1));total=int(paging.get('total',1))
    except (ValueError,TypeError):return out
    rows=response['response']
    out['bookmakers_seen']=len(rows)
    if current!=1 or total!=1:
        out['status']='PAGED_CATALOGUE_NOT_COMPLETE';return out
    if not all(isinstance(r,dict) and isinstance(r.get('name'),str) and r.get('id') is not None for r in rows):
        out['status']='INVALID_CATALOGUE_ITEMS';return out
    out['directory_complete']=True
    for r in rows:
        name=r['name'].strip();low=name.casefold()
        if 'caliente' in low:out['caliente_matches'].append({'name':name,'id':r['id']})
        if 'playdoit' in low or 'play doit' in low:out['playdoit_matches'].append({'name':name,'id':r['id']})
    out['status']='NAMED_BOOKMAKER_REFERENCES_FOUND' if out['caliente_matches'] or out['playdoit_matches'] else 'NO_NAMED_BOOKMAKER_IN_COMPLETE_CATALOGUE'
    return out

def fetch_bookmaker_directory(request_fn):
    try:
        out=inspect_bookmaker_directory(request_fn('odds/bookmakers',{}))
    except (OSError,ValueError,RuntimeError,KeyError,TypeError) as e:
        out=inspect_bookmaker_directory(None)
        out['status']='BOOKMAKER_DIRECTORY_REQUEST_FAILED'
        out['error_type']=type(e).__name__
    out['checked_at_utc']=datetime.now(timezone.utc).isoformat()
    return out
