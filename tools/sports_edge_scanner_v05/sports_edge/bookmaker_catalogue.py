"""Verify bookmaker NAME directory via authorized API-Football Pro searches.

A name match is only provider REFERENCE coverage, not an actionable price,
sportsbook login, betslip or authorization. No sportsbook scraping.
"""
from __future__ import annotations
from datetime import datetime, timezone

def inspect_bookmaker_directory(response):
    out={'source':'API_FOOTBALL_ODDS_BOOKMAKERS_DIRECTORY','status':'UNVERIFIED',
         'bookmakers_seen':0,'directory_complete':False,
         'caliente_matches':[],'playdoit_matches':[],
         'authenticated_sportsbook_price':False,'actionable_offers':0}
    if not isinstance(response,dict) or not isinstance(response.get('response'),list):
        return out
    if response.get('errors') not in ({},[],None):
        out['status']='PROVIDER_ERRORS';return out
    paging=response.get('paging') or {}
    try: current=int(paging.get('current',1));total=int(paging.get('total',1))
    except (ValueError,TypeError):return out
    rows=response['response'];out['bookmakers_seen']=len(rows)
    if current!=1 or total!=1:
        out['status']='PAGED_CATALOGUE_NOT_COMPLETE';return out
    invalid=0
    for r in rows:
        if not isinstance(r,dict) or not isinstance(r.get('name'),str) or not r['name'].strip():
            invalid+=1;continue
        name=r['name'].strip();low=name.casefold()
        if 'caliente' in low:
            out['caliente_matches'].append({'name':name,'id':r.get('id')})
        if 'playdoit' in low or 'play doit' in low:
            out['playdoit_matches'].append({'name':name,'id':r.get('id')})
    out['unparsed_items']=invalid
    if invalid:
        out['status']='CATALOGUE_PARTIAL_UNPARSED_ITEMS';return out
    out['directory_complete']=True
    out['status']='NAMED_BOOKMAKER_REFERENCES_FOUND' if out['caliente_matches'] or out['playdoit_matches'] else 'NO_NAMED_BOOKMAKER_IN_COMPLETE_CATALOGUE'
    return out

def fetch_bookmaker_directory(request_fn):
    """Targeted authorized directory search; two requests, no offered odds."""
    records=[];errors=[]
    for term in ('Caliente','Playdoit'):
        try:
            reply=request_fn('odds/bookmakers',{'search':term})
            view=inspect_bookmaker_directory(reply)
            records.append({'query':term,'status':view['status'],
                            'results':view['bookmakers_seen'],
                            'caliente_matches':view['caliente_matches'],
                            'playdoit_matches':view['playdoit_matches']})
            if not view['directory_complete']:
                errors.append(term+':'+view['status'])
        except (OSError,ValueError,TypeError,RuntimeError,KeyError) as exc:
            errors.append(term+':'+type(exc).__name__)
    matches1=[x for r in records for x in r['caliente_matches']]
    matches2=[x for r in records for x in r['playdoit_matches']]
    out={'source':'API_FOOTBALL_TARGETED_BOOKMAKER_SEARCH',
         'status':('TARGETED_SEARCH_UNVERIFIED' if errors else
                   'REFERENCE_NAME_MATCHES_FOUND' if matches1 or matches2 else
                   'NO_REFERENCE_NAMES_RETURNED_IN_TARGETED_SEARCH'),
         'checked_at_utc':datetime.now(timezone.utc).isoformat(),
         'queries':records,'query_errors':errors,
         'caliente_matches':matches1,'playdoit_matches':matches2,
         'directory_complete':not errors,'authenticated_sportsbook_price':False,
         'actionable_offers':0,
         'note':'API-Football bookmaker search is NOT a Caliente/Playdoit live offer feed.'}
    return out
