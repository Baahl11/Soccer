"""Command line for source collection and conservative market screening."""
from __future__ import annotations
import argparse,csv,json,sys,os
from pathlib import Path
from datetime import datetime,timezone
from .core import evaluate,generate_parlays,attach_market_fair_probabilities,parse_iso
from .providers import (api_football_fixtures,api_football_odds,normalize_api_football_odds,
                        propline_odds,normalize_propline_odds,ProviderFailure)
from .coverage import audit_capabilities, audit_connector_counts
from .pro_pipeline import build_pro_slate, sporting_research
from .daily import run_daily, screen_complete_slate, fetch_research_reference_odds
from .refresh import pregame_refresh
from .free_sources import (the_odds_api,normalize_the_odds_api,
                           normalize_connected_soccer_odds,summarize_connected_soccer_fixtures,
                           _read_ledger,SPORT_KEYS)

def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Wrote {p} ({len(obj) if hasattr(obj,"__len__") else 1} items)')

def cli():
    p=argparse.ArgumentParser(description='SPORTS EDGE ENGINE read-only discovery and price monitor')
    subs=p.add_subparsers(dest='cmd',required=True)
    f=subs.add_parser('fetch-soccer-fixtures');f.add_argument('--date',required=True);f.add_argument('--out',required=True)
    o=subs.add_parser('fetch-soccer-odds');o.add_argument('--fixture-id',type=int,required=True);o.add_argument('--out',required=True)
    pp=subs.add_parser('fetch-provider-odds');pp.add_argument('--sport-key',required=True)
    pp.add_argument('--sport',choices=['soccer','nfl','ncaaf'],required=True)
    pp.add_argument('--books');pp.add_argument('--markets',default='h2h,spreads,totals');pp.add_argument('--out',required=True)
    free=subs.add_parser('fetch-free-odds',help='Optional $0/free The Odds API key, capped quota')
    free.add_argument('--sport-key',required=True,choices=list(SPORT_KEYS))
    free.add_argument('--markets',default='h2h')
    free.add_argument('--regions',default='us')
    free.add_argument('--quota-file',default='data/free_quota.json')
    free.add_argument('--out',required=True)
    ci=subs.add_parser('import-connected-soccer-odds',help='Normalize Soccer Edge API get_odds JSON')
    ci.add_argument('--input',required=True);ci.add_argument('--out',required=True)
    cf=subs.add_parser('import-connected-soccer-fixtures',help='Audit slate coverage without assuming complete list')
    cf.add_argument('--input',required=True);cf.add_argument('--out',required=True)
    cc=subs.add_parser('audit-connector-counts',help='Audit persisted connector counts without pretending rows exist')
    cc.add_argument('--input',required=True);cc.add_argument('--out',required=True)
    ca=subs.add_parser('audit-connected-access',help='Read saved endpoint probes; never spend API quota')
    ca.add_argument('--input',required=True);ca.add_argument('--out',required=True)
    cq=subs.add_parser('free-quota-status',help='Read local free quota ledger')
    cq.add_argument('--quota-file',default='data/free_quota.json')
    pd=subs.add_parser('pro-soccer-discovery',help='Direct Pro full-day fixture capture + selected competition slate')
    pd.add_argument('--date',required=True);pd.add_argument('--leagues',required=True,help='Comma-separated API-Football league IDs')
    pd.add_argument('--out',required=True);pd.add_argument('--raw-out',help='Optional provider source snapshot for audit')
    pb=subs.add_parser('pro-soccer-build-slate',help='Offline validation of saved raw API/connector fixture replies')
    pb.add_argument('--input',required=True);pb.add_argument('--source',choices=['direct','connected'],required=True)
    pb.add_argument('--date',required=True);pb.add_argument('--leagues',required=True);pb.add_argument('--out',required=True)
    ps=subs.add_parser('pro-soccer-screen',help='Sports data FIRST, no odds; requires full slate audit')
    ps.add_argument('--slate',required=True);ps.add_argument('--fixture-id',type=int,required=True);ps.add_argument('--out',required=True)
    po=subs.add_parser('pro-soccer-odds',help='Investigate bookmaker prices only after sport screen')
    po.add_argument('--screen',required=True);po.add_argument('--out',required=True)
    po.add_argument('--allow-small-sample',action='store_true',help='Still WATCH; extra usage must be intentional')
    dy=subs.add_parser('pro-soccer-daily',help='Full date + EVERY eligible game sports screen, paper-only')
    dy.add_argument('--date',required=True);dy.add_argument('--leagues',required=True,
                    help='Explicit league IDs; do not treat this as every sport/league')
    dy.add_argument('--output-dir',required=True)
    dy.add_argument('--odds-limit',type=int,default=0,help='Reference-only shortlisted odds calls, 0..25; default 0')
    dy.add_argument('--raw-file',help='Offline replay of an existing full direct fixture source; still needs live team stats')
    ba=subs.add_parser('pro-soccer-batch-screen',help='Sport-first research of ALL eligible games in saved slate')
    ba.add_argument('--slate',required=True);ba.add_argument('--out',required=True)
    ro=subs.add_parser('pro-soccer-reference-odds',help='Paper-only price research AFTER full sporting screen')
    ro.add_argument('--batch',required=True);ro.add_argument('--output-dir',required=True)
    ro.add_argument('--max-fixtures',type=int,default=8);ro.add_argument('--out',required=True)
    pg=subs.add_parser('pro-soccer-refresh',help='T-90 / T-40 / T-20 shortlist pregame research')
    pg.add_argument('--batch',required=True);pg.add_argument('--fixture-id',type=int,required=True)
    pg.add_argument('--output-dir',required=True)
    pg.add_argument('--reference-odds',action='store_true',help='Retrieve third-party odds as WATCH only')
    s=subs.add_parser('scan');s.add_argument('--odds',required=True);s.add_argument('--projections')
    s.add_argument('--independence');s.add_argument('--out',required=True)
    s.add_argument('--max-age-minutes',type=int,default=180)
    s.add_argument('--coverage-audit',help='Day-specific fixture audit JSON; missing/incomplete blocks actionable status')
    s.add_argument('--actionable-books',default='',help='Comma-separated bookmakers confirmed accessible to user; default none')
    a=p.parse_args()
    try:
        if a.cmd=='fetch-soccer-fixtures':write(a.out,api_football_fixtures(a.date));return 0
        if a.cmd=='fetch-soccer-odds':write(a.out,normalize_api_football_odds(api_football_odds(a.fixture_id)));return 0
        if a.cmd=='fetch-provider-odds':
            obj=propline_odds(a.sport_key,a.markets,a.books)
            write(a.out,normalize_propline_odds(obj,a.sport));return 0
        if a.cmd=='fetch-free-odds':
            mapping={'nfl':'nfl','ncaaf':'ncaaf','mlb':'mlb','nba':'nba'}
            sp=mapping.get(a.sport_key,'soccer')
            obj=the_odds_api(SPORT_KEYS[a.sport_key],a.markets,a.regions,a.quota_file)
            write(a.out,normalize_the_odds_api(obj,sp))
            print('Source: The Odds API; actual bookmaker coverage not assumed. Quota protected.')
            return 0
        if a.cmd=='import-connected-soccer-odds':
            write(a.out,normalize_connected_soccer_odds(load(a.input)));return 0
        if a.cmd=='import-connected-soccer-fixtures':
            result=summarize_connected_soccer_fixtures(load(a.input))
            write(a.out,result)
            if not result['complete_slate_verified']:
                print('WARNING: INCOMPLETE_SLATE; must not claim all fixtures screened')
            return 0
        if a.cmd=='audit-connector-counts':
            write(a.out,audit_connector_counts(load(a.input)));return 0
        if a.cmd=='audit-connected-access':
            write(a.out,audit_capabilities(load(a.input)));return 0
        if a.cmd=='free-quota-status':
            print(json.dumps(_read_ledger(a.quota_file),ensure_ascii=False,indent=2));return 0
        if a.cmd=='pro-soccer-discovery':
            lids={int(v.strip()) for v in a.leagues.split(',') if v.strip()}
            payload=api_football_fixtures(a.date)
            if a.raw_out:write(a.raw_out,payload)
            doc=build_pro_slate(payload,a.date,lids)
            write(a.out,doc)
            print(f"Pro discovery: {doc['all_received_fixture_count']} global rows; "
                  f"{doc['eligible_count']} eligible in configured leagues; "
                  f"complete={doc['eligible_slate_verified']}")
            return 0
        if a.cmd=='pro-soccer-build-slate':
            lids={int(v.strip()) for v in a.leagues.split(',') if v.strip()}
            doc=build_pro_slate(load(a.input),a.date,lids,source_kind=a.source)
            write(a.out,doc)
            return 0
        if a.cmd=='pro-soccer-screen':
            doc=load(a.slate)
            if not doc.get('eligible_slate_verified'):
                raise ProviderFailure('Full slate incomplete: sporting screen must not imply all-match coverage')
            candidate=next((f for f in doc.get('eligible_fixtures',[])
                            if f.get('fixture_id')==a.fixture_id),None)
            if candidate is None:raise ProviderFailure('Fixture not eligible in configured full slate')
            result=sporting_research(candidate)
            write(a.out,result)
            print('Sport-first research only: WATCH, no prices/probabilities inferred.')
            return 0
        if a.cmd=='pro-soccer-odds':
            screen=load(a.screen)
            if screen.get('stage')!='SPORT_FIRST_RESEARCH' or screen.get('status')!='WATCH':
                raise ProviderFailure('Verified sport-first research file required')
            if not screen.get('min_sample_met') and not a.allow_small_sample:
                raise ProviderFailure('Team season sample insufficient; avoid spending odds quota')
            fixture_id=int(screen['fixture_id'])
            prices=normalize_api_football_odds(api_football_odds(fixture_id),
                      {'fixture_id':fixture_id,'event_name':screen.get('match','')})
            write(a.out,prices)
            print('BOOKMAKER COVERAGE IS NOT CALIENTE/PLAYDOIT VERIFIED; WATCH until model & price review.')
            return 0
        if a.cmd=='pro-soccer-daily':
            lids={int(v.strip()) for v in a.leagues.split(',') if v.strip()}
            if a.raw_file:
                source=lambda _day:load(a.raw_file)
            else:
                source=api_football_fixtures
            report=run_daily(a.date,lids,a.output_dir,source,odds_limit=a.odds_limit)
            print(json.dumps(report,ensure_ascii=False,indent=2))
            return 0 if report.get('source_slate_complete_verified') else 3
        if a.cmd=='pro-soccer-batch-screen':
            batch=screen_complete_slate(load(a.slate))
            write(a.out,batch)
            return 0
        if a.cmd=='pro-soccer-reference-odds':
            report=fetch_research_reference_odds(load(a.batch),a.output_dir,a.max_fixtures)
            write(a.out,report)
            return 0
        if a.cmd=='pro-soccer-refresh':
            report=pregame_refresh(load(a.batch),a.fixture_id,a.output_dir,
                                   reference_odds=a.reference_odds)
            print(json.dumps(report,ensure_ascii=False,indent=2))
            return 0
        if a.cmd=='scan':
            odds=load(a.odds);projs=load(a.projections) if a.projections else []
            if isinstance(odds,dict):odds=odds.get('odds',[])
            if isinstance(projs,dict):projs=projs.get('projections',[])
            if not isinstance(odds,list) or not isinstance(projs,list):raise ValueError('Odds and projections must be arrays')
            ix={tuple(str(p.get(k,'')) for k in ('sport','event_id','market','selection','point')):p for p in projs}
            now=datetime.now(timezone.utc)
            odds=attach_market_fair_probabilities(odds)
            rows=[evaluate(q,ix.get(tuple(str(q.get(k,'')) for k in ('sport','event_id','market','selection','point'))),now,a.max_age_minutes) for q in odds]
            indep=load(a.independence) if a.independence else {}
            # Slate-first policy: if there is no verified full slate, downgrade BET/LEAN.
            # A quote cannot repair the absence of the sporting screen of all eligible events.
            coverage=load(a.coverage_audit) if a.coverage_audit else None
            # This CLI currently accepts a Soccer fixture audit only. NFL/NCAAF
            # cannot inherit Soccer's completeness status from the same report.
            # Accept both a standalone coverage audit and the v0.4 full Pro slate.
            if coverage and 'provider_world_slate_audit' in coverage:
                raw_audit=coverage['provider_world_slate_audit']
                slate_complete=bool(coverage.get('eligible_slate_verified') and raw_audit.get('complete_slate_verified'))
                soccer_day=coverage.get('match_date')
            else:
                slate_complete = bool(coverage and coverage.get('complete_slate_verified') is True)
                soccer_day = coverage.get('match_date') if coverage else None
            for row in rows:
                starts = row.get('starts_at_utc')
                try:
                    from zoneinfo import ZoneInfo
                    row_local_date = parse_iso(starts).astimezone(ZoneInfo('America/Mexico_City')).date().isoformat()
                except (ValueError, TypeError):
                    row_local_date = None
                slate_valid_for_row = (slate_complete and row.get('sport') == 'soccer'
                                      and soccer_day == row_local_date)
                if not slate_valid_for_row and row['status'] in ('BET','LEAN'):
                    row['status']='WATCH'
                    row['reason']='SPORT_DATE_SLATE_COVERAGE_NOT_VERIFIED'
                    row['stake_units']=0
            allowed={v.strip().lower() for v in a.actionable_books.split(',') if v.strip()}
            for row in rows:
                if row['status'] in ('BET','LEAN') and str(row.get('bookmaker','')).lower() not in allowed:
                    row.update(status='WATCH',reason='ACTIONABLE_BOOK_NOT_CONFIRMED',stake_units=0)
            parlays=generate_parlays(rows,evidence=indep)
            report={'generated_at_utc':now.isoformat(),'mode':'READ_ONLY_PAPER_RESEARCH','prices_checked':len(rows),
                    'wagers_placed':0,'singles':rows,'parlays':parlays,
                    'slate_complete_verified':slate_complete,
                    'slate_coverage_gate':'COMPLETE' if slate_complete else 'INCOMPLETE_OR_NOT_PROVIDED',
                    'notes':['No bet placing; BET is a research classification only.',
                             'Parlays remain WATCH until real combined price, vig, portfolio and availability review.',
                             'Missing or uncalibrated sporting probabilities produce WATCH, never invented probabilities.',
                             'Incomplete or missing full-slate audit blocks all actionable classifications.',
                             'No sportsbook is actionable unless explicitly listed in --actionable-books.']}
            write(a.out,report)
            print(f'Summary: {len(rows)} quotes / {sum(x["status"]=="BET" for x in rows)} verified single candidates / {len(parlays)} watch-only parlays')
            return 0
    except (ProviderFailure,ValueError,OSError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr);return 2

if __name__=='__main__':sys.exit(cli())