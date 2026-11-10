"""Prospective, exact-instrument research odds visits AFTER frozen SPORT FIRST.

API-Football bookmaker ID 8 prices are provider-reported, NOT independently
confirmed executable Bet365 betslip quotes. An authentic second visit only
qualifies as research price movement when provider_update strictly advances.
No retrospective quote, CLV promotion, bet or synthetic close.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import urllib.parse
import urllib.request

SOURCE="API_FOOTBALL_ODDS_V3"
MAX_CALLS=2
MAX_FIXTURES=2
MAX_ATTEMPTS_PER_FIXTURE=4
MAX_RESPONSE_BYTES=950_000
MIN_DAILY_REMAINING=1000
BOOKMAKER_ID=8
PRICE_SOURCE="API_FOOTBALL_BET365_PROXY_NOT_DIRECT_BETSLIP"
MAX_LEAD=timedelta(hours=12)
MIN_LEAD=timedelta(minutes=10)
CLOSE_WINDOW=timedelta(minutes=40)
MAX_REGISTRY_ROWS=5000


def _dt(v):
    if not isinstance(v,str):
        return None
    try:
        z=datetime.fromisoformat(v.replace("Z","+00:00"))
    except (TypeError,ValueError):
        return None
    return z.astimezone(timezone.utc) if z.tzinfo else None


def _decimal(x):
    if type(x) is not str and type(x) not in (float,int):
        return None
    try:
        n=float(x)
        return round(n,6) if math.isfinite(n) and 1.001<=n<=1000 else None
    except (ValueError,TypeError):
        return None


def _load_json(path:Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _quotes_in_recent_files(folder:Path,now:datetime):
    rows=[]
    # Only the three recent UTC days can contain pregame attempts for
    # fixtures in our 12h horizon. Historical files remain untouched.
    for day in [now.date()-timedelta(days=2),now.date()-timedelta(days=1),now.date()]:
        file=folder/(day.isoformat()+".jsonl")
        if not file.is_file():
            continue
        if file.stat().st_size>1_500_000:
            raise RuntimeError("RECENT_QUOTE_SOURCE_FILE_UNBOUNDED")
        for line in file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                x=json.loads(line)
                if not isinstance(x,dict) or x.get("source")!=SOURCE:
                    raise RuntimeError("UNVERIFIED_PRIOR_PRICE_ATTEMPT")
                rows.append(x)
    if len(rows)>1500:
        raise RuntimeError("RECENT_QUOTE_REGISTRY_UNBOUNDED")
    return rows


def eligible_frozen(frozen,receipt,previous,now):
    if receipt.get("status")!="RESEARCH_ONLY_FROZEN_PG_IMPORT":
        raise RuntimeError("FROZEN_PG_ORIGINAL_PROVENANCE_NOT_IMPORTED")
    imported=receipt.get("imported_keys") or []
    if not isinstance(imported,list) or len(imported)>MAX_REGISTRY_ROWS:
        raise RuntimeError("FROZEN_PG_IMPORT_RECEIPT_NOT_VERIFIED")
    verified=set(imported)
    if len(frozen)>MAX_REGISTRY_ROWS:
        raise RuntimeError("FROZEN_MODELS_UNBOUNDED")
    candidates=[]
    for r in frozen:
        fid=r.get("fixture_id")
        freeze=_dt(r.get("captured_at_utc"))
        kickoff=_dt(r.get("kickoff"))
        if not all([type(fid) is int,freeze,kickoff]) or not isinstance(r.get("fixture"),dict):
            raise RuntimeError("FROZEN_SOURCE_INPUT_INVALID")
        if (f"{fid}:{r['captured_at_utc']}" not in verified
            or r.get("status")!="FROZEN_RESEARCH_PREMATCH_SPORT_PROJECTION"
            or r.get("raw_projection_before_market") is not True
            or r.get("bet_eligible") is not False):
            continue
        if not (freeze<now and freeze<=kickoff-timedelta(minutes=45)
                and MIN_LEAD <= kickoff-now <= MAX_LEAD):
            continue
        past=[q for q in previous if q.get("fixture_id")==fid and q.get("frozen_at_utc")==r["captured_at_utc"]]
        if len(past)>=MAX_ATTEMPTS_PER_FIXTURE:
            continue
        latest=max((_dt(q.get("captured_at_utc")) for q in past),default=None)
        if latest is not None and now-latest<timedelta(minutes=18):
            continue
        actual_priced=any(isinstance(q.get("instruments"),list) and q["instruments"] for q in past)
        candidates.append((r,len(past),actual_priced))
    # One previous verified quote deserves a timely second visit before
    # adding an entirely new game. Never select by odds or predicted edge.
    candidates.sort(key=lambda x:(-int(x[2]),x[1],x[0]["kickoff"],x[0]["fixture_id"]))
    return candidates[:MAX_FIXTURES]


class OfficialOdds:
    def __init__(self,key):
        if not key:
            raise RuntimeError("OFFICIAL_ODDS_API_KEY_NOT_CONFIGURED")
        self.key=key
        self.calls=0
        self.remaining=None

    def fixture(self,fid):
        if self.calls>=MAX_CALLS or type(fid) is not int or fid<=0:
            raise RuntimeError("PRICE_VISIT_PROVIDER_CAP_OR_ID_FAILED")
        if self.remaining is not None and self.remaining<=MIN_DAILY_REMAINING:
            raise RuntimeError("PRICE_PROVIDER_DAILY_RESERVE")
        url="https://v3.football.api-sports.io/odds?"+urllib.parse.urlencode({
            "fixture":fid,"bookmaker":BOOKMAKER_ID,
        })
        req=urllib.request.Request(url,headers={
            "x-apisports-key":self.key,"Accept":"application/json",
        })
        with urllib.request.urlopen(req,timeout=17) as response:
            body=response.read(MAX_RESPONSE_BYTES+1)
            quota=response.headers.get("x-ratelimit-requests-remaining")
        self.calls+=1
        if len(body)>MAX_RESPONSE_BYTES:
            raise RuntimeError("ODDS_PROVIDER_RESPONSE_TOO_LARGE")
        try:
            remain=int(quota)
        except (ValueError,TypeError):
            raise RuntimeError("ODDS_DAILY_REMAINING_NOT_VERIFIED") from None
        if remain<MIN_DAILY_REMAINING:
            raise RuntimeError("ODDS_QUOTA_BELOW_RESERVE")
        self.remaining=remain if self.remaining is None else min(self.remaining,remain)
        obj=json.loads(body)
        if not isinstance(obj,dict) or not isinstance(obj.get("response"),list) or obj.get("errors"):
            raise RuntimeError("ODDS_PROVIDER_RETURN_NOT_VERIFIED")
        return obj


def _instrument(market_id,market,selection,line,raw_price):
    if type(market_id) is not int or market_id<=0:
        return None
    p=_decimal(raw_price)
    if p is None:
        return None
    return {"bookmaker_id":BOOKMAKER_ID,"market_id":market_id,
            "market":market,"selection":selection,"line":line,
            "decimal_price":p}


def normalize_exact(payload,fixture_id,observed_at):
    all_quotes=[]
    updates=[]
    for row in payload.get("response") or []:
        if not isinstance(row,dict) or (row.get("fixture") or {}).get("id")!=fixture_id:
            raise RuntimeError("ODDS_FIXTURE_ID_NOT_VERIFIED")
        update=_dt(row.get("update"))
        if update and update>observed_at+timedelta(minutes=5):
            raise RuntimeError("ODDS_PROVIDER_TIMESTAMP_IN_FUTURE")
        for book in row.get("bookmakers") or []:
            if book.get("id")!=BOOKMAKER_ID:
                continue
            if not update:
                # A price without a provider timestamp is not an as-of quote.
                continue
            updates.append(update)
            for bet in book.get("bets") or []:
                name=str(bet.get("name") or "").strip().lower()
                market_id=bet.get("id")
                outcomes={}
                for item in bet.get("values") or []:
                    value=str(item.get("value") or "").strip()
                    price=item.get("odd")
                    if name=="match winner" and value in ("Home","Draw","Away"):
                        key=("1X2",value,None)
                    elif name in ("both teams score","both teams to score") and value in ("Yes","No"):
                        key=("BTTS",value,None)
                    elif name in ("goals over/under","over/under") and value in ("Over 2.5","Under 2.5"):
                        key=("FT_TOTALS","Over" if value.startswith("Over") else "Under",2.5)
                    else:
                        continue
                    instrument=_instrument(market_id,name,key[1],key[2],price)
                    if instrument:
                        instrument["family"]=key[0]
                        instrument["provider_update_utc"]=update.isoformat()
                        outcomes[key]=instrument
                by_family={}
                for key,item in outcomes.items():
                    by_family.setdefault(key[0],[]).append(item)
                # Retain only complete two-way/three-way mutually exclusive
                # groups, never a single-sided quote disguised as a market.
                for family,values in by_family.items():
                    sides={x["selection"] for x in values}
                    need={"1X2":{"Home","Draw","Away"},
                          "BTTS":{"Yes","No"},
                          "FT_TOTALS":{"Over","Under"}}[family]
                    if sides==need:
                        all_quotes.extend(values)
    distinct={}
    for q in all_quotes:
        key=(q["market_id"],q["family"],q["selection"],q["line"])
        if key in distinct and distinct[key]["decimal_price"]!=q["decimal_price"]:
            raise RuntimeError("CONFLICTING_SAME_PROVIDER_INSTRUMENT")
        distinct[key]=q
    return list(distinct.values()),max(updates).isoformat() if updates else None


def _matching_price_pairs(attempts,fixture_id,frozen,kickoff):
    sample=[p for p in attempts if p.get("fixture_id")==fixture_id and p.get("frozen_at_utc")==frozen]
    sample.sort(key=lambda x:x.get("captured_at_utc",""))
    firsts={}
    pairs=[]
    for a in sample:
        visited=_dt(a.get("captured_at_utc"))
        update=_dt(a.get("provider_update_utc"))
        if not visited or not update or not (_dt(frozen)<visited<kickoff):
            continue
        for q in a.get("instruments") or []:
            when=_dt(q.get("provider_update_utc"))
            if when is None or when!=update:
                continue
            k=(q.get("bookmaker_id"),q.get("market_id"),q.get("family"),
               q.get("selection"),q.get("line"))
            if k not in firsts:
                firsts[k]=(visited,update,q)
                continue
            first_time,first_update,first=firsts[k]
            if (first_time<visited<kickoff and first_update<update
                    and first_update>_dt(frozen) and update>_dt(frozen)):
                pairs.append({
                    "bookmaker_id":BOOKMAKER_ID,"market_id":q["market_id"],
                    "family":q["family"],"selection":q["selection"],"line":q["line"],
                    "entry_price":first["decimal_price"],"later_price":q["decimal_price"],
                    "entry_captured_at_utc":first_time.isoformat(),
                    "later_captured_at_utc":visited.isoformat(),
                    "entry_provider_update_utc":first_update.isoformat(),
                    "later_provider_update_utc":update.isoformat(),
                    "within_40min_kickoff":kickoff-visited<=CLOSE_WINDOW,
                    "classification":"RESEARCH_PRICE_MOVEMENT_NOT_CANONICAL_CLV",
                    "bet_eligible":False,
                })
    seen={}
    for p in pairs:
        k=(p["market_id"],p["family"],p["selection"],p["line"])
        seen[k]=p
    return list(seen.values())


def _quote_provenance_status(instruments,updated,frozen_at):
    if not instruments:
        return "NO_EXACT_PROVIDER_MARKETS"
    source=_dt(updated)
    frozen=_dt(frozen_at)
    if not source or not frozen or source<=frozen:
        return "PROXY_QUOTE_VISIBLE_BUT_LAST_PROVIDER_UPDATE_NOT_POST_MODEL"
    return "PROXY_QUOTE_SOURCE_UPDATED_AFTER_FROZEN_MODEL"


def capture(frozen,receipt,previous,provider,now):
    eligible=eligible_frozen(frozen,receipt,previous,now)
    output=[]
    for model,prior_count,_ in eligible:
        fid=model["fixture_id"]
        if now>=_dt(model["kickoff"])-MIN_LEAD:
            continue
        result=provider.fixture(fid)
        captured=datetime.now(timezone.utc)
        if captured>=_dt(model["kickoff"]):
            raise RuntimeError("PRICE_CAPTURED_AFTER_KICKOFF")
        instruments,updated=normalize_exact(result,fid,captured)
        output.append({
            "source":SOURCE,"fixture_id":fid,
            "frozen_at_utc":model["captured_at_utc"],"kickoff":model["kickoff"],
            "captured_at_utc":captured.isoformat(),
            "provider_update_utc":updated,
            "bookmaker_id":BOOKMAKER_ID,
            "price_source":PRICE_SOURCE,
            "prior_attempts":prior_count,
            "instruments":instruments,
            "status":_quote_provenance_status(instruments,updated,model["captured_at_utc"]),
            "provider_update_post_model_verified":bool(
                instruments and _dt(updated) and _dt(updated)>_dt(model["captured_at_utc"])
            ),
            "direct_executable_betslip_verified":False,
            "canonical_clv_eligible":False,
            "production_promotion_allowed":False,"bet_eligible":False,
        })
    combined=previous+output
    pairs=[]
    for r in output:
        pairs.extend(_matching_price_pairs(combined,r["fixture_id"],r["frozen_at_utc"],_dt(r["kickoff"])))
    return {
        "status":"RESEARCH_ONLY_REAL_ODDS_VISIT",
        "generated_at_utc":datetime.now(timezone.utc).isoformat(),
        "eligible_model_fixtures":len(eligible),
        "price_visits_added":len(output),
        "visits_with_complete_exact_market":sum(bool(x["instruments"]) for x in output),
        "visits_with_provider_update_after_model":sum(x["provider_update_post_model_verified"] for x in output),
        "visits_with_stale_provider_update_at_model":sum(
            x["status"]=="PROXY_QUOTE_VISIBLE_BUT_LAST_PROVIDER_UPDATE_NOT_POST_MODEL"
            for x in output
        ),
        "visits_without_exact_bookmaker_markets":sum(not x["instruments"] for x in output),
        "verified_provider_update_pairs_this_run":len(pairs),
        "research_pairs":pairs,
        "new":output,
        "provider_requests_added":provider.calls,
        "daily_remaining_reported":provider.remaining,
        "bookmaker_id":BOOKMAKER_ID,
        "direct_betslip_price_verified":False,
        "canonical_clv_rows_added":0,
        "retroactive_prices_created":False,
        "production_promotion_allowed":False,"bet_eligible":False,
    }


def self_test():
    now=datetime(2026,11,12,19,tzinfo=timezone.utc)
    kickoff=now+timedelta(hours=2)
    freeze=now-timedelta(hours=1)
    model={"fixture_id":42,"kickoff":kickoff.isoformat(),
           "captured_at_utc":freeze.isoformat(),
           "raw_projection_before_market":True,
           "status":"FROZEN_RESEARCH_PREMATCH_SPORT_PROJECTION",
           "bet_eligible":False,"fixture":{"fixture_id":42}}
    receipt={"status":"RESEARCH_ONLY_FROZEN_PG_IMPORT",
             "imported_keys":[f"42:{freeze.isoformat()}"]}
    assert len(eligible_frozen([model],receipt,[],now))==1
    try:
        eligible_frozen([model],{"status":"NOT_IMPORTED"},[],now)
        raise AssertionError("Unimported frozen source admitted")
    except RuntimeError:
        pass
    def payload(update,over="1.95"):
        return {"response":[{"fixture":{"id":42},"update":update.isoformat(),
          "bookmakers":[{"id":8,"bets":[
            {"id":1,"name":"Match Winner","values":[{"value":"Home","odd":"1.95"},{"value":"Draw","odd":"3.2"},{"value":"Away","odd":"4.1"}]},
            {"id":8,"name":"Both Teams Score","values":[{"value":"Yes","odd":"1.7"},{"value":"No","odd":"2.0"}]},
            {"id":5,"name":"Goals Over/Under","values":[{"value":"Over 2.5","odd":over},{"value":"Under 2.5","odd":"1.85"}]},
          ]}]}]}
    quotes,u=normalize_exact(payload(now),42,now)
    assert _quote_provenance_status(quotes,u,freeze.isoformat())=="PROXY_QUOTE_SOURCE_UPDATED_AFTER_FROZEN_MODEL"
    assert _quote_provenance_status(quotes,(freeze-timedelta(hours=1)).isoformat(),freeze.isoformat())=="PROXY_QUOTE_VISIBLE_BUT_LAST_PROVIDER_UPDATE_NOT_POST_MODEL"
    assert _quote_provenance_status([],None,freeze.isoformat())=="NO_EXACT_PROVIDER_MARKETS"
    assert len(quotes)==7 and u==now.isoformat()
    assert all(x["bookmaker_id"]==8 for x in quotes)
    first={"source":SOURCE,"fixture_id":42,"frozen_at_utc":freeze.isoformat(),
           "kickoff":kickoff.isoformat(),"captured_at_utc":now.isoformat(),
           "provider_update_utc":now.isoformat(),"instruments":quotes}
    later=now+timedelta(minutes=30)
    later_quotes,_=normalize_exact(payload(later,over="1.90"),42,later)
    second={**first,"captured_at_utc":later.isoformat(),
            "provider_update_utc":later.isoformat(),"instruments":later_quotes}
    assert len(_matching_price_pairs([first,second],42,freeze.isoformat(),kickoff))==7
    stale={**second,"provider_update_utc":now.isoformat(),
           "instruments":[{**q,"provider_update_utc":now.isoformat()} for q in later_quotes]}
    assert _matching_price_pairs([first,stale],42,freeze.isoformat(),kickoff)==[]
    old_entry={**first,"provider_update_utc":(freeze-timedelta(minutes=30)).isoformat(),
               "instruments":[{**q,"provider_update_utc":(freeze-timedelta(minutes=30)).isoformat()} for q in quotes]}
    assert _matching_price_pairs([old_entry,second],42,freeze.isoformat(),kickoff)==[]
    false_second={**second,"captured_at_utc":(kickoff+timedelta(minutes=1)).isoformat()}
    assert _matching_price_pairs([first,false_second],42,freeze.isoformat(),kickoff)==[]
    wrong=payload(now)
    wrong["response"][0]["bookmakers"][0]["id"]=7
    assert normalize_exact(wrong,42,now)[0]==[]
    future=payload(now+timedelta(hours=1))
    try:
        normalize_exact(future,42,now)
        raise AssertionError("Future provider update accepted")
    except RuntimeError:
        pass
    print("REAL_ODDS_VISIT_OFFLINE_PASS: SAME_INSTRUMENT_STRICT_UPDATE_BEFORE_KICKOFF_NO_BET")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--frozen",type=Path)
    p.add_argument("--import-receipt",type=Path)
    p.add_argument("--quote-dir",type=Path)
    p.add_argument("--receipt",type=Path)
    args=p.parse_args()
    if args.self_test:
        self_test()
        return
    if not all([args.frozen,args.import_receipt,args.quote_dir,args.receipt]):
        p.error("frozen,import-receipt,quote-dir,receipt required")
    frozen=[json.loads(line) for line in args.frozen.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(frozen)>MAX_REGISTRY_ROWS:
        raise RuntimeError("FROZEN_REGISTRY_UNBOUNDED")
    receipt=_load_json(args.import_receipt)
    now=datetime.now(timezone.utc)
    previous=_quotes_in_recent_files(args.quote_dir,now)
    eligible=eligible_frozen(frozen,receipt,previous,now)
    if eligible:
        provider=OfficialOdds(os.getenv("API_FOOTBALL_KEY"))
        report=capture(frozen,receipt,previous,provider,now)
    else:
        report={"status":"RESEARCH_ONLY_REAL_ODDS_VISIT","generated_at_utc":now.isoformat(),
                "eligible_model_fixtures":0,"price_visits_added":0,
                "visits_with_complete_exact_market":0,
                "visits_with_provider_update_after_model":0,
                "visits_with_stale_provider_update_at_model":0,
                "visits_without_exact_bookmaker_markets":0,
                "verified_provider_update_pairs_this_run":0,"research_pairs":[],"new":[],
                "provider_requests_added":0,"daily_remaining_reported":None,
                "bookmaker_id":BOOKMAKER_ID,"direct_betslip_price_verified":False,
                "canonical_clv_rows_added":0,"retroactive_prices_created":False,
                "production_promotion_allowed":False,"bet_eligible":False}
    if report["new"]:
        args.quote_dir.mkdir(parents=True,exist_ok=True)
        file=args.quote_dir/(now.date().isoformat()+".jsonl")
        with file.open("a",encoding="utf-8") as out:
            for row in report["new"]:
                out.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n")
    info={k:v for k,v in report.items() if k not in ("new","research_pairs")}
    info["research_pair_summaries"]=report["research_pairs"][:16]
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(info,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in info.items() if k!="research_pair_summaries"},sort_keys=True))


if __name__=="__main__":
    main()
