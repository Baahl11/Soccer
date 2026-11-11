#!/usr/bin/env python3
"""Highlightly as-of SPORT evidence adapter — opt-in research ONLY.

- GitHub Actions secret is never logged or transported.
- Hard budget 10 requests, reserve 70/day, no price/odds paths.
- Canonical fixtures are pinned from a prior independent API-Football call.
- Strict team+kickoff+country+competition+season identity or REVIEW.
- Historic per-team aggregates are sampled, but NEVER backfilled as
  legitimate pregame inputs if captured after that fixture's kickoff.
- Only redacted evidence summaries may be shared. No raw provider JSON.
- This module never imports model/probability/market/persistence packages.
"""
from __future__ import annotations
import json
import os
import re
import sys
import time
import hashlib
import unicodedata
import urllib.request
import urllib.error
import urllib.parse
from datetime import datetime, timezone, timedelta

HOST = "https://soccer.highlightly.net"
MAX_REQUESTS = 10
# One-time authorized manual research run: no scheduler, no retries.
QUOTA_FLOOR = 70
# The canonical identity was independently observed via Soccer_Edge_API on
# 2026-11-11T04:41Z. These references do not contain odds or outcome data.
REFERENCE = (
    {"id":1570011,"date":"2026-11-09T19:30:00Z","home":"Las Palmas","away":"Eldense","country":"Spain","league":"Segunda División","season":2026},
    {"id":1576529,"date":"2026-11-09T20:15:00Z","home":"Torreense","away":"União de Leiria","country":"Portugal","league":"Segunda Liga","season":2026},
    {"id":1608737,"date":"2026-11-09T19:30:00Z","home":"Pineto","away":"Vis Pesaro","country":"Italy","league":"Serie C - Girone B","season":2026},
    {"id":1636031,"date":"2026-11-10T02:05:00Z","home":"Chapulineros","away":"Cordobés","country":"Mexico","league":"Primera Premier","season":2026},
    {"id":1600289,"date":"2026-11-09T17:00:00Z","home":"Nova Cidade","away":"Carapebus","country":"Brazil","league":"Carioca - 2","season":2026},
    {"id":1638304,"date":"2026-11-10T17:45:00Z","home":"Häcken W","away":"Chelsea W","country":"World","league":"UEFA Champions League Women","season":2026},
    {"id":1638332,"date":"2026-11-10T17:45:00Z","home":"Real Madrid W","away":"Paris FC W","country":"World","league":"UEFA Champions League Women","season":2026},
    {"id":1602548,"date":"2026-11-10T17:30:00Z","home":"Metalul Buzău","away":"CSA Steaua Bucureşti","country":"Romania","league":"Liga II","season":2026},
    {"id":1619186,"date":"2026-11-10T13:00:00Z","home":"Hull City U21","away":"Fleetwood Town U21","country":"England","league":"Professional Development League","season":2026},
    {"id":1581906,"date":"2026-11-11T09:30:00Z","home":"Chernihiv","away":"Inhulets","country":"Ukraine","league":"Persha Liga","season":2026},
    {"id":1648706,"date":"2026-11-11T09:30:00Z","home":"India","away":"Maldives","country":"World","league":"SAFF Championship","season":2026},
    {"id":1559458,"date":"2026-11-11T10:00:00Z","home":"Germany U17","away":"Azerbaijan U17","country":"World","league":"UEFA U17 Championship - Qualification","season":2027},
)
TARGET_DATES = ("2026-11-09","2026-11-10","2026-11-11")
SEASON_START = "2026-08-01"
ALLOWED_FAMILIES = ("GOALS", "FORM", "CORNERS", "CARDS", "SHOTS", "SOT", "POSSESSION", "PASSES", "XG")
ALLOWED_PATHS = re.compile(r"^(/matches|/teams/statistics/[0-9]{1,12}|/last-five-games|/statistics/[0-9]{1,12})$")
NOW = lambda: datetime.now(timezone.utc)


def canon(value):
    if not isinstance(value,str): return ""
    return re.sub(r"[^a-z0-9]", "",unicodedata.normalize("NFKD", value).encode("ascii","ignore").decode().lower())


def timestamp(value):
    if not isinstance(value,str): return None
    try:
        t=datetime.fromisoformat(value.strip().replace("Z","+00:00"))
    except (ValueError,TypeError): return None
    return t.astimezone(timezone.utc) if t.tzinfo else None


def match_identities(ref, row):
    """Conservative and reversible; no fuzzy aliases or posthoc hand-waving."""
    h=row.get("homeTeam") or {}
    a=row.get("awayTeam") or {}
    l=row.get("league") or {}
    c=row.get("country") or {}
    if not all(isinstance(v,dict) for v in (h,a,l,c)): return "UNVERIFIED_SHAPE"
    if canon(h.get("name")) != canon(ref["home"]) or canon(a.get("name")) != canon(ref["away"]):
        return "TEAM_NAMES_DIFFER"
    at=timestamp(row.get("date"))
    expected=timestamp(ref["date"])
    if at is None or expected is None or abs((at-expected).total_seconds()) > 900:
        return "KICKOFF_NOT_VERIFIED"
    if canon(c.get("name")) != canon(ref["country"]):
        return "COUNTRY_NOT_VERIFIED"
    if canon(l.get("name")) != canon(ref["league"]):
        return "LEAGUE_NOT_VERIFIED"
    if type(l.get("season")) is not int or l["season"] != ref["season"]:
        return "SEASON_NOT_VERIFIED"
    if type(row.get("id")) is not int or row["id"] <= 0:
        return "HIGHLIGHTLY_ID_INVALID"
    if type(h.get("id")) is not int or type(a.get("id")) is not int or h["id"] == a["id"]:
        return "TEAM_ID_INVALID"
    return "IDENTITY_EXACT_VERIFIED"


def capture_status(ref, captured_at):
    kickoff=timestamp(ref["date"])
    assert kickoff is not None
    return "PREMATCH_CAPTURE" if captured_at < kickoff-timedelta(minutes=5) else "AFTER_KICKOFF_RESEARCH_ONLY"


def inspect_families(payload):
    """Only verified *presence* of non-null values. Never fabricate missing 0."""
    dictionary={
        "GOALS": ("goal",),
        "FORM": ("wins","draws","loses","played"),
        "CORNERS": ("corner",),
        "CARDS": ("yellowcard","redcard","card"),
        "SHOTS": ("shot",),
        "SOT": ("shotsontarget","shotontarget"),
        "POSSESSION": ("possession",),
        "PASSES": ("pass",),
        "XG": ("expectedgoal","xg"),
    }
    present=set()
    queue=[payload]
    scanned=0
    while queue and scanned<2500:
        node=queue.pop()
        scanned+=1
        if isinstance(node,dict):
            name=canon(str(node.get("displayName") or node.get("name") or ""))
            value=node.get("value")
            if value is not None:
                for family,keys in dictionary.items():
                    if any(k in name for k in keys):present.add(family)
            for key,val in node.items():
                if val is None or val=="" or val==[]:continue
                # Dictionary keys such as 'goals' are evaluated only with a
                # populated scalar or stats sub-structure, never null.
                if not isinstance(val,(dict,list)):
                    name_key=canon(key)
                    for family,keys in dictionary.items():
                        if any(k in name_key for k in keys):present.add(family)
                else:queue.append(val)
        elif isinstance(node,list):
            queue.extend(node[:35])
    return sorted(present)


class Source:
    def __init__(self,key):
        self.key=key
        self.calls=0
        self.remaining=None
        self.records=[]
        self.stopped=False

    def get(self,path,params=None):
        if self.stopped or self.calls>=MAX_REQUESTS: return {"status":"BUDGET_STOP","data":None}
        if not ALLOWED_PATHS.fullmatch(path):raise ValueError("NON_SPORT_PATH_BLOCKED")
        if self.remaining is not None and self.remaining<=QUOTA_FLOOR:
            self.stopped=True
            return {"status":"QUOTA_FLOOR_STOP","data":None}
        self.calls+=1
        url=HOST+path+("?"+urllib.parse.urlencode(params) if params else "")
        req=urllib.request.Request(url,headers={
            "x-rapidapi-key":self.key,"Accept":"application/json",
            "User-Agent":"SoccerEdgeResearch/1.0"
        },method="GET")
        start=time.monotonic()
        at=NOW()
        try:
            with urllib.request.urlopen(req,timeout=12) as r:
                data_raw=r.read(1_500_001)
                hdr=r.headers
                http=r.status
                if len(data_raw)>1_500_000:raise ValueError("UNBOUNDED_PROVIDER_RESPONSE")
                data=json.loads(data_raw)
        except urllib.error.HTTPError as exc:
            http=exc.code
            hdr=exc.headers
            data=None
        except Exception as exc:
            out={"status":"NETWORK_OR_PARSE_ERROR","error_type":type(exc).__name__,"captured_at_utc":at.isoformat()}
            self.records.append({"endpoint":path,"http":None,"status":out["status"]})
            self.stopped=True
            return out
        rem=hdr.get("x-ratelimit-requests-remaining")
        if rem is not None and str(rem).isdigit():
            self.remaining=int(rem)
        if http in (401,403,429):
            self.stopped=True
        out={"status":"HTTP_OK" if http==200 else "HTTP_NOT_OK","data":data,
             "http":http,"captured_at_utc":at.isoformat()}
        self.records.append({"endpoint":path,"http":http,
                             "ms":round((time.monotonic()-start)*1000),
                             "quota_remaining":self.remaining})
        return out


def mapping_report(entries):
    """Return one deterministic state per canonical fixture, never 2 selections."""
    rows=[]
    for ref in REFERENCE:
        same_names=[]
        for event in entries:
            if not isinstance(event,dict):continue
            h=event.get("homeTeam");a=event.get("awayTeam")
            if not isinstance(h,dict) or not isinstance(a,dict):continue
            if canon(h.get("name")) == canon(ref["home"]) and canon(a.get("name")) == canon(ref["away"]):
                same_names.append((match_identities(ref,event),event))
        verified=[row for status,row in same_names if status=="IDENTITY_EXACT_VERIFIED"]
        if len(verified)==1:
            status="IDENTITY_EXACT_VERIFIED"
            event=verified[0]
            provider_id=event["id"]
            league=event.get("league") or {}
            fields={"provider_id":provider_id,"home_id":event["homeTeam"]["id"],
                    "away_id":event["awayTeam"]["id"],"league_id":league.get("id")}
        elif len(verified)>1:
            status="AMBIGUOUS_PROVIDER_FIXTURE"
            fields={}
        elif same_names:
            # Never select a name-only match. No provider ID survives.
            status=sorted(s for s,_ in same_names)[0]
            fields={}
        else:
            status="NO_MATCH_ON_FETCHED_PAGES"
            fields={}
        rows.append({"canonical_fixture_id":ref["id"],"identity_status":status,
                     "kickoff_utc":ref["date"],"competition":ref["league"],
                     "country":ref["country"],"season":ref["season"],
                     **fields})
    return rows


def inspect_team_stats(source, matched, ref):
    """Capture only feature availability and sample counts; not values."""
    observed=NOW()
    status=capture_status(ref,observed)
    if status!="PREMATCH_CAPTURE":
        return {"status":"AFTER_KICKOFF_NOT_VALID_PREGAME","captured_at_utc":observed.isoformat(),
                "team_statistics_not_backfilled":True}
    out={"status":"PREMATCH_RESEARCH_SAMPLE","captured_at_utc":observed.isoformat(),
         "source_updated_at":"NOT_VERIFIED","raw_projection_allowed":False,
         "availability_confidence_increased":False,
         "teams":{}}
    for side in ("home","away"):
        r=source.get(f"/teams/statistics/{matched[side+'_id']}",{"fromDate":SEASON_START})
        records=r.get("data") if isinstance(r.get("data"),list) else []
        # Only same-season SAME-LEAGUE entry allowed.
        selected=[x for x in records if isinstance(x,dict)
                  and x.get("season")==ref["season"]
                  and x.get("leagueId")==matched.get("league_id")]
        out["teams"][side]={"http":r.get("http"),
                            "same_league_season_records":len(selected),
                            "available_families":inspect_families(selected),
                            "data_source":"HIGHLIGHTLY_TEAM_SEASON_AGGREGATE",
                            "asof":r.get("captured_at_utc"),
                            "source_updated_at":"NOT_VERIFIED"}
        if source.stopped:break
    out["historical_record_before_kickoff_verified"]=False
    # A current aggregate with no explicit source_update is an observational
    # research input; cannot be used in official model backtest automatically.
    return out


def run(key):
    source=Source(key)
    pages=[]
    entries=[]
    for date in TARGET_DATES:
        if source.stopped:break
        offset=0
        # Nov11 has a known second page; only ONE extra page is allowed.
        for attempt in range(2 if date=="2026-11-11" else 1):
            response=source.get("/matches",{"date":date,"limit":100,"offset":offset,"timezone":"Etc/UTC"})
            body=response.get("data")
            data=body.get("data") if isinstance(body,dict) else None
            page=data if isinstance(data,list) else []
            meta=body.get("pagination") if isinstance(body,dict) else {}
            if not isinstance(meta,dict):meta={}
            total=meta.get("totalCount")
            pages.append({"date":date,"offset":offset,"http":response.get("http"),
                          "received":len(page),"declared_total":total,
                          "truncated":type(total) is int and total>offset+len(page)})
            entries.extend(page)
            if response.get("http")!=200 or source.stopped:break
            if type(total) is not int or total<=offset+len(page):break
            offset+=100
    mapped=mapping_report(entries)
    target=[row for row in mapped
            if row["identity_status"]=="IDENTITY_EXACT_VERIFIED" and
            capture_status(next(x for x in REFERENCE if x["id"]==row["canonical_fixture_id"]), NOW())=="PREMATCH_CAPTURE"]
    target.sort(key=lambda x:x["kickoff_utc"])
    studied=[]
    if target and source.calls+2<=MAX_REQUESTS and not source.stopped:
        row=target[0]
        ref=next(x for x in REFERENCE if x["id"]==row["canonical_fixture_id"])
        studied=[{"canonical_fixture_id":row["canonical_fixture_id"],
                  "identity_status":row["identity_status"],
                  "capture":inspect_team_stats(source,row,ref)}]
    report={"schema":"SOCCER_HIGHLIGHTLY_ASOF_RESEARCH_v1",
            "generated_at_utc":NOW().isoformat(),"source":"HIGHLIGHTLY_BASIC",
            "reference_source":"API_FOOTBALL_INDEPENDENT_FIXTURE_IDENTITY",
            "reference_count":len(REFERENCE),"scan_dates":list(TARGET_DATES),
            "pages":pages,"identity_results":[{k:v for k,v in x.items() if k not in ("provider_id","home_id","away_id","league_id")} for x in mapped],
            "prematch_team_stats_research":studied,
            "requests_executed":source.calls,"requests_cap":MAX_REQUESTS,
            "quota_floor":QUOTA_FLOOR,"quota_remaining":source.remaining,
            "calls_by_endpoint":source.records,
            "provider_key_logged":False,"raw_provider_json_persisted":False,
            "canonical_database_writes":0,"bet_eligible":False,
            "odds_queried":False,"source_license_for_customer_display":"NOT_VERIFIED",
            "model_input_promotion_allowed":False}
    return report


def self_test():
    r=REFERENCE[0]
    row={"id":9999,"date":r["date"],"homeTeam":{"id":111,"name":r["home"]},
         "awayTeam":{"id":222,"name":r["away"]},
         "league":{"id":3,"name":r["league"],"season":r["season"]},
         "country":{"name":r["country"]}}
    assert match_identities(r,row)=="IDENTITY_EXACT_VERIFIED"
    assert match_identities(r,{**row,"date":"2026-12-01T12:00:00Z"})=="KICKOFF_NOT_VERIFIED"
    assert match_identities(r,{**row,"league":{**row["league"],"season":2025}})=="SEASON_NOT_VERIFIED"
    assert match_identities(r,{**row,"league":{**row["league"],"name":"Another"}})=="LEAGUE_NOT_VERIFIED"
    assert match_identities(r,{**row,"country":{"name":"Wrong"}})=="COUNTRY_NOT_VERIFIED"
    assert mapping_report([row])[0]["identity_status"]=="IDENTITY_EXACT_VERIFIED"
    assert inspect_families([{"statistics":[{"displayName":"Corners","value":5},{"displayName":"Expected Goals","value":None}]}])==["CORNERS"]
    assert capture_status(r,timestamp(r["date"])+timedelta(minutes=1))=="AFTER_KICKOFF_RESEARCH_ONLY"
    assert not ALLOWED_PATHS.fullmatch("/odds")
    assert not ALLOWED_PATHS.fullmatch("/players/1/odds")
    print("HIGHLIGHTLY_ASOF_RESEARCH_LOCAL_GATES_PASS")


def main():
    if "--self-test" in sys.argv:
        self_test();return 0
    key=os.environ.get("SOCCER_HIGHLIGHTLY_API_KEY","").strip()
    if not key:
        print(json.dumps({"schema":"SOCCER_HIGHLIGHTLY_ASOF_RESEARCH_v1",
                          "status":"SECRET_NOT_AVAILABLE","requests_executed":0}))
        return 0
    report=run(key)
    print(json.dumps(report,sort_keys=True,ensure_ascii=False,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
