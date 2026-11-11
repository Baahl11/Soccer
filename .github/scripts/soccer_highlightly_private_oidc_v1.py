#!/usr/bin/env python3
"""One-shot Highlightly GitHub BASIC → private Render OIDC research bridge.

Main-only, no cron, no public customer stats, zero odds, zero raw provider
values transferred, source timestamps and license gates explicit. The receiver
stores only ephemeral audited coverage flags in its own PG research table.
"""
from __future__ import annotations
import json
import os
import sys
import urllib.parse
import urllib.request
import urllib.error
from datetime import datetime,timezone,timedelta
import soccer_highlightly_asof_research_v1 as research

AUDIENCE="soccer-edge-render"
ENDPOINT="https://soccer-edge-api.onrender.com/internal/highlightly-sport-research-v1/import"
# GitHub squash commits include a multi-line body; the live workflow now matches the title prefix.
SCHEMA="SOCCER_HIGHLIGHTLY_PRIVATE_RESEARCH_v1"
SOURCE="HIGHLIGHTLY_BASIC_RESEARCH_ONLY"


def dt(value):
    return research.timestamp(value)


def build_import(report, *, now=None):
    now=now or datetime.now(timezone.utc)
    if report.get("schema")!="SOCCER_HIGHLIGHTLY_ASOF_RESEARCH_v1":
        raise ValueError("RESEARCH_FEED_SCHEMA_NOT_VERIFIED")
    if report.get("source")!="HIGHLIGHTLY_BASIC":
        raise ValueError("RESEARCH_FEED_SOURCE_NOT_VERIFIED")
    if any(report.get(x) is not False for x in (
        "bet_eligible","odds_queried","model_input_promotion_allowed",
        "raw_provider_json_persisted","provider_key_logged")):
        raise ValueError("NON_COMMERCIAL_RESEARCH_ONLY_GATE_FAILED")
    if report.get("source_license_for_customer_display")!="NOT_VERIFIED":
        raise ValueError("SOURCE_LICENSE_GATE_FAILED")
    if report.get("canonical_database_writes")!=0:
        raise ValueError("CANONICAL_DB_INPUT_FORBIDDEN")
    calls=report.get("requests_executed")
    remaining=report.get("quota_remaining")
    if type(calls) is not int or not 1<=calls<=10 or type(remaining) is not int or remaining<70:
        raise ValueError("RESEARCH_QUOTA_GATE_FAILED")
    report_at=dt(report.get("generated_at_utc"))
    if not report_at or not now-timedelta(minutes=20)<=report_at<=now+timedelta(seconds=3):
        raise ValueError("RESEARCH_SOURCE_NOT_RECENT")
    rows=report.get("identity_results")
    if not isinstance(rows,list) or len(rows)!=len(research.REFERENCE):
        raise ValueError("MISSING_FIXTURE_REFERENCE_ROWS")
    reference={r["id"]:r for r in research.REFERENCE}
    seen=set()
    studied={row["canonical_fixture_id"]:row for row in report.get("prematch_team_stats_research",[])}
    out=[]
    for r in rows:
        fid=r.get("canonical_fixture_id")
        if type(fid) is not int or fid not in reference or fid in seen:
            raise ValueError("INVALID_OR_DUPLICATE_CANONICAL_REFERENCE")
        seen.add(fid)
        ident=r.get("identity_status")
        if ident not in {
            "IDENTITY_EXACT_VERIFIED", "LEAGUE_NOT_VERIFIED",
            "NO_MATCH_ON_FETCHED_PAGES", "SEASON_NOT_VERIFIED",
            "COUNTRY_NOT_VERIFIED", "KICKOFF_NOT_VERIFIED",
            "TEAM_ID_INVALID", "HIGHLIGHTLY_ID_INVALID",
            "UNVERIFIED_SHAPE", "AMBIGUOUS_PROVIDER_FIXTURE",
        }:
            raise ValueError("UNKNOWN_IDENTITY_STATE")
        ref=reference[fid]
        if r.get("kickoff_utc")!=ref["date"] or r.get("season")!=ref["season"]:
            raise ValueError("CANONICAL_REFERENCE_TAMPERED")
        kickoff=dt(ref["date"])
        captured_at=report_at
        evidence="IDENTITY_PENDING"
        families=[]
        if ident=="NO_MATCH_ON_FETCHED_PAGES":
            evidence="NO_MATCH"
        elif ident=="IDENTITY_EXACT_VERIFIED":
            if now>=kickoff-timedelta(minutes=5):
                evidence="HISTORY_NOT_PREGAME"
            else:
                evidence="IDENTITY_EXACT_NO_FEATURES"
                study=studied.get(fid)
                if isinstance(study,dict) and study.get("identity_status")=="IDENTITY_EXACT_VERIFIED":
                    capture=study.get("capture")
                    if isinstance(capture,dict) and capture.get("status")=="PREMATCH_RESEARCH_SAMPLE":
                        ct=dt(capture.get("captured_at_utc"))
                        teams=capture.get("teams")
                        if ct and ct<kickoff-timedelta(minutes=5) and ct<=report_at and isinstance(teams,dict):
                            home,away=teams.get("home"),teams.get("away")
                            if isinstance(home,dict) and isinstance(away,dict):
                                if all(team.get("http")==200 and team.get("same_league_season_records",0)>=1
                                       and team.get("source_updated_at")=="NOT_VERIFIED" for team in (home,away)):
                                    a,b=home.get("available_families"),away.get("available_families")
                                    if isinstance(a,list) and isinstance(b,list):
                                        families=sorted(set(a)&set(b))
                                        families=[k for k in families if k in research.ALLOWED_FAMILIES]
                                        if families:
                                            evidence="PREMATCH_RESEARCH_LIMITED"
                                            captured_at=ct
        out.append({
            "fixture_id":fid,
            "kickoff_utc":ref["date"],
            "captured_at_utc":captured_at.isoformat(),
            "identity_status":ident,
            "evidence_status":evidence,
            "families":families,
            "source_updated_at_verified":False,
            "is_model_input":False,
            "bet_eligible":False,
            "commercial_display_allowed":False,
        })
    return {
        "schema":SCHEMA,"source":SOURCE,
        "report_generated_at_utc":report_at.isoformat(),
        "source_rights":"NOT_VERIFIED",
        "model_input_promotion_allowed":False,
        "bet_eligible":False,"odds_queried":False,
        "canonical_database_writes":0,
        "requests_executed":calls,
        "quota_remaining":remaining,
        "rows":out,
    }


def github_oidc_token():
    if os.getenv("GITHUB_REF")!="refs/heads/main" or os.getenv("GITHUB_REPOSITORY")!="Baahl11/Soccer":
        raise RuntimeError("ONLY_MAIN_REPOSITORY_WORKFLOW_CAN_SUBMIT")
    url=os.environ.get("ACTIONS_ID_TOKEN_REQUEST_URL")
    bearer=os.environ.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN")
    if not url or not bearer:raise RuntimeError("OIDC_IDENTITY_NOT_AVAILABLE")
    joiner="&" if "?" in url else "?"
    req=urllib.request.Request(url+joiner+urllib.parse.urlencode({"audience":AUDIENCE}),
                               headers={"Authorization":"bearer "+bearer,"Accept":"application/json"})
    with urllib.request.urlopen(req,timeout=12) as response:
        data=json.loads(response.read(40_000))
    token=data.get("value")
    if not isinstance(token,str) or token.count(".")!=2:
        raise RuntimeError("GITHUB_OIDC_TOKEN_NOT_VERIFIED")
    return token


def submit(payload):
    token=github_oidc_token()
    encoded=json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(",",":")).encode()
    if len(encoded)>28_000:raise RuntimeError("SOURCE_RESEARCH_OVERSIZE")
    req=urllib.request.Request(ENDPOINT,data=encoded,method="POST",headers={
        "Content-Type":"application/json",
        "Authorization":"Bearer "+token,
        "Accept":"application/json",
    })
    with urllib.request.urlopen(req,timeout=25) as response:
        data=json.loads(response.read(15_000))
        status=response.status
    if status!=200 or data.get("status")!="HIGHLIGHTLY_RESEARCH_COVERAGE_PERSISTED" or data.get("records_received")!=len(payload["rows"]):
        raise RuntimeError("PRIVATE_RESEARCH_IMPORT_ACK_NOT_VERIFIED")
    return {"status":"VERIFIED_PRIVATE_RESEARCH_IMPORT",
            "records_received":data["records_received"],
            "rows_inserted":data["rows_inserted"],
            "source":"HIGHLIGHTLY_BASIC_RESEARCH_ONLY",
            "model_input_promoted":False,
            "bet_eligible":False}


def self_test():
    report={
        "schema":"SOCCER_HIGHLIGHTLY_ASOF_RESEARCH_v1",
        "source":"HIGHLIGHTLY_BASIC",
        "generated_at_utc":"2026-11-11T05:03:00Z",
        "bet_eligible":False,
        "odds_queried":False,
        "model_input_promotion_allowed":False,
        "raw_provider_json_persisted":False,
        "provider_key_logged":False,
        "source_license_for_customer_display":"NOT_VERIFIED",
        "canonical_database_writes":0,
        "requests_executed":6,"quota_remaining":85,
        "identity_results":[{
            "canonical_fixture_id":r["id"],
            "identity_status":"IDENTITY_EXACT_VERIFIED",
            "kickoff_utc":r["date"],
            "season":r["season"]
        } for r in research.REFERENCE],
        "prematch_team_stats_research":[{"canonical_fixture_id":1581906,
            "identity_status":"IDENTITY_EXACT_VERIFIED",
            "capture":{"status":"PREMATCH_RESEARCH_SAMPLE",
                       "captured_at_utc":"2026-11-11T05:02:30Z",
                       "teams":{
                           "home":{"http":200,"same_league_season_records":1,"source_updated_at":"NOT_VERIFIED","available_families":["FORM"]},
                           "away":{"http":200,"same_league_season_records":1,"source_updated_at":"NOT_VERIFIED","available_families":["FORM"]},
                       }}
        }],
    }
    at=dt("2026-11-11T05:04:00Z")
    payload=build_import(report,now=at)
    assert len(payload["rows"])==12
    by_id={x["fixture_id"]:x for x in payload["rows"]}
    assert by_id[1581906]["evidence_status"]=="PREMATCH_RESEARCH_LIMITED"
    assert by_id[1581906]["families"]==["FORM"]
    assert by_id[1570011]["evidence_status"]=="HISTORY_NOT_PREGAME"
    assert by_id[1648706]["evidence_status"]=="IDENTITY_EXACT_NO_FEATURES"
    assert payload["bet_eligible"] is False
    assert payload["source_rights"]=="NOT_VERIFIED"
    report["odds_queried"]=True
    try:
        build_import(report,now=at)
    except ValueError:
        pass
    else:
        raise AssertionError("ODDS_GATE_NOT_BLOCKED")
    print("HIGHLIGHTLY_OIDC_BRIDGE_OFFLINE_GATES_PASS")


def main():
    if "--self-test" in sys.argv:
        self_test()
        return 0
    key=os.getenv("SOCCER_HIGHLIGHTLY_API_KEY","").strip()
    if not key:raise RuntimeError("HIGHLIGHTLY_KEY_NOT_AVAILABLE_IN_MAIN_WORKFLOW")
    report=research.run(key)
    payload=build_import(report)
    # Both artifact and logs carry derived flags ONLY; provider JSON remains in memory.
    receipt={
        "report_status":"HIGHLIGHTLY_STRICT_ASOF_CAPTURED",
        "requests_executed":report["requests_executed"],
        "daily_quota_remaining":report["quota_remaining"],
        "identity_exact":sum(x["identity_status"]=="IDENTITY_EXACT_VERIFIED" for x in payload["rows"]),
        "research_prematch_fixtures":sum(x["evidence_status"]=="PREMATCH_RESEARCH_LIMITED" for x in payload["rows"]),
        "model_input_promoted":False,
        "bet_eligible":False,
    }
    try:
        receipt["private_postgres"]=submit(payload)
    except Exception as exc:
        # No credentials, JWT, response body or provider payload in error output.
        receipt["private_postgres"]={"status":"NOT_VERIFIED","error_type":type(exc).__name__}
        print(json.dumps(receipt,sort_keys=True,indent=2))
        return 1
    print(json.dumps(receipt,sort_keys=True,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
