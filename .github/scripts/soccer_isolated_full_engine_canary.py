"""One-shot full canonical soccer sporting-engine memory isolation canary.

Runs actual pinned runtime automation_v129 outside the 512MiB Render web service.
No Postgres credentials, betting outputs, retroactive source time or app writes.
Quote resolver may execute internally only AFTER its sporting model as normal,
but output preserves ONLY verified raw sport research forecasts, not prices.
"""
from __future__ import annotations
import argparse
import asyncio
from datetime import datetime,timezone,timedelta
import json
import os
from pathlib import Path
import re
import sys

MAX_PROVIDER_REQUESTS=40
MAX_FROZEN_EVENTS=20
MIN_BEFORE_KICKOFF=timedelta(minutes=5)



def _official_transport_budget_guard():
    """Hard cap on actual HTTPX provider transport across all canonical phases.

    Important: v129's reported api_calls_this_tick may include the downstream
    price-resolution phase, outside v2's upstream budget. This guard ensures
    no more than 40 *outgoing* API-Football requests even if a component fails
    to respect the upstream-only cap. It is local to one disposable runner.
    """
    import httpx
    import threading
    lock=threading.Lock()
    actual={"outgoing":0,"blocked":0,"quota_headers_verified":0}
    original_async=httpx.AsyncClient.send
    original_sync=httpx.Client.send
    hosts={"v3.football.api-sports.io","api-football-v1.p.rapidapi.com"}
    def admit(request):
        if request.url.host not in hosts:
            return False
        with lock:
            if actual["outgoing"]>=MAX_PROVIDER_REQUESTS:
                actual["blocked"]+=1
                raise RuntimeError("ISOLATED_CANONICAL_FULL_ENGINE_TRANSPORT_LIMIT")
            actual["outgoing"]+=1
        return True
    def observe(response, tracked):
        if tracked and response.headers.get("x-ratelimit-requests-remaining") is not None:
            with lock:
                actual["quota_headers_verified"]+=1
        return response
    async def async_send(client, request, *a, **k):
        tracked=admit(request)
        return observe(await original_async(client, request, *a, **k),tracked)
    def sync_send(client, request, *a, **k):
        tracked=admit(request)
        return observe(original_sync(client, request, *a, **k),tracked)
    httpx.AsyncClient.send=async_send
    httpx.Client.send=sync_send
    def restore():
        httpx.AsyncClient.send=original_async
        httpx.Client.send=original_sync
    return actual,restore


def _utc(v):
    if not isinstance(v,str):
        return None
    try:
        dt=datetime.fromisoformat(v.replace("Z","+00:00"))
        return dt.astimezone(timezone.utc) if dt.tzinfo else None
    except ValueError:
        return None


def _p(v):
    return type(v) in (float,int) and 0<=v<=1


def extract_sport_only(payload,after,sha):
    if not isinstance(payload,dict) or not isinstance(payload.get("events"),list):
        raise ValueError("CANONICAL_ENGINE_PAYLOAD_NOT_VERIFIED")
    before=_utc(payload.get("generated_at_utc"))
    if before is None or not before<=after:
        raise ValueError("START_TIMESTAMP_NOT_VERIFIED")
    if len(payload["events"])>3000:
        raise ValueError("CANONICAL_EVENTS_UNBOUNDED")
    due=payload.get("fair_scheduler") if isinstance(payload.get("fair_scheduler"),dict) else {}
    out=[];seen=set();nonmodel=0;post=0
    for e in payload["events"]:
        if not isinstance(e,dict):
            continue
        f=e.get("fixture")
        raw=e.get("raw_projection")
        if not isinstance(f,dict) or not isinstance(raw,dict):
            nonmodel+=1
            continue
        fid=f.get("fixture_id")
        kick=_utc(f.get("kickoff"))
        if type(fid) is not int or fid<=0 or kick is None:
            nonmodel+=1
            continue
        if not kick-after>=MIN_BEFORE_KICKOFF:
            post+=1
            continue
        if not all(_p(raw.get(x)) for x in ("raw_home_win_prob","raw_draw_prob","raw_away_win_prob","raw_btts_yes_prob","raw_over_2_5_prob")):
            nonmodel+=1
            continue
        if abs(sum(raw[x] for x in ("raw_home_win_prob","raw_draw_prob","raw_away_win_prob"))-1)>0.002:
            nonmodel+=1
            continue
        if fid in seen:
            continue
        seen.add(fid)
        out.append({
            "schema_version":"1.0.0",
            "status":"ISOLATED_CANONICAL_SPORT_RAW_RESEARCH_ONLY",
            "fixture_id":fid,"kickoff":kick.isoformat(),
            "observed_after_full_model_at_utc":after.isoformat(),
            "full_engine_started_at_utc":before.isoformat(),
            "original_fixture":{"home_team_id":f.get("home_team_id"),"away_team_id":f.get("away_team_id"),
                                "league_id":f.get("league_id"),"data_tier":(e.get("coverage") or {}).get("data_tier")},
            "model_version":raw.get("model_version") or e.get("model_version"),
            "source_runtime_commit":sha,
            "raw_sport_projection":{k:raw.get(k) for k in (
                "raw_home_win_prob","raw_draw_prob","raw_away_win_prob","raw_btts_yes_prob","raw_over_2_5_prob",
                "raw_home_goal_rate","raw_away_goal_rate","projection_model","status",
            )},
            "availability_confidence":e.get("availability_confidence"),
            "source_stage":e.get("stage"),
            "source":"PINNED_CANONICAL_V129_LIVE_ENGINE_GITHUB_RUNNER",
            "market_used_to_generate_raw_probability":False,
            "market_price_verified":False,"canonical_clv_eligible":False,
            "bet_eligible":False,"production_promotion_allowed":False,
            "retroactive_forecast_created":False,
        })
        if len(out)>=MAX_FROZEN_EVENTS:
            break
    return out,{
        "fixture_scan_count":payload.get("fixture_scan_count"),
        "due_fixture_count":payload.get("due_fixture_count"),
        "tier_ab_due":due.get("eligible_tier_ab_due_count"),
        "tier_ab_with_deep_dive":due.get("due_with_any_deep_dive_count"),
        "full_engine_reported_deep_dive_processed":payload.get("deep_dive_processed_count"),
        "raw_sport_snapshots_extracted":len(out),
        "event_rows_without_usable_sport_model":nonmodel,
        "model_events_not_safely_pregame":post,
        "pipeline_errors":sum(
            e.get("event_type")=="PIPELINE_ERROR" for e in payload["events"] if isinstance(e,dict)
        ),
    }


def self_test():
    a=datetime(2026,11,10,16,tzinfo=timezone.utc)
    kick=(a+timedelta(hours=2)).isoformat()
    p={"generated_at_utc":(a-timedelta(minutes=3)).isoformat(),
       "fair_scheduler":{"eligible_tier_ab_due_count":25,"due_with_any_deep_dive_count":6},
       "events":[{"fixture":{"fixture_id":44,"kickoff":kick,"home_team_id":1,
                             "away_team_id":2,"league_id":55},
                  "coverage":{"data_tier":"A"},"stage":"T-90",
                  "raw_projection":{"status":"MODELED_LIMITED",
                  "raw_home_win_prob":0.45,"raw_draw_prob":0.3,
                  "raw_away_win_prob":0.25,"raw_btts_yes_prob":0.6,
                  "raw_over_2_5_prob":0.55}},
                 {"fixture":{"fixture_id":45,"kickoff":(a-timedelta(minutes=1)).isoformat()},
                  "raw_projection":{"raw_home_win_prob":0.1}}]}
    rows,health=extract_sport_only(p,a,"a"*40)
    assert len(rows)==1 and rows[0]["fixture_id"]==44
    assert rows[0]["bet_eligible"] is False
    assert rows[0]["canonical_clv_eligible"] is False
    assert health["tier_ab_with_deep_dive"]==6
    assert health["model_events_not_safely_pregame"]==1
    assert "quota" not in rows[0] and "price" not in rows[0]
    assert MAX_PROVIDER_REQUESTS==40
    assert callable(_official_transport_budget_guard)
    print("ISOLATED_FULL_CANONICAL_ENGINE_SPORT_ONLY_ASOF_QA_PASS")


async def run(output,records,sha):
    if not re.fullmatch("[a-f0-9]{40}",sha):
        raise RuntimeError("RUNTIME_SHA_NOT_VERIFIED")
    if not os.environ.get("API_FOOTBALL_KEY"):
        raise RuntimeError("OFFICIAL_PROVIDER_KEY_MISSING")
    if os.environ.get("DATABASE_URL"):
        raise RuntimeError("CANARY_MUST_NOT_HAVE_DB_CREDENTIALS")
    if os.environ.get("SOCCER_EDGE_MAX_API_CALLS_PER_TICK")!=str(MAX_PROVIDER_REQUESTS):
        raise RuntimeError("PROVIDER_CALL_CAP_NOT_ENFORCED")
    # Import AFTER env setup, so v2/v6 budget reads the canary limit.
    from mcp_gateway import automation_v129
    finished_at=None
    traffic,restore=_official_transport_budget_guard()
    try:
        payload=await asyncio.wait_for(automation_v129.run_tick(),timeout=180)
    finally:
        restore()
    finished_at=datetime.now(timezone.utc)
    models,summary=extract_sport_only(payload,finished_at,sha)
    api_calls=payload.get("api_calls_this_tick")
    remain=payload.get("last_daily_remaining")
    provider_report=payload.get("quota") if isinstance(payload.get("quota"),dict) else {}
    if type(remain) is not int:
        remain=provider_report.get("daily_remaining")
    quota_verified=(type(remain) is int and remain>=1000)
    call_count_verified=(
        type(api_calls) is int
        and 0<=api_calls
        and 0<=traffic["outgoing"]<=MAX_PROVIDER_REQUESTS
        and traffic["quota_headers_verified"]>0
        and traffic["blocked"]==0
    )
    if not (quota_verified and call_count_verified):
        # The complete engine ran, but without quota proof we MUST NOT save
        # sport snapshots or claim an operational success. Publish a bounded,
        # no-secrets forensic summary so this can be diagnosed.
        print(json.dumps({
            "status":"CANONICAL_FULL_ENGINE_QUOTA_GATE_FAILED",
            "api_calls_type":type(api_calls).__name__,
            "api_calls_observed":api_calls if type(api_calls) is int else None,
            "last_daily_remaining_type":type(payload.get("last_daily_remaining")).__name__,
            "quota_daily_remaining_type":type(provider_report.get("daily_remaining")).__name__,
            "quota_verified":quota_verified,
            "call_count_verified":call_count_verified,
            "transport_outbound_requests":traffic["outgoing"],
            "transport_budget_denied":traffic["blocked"],
            "transport_quota_headers_verified":traffic["quota_headers_verified"],
            "sport_only_summary":summary,
            "canonical_clv_rows_added":0,
            "bet_eligible":False,
            "production_promotion_allowed":False,
        },sort_keys=True),flush=True)
        raise RuntimeError("CANONICAL_PROVIDER_QUOTA_PROOF_NOT_VERIFIED")
    report={
        "schema_version":"1.0.0","status":"ISOLATED_FULL_ENGINE_RESEARCH_CANARY_COMPLETED",
        "generated_at_utc":finished_at.isoformat(),
        "canonical_runtime_commit":sha,"api_calls_this_run":api_calls,
        "actual_httpx_provider_calls":traffic["outgoing"],
        "provider_transport_budget_denied":traffic["blocked"],
        "transport_quota_headers_verified":traffic["quota_headers_verified"],
        "daily_remaining_reported":remain,
        "no_db_credentials":True,"postgres_write_queries":0,
        "sport_model_weights_changed":False,"market_gates_changed":False,
        "canonical_clv_rows_added":0,"bet_eligible":False,
        "production_promotion_allowed":False,
        "raw_snapshots":summary,
    }
    if len(models)>MAX_FROZEN_EVENTS:
        raise RuntimeError("CANARY_MODEL_OUTPUT_LIMIT_EXCEEDED")
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    records.parent.mkdir(parents=True,exist_ok=True)
    with records.open("a",encoding="utf-8") as fp:
        for x in models:
            fp.write(json.dumps(x,sort_keys=True,ensure_ascii=False)+"\n")
    print(json.dumps(report,sort_keys=True))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--report",type=Path)
    p.add_argument("--frozen",type=Path)
    p.add_argument("--runtime-sha")
    args=p.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.report or not args.frozen or not args.runtime_sha:
        p.error("report, frozen, runtime-sha required")
    asyncio.run(run(args.report,args.frozen,args.runtime_sha))


if __name__=="__main__":
    main()
