"""Daily no-provider market evidence continuity watchdog.

This is NOT a model, a historical backfill, a market price source, or an
authorization to bet. It measures whether the *process* that should produce
new market evidence is running and whether evidence counters advance.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

FAMILIES = ("1X2", "BTTS", "FT_TOTALS", "HOME_TT", "AWAY_TT")
LEDGER_MAX_SUCCESS_AGE_HOURS = 48
CLV_MAX_SUCCESS_AGE_HOURS = 48
SCHEDULER_MAX_AGE_HOURS = 30
LEDGER_DATA_MAX_LAG_HOURS = 72


def timestamp(raw):
    if not isinstance(raw, str):
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.astimezone(timezone.utc) if dt.tzinfo else None


def recent_workflow(runs, now, max_age_hours):
    if not isinstance(runs, dict) or not isinstance(runs.get("workflow_runs"), list):
        return {"status": "UNVERIFIED_RUN_HISTORY", "run_id": None, "last_success": None}
    current = [r for r in runs["workflow_runs"]
               if isinstance(r, dict) and r.get("head_branch") == "main"]
    current.sort(key=lambda r: timestamp(r.get("created_at")) or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    if not current:
        return {"status": "NO_MAIN_RUNS", "run_id": None, "last_success": None}
    latest = current[0]
    successful = [r for r in current if r.get("status") == "completed" and r.get("conclusion") == "success"]
    if not successful:
        return {"status": "NO_SUCCESSFUL_RUN", "run_id": latest.get("id"), "last_success": None}
    success = successful[0]
    completed = timestamp(success.get("updated_at"))
    if completed is None or not timedelta(0) <= now-completed <= timedelta(hours=max_age_hours):
        state = "SUCCESS_TOO_OLD_OR_TIME_INVALID"
    elif latest.get("status") == "completed" and latest.get("conclusion") == "failure" and latest.get("id") != success.get("id"):
        state = "LATEST_RUN_FAILED"
    else:
        state = "RECENT_SUCCESS"
    return {"status": state, "run_id": success.get("id"), "last_success": completed.isoformat() if completed else None}




def _count(raw):
    return raw if type(raw) is int and raw >= 0 else None


def sport_to_market_admission(latest, previous, *, now):
    """Expose the sport-first -> quote -> close path, without reclassifying bets.

    CLV absence on a single snapshot cannot be attributed to provider failure:
    it may reflect no frozen sport signal, no upcoming kickoff, admission caps,
    price freshness, or no later authentic provider update. These are distinct.
    """
    previous = previous if isinstance(previous, dict) else {}
    if not isinstance(latest, dict):
        return {
            "status": "SOURCE_NOT_VERIFIED",
            "warnings": ["LATEST_SCHEDULER_PAYLOAD_NOT_VERIFIED"],
            "three_daily_low_coverage_checks": False,
        }
    observed = timestamp(latest.get("generated_at_utc"))
    if observed is None or observed > now or now - observed > timedelta(hours=SCHEDULER_MAX_AGE_HOURS):
        return {
            "status": "SOURCE_NOT_VERIFIED",
            "warnings": ["LATEST_SCHEDULER_TIMESTAMP_INVALID_OR_STALE"],
            "three_daily_low_coverage_checks": False,
        }
    fair = latest.get("fair_scheduler") if isinstance(latest.get("fair_scheduler"), dict) else {}
    price = latest.get("price_resolution_v4") if isinstance(latest.get("price_resolution_v4"), dict) else {}
    guard = (
        ((price.get("v209_post_reserve_verification") or {}).get("upstream_phase_lock") or {})
        .get("resource_memory_guard")
    )
    if not isinstance(guard, dict):
        guard = {}
    eligible = _count(fair.get("eligible_tier_ab_due_count"))
    analyzed = _count(fair.get("due_with_any_deep_dive_count"))
    due = _count(latest.get("due_fixture_count"))
    entry = _count(price.get("candidate_rows"))
    maturing = _count(price.get("primary_clv_maturation_candidates"))
    priced = _count(price.get("api_calls_added"))
    price_budget = _count(price.get("max_api_calls"))
    remaining = _count(latest.get("last_daily_remaining"))
    mandatory = (eligible, analyzed, due, entry, maturing, priced, price_budget)
    invalid = any(v is None for v in mandatory)
    warnings = []
    if invalid:
        warnings.append("SPORT_TO_PRICE_COUNTS_NOT_VERIFIED")
    sport_coverage = round(analyzed / eligible, 6) if eligible and analyzed is not None else None
    low_coverage = eligible is not None and eligible >= 10 and analyzed is not None and analyzed * 4 < eligible
    entry_gap = due is not None and due >= 10 and entry == 0
    leftover = price_budget is not None and price_budget > 0 and priced == 0 and entry == 0 and maturing == 0
    if low_coverage:
        warnings.append("SPORT_FIRST_DEEP_DIVE_COVERAGE_UNDER_25_PCT")
    if entry_gap:
        warnings.append("PRICE_ENTRY_CANDIDATES_ZERO_WITH_DUE_FIXTURES")
    if leftover:
        warnings.append("PRICE_RESERVED_CAPACITY_UNUSED_WITH_NO_ELIGIBLE_TARGETS")
    if guard.get("applied") is True:
        warnings.append("UPSTREAM_LOW_MEMORY_GUARD_APPLIED")
    prev = previous.get("sport_to_market_admission")
    old = prev if isinstance(prev, dict) else {}
    prior_streak = _count(old.get("consecutive_low_coverage_daily_checks")) or 0
    old_check = timestamp(previous.get("observed_at_utc"))
    daily = old_check is not None and timedelta(hours=20) <= now - old_check <= timedelta(days=4)
    streak = prior_streak + 1 if daily and low_coverage and old.get("low_coverage") is True else (1 if low_coverage else 0)
    return {
        "status": "SOURCE_NOT_VERIFIED" if invalid else ("GAPS_OBSERVED" if warnings else "NO_GAPS_OBSERVED"),
        "latest_tick_utc": observed.isoformat(),
        "eligible_tier_ab_due": eligible,
        "deep_dives_for_due_tier_ab": analyzed,
        "sport_deep_dive_coverage": sport_coverage,
        "due_fixtures_all_tiers": due,
        "direct_price_candidate_rows": entry,
        "primary_clv_maturation_candidate_fixtures": maturing,
        "price_provider_calls_this_tick": priced,
        "price_provider_call_cap_this_tick": price_budget,
        "provider_daily_remaining_observed": remaining,
        "low_memory_guard_applied": guard.get("applied") is True,
        "low_memory_cgroup_limit_bytes": _count(guard.get("cgroup_limit_bytes")),
        "upstream_calls_deferred_by_resource_guard": _count(guard.get("deferred_extra_upstream_calls_max")),
        "low_coverage": low_coverage,
        "consecutive_low_coverage_daily_checks": streak,
        "three_daily_low_coverage_checks": streak >= 3,
        "warnings": warnings,
        "bet_eligible": False,
        "no_new_market_evidence_inferred": True,
        "interpretation": "One tick is descriptive only. A zero CLV candidate count is not proof of odds-provider outage. Source football projection must precede price resolution; CLV additionally requires a later comparable real quote.",
    }

def assess(health, ledger, clv, ledger_runs, clv_runs, previous, *, now, latest=None):
    if now.tzinfo is None:
        raise ValueError("AUDIT_TIME_MUST_BE_UTC_AWARE")
    now = now.astimezone(timezone.utc)
    issues = []
    alarms = {}
    workflows = {
        "postgre_delta_ledger": recent_workflow(ledger_runs, now, LEDGER_MAX_SUCCESS_AGE_HOURS),
        "market_clv": recent_workflow(clv_runs, now, CLV_MAX_SUCCESS_AGE_HOURS),
    }
    for name, entry in workflows.items():
        if entry["status"] != "RECENT_SUCCESS":
            issues.append(f"WORKFLOW_{name.upper()}_{entry['status']}")
    tick = timestamp(health.get("generated_at_utc"))
    evidence = timestamp(ledger.get("last_evidence_at_utc"))
    if tick is None or tick > now or now-tick > timedelta(hours=SCHEDULER_MAX_AGE_HOURS):
        issues.append("SCHEDULER_HEALTH_STALE_OR_INVALID")
    if health.get("database_persisted") is not True:
        issues.append("SCHEDULER_DATABASE_PERSISTENCE_NOT_VERIFIED")
    if evidence is None or evidence > now:
        issues.append("LEDGER_EVIDENCE_TIME_NOT_VERIFIED")
    elif tick is not None and tick-evidence > timedelta(hours=LEDGER_DATA_MAX_LAG_HOURS):
        issues.append("LEDGER_NOT_MATERIALIZED_AFTER_LIVE_SCHEDULER")
    if ledger.get("historical_probabilities_recomputed") is True:
        issues.append("HISTORICAL_PREDICTION_REWRITE_DETECTED")

    counts = clv.get("family_counts")
    if not isinstance(counts, dict):
        counts = {}
        issues.append("NO_FAMILY_CLV_COUNTS")
    if clv.get("production_promotion_allowed") is not False:
        issues.append("CLV_SOURCE_PRODUCTION_POLICY_NOT_VERIFIED")
    previous_families = previous.get("families") if isinstance(previous, dict) else {}
    if not isinstance(previous_families, dict):
        previous_families = {}
    old_time = timestamp(previous.get("observed_at_utc")) if isinstance(previous, dict) else None
    for family in FAMILIES:
        actual = counts.get(family)
        if type(actual) is not int or actual < 0:
            issues.append(f"{family}_CLV_COUNT_NOT_VERIFIED")
            continue
        prior = previous_families.get(family)
        old = prior.get("clv_rows") if isinstance(prior, dict) else None
        consecutive = prior.get("consecutive_unchanged_daily_checks", 0) if isinstance(prior, dict) else 0
        if type(consecutive) is not int or consecutive < 0:
            consecutive = 0
        if type(old) is int and old > actual:
            issues.append(f"{family}_HISTORICAL_CLV_COUNT_REGRESSION")
        daily_check = old_time is not None and timedelta(hours=20) <= now-old_time <= timedelta(days=4)
        no_growth = type(old) is int and old == actual
        checks = consecutive + 1 if daily_check and no_growth else 0
        alarms[family] = {
            "clv_rows": actual,
            "prior_clv_rows": old if type(old) is int else None,
            "new_verified_clv_rows_since_prior_check": max(actual-old, 0) if type(old) is int else None,
            "consecutive_unchanged_daily_checks": checks,
            "stalled_3plus_days": checks >= 3,
            "assessment": "OBSERVED_COUNTER_ONLY_NOT_EVIDENCE_OF_PROVIDER_AVAILABILITY",
        }
    total = clv.get("comparable_true_clv_rows")
    if type(total) is not int or total < 0:
        issues.append("TOTAL_COMPARABLE_CLV_NOT_VERIFIED")
    elif total < sum(v["clv_rows"] for v in alarms.values()):
        issues.append("TOTAL_COMPARABLE_CLV_BELOW_FAMILY_SUM")
    ledger_rows = ledger.get("rows")
    if type(ledger_rows) is not int or ledger_rows < 0:
        issues.append("LEDGER_ROW_COUNT_NOT_VERIFIED")
    source_funnel = sport_to_market_admission(latest, previous, now=now)
    if source_funnel["status"] == "SOURCE_NOT_VERIFIED":
        issues.append("SPORT_TO_PRICE_FUNNEL_SOURCE_NOT_VERIFIED")
    prev_ledger_rows = previous.get("ledger_rows") if isinstance(previous, dict) else None
    if type(ledger_rows) is int and type(prev_ledger_rows) is int and ledger_rows < prev_ledger_rows:
        issues.append("HISTORICAL_LEDGER_ROW_COUNT_REGRESSION")
    return {
        "schema_version": "1.0.0",
        "status": "PASS" if not issues else "FAIL_CLOSED",
        "observed_at_utc": now.isoformat(),
        "scheduler_latest_tick_utc": tick.isoformat() if tick else None,
        "scheduler_database_persisted": health.get("database_persisted") is True,
        "ledger_latest_evidence_utc": evidence.isoformat() if evidence else None,
        "ledger_rows": ledger_rows,
        "previous_ledger_rows": prev_ledger_rows,
        "clv_total_rows": total,
        "sport_to_market_admission": source_funnel,
        "families": alarms,
        "source_workflows": workflows,
        "blockers": issues,
        "stalled_market_families_3plus_days": [k for k,v in alarms.items() if v["stalled_3plus_days"]],
        "provider_requests_added": 0,
        "postgres_write_queries": 0,
        "historical_predictions_recomputed": False,
        "production_promotion_allowed": False,
        "notes": "Successful jobs and historical rows do not prove a new close-price exists. Missing strictly later quotes remain missing. No thresholds changed.",
    }


def self_test():
    now = datetime(2026,11,9,22,0,tzinfo=timezone.utc)
    iso=lambda d: d.isoformat()
    health = {"generated_at_utc":iso(now-timedelta(hours=2)),"database_persisted":True}
    ledger = {"last_evidence_at_utc":iso(now-timedelta(hours=2)),"rows":32178,"historical_probabilities_recomputed":False}
    clv = {"status":"ACTIVE_TRUE_CLV_SAMPLE","comparable_true_clv_rows":789,
           "family_counts":{"1X2":53,"BTTS":20,"FT_TOTALS":8,"HOME_TT":343,"AWAY_TT":365},
           "production_promotion_allowed":False}
    runs={"workflow_runs":[{"head_branch":"main","id":900,"created_at":iso(now-timedelta(hours=1)),
            "updated_at":iso(now-timedelta(minutes=5)),"conclusion":"success","status":"completed"}]}
    latest = {
        "generated_at_utc": iso(now-timedelta(hours=2)), "due_fixture_count":34,
        "last_daily_remaining":7427,
        "fair_scheduler":{"eligible_tier_ab_due_count":25,"due_with_any_deep_dive_count":2},
        "price_resolution_v4":{
            "candidate_rows":0,"primary_clv_maturation_candidates":0,
            "api_calls_added":0,"max_api_calls":8,
            "v209_post_reserve_verification":{
                "upstream_phase_lock":{"resource_memory_guard":{
                    "applied":True,"cgroup_limit_bytes":536870912,
                    "deferred_extra_upstream_calls_max":34
                }}
            }
        },
    }
    source=sport_to_market_admission(latest, {}, now=now)
    assert source["status"]=="GAPS_OBSERVED"
    assert source["eligible_tier_ab_due"]==25
    assert source["deep_dives_for_due_tier_ab"]==2
    assert source["sport_deep_dive_coverage"]==0.08
    assert source["primary_clv_maturation_candidate_fixtures"]==0
    assert source["price_provider_call_cap_this_tick"]==8
    assert source["price_provider_calls_this_tick"]==0
    assert "SPORT_FIRST_DEEP_DIVE_COVERAGE_UNDER_25_PCT" in source["warnings"]
    assert "PRICE_ENTRY_CANDIDATES_ZERO_WITH_DUE_FIXTURES" in source["warnings"]
    assert "UPSTREAM_LOW_MEMORY_GUARD_APPLIED" in source["warnings"]
    assert sport_to_market_admission({}, {}, now=now)["status"]=="SOURCE_NOT_VERIFIED"
    good=assess(health,ledger,clv,runs,runs,{},now=now,latest=latest)
    assert good["status"]=="PASS" and good["provider_requests_added"]==0
    assert good["clv_total_rows"]==789 and good["families"]["1X2"]["clv_rows"]==53
    old={**good,"observed_at_utc":iso(now-timedelta(days=1))}
    old["families"]["1X2"]["consecutive_unchanged_daily_checks"]=2
    stalled=assess(health,ledger,clv,runs,runs,old,now=now,latest=latest)
    assert stalled["status"]=="PASS"
    assert stalled["families"]["1X2"]["stalled_3plus_days"] is True
    regressed={**clv,"family_counts":{**clv["family_counts"],"1X2":52}}
    bad=assess(health,ledger,regressed,runs,runs,old,now=now,latest=latest)
    assert "1X2_HISTORICAL_CLV_COUNT_REGRESSION" in bad["blockers"]
    stale=assess(health,ledger,clv,{"workflow_runs":[]},runs,{},now=now,latest=latest)
    assert stale["status"]=="FAIL_CLOSED"
    print("SOCCER_MARKET_MATURATION_CONTINUITY_OFFLINE_PASS")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    for k in ["health","latest","ledger","clv","ledger-runs","clv-runs","previous","output"]:
        p.add_argument("--"+k)
    args=p.parse_args()
    if args.self_test:
        self_test()
        return
    required=["health","latest","ledger","clv","ledger_runs","clv_runs","output"]
    if not all(getattr(args,k) for k in required):
        p.error("health, latest, ledger, clv, ledger-runs, clv-runs, output required")
    read=lambda path:json.loads(Path(path).read_text(encoding="utf-8"))
    previous=read(args.previous) if args.previous and Path(args.previous).exists() else {}
    result=assess(read(args.health),read(args.ledger),read(args.clv),
                  read(args.ledger_runs),read(args.clv_runs),previous,now=datetime.now(timezone.utc),latest=read(args.latest))
    target=Path(args.output)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(result,sort_keys=True,ensure_ascii=False))


if __name__ == "__main__":
    main()
