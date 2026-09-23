from __future__ import annotations

import gc
import functools
import os
import resource
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone as dt_timezone
from typing import Any, Awaitable, Callable

import httpx

from mcp_gateway import automation as base
from mcp_gateway import automation_v2 as v2
from mcp_gateway import automation_v3 as v3
from mcp_gateway import automation_v4 as v4
from mcp_gateway import automation_v5 as v5
from mcp_gateway import automation_v6 as v6
from mcp_gateway import automation_v89 as v89

MODEL_VERSION = v89.MODEL_VERSION
AUTOMATION_VERSION = "3.62.1"

CORE_SLATE_FLOOR_MIN_FIXTURES = int(os.getenv("SOCCER_EDGE_SLATE_FLOOR_MIN_FIXTURES", "12"))
CORE_SLATE_FLOOR_MIN_DAILY_REMAINING = int(
    os.getenv("SOCCER_EDGE_SLATE_FLOOR_MIN_DAILY_REMAINING", "4000")
)
ELASTIC_GC_BATCH_SIZE = max(1, int(os.getenv("SOCCER_EDGE_ELASTIC_GC_BATCH_SIZE", "4")))
POSTGAME_STATS_MAX_ROWS = max(1, int(os.getenv("SOCCER_EDGE_POSTGAME_STATS_MAX_ROWS", "2")))

RunTick = Callable[[], Awaitable[dict[str, Any]]]


def _elastic_deep_dive_cap(daily_remaining: Any, due_count: int) -> tuple[int, str]:
    """Scale fixture coverage from verified quota; caches still govern actual calls."""
    try:
        remaining = int(daily_remaining)
    except (TypeError, ValueError):
        return 12, "QUOTA_UNKNOWN_SAFE"
    if remaining > 6000:
        return min(max(due_count, 24), 48), "GT_6000"
    if remaining > 4500:
        return min(max(due_count, 20), 36), "GT_4500"
    if remaining > 3000:
        return min(max(due_count, 16), 28), "GT_3000"
    if remaining > 1500:
        return min(max(due_count, 12), 20), "GT_1500"
    return min(max(due_count, 8), 12), "RESERVE_MODE"


def _elastic_request_cap(daily_remaining: Any) -> tuple[int, str]:
    """Hard per-tick request ceiling; cache hits consume zero provider requests."""
    try:
        remaining = int(daily_remaining)
    except (TypeError, ValueError):
        return 35, "QUOTA_UNKNOWN_LEGACY"
    if remaining > 6000:
        return 70, "GT_6000"
    if remaining > 4500:
        return 55, "GT_4500"
    if remaining > 3000:
        return 45, "GT_3000"
    if remaining > 1500:
        return 35, "GT_1500"
    return 25, "RESERVE_MODE"


def _new_core_metrics() -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "policy": "CORE_SCHEDULER_DATE_SLATE_FLOOR_AFTER_SOURCE_SELECTION",
        "min_fixture_count": CORE_SLATE_FLOOR_MIN_FIXTURES,
        "min_daily_remaining": CORE_SLATE_FLOOR_MIN_DAILY_REMAINING,
        "triggered": False,
        "reason": None,
        "primary_date": None,
        "reconciliation_date": None,
        "primary_slate_count": None,
        "reconciliation_slate_count": 0,
        "merged_slate_count": None,
        "provider_requests_added_max": 1,
        "provider_requests_added": 0,
        "scope": "core scheduler fixtures?date slate only; no odds request; no market/tier/stake/model-weight change",
    }


def _daily_remaining_allows(value: Any) -> bool:
    try:
        if value is None:
            return True
        return int(value) > CORE_SLATE_FLOOR_MIN_DAILY_REMAINING
    except (TypeError, ValueError):
        return True


def _tick_budget_allows_extra_call() -> bool:
    try:
        cap = int(v2.MAX_API_CALLS_PER_TICK or 0)
        used = int(v2._API_CALLS_THIS_TICK or 0)
    except (TypeError, ValueError):
        return False
    return cap <= 0 or used < cap


def _compact_valid_fixture(row: Any) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    fx = base._compact_fixture(row)
    if (
        fx.get("fixture_id")
        and fx.get("kickoff")
        and fx.get("league_id")
        and fx.get("season")
    ):
        return fx
    return None


def _v90_mem(label: str) -> None:
    rss = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    line = f"V90_MEM {label} peak_rss_mb={rss}"
    print(line, file=sys.stderr, flush=True)
    trace_path = os.getenv("SOCCER_EDGE_V90_TRACE_FILE", "/tmp/soccer_v90_mem.trace")
    try:
        with open(trace_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\\n")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError:
        pass


def _compact_runtime_event(event: dict[str, Any]) -> dict[str, Any]:
    """Bound event retention without changing decision inputs or provider calls."""
    if event.get("stage") == "POSTGAME" and isinstance(event.get("match_stats"), list):
        # Full provider statistics are already durable/cached elsewhere; the tick only
        # needs a bounded diagnostic snapshot for downstream presentation/persistence.
        compact_rows = []
        for row in event["match_stats"][:POSTGAME_STATS_MAX_ROWS]:
            if not isinstance(row, dict):
                continue
            team = row.get("team") or {}
            compact_rows.append({
                "team": {"id": team.get("id"), "name": team.get("name")},
                "statistics": (row.get("statistics") or [])[:24],
            })
        event["match_stats"] = compact_rows
    return event


def _append_fixture_payload(
    fixtures: list[dict[str, Any]],
    seen_fixture_ids: set[int],
    payload: dict[str, Any],
) -> int:
    added = 0
    for row in payload.get("response", []) or []:
        fx = _compact_valid_fixture(row)
        if not fx:
            continue
        fixture_id = int(fx["fixture_id"])
        if fixture_id in seen_fixture_ids:
            continue
        seen_fixture_ids.add(fixture_id)
        fixtures.append(fx)
        added += 1
    return added


async def _run_tick_with_core_slate_floor() -> dict[str, Any]:
    _v90_mem("entered_core_slate_floor")
    v2._API_CALLS_THIS_TICK = 0
    v2._LAST_DAILY_REMAINING = None

    # Preserve v1.4/v1.6 protections. At runtime v6 has already replaced
    # v4._paced_api_get with the adaptive wrapper, so this assignment keeps the
    # normal budget/pacing chain active.
    base._api_get = v4._paced_api_get
    v2.evaluate_market = v4._safe_evaluate_market

    now_utc = datetime.now(dt_timezone.utc)
    local_now = now_utc.astimezone(base.TIMEZONE)
    base._prune_cache(now_utc)

    dates = [local_now.date()]
    if local_now.hour >= 22:
        dates.append((local_now + timedelta(days=1)).date())

    fixtures: list[dict[str, Any]] = []
    seen_fixture_ids: set[int] = set()
    quota: dict[str, Any] = {}
    core_metrics = _new_core_metrics()
    core_metrics["primary_date"] = dates[0].isoformat()

    original_coverage = base._coverage
    base._coverage = v3._coverage_fast

    base._HTTP_CLIENT = httpx.AsyncClient(
        timeout=base.TIMEOUT,
        limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
    )
    try:
        for index, d in enumerate(dates):
            payload = await base._api_get(
                "fixtures", {"date": d.isoformat(), "timezone": base.TIMEZONE_NAME}
            )
            quota = payload.get("quota", quota)
            # The primary slate is the first real API-Football request in this path.
            # Capture its verified header quota immediately so elastic caps are based
            # on the live provider budget rather than the unknown-safe fallback.
            if v2._LAST_DAILY_REMAINING is None:
                try:
                    remaining = (quota or {}).get("daily_remaining")
                    if remaining is not None:
                        v2._LAST_DAILY_REMAINING = int(remaining)
                except (TypeError, ValueError):
                    pass
            added = _append_fixture_payload(fixtures, seen_fixture_ids, payload)
            if index == 0:
                core_metrics["primary_slate_count"] = added
                core_metrics["merged_slate_count"] = len(fixtures)

        if len(dates) == 1:
            primary_count = int(core_metrics.get("primary_slate_count") or 0)
            if primary_count >= CORE_SLATE_FLOOR_MIN_FIXTURES:
                core_metrics["reason"] = "PRIMARY_SLATE_MEETS_FLOOR"
            elif local_now.hour >= 22:
                core_metrics["reason"] = "LEGACY_LATE_DAY_NEXT_DATE_ALREADY_ENABLED"
            elif not _tick_budget_allows_extra_call():
                core_metrics["reason"] = "TICK_BUDGET_BLOCKED"
            elif not _daily_remaining_allows(v2._LAST_DAILY_REMAINING):
                core_metrics["reason"] = "DAILY_BUDGET_BLOCKED"
            else:
                tomorrow = (local_now.date() + timedelta(days=1)).isoformat()
                reconciliation = await base._api_get(
                    "fixtures", {"date": tomorrow, "timezone": base.TIMEZONE_NAME}
                )
                quota = reconciliation.get("quota", quota)
                added = _append_fixture_payload(fixtures, seen_fixture_ids, reconciliation)
                core_metrics["triggered"] = True
                core_metrics["reason"] = "PRIMARY_SLATE_BELOW_FLOOR_RECONCILED_WITH_NEXT_DATE"
                core_metrics["reconciliation_date"] = tomorrow
                core_metrics["reconciliation_slate_count"] = added
                core_metrics["merged_slate_count"] = len(fixtures)
                core_metrics["provider_requests_added"] = 1
        else:
            core_metrics["reason"] = "LEGACY_MULTI_DATE_SLATE_ALREADY_ENABLED"
            core_metrics["merged_slate_count"] = len(fixtures)

        _v90_mem("after_slate")
        events: list[dict[str, Any]] = []
        discovery = await v3._daily_discovery_event(fixtures, now_utc, local_now)
        if discovery is not None:
            events.append(discovery)
        _v90_mem("after_discovery")

        due: list[dict[str, Any]] = []
        low_data_counts: Counter[str] = Counter()

        for fx in fixtures:
            kickoff = base._dt(fx["kickoff"])
            minutes_to = (kickoff - now_utc).total_seconds() / 60.0
            stage = base._stage_for(minutes_to, fx.get("status") or "")
            if not stage:
                continue

            if stage == "POSTGAME":
                minutes_since = -minutes_to
                if minutes_since < 95 or minutes_since > 240:
                    continue
                prior = v5._shortlist_get(fx["fixture_id"], now_utc)
                if not prior or not prior.get("shortlisted"):
                    continue

            coverage = await v3._coverage_fast(fx["league_id"], fx["season"], now_utc)
            tier = coverage.get("data_tier") or "D"
            prior = v5._shortlist_get(fx["fixture_id"], now_utc)
            prior_rank = v5._num((prior or {}).get("rank")) or 0.0
            target = v5.STAGE_TARGET.get(stage)
            proximity = abs(minutes_to - target) if target is not None else 0.0

            due.append(
                {
                    "fx": fx,
                    "stage": stage,
                    "coverage": coverage,
                    "tier": tier,
                    "priority": (
                        0 if tier in {"A", "B"} else 1,
                        v5.STAGE_PRIORITY.get(stage, 99),
                        0 if prior and prior.get("shortlisted") else 1,
                        0 if tier == "A" else 1 if tier == "B" else 2 if tier == "C" else 3,
                        v5._competition_priority(fx),
                        -v5._coverage_strength(coverage),
                        -prior_rank,
                        proximity,
                        kickoff,
                    ),
                }
            )

        due.sort(key=lambda item: item["priority"])
        _v90_mem(f"after_due_build count={len(due)}")

        quota_remaining_basis = v2._LAST_DAILY_REMAINING
        if quota_remaining_basis is None:
            for key in ("daily_remaining", "requests_remaining", "remaining"):
                try:
                    value = (quota or {}).get(key)
                    if value is not None:
                        quota_remaining_basis = int(value)
                        break
                except (TypeError, ValueError):
                    pass
        elastic_deep_dive_cap, elastic_cap_reason = _elastic_deep_dive_cap(quota_remaining_basis, len(due))
        elastic_request_cap, elastic_request_reason = _elastic_request_cap(quota_remaining_basis)
        # The budget wrapper reads this value dynamically. Cached reads do not increment it.
        v2.MAX_API_CALLS_PER_TICK = elastic_request_cap
        deep_dive_processed = 0
        deferred_due_to_priority = 0
        deferred_due_to_budget = 0
        market_requests_avoided_by_screen = 0
        shortlist_events = 0
        gc_batches_completed = 0

        for item in due:
            fx = item["fx"]
            stage = item["stage"]
            coverage = item["coverage"]
            tier = item["tier"]

            if v5._stage_already_processed(fx["fixture_id"], stage, now_utc):
                continue

            if tier not in {"A", "B"}:
                low_data_counts[f"{tier}:{stage}"] += 1
                v5._mark_stage_processed(fx["fixture_id"], stage, now_utc)
                continue

            if deep_dive_processed >= elastic_deep_dive_cap:
                deferred_due_to_priority += 1
                continue

            try:
                event = await v5._priority_event(fx, stage, coverage, now_utc)
            except v2.TickBudgetExceeded as exc:
                deferred_due_to_budget += 1
                events.append(
                    {
                        "event_type": "QUOTA_GUARD",
                        "stage": stage,
                        "fixture": fx,
                        "model_version": v5.MODEL_VERSION,
                        "classification": "WATCH",
                        "error": str(exc),
                    }
                )
                break
            except Exception as exc:
                events.append(
                    {
                        "event_type": "PIPELINE_ERROR",
                        "stage": stage,
                        "fixture": fx,
                        "model_version": v5.MODEL_VERSION,
                        "classification": "WATCH",
                        "error": str(exc)[:500],
                    }
                )
                continue

            deep_dive_processed += 1
            v5._mark_stage_processed(fx["fixture_id"], stage, now_utc)
            if event.get("market_skipped_by_sport_screen"):
                market_requests_avoided_by_screen += 1
            if (event.get("sporting_shortlist") or {}).get("shortlisted"):
                shortlist_events += 1
            events.append(_compact_runtime_event(event))
            event = None

            # Keep elastic coverage independent from worker memory growth. Persistent
            # provider caches live in SQLite, so collecting transient Python objects
            # between small batches does not discard cached API responses.
            if deep_dive_processed % ELASTIC_GC_BATCH_SIZE == 0:
                gc.collect()
                gc_batches_completed += 1

        if deep_dive_processed and deep_dive_processed % ELASTIC_GC_BATCH_SIZE:
            gc.collect()
            gc_batches_completed += 1
        _v90_mem(f"after_deep_dive_loop processed={deep_dive_processed} events={len(events)}")

        if low_data_counts:
            events.append(
                {
                    "event_type": "LOW_DATA_SCREEN_SUMMARY",
                    "stage": "SLATE_SCREEN",
                    "classification": "PASS",
                    "model_version": v5.MODEL_VERSION,
                    "screened_out_count": sum(low_data_counts.values()),
                    "by_tier_stage": dict(low_data_counts),
                    "notes": [
                        "Data Tier C/D fixtures remain counted in the slate but do not consume deep-dive API budget.",
                        "Detailed market/lineup refresh is reserved for Data Tier A/B sporting candidates.",
                    ],
                }
            )

        _v90_mem("before_actionable_lists")
        actionable = [
            e for e in events if e.get("stage") in {"T-40", "T-20", "T-10", "CLOSE"}
        ]
        bets = [e for e in events if e.get("classification") == "BET"]
        _v90_mem(f"after_actionable_lists actionable={len(actionable)} bets={len(bets)}")

        _v90_mem("before_return_payload")
        return {
            "service": "soccer-edge-automation",
            "version": v5.AUTOMATION_VERSION,
            "model_version": v5.MODEL_VERSION,
            "generated_at_utc": now_utc.isoformat(),
            "generated_at_local": local_now.isoformat(),
            "timezone": base.TIMEZONE_NAME,
            "fixture_scan_count": len(fixtures),
            "due_fixture_count": len(due),
            "event_count": len(events),
            "actionable_refresh_count": len(actionable),
            "bet_candidate_count": len(bets),
            "api_calls_this_tick": v2._API_CALLS_THIS_TICK,
            "max_api_calls_per_tick": v2.MAX_API_CALLS_PER_TICK,
            "last_daily_remaining": v2._LAST_DAILY_REMAINING,
            "deferred_due_to_budget": deferred_due_to_budget,
            "deferred_due_to_priority": deferred_due_to_priority,
            "deep_dive_processed_count": deep_dive_processed,
            "shortlist_event_count": shortlist_events,
            "screened_out_low_data_count": sum(low_data_counts.values()),
            "market_requests_avoided_by_sport_screen": market_requests_avoided_by_screen,
            "max_deep_dive_fixtures_per_tick": elastic_deep_dive_cap,
            "configured_deep_dive_floor": v5.MAX_DEEP_DIVE_FIXTURES_PER_TICK,
            "elastic_deep_dive_cap": elastic_deep_dive_cap,
            "elastic_deep_dive_cap_reason": elastic_cap_reason,
            "elastic_request_cap": elastic_request_cap,
            "elastic_request_cap_reason": elastic_request_reason,
            "elastic_quota_remaining_basis": quota_remaining_basis,
            "elastic_gc_batch_size": ELASTIC_GC_BATCH_SIZE,
            "elastic_gc_batches_completed": gc_batches_completed,
            "runtime_event_retention_policy": "COMPACT_POSTGAME_STATS_BOUNDED",
            "priority_queue": "DATA_TIER_THEN_STAGE_THEN_PRIOR_SHORTLIST_THEN_COMPETITION_THEN_COVERAGE",
            "request_pacing_seconds": v4.MIN_REQUEST_INTERVAL_SECONDS,
            "rate_limit_max_retries": v4.RATE_LIMIT_MAX_RETRIES,
            "first_half_market_model": "BLOCKED_PENDING_EXPLICIT_1H_MODEL",
            "quota": quota,
            "events": events,
            "database_persistence": "OPTIONAL_NOT_REQUIRED_FOR_SCHEDULER",
            "core_slate_floor_reconciliation": core_metrics,
        }
    finally:
        base._coverage = original_coverage
        if base._HTTP_CLIENT is not None:
            await base._HTTP_CLIENT.aclose()
            base._HTTP_CLIENT = None
        if base._CACHE_CONN is not None:
            base._CACHE_CONN.commit()


def _apply_top_level_metrics(payload: dict[str, Any]) -> None:
    core = payload.get("core_slate_floor_reconciliation")
    if not isinstance(core, dict):
        return

    core_calls = int(core.get("provider_requests_added") or 0)
    raw_calls = int(payload.get("api_slate_reconciliation_calls") or 0)
    core_triggered = bool(core.get("triggered"))

    payload["core_slate_floor_reconciliation"] = core
    payload["core_slate_floor_triggered"] = core_triggered
    payload["core_slate_floor_reason"] = core.get("reason")
    payload["core_slate_floor_min_fixture_count"] = core.get("min_fixture_count")
    payload["core_slate_floor_primary_count"] = core.get("primary_slate_count")
    payload["core_slate_floor_reconciliation_count"] = core.get("reconciliation_slate_count")
    payload["core_slate_floor_merged_count"] = core.get("merged_slate_count")
    payload["core_slate_floor_provider_requests_added"] = core_calls

    if core_triggered:
        payload["slate_floor_triggered"] = True
        payload["slate_floor_reason"] = core.get("reason")
        payload["slate_floor_min_fixture_count"] = core.get("min_fixture_count")
        payload["api_raw_slate_count"] = core.get("primary_slate_count")
        payload["api_raw_reconciliation_slate_count"] = core.get("reconciliation_slate_count")
        payload["merged_slate_count"] = core.get("merged_slate_count")

    payload["api_slate_reconciliation_calls"] = raw_calls + core_calls
    payload["slate_source_policy"] = (
        "CORE_SCHEDULER_TODAY_FIRST; IF_SOURCE_SELECTED_FIXTURE_COUNT_BELOW_FLOOR_AND_BUDGET_ALLOWS_ADD_TOMORROW"
    )


async def run_tick() -> dict[str, Any]:
    # v7 owns the effective scheduler loop and bypasses v6.run_tick/v5.run_tick.
    # Apply V4-005 elasticity at the v6 adaptive-request symbol that v7 installs
    # into the live path, preserving v7 queue/market-safety behavior.
    original_adaptive = v6._adaptive_paced_api_get
    original_base_cap = v6._BASE_MAX_API_CALLS_PER_TICK
    original_deep_cap = v5.MAX_DEEP_DIVE_FIXTURES_PER_TICK
    elastic_state: dict[str, Any] = {
        "basis": None,
        "request_cap": 35,
        "request_reason": "QUOTA_UNKNOWN_LEGACY",
        "deep_cap": 12,
        "deep_reason": "QUOTA_UNKNOWN_SAFE",
    }

    async def elastic_adaptive_api_get(endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        payload = await original_adaptive(endpoint, params)
        remaining = v2._LAST_DAILY_REMAINING
        if remaining is None:
            try:
                raw = (payload.get("quota") or {}).get("daily_remaining")
                remaining = int(raw) if raw is not None else None
            except (TypeError, ValueError):
                remaining = None
        if remaining is not None:
            request_cap, request_reason = _elastic_request_cap(remaining)
            deep_cap, deep_reason = _elastic_deep_dive_cap(remaining, 1_000_000)
            v2.MAX_API_CALLS_PER_TICK = request_cap
            v5.MAX_DEEP_DIVE_FIXTURES_PER_TICK = deep_cap
            elastic_state.update(
                {
                    "basis": int(remaining),
                    "request_cap": request_cap,
                    "request_reason": request_reason,
                    "deep_cap": deep_cap,
                    "deep_reason": deep_reason,
                }
            )
        return payload

    # v7 resets to v6._BASE_MAX_API_CALLS_PER_TICK before its first provider call.
    # Start permissive, then immediately tighten/expand from the verified quota
    # returned by that first API-Football response.
    v6._BASE_MAX_API_CALLS_PER_TICK = 70
    v6._adaptive_paced_api_get = elastic_adaptive_api_get
    v5.MAX_DEEP_DIVE_FIXTURES_PER_TICK = 12
    try:
        payload = await v89.run_tick()
    finally:
        v6._adaptive_paced_api_get = original_adaptive
        v6._BASE_MAX_API_CALLS_PER_TICK = original_base_cap
        v5.MAX_DEEP_DIVE_FIXTURES_PER_TICK = original_deep_cap

    basis = elastic_state.get("basis")
    if basis is None:
        try:
            last_remaining = payload.get("last_daily_remaining")
            basis = int(last_remaining) if last_remaining is not None else None
        except (TypeError, ValueError):
            basis = None

    due_count = int(payload.get("due_fixture_count") or 0)
    deep_cap, deep_reason = _elastic_deep_dive_cap(basis, due_count)
    request_cap, request_reason = _elastic_request_cap(basis)

    payload["elastic_quota_remaining_basis"] = basis
    payload["elastic_deep_dive_cap"] = deep_cap
    payload["elastic_deep_dive_cap_reason"] = deep_reason
    payload["elastic_request_cap"] = request_cap
    payload["elastic_request_cap_reason"] = request_reason
    payload["max_deep_dive_fixtures_per_tick"] = deep_cap
    payload["configured_deep_dive_floor"] = original_deep_cap
    payload["max_api_calls_per_tick"] = request_cap
    payload["effective_max_api_calls_per_tick"] = request_cap
    payload["v4_005_effective_scheduler_path"] = "V7_LOOP_WITH_V6_ADAPTIVE_ELASTIC_WRAPPER"

    _apply_top_level_metrics(payload)
    payload["v3621_provider_requests_added"] = int(
        (payload.get("core_slate_floor_reconciliation") or {}).get("provider_requests_added") or 0
    )
    payload["v3621_provider_request_scope"] = "fixtures?date=tomorrow only when core scheduler selected slate is below floor"
    payload["v3621_model_weights_changed"] = False
    payload["v3621_canonical_bet_logic_changed"] = False
    payload["v3621_runtime_promotion_added"] = False
    payload["v3621_stake_or_tier_change"] = False
    payload["v3621_core_slate_floor_checkpoint"] = (
        "CORE SLATE FLOOR GUARD: A SOURCE-SELECTED TODAY SLATE BELOW FLOOR NO LONGER LETS "
        "THE SCHEDULER END WITH ONLY A TINY FIXTURE UNIVERSE; TOMORROW IS ADDED WHEN BUDGET ALLOWS."
    )
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
