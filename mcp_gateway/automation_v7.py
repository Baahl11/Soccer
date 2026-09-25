from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone as dt_timezone
from typing import Any

import httpx

from mcp_gateway import automation as base
from mcp_gateway import automation_v2 as v2
from mcp_gateway import automation_v3 as v3
from mcp_gateway import automation_v4 as v4
from mcp_gateway import automation_v5 as v5
from mcp_gateway import automation_v6 as v6
from mcp_gateway import fair_scheduler

MODEL_VERSION = "SOCCER EDGE ENGINE v1.0"
AUTOMATION_VERSION = "1.7.0"

# Once a fixture has passed the sporting screen, protect its late verification
# windows before spending the scarce tick budget discovering new candidates.
# This preserves T-40 as the main discovery window without allowing a new T-40
# batch to starve an existing shortlist at T-30/T-20/T-10/CLOSE.
LATE_SHORTLIST_STAGE_PRIORITY = {
    "T-10": 0,
    "T-20": 1,
    "CLOSE": 2,
    "T-30": 3,
}


def _upcoming_market_capture_fixtures(
    fixtures: list[dict[str, Any]],
    now_utc: datetime,
) -> list[dict[str, Any]]:
    return sorted(
        [
            fx
            for fx in fixtures
            if fx.get("kickoff")
            and base._dt(fx["kickoff"]) > now_utc
            and fx.get("status") not in base.CANCELLED_STATUSES | base.POSTPONED_STATUSES
        ],
        key=lambda fx: base._dt(fx["kickoff"]),
    )[:v5.MAX_UPCOMING_MARKET_CAPTURE_FIXTURES]


def _queue_priority(
    fx: dict[str, Any],
    stage: str,
    coverage: dict[str, Any],
    tier: str,
    prior: dict[str, Any] | None,
    prior_rank: float,
    proximity: float,
    kickoff: datetime,
) -> tuple[Any, ...]:
    prior_shortlisted = bool(prior and prior.get("shortlisted"))
    late_existing = prior_shortlisted and stage in LATE_SHORTLIST_STAGE_PRIORITY

    # Bucket 0: already-shortlisted fixtures at a late verification/closing gate.
    # Bucket 1: the main T-40 discovery window.
    # Bucket 2: other already-shortlisted lifecycle refreshes.
    # Bucket 3: remaining eligible discovery/maintenance work.
    if late_existing:
        lifecycle_bucket = 0
        lifecycle_order = LATE_SHORTLIST_STAGE_PRIORITY[stage]
    elif stage == "T-40":
        lifecycle_bucket = 1
        lifecycle_order = 0
    elif prior_shortlisted:
        lifecycle_bucket = 2
        lifecycle_order = v5.STAGE_PRIORITY.get(stage, 99)
    else:
        lifecycle_bucket = 3
        lifecycle_order = v5.STAGE_PRIORITY.get(stage, 99)

    return (
        0 if tier in {"A", "B"} else 1,
        lifecycle_bucket,
        lifecycle_order,
        0 if tier == "A" else 1 if tier == "B" else 2 if tier == "C" else 3,
        v5._competition_priority(fx),
        -v5._coverage_strength(coverage),
        -prior_rank,
        kickoff,  # earlier kickoff wins after quality/shortlist criteria
        proximity,
    )


async def run_tick() -> dict[str, Any]:
    v2._API_CALLS_THIS_TICK = 0
    v2._LAST_DAILY_REMAINING = None
    v2.MAX_API_CALLS_PER_TICK = v6._BASE_MAX_API_CALLS_PER_TICK
    v6._BUDGET_MODE = "NORMAL"

    previous_paced_symbol = v4._paced_api_get
    previous_market_symbol = v4._safe_evaluate_market
    original_coverage = base._coverage

    # Keep v1.6 quota pacing / market-safety protections while replacing only
    # the due-fixture queue ordering.
    v4._paced_api_get = v6._adaptive_paced_api_get
    v4._safe_evaluate_market = v6._safe_market_evaluate
    base._api_get = v4._paced_api_get
    v2.evaluate_market = v4._safe_evaluate_market
    base._coverage = v3._coverage_fast

    now_utc = datetime.now(dt_timezone.utc)
    local_now = now_utc.astimezone(base.TIMEZONE)
    base._prune_cache(now_utc)

    dates = [local_now.date()]
    if local_now.hour >= 22:
        from datetime import timedelta
        dates.append((local_now + timedelta(days=1)).date())

    fixtures: list[dict[str, Any]] = []
    quota: dict[str, Any] = {}

    base._HTTP_CLIENT = httpx.AsyncClient(
        timeout=base.TIMEOUT,
        limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
    )

    try:
        for d in dates:
            payload = await base._api_get(
                "fixtures", {"date": d.isoformat(), "timezone": base.TIMEZONE_NAME}
            )
            quota = payload.get("quota", quota)
            for row in payload.get("response", []):
                fx = base._compact_fixture(row)
                if (
                    fx.get("fixture_id")
                    and fx.get("kickoff")
                    and fx.get("league_id")
                    and fx.get("season")
                ):
                    fixtures.append(fx)

        upcoming_market_capture_fixtures = _upcoming_market_capture_fixtures(fixtures, now_utc)

        events: list[dict[str, Any]] = []
        discovery = await v3._daily_discovery_event(fixtures, now_utc, local_now)
        if discovery is not None:
            events.append(discovery)

        due: list[dict[str, Any]] = []
        low_data_counts: Counter[str] = Counter()
        urgent_late_shortlist_due = 0

        for fx in fixtures:
            kickoff = base._dt(fx["kickoff"])
            minutes_to = (kickoff - now_utc).total_seconds() / 60.0
            stage = base._stage_for(minutes_to, fx.get("status") or "")
            if not stage:
                continue

            prior = v5._shortlist_get(fx["fixture_id"], now_utc)
            if stage == "POSTGAME":
                minutes_since = -minutes_to
                if minutes_since < 95 or minutes_since > 240:
                    continue
                if not prior or not prior.get("shortlisted"):
                    continue

            coverage = await v3._coverage_fast(fx["league_id"], fx["season"], now_utc)
            tier = coverage.get("data_tier") or "D"
            prior_rank = v5._num((prior or {}).get("rank")) or 0.0
            target = v5.STAGE_TARGET.get(stage)
            proximity = abs(minutes_to - target) if target is not None else 0.0

            if prior and prior.get("shortlisted") and stage in LATE_SHORTLIST_STAGE_PRIORITY:
                urgent_late_shortlist_due += 1

            due.append(
                {
                    "fx": fx,
                    "stage": stage,
                    "coverage": coverage,
                    "tier": tier,
                    "prior_shortlisted": bool(prior and prior.get("shortlisted")),
                    "priority": _queue_priority(
                        fx,
                        stage,
                        coverage,
                        tier,
                        prior,
                        prior_rank,
                        proximity,
                        kickoff,
                    ),
                }
            )

        due.sort(key=lambda item: item["priority"])

        pass_through_due: list[dict[str, Any]] = []
        fair_eligible_due: list[dict[str, Any]] = []
        for item in due:
            fx = item["fx"]
            stage = item["stage"]
            already_processed = v5._stage_already_processed(fx["fixture_id"], stage, now_utc)
            item["stage_already_processed"] = already_processed
            if already_processed or item["tier"] not in {"A", "B"}:
                pass_through_due.append(item)
                continue
            state = fair_scheduler.get_state(fx["fixture_id"], now_utc)
            item["fairness_state"] = state
            item["fairness_category"] = fair_scheduler.category(
                bool(item.get("prior_shortlisted")),
                state,
            )
            fair_eligible_due.append(item)

        fair_selected, fair_deferred, fair_plan = fair_scheduler.fair_order(
            fair_eligible_due,
            v5.MAX_DEEP_DIVE_FIXTURES_PER_TICK,
        )
        due_execution_order = pass_through_due + fair_selected + fair_deferred

        deep_dive_processed = 0
        deferred_due_to_priority = 0
        deferred_due_to_budget = 0
        market_requests_avoided_by_screen = 0
        shortlist_events = 0
        urgent_late_shortlist_processed = 0
        fair_processed_counts: Counter[str] = Counter()

        for item in due_execution_order:
            fx = item["fx"]
            stage = item["stage"]
            coverage = item["coverage"]
            tier = item["tier"]

            if item.get("stage_already_processed"):
                continue

            if tier not in {"A", "B"}:
                low_data_counts[f"{tier}:{stage}"] += 1
                v5._mark_stage_processed(fx["fixture_id"], stage, now_utc)
                continue

            if deep_dive_processed >= v5.MAX_DEEP_DIVE_FIXTURES_PER_TICK:
                deferred_due_to_priority += 1
                continue

            prior = v5._shortlist_get(fx["fixture_id"], now_utc)
            is_urgent_existing = bool(
                prior and prior.get("shortlisted") and stage in LATE_SHORTLIST_STAGE_PRIORITY
            )

            try:
                event = await v5._priority_event(fx, stage, coverage, now_utc)
            except v2.TickBudgetExceeded as exc:
                deferred_due_to_budget += 1
                events.append(
                    {
                        "event_type": "QUOTA_GUARD",
                        "stage": stage,
                        "fixture": fx,
                        "model_version": MODEL_VERSION,
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
                        "model_version": MODEL_VERSION,
                        "classification": "WATCH",
                        "error": str(exc)[:500],
                    }
                )
                continue

            deep_dive_processed += 1
            fairness_category = str(item.get("fairness_category") or "exploratory")
            fair_processed_counts[fairness_category] += 1
            fair_scheduler.record_deep_dive(fx["fixture_id"], stage, now_utc)
            if is_urgent_existing:
                urgent_late_shortlist_processed += 1
            v5._mark_stage_processed(fx["fixture_id"], stage, now_utc)
            if event.get("market_skipped_by_sport_screen"):
                market_requests_avoided_by_screen += 1
            if (event.get("sporting_shortlist") or {}).get("shortlisted"):
                shortlist_events += 1
            events.append(event)

        if low_data_counts:
            events.append(
                {
                    "event_type": "LOW_DATA_SCREEN_SUMMARY",
                    "stage": "SLATE_SCREEN",
                    "classification": "PASS",
                    "model_version": MODEL_VERSION,
                    "screened_out_count": sum(low_data_counts.values()),
                    "by_tier_stage": dict(low_data_counts),
                    "notes": [
                        "Data Tier C/D fixtures remain counted in the slate but do not consume deep-dive API budget.",
                        "Detailed market/lineup refresh is reserved for Data Tier A/B sporting candidates.",
                    ],
                }
            )

        actionable = [
            e for e in events if e.get("stage") in {"T-40", "T-20", "T-10", "CLOSE"}
        ]
        bets = [e for e in events if e.get("classification") == "BET"]
        shortlist_state = v6.export_shortlist_state()
        scheduler_fairness_state = fair_scheduler.export_state()
        fair_metrics = {
            **fair_plan,
            **fair_scheduler.coverage_metrics(
                due,
                dict(fair_processed_counts),
                v2._API_CALLS_THIS_TICK,
                deep_dive_processed,
                now_utc,
            ),
            "processed_category_counts": {
                key: int(fair_processed_counts.get(key, 0))
                for key in fair_scheduler.CATEGORY_WEIGHTS
            },
            "state_count": len(scheduler_fairness_state),
            "provider_requests_added": 0,
            "model_weights_changed": False,
            "canonical_bet_logic_changed": False,
        }

        return {
            "service": "soccer-edge-automation",
            "version": AUTOMATION_VERSION,
            "model_version": MODEL_VERSION,
            "generated_at_utc": now_utc.isoformat(),
            "generated_at_local": local_now.isoformat(),
            "timezone": base.TIMEZONE_NAME,
            "fixture_scan_count": len(fixtures),
            "upcoming_market_capture_fixture_count": len(upcoming_market_capture_fixtures),
            "upcoming_market_capture_fixtures": upcoming_market_capture_fixtures,
            "due_fixture_count": len(due),
            "event_count": len(events),
            "actionable_refresh_count": len(actionable),
            "bet_candidate_count": len(bets),
            "api_calls_this_tick": v2._API_CALLS_THIS_TICK,
            "max_api_calls_per_tick": v2.MAX_API_CALLS_PER_TICK,
            "effective_max_api_calls_per_tick": v2.MAX_API_CALLS_PER_TICK,
            "last_daily_remaining": v2._LAST_DAILY_REMAINING,
            "daily_budget_mode": v6._BUDGET_MODE,
            "daily_budget_policy": {
                "normal_above": 4000,
                "reduced_at_or_below": 4000,
                "priority_only_at_or_below": 2500,
                "emergency_at_or_below": 1500,
                "reserve_at_or_below": 500,
            },
            "deferred_due_to_budget": deferred_due_to_budget,
            "deferred_due_to_priority": deferred_due_to_priority,
            "deep_dive_processed_count": deep_dive_processed,
            "shortlist_event_count": shortlist_events,
            "screened_out_low_data_count": sum(low_data_counts.values()),
            "market_requests_avoided_by_sport_screen": market_requests_avoided_by_screen,
            "max_deep_dive_fixtures_per_tick": v5.MAX_DEEP_DIVE_FIXTURES_PER_TICK,
            "priority_queue": "WEIGHTED_FAIR_60_ACTIONABLE_25_UNSEEN_15_EXPLORATORY_WITH_LIFECYCLE_URGENCY",
            "fair_scheduler": fair_metrics,
            "urgent_late_shortlist_due": urgent_late_shortlist_due,
            "urgent_late_shortlist_processed": urgent_late_shortlist_processed,
            "request_pacing_seconds": v4.MIN_REQUEST_INTERVAL_SECONDS,
            "rate_limit_max_retries": v4.RATE_LIMIT_MAX_RETRIES,
            "first_half_market_model": "BLOCKED_PENDING_EXPLICIT_1H_MODEL",
            "period_market_model": "BLOCKED_PENDING_EXPLICIT_PERIOD_MODELS",
            "automated_market_scope": "CANONICAL_FT_1X2_TOTAL_BTTS_ONLY",
            "shortlist_persistence": "GITHUB_STATE_SEEDED",
            "shortlist_state_count": len(shortlist_state),
            "shortlist_state": shortlist_state,
            "scheduler_fairness_state": scheduler_fairness_state,
            "quota": quota,
            "events": events,
            "database_persistence": "OPTIONAL_NOT_REQUIRED_FOR_SCHEDULER",
        }
    finally:
        base._coverage = original_coverage
        v4._paced_api_get = previous_paced_symbol
        v4._safe_evaluate_market = previous_market_symbol
        if base._HTTP_CLIENT is not None:
            await base._HTTP_CLIENT.aclose()
            base._HTTP_CLIENT = None
        if base._CACHE_CONN is not None:
            base._CACHE_CONN.commit()
            base._CACHE_CONN.close()
            base._CACHE_CONN = None
