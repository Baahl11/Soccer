# v125 runtime redeploy trigger
from __future__ import annotations

import os
from typing import Any

from mcp_gateway import automation_v2 as v2
from mcp_gateway import automation_v6 as v6
from mcp_gateway import automation_v90 as v90
from mcp_gateway import automation_v121 as v121
from mcp_gateway import automation_v92 as v92
from mcp_gateway import automation_v112 as v112
from mcp_gateway import price_resolver_v4
from mcp_gateway import team_totals_intelligence

MODEL_VERSION = v121.MODEL_VERSION
AUTOMATION_VERSION = "4.32.6-primary-clv-maturation"
PRIMARY_PRICE_RESERVE_CALLS = max(
    0,
    int(os.getenv("SOCCER_PRIMARY_PRICE_RESERVE_CALLS", "20")),
)
MIN_UPSTREAM_API_CALLS = max(
    1,
    int(os.getenv("SOCCER_MIN_UPSTREAM_API_CALLS", "8")),
)


def _reserve_from_elastic_cap(global_cap: int) -> tuple[int, int]:
    cap = max(1, int(global_cap))
    reserve = min(PRIMARY_PRICE_RESERVE_CALLS, max(0, cap - MIN_UPSTREAM_API_CALLS))
    return max(1, cap - reserve), reserve
# Deployment marker: v128 guarded diversity catch-up overflow.
# Deployment marker: v126 active-v7 upcoming fixture handoff.
# Deployment marker: v125 scanned-upcoming FT Team Totals capture.


def _leftover_price_budget(payload: dict[str, Any]) -> int:
    configured = int(price_resolver_v4.DEFAULT_MAX_API_CALLS)
    try:
        used = int(payload.get("api_calls_this_tick") or 0)
    except (TypeError, ValueError):
        used = 0

    cap_value = payload.get("effective_max_api_calls_per_tick")
    if cap_value is None:
        cap_value = payload.get("max_api_calls_per_tick")
    try:
        cap = int(cap_value)
    except (TypeError, ValueError):
        return configured

    return max(0, min(configured, cap - used))


def _price_budget_plan(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a strict leftover-only price budget.

    Any capacity used here must fit inside the same global per-tick cap. There is
    no Team Totals overflow above the cap, even when daily quota is abundant.
    """
    configured = max(0, int(price_resolver_v4.DEFAULT_MAX_API_CALLS))
    standard_leftover = _leftover_price_budget(payload)

    try:
        daily_remaining = int(payload.get("last_daily_remaining"))
    except (TypeError, ValueError):
        quota = payload.get("quota") if isinstance(payload.get("quota"), dict) else {}
        try:
            daily_remaining = int(quota.get("daily_remaining"))
        except (TypeError, ValueError):
            daily_remaining = None

    mode = str(payload.get("daily_budget_mode") or "").upper()
    return {
        "configured_price_cap": configured,
        "standard_leftover_budget": standard_leftover,
        "diversity_catchup_overflow_budget": 0,
        "total_price_resolver_budget": standard_leftover,
        "daily_remaining_before_price_resolver": daily_remaining,
        "daily_budget_mode": mode or None,
        "catchup_enabled": False,
        "overflow_above_global_tick_cap_allowed": False,
        "primary_tick_cap_unchanged": False,
        "elastic_global_cap_used": True,
    }


def _annotate_checkpoint(payload: dict[str, Any]) -> None:
    resolution = payload.get("price_resolution_v4") if isinstance(payload.get("price_resolution_v4"), dict) else {}
    payload["price_resolution_checkpoint"] = {
        "schema_version": "1.0.0",
        "status": resolution.get("status") or "NOT_RUN",
        "model_version": price_resolver_v4.MODEL_VERSION,
        "candidate_rows": int(resolution.get("candidate_rows") or 0),
        "unique_candidate_fixtures": int(resolution.get("unique_candidate_fixtures") or 0),
        "api_calls_added": int(resolution.get("api_calls_added") or 0),
        "resolution_counts": dict(resolution.get("resolution_counts") or {}),
        "simulated_odds_allowed": False,
        "calibrated_probability_fabricated": False,
        "phase16_recomputed_after_price_resolution": True,
        "team_totals_recomputed_after_price_resolution": True,
        "team_totals_post_resolution": dict(payload.get("team_totals_post_resolution") or {}),
        "canonical_bet_logic_changed": False,
        "model_weights_changed": False,
        "production_promotion_allowed": False,
        "primary_clv_maturation_source": resolution.get("primary_clv_maturation_source"),
        "primary_clv_maturation_candidates": resolution.get("primary_clv_maturation_candidates", 0),
        "primary_clv_maturation_candidate_family_counts": dict(resolution.get("primary_clv_maturation_candidate_family_counts") or {}),
        "primary_clv_maturation_max_calls_per_tick": resolution.get("primary_clv_maturation_max_calls_per_tick", 0),
        "primary_clv_maturation_api_calls_added": resolution.get("primary_clv_maturation_api_calls_added", 0),
        "primary_clv_maturation_fixtures_refreshed": resolution.get("primary_clv_maturation_fixtures_refreshed", 0),
        "primary_clv_maturation_family_refresh_counts": dict(resolution.get("primary_clv_maturation_family_refresh_counts") or {}),
        "primary_clv_maturation_cache_replays_ignored": resolution.get("primary_clv_maturation_cache_replays_ignored", 0),
        "primary_clv_maturation_unchanged_provider_updates": resolution.get("primary_clv_maturation_unchanged_provider_updates", 0),
        "primary_clv_maturation_budget_exhausted": resolution.get("primary_clv_maturation_budget_exhausted", 0),
        "primary_clv_maturation_primary_payload_reuse_fixtures": resolution.get("primary_clv_maturation_primary_payload_reuse_fixtures", 0),
        "primary_clv_maturation_synthetic_events_added": resolution.get("primary_clv_maturation_synthetic_events_added", 0),
        "primary_clv_maturation_policy": resolution.get("primary_clv_maturation_policy"),
        "research_spillover_candidate_fixtures": resolution.get("research_spillover_candidate_fixtures", 0),
        "research_spillover_cache_hits": resolution.get("research_spillover_cache_hits", 0),
        "research_spillover_api_calls_added": resolution.get("research_spillover_api_calls_added", 0),
        "research_spillover_fixtures_fetched": resolution.get("research_spillover_fixtures_fetched", 0),
        "research_spillover_market_rows_fetched": resolution.get("research_spillover_market_rows_fetched", 0),
        "research_spillover_unique_fixture_target": resolution.get("research_spillover_unique_fixture_target", 20),
        "research_spillover_existing_unique_fixtures": resolution.get("research_spillover_existing_unique_fixtures", 0),
        "research_spillover_legacy_observed_unique_fixtures": resolution.get("research_spillover_legacy_observed_unique_fixtures", 0),
        "research_spillover_diversity_counter_semantics": resolution.get("research_spillover_diversity_counter_semantics"),
        "research_spillover_phase19_true_clv_gate_separate": resolution.get("research_spillover_phase19_true_clv_gate_separate", True),
        "research_spillover_new_unique_fixtures_this_tick": resolution.get("research_spillover_new_unique_fixtures_this_tick", 0),
        "research_spillover_projected_unique_fixtures": resolution.get("research_spillover_projected_unique_fixtures", 0),
        "research_spillover_diversity_gap_remaining": resolution.get("research_spillover_diversity_gap_remaining", 20),
        "research_spillover_persisted_backlog_candidates": resolution.get("research_spillover_persisted_backlog_candidates", 0),
        "research_spillover_scanned_upcoming_candidates": resolution.get("research_spillover_scanned_upcoming_candidates", 0),
        "research_spillover_market_capture_only_candidates": resolution.get("research_spillover_market_capture_only_candidates", 0),
        "research_spillover_exact_team_total_fixtures_attached": resolution.get("research_spillover_exact_team_total_fixtures_attached", 0),
        "research_spillover_ft_team_total_market_rows_attached": resolution.get("research_spillover_ft_team_total_market_rows_attached", 0),
        "research_spillover_primary_payload_reuse_fixtures": resolution.get("research_spillover_primary_payload_reuse_fixtures", 0),
        "research_spillover_primary_payload_reuse_market_rows": resolution.get("research_spillover_primary_payload_reuse_market_rows", 0),
        "research_spillover_maturation_source": resolution.get("research_spillover_maturation_source"),
        "research_spillover_maturation_candidates": resolution.get("research_spillover_maturation_candidates", 0),
        "research_spillover_maturation_max_calls_per_tick": resolution.get("research_spillover_maturation_max_calls_per_tick", 0),
        "research_spillover_maturation_api_calls_added": resolution.get("research_spillover_maturation_api_calls_added", 0),
        "research_spillover_maturation_later_real_quote_refreshes": resolution.get("research_spillover_maturation_later_real_quote_refreshes", 0),
        "research_spillover_maturation_cache_replays_ignored": resolution.get("research_spillover_maturation_cache_replays_ignored", 0),
        "research_spillover_maturation_unchanged_provider_updates": resolution.get("research_spillover_maturation_unchanged_provider_updates", 0),
        "research_spillover_maturation_budget_exhausted": resolution.get("research_spillover_maturation_budget_exhausted", 0),
        "research_spillover_clv_maturation_continues_after_diversity_target": resolution.get("research_spillover_clv_maturation_continues_after_diversity_target", False),
        "research_spillover_primary_markets_preempted": resolution.get("research_spillover_primary_markets_preempted", False),
        "standard_leftover_budget": payload.get("price_resolver_leftover_budget", 0),
        "diversity_catchup_overflow_budget": payload.get("team_totals_diversity_catchup_overflow_budget", 0),
        "total_price_resolver_budget": payload.get("price_resolver_total_budget", 0),
        "price_resolver_budget_plan": dict(payload.get("price_resolver_budget_plan") or {}),
        "pre_price_pipeline_api_cap": payload.get("pre_price_pipeline_api_cap"),
        "global_api_cap_after_daily_policy": payload.get("global_api_cap_after_daily_policy"),
        "primary_price_reserve_calls": payload.get("primary_price_reserve_calls"),
        "elastic_request_cap_upstream_observed": payload.get("elastic_request_cap_upstream_observed"),
        "overflow_above_global_tick_cap_allowed": False,
        "note": (
            "The upstream sporting/deep-dive chain is temporarily capped below the SAME global tick ceiling so "
            "primary FT Totals/BTTS/1X2 price targets retain reserved capacity on busy slates; no overflow above "
            "the global cap is allowed. Price resolver uses real API-Football /odds fixture quotes or fresh Postgres market snapshots. "
            "Resolved rows are re-evaluated by Team Totals research intelligence, execution-status separation "
            "and Phase16. Team Totals first reuses exact FT Team Totals already present in paid primary /odds "
            "payloads at zero extra provider cost; cache comes next, and only then may it use provider budget left after every primary "
            "price target; those spillover calls remain research-only and cannot pre-empt FT Totals/BTTS/1X2. "
            "Uncovered fixtures with persisted pre-kickoff team lambdas are prioritized first; then already-scanned "
            "upcoming fixtures may receive market-capture-only /odds hydration until 20 explicit strict FT Team Totals "
            "capture fixtures are collected. A market-only capture is not Phase19 directional evidence. Legacy observed "
            "rows are diagnostic only and cannot "
            "satisfy this gate; Phase19 true-CLV uniqueness remains a separate downstream gate that needs a later "
            "pre-kickoff close. No calibrated probability is fabricated and no BET/tier/stake/model threshold is changed."
        ),
    }


async def run_tick() -> dict[str, Any]:
    # v90 owns the live elastic request ceiling (70/55/45/35/25 from verified
    # quota). Reserve capacity at THAT layer, not at v6's nominal base cap,
    # otherwise v90 simply expands the upstream loop again and consumes the
    # intended pricing reserve.
    original_elastic_request_cap = v90._elastic_request_cap

    def _reserved_elastic_request_cap(remaining: Any) -> tuple[int, str]:
        global_cap, reason = original_elastic_request_cap(remaining)
        upstream_cap, reserve = _reserve_from_elastic_cap(int(global_cap))
        return upstream_cap, f"{reason}_PRICE_RESERVE_{reserve}"

    v90._elastic_request_cap = _reserved_elastic_request_cap
    try:
        payload = await v121.run_tick()
    finally:
        v90._elastic_request_cap = original_elastic_request_cap

    remaining_raw = payload.get("last_daily_remaining")
    try:
        remaining = int(remaining_raw) if remaining_raw is not None else None
    except (TypeError, ValueError):
        remaining = None

    global_cap, global_reason = original_elastic_request_cap(remaining)
    global_cap = int(global_cap)
    pre_price_cap, reserve_requested = _reserve_from_elastic_cap(global_cap)
    observed_upstream_cap = payload.get("elastic_request_cap")
    try:
        observed_upstream_cap = int(observed_upstream_cap)
    except (TypeError, ValueError):
        observed_upstream_cap = pre_price_cap

    # The actual upstream wrapper should have used the reserved cap. Keep the
    # observed value visible, but restore top-level max/effective to the true
    # global elastic ceiling before computing post-primary leftover.
    v2.MAX_API_CALLS_PER_TICK = global_cap
    payload["pre_price_pipeline_api_cap"] = observed_upstream_cap
    payload["global_api_cap_after_daily_policy"] = global_cap
    payload["primary_price_reserve_calls"] = max(0, global_cap - observed_upstream_cap)
    payload["elastic_request_cap_upstream_observed"] = observed_upstream_cap
    payload["elastic_request_cap"] = global_cap
    payload["elastic_request_cap_reason"] = global_reason
    payload["max_api_calls_per_tick"] = global_cap
    payload["effective_max_api_calls_per_tick"] = global_cap

    budget_plan = _price_budget_plan(payload)
    payload["price_resolver_leftover_budget"] = budget_plan["standard_leftover_budget"]
    payload["team_totals_diversity_catchup_overflow_budget"] = 0
    payload["price_resolver_total_budget"] = budget_plan["total_price_resolver_budget"]
    payload["price_resolver_budget_plan"] = budget_plan
    payload["price_resolver_budget_policy"] = (
        "SAME_GLOBAL_TICK_CAP_ONLY; UPSTREAM_SPORT_DEEP_DIVE_TEMP_CAP_RESERVES_CAPACITY; "
        "FT_TOTALS_BTTS_1X2_PRICE_TARGETS_RESOLVE_FIRST; TEAM_TOTALS_USES_ONLY_POST_PRIMARY_LEFTOVER; "
        "NO_DIVERSITY_OVERFLOW_ABOVE_GLOBAL_CAP"
    )
    await price_resolver_v4.resolve_payload(
        payload,
        max_api_calls=budget_plan["total_price_resolver_budget"],
    )

    payload["team_totals_post_resolution"] = team_totals_intelligence.attach(payload)

    v92._annotate_decision_separation(payload)
    v112._annotate_phase16(payload)

    _annotate_checkpoint(payload)
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
