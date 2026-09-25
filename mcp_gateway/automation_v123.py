# v125 runtime redeploy trigger
from __future__ import annotations

import os
from typing import Any

from mcp_gateway import automation_v121 as v121
from mcp_gateway import automation_v92 as v92
from mcp_gateway import automation_v112 as v112
from mcp_gateway import price_resolver_v4
from mcp_gateway import team_totals_intelligence

MODEL_VERSION = v121.MODEL_VERSION
AUTOMATION_VERSION = "4.31.8-team-totals-diversity-catchup"
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


TEAM_TOTALS_DIVERSITY_CATCHUP_MAX_CALLS = max(
    0,
    int(os.getenv("SOCCER_TEAM_TOTALS_DIVERSITY_CATCHUP_MAX_CALLS", "20")),
)
TEAM_TOTALS_DIVERSITY_CATCHUP_DAILY_FLOOR = max(
    0,
    int(os.getenv("SOCCER_TEAM_TOTALS_DIVERSITY_CATCHUP_DAILY_FLOOR", "5000")),
)


def _price_budget_plan(payload: dict[str, Any]) -> dict[str, Any]:
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
    catchup = 0
    if (
        configured > standard_leftover
        and TEAM_TOTALS_DIVERSITY_CATCHUP_MAX_CALLS > 0
        and mode == "NORMAL"
        and daily_remaining is not None
        and daily_remaining > TEAM_TOTALS_DIVERSITY_CATCHUP_DAILY_FLOOR
    ):
        daily_surplus = max(0, daily_remaining - TEAM_TOTALS_DIVERSITY_CATCHUP_DAILY_FLOOR)
        catchup = min(
            TEAM_TOTALS_DIVERSITY_CATCHUP_MAX_CALLS,
            configured - standard_leftover,
            daily_surplus,
        )

    total = standard_leftover + catchup
    return {
        "configured_price_cap": configured,
        "standard_leftover_budget": standard_leftover,
        "diversity_catchup_overflow_budget": catchup,
        "total_price_resolver_budget": total,
        "daily_remaining_before_price_resolver": daily_remaining,
        "daily_floor": TEAM_TOTALS_DIVERSITY_CATCHUP_DAILY_FLOOR,
        "daily_budget_mode": mode or None,
        "catchup_enabled": catchup > 0,
        "primary_tick_cap_unchanged": True,
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
        "research_spillover_primary_markets_preempted": resolution.get("research_spillover_primary_markets_preempted", False),
        "standard_leftover_budget": payload.get("price_resolver_leftover_budget", 0),
        "diversity_catchup_overflow_budget": payload.get("team_totals_diversity_catchup_overflow_budget", 0),
        "total_price_resolver_budget": payload.get("price_resolver_total_budget", 0),
        "price_resolver_budget_plan": dict(payload.get("price_resolver_budget_plan") or {}),
        "note": (
            "Price resolver uses real API-Football /odds fixture quotes or fresh Postgres market snapshots. "
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
    payload = await v121.run_tick()

    budget_plan = _price_budget_plan(payload)
    payload["price_resolver_leftover_budget"] = budget_plan["standard_leftover_budget"]
    payload["team_totals_diversity_catchup_overflow_budget"] = budget_plan["diversity_catchup_overflow_budget"]
    payload["price_resolver_total_budget"] = budget_plan["total_price_resolver_budget"]
    payload["price_resolver_budget_plan"] = budget_plan
    payload["price_resolver_budget_policy"] = (
        "PRIMARY SCHEDULER KEEPS ITS FULL EFFECTIVE TICK CAP; STANDARD PRICE BUDGET USES ONLY CAP LEFTOVER. "
        "WHILE STRICT FT TEAM TOTAL DIVERSITY IS CATCHING UP, NORMAL DAILY-BUDGET MODE WITH >5000 REQUESTS "
        "REMAINING MAY ADD UP TO 20 TEMPORARY RESEARCH-ONLY /ODDS CALLS AFTER PRIMARY WORK. "
        "PRIMARY PRICE TARGETS STILL RUN BEFORE TEAM TOTALS; OVERFLOW NEVER REDUCES THE PRIMARY CAP."
    )
    await price_resolver_v4.resolve_payload(
        payload,
        max_api_calls=budget_plan["total_price_resolver_budget"],
    )

    # Team Totals is built earlier in the automation chain, before the price
    # resolver may attach real fixture /odds markets. Rebuild this research-only
    # derivative after price enrichment. The resolver may use only budget left
    # after all primary price targets, so derivative collection never pre-empts
    # FT Totals / BTTS / 1X2 price resolution.
    payload["team_totals_post_resolution"] = team_totals_intelligence.attach(payload)

    # Price enrichment changes exact market/selection/price/fair probability on
    # research visibility rows. Recompute readiness and Phase16 against that
    # enriched point-in-time view. Both functions remain research-only.
    v92._annotate_decision_separation(payload)
    v112._annotate_phase16(payload)

    _annotate_checkpoint(payload)
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
