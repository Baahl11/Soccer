from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v128 as v128
from mcp_gateway import dynamic_strength_challenger_v4
from mcp_gateway import price_resolver_v4
from mcp_gateway import primary_clv_anchor_v4
from mcp_gateway import derivative_clv_anchor_v4
from mcp_gateway import team_totals_clv_anchor_v4

MODEL_VERSION = v128.MODEL_VERSION
AUTOMATION_VERSION = "4.38.3-team-totals-clv-anchor-repair"


async def run_tick() -> dict[str, Any]:
    original_primary_loader = price_resolver_v4._load_primary_clv_maturation_backlog
    original_one_h_loader = price_resolver_v4._load_one_h_clv_maturation_backlog
    original_two_h_loader = price_resolver_v4._load_two_h_clv_maturation_backlog
    original_corners_loader = price_resolver_v4._load_corners_clv_maturation_backlog
    original_team_totals_loader = price_resolver_v4._load_team_totals_maturation_backlog
    price_resolver_v4._load_primary_clv_maturation_backlog = (
        primary_clv_anchor_v4.load_primary_clv_maturation_backlog
    )
    price_resolver_v4._load_one_h_clv_maturation_backlog = (
        derivative_clv_anchor_v4.load_one_h_clv_maturation_backlog
    )
    price_resolver_v4._load_two_h_clv_maturation_backlog = (
        derivative_clv_anchor_v4.load_two_h_clv_maturation_backlog
    )
    price_resolver_v4._load_corners_clv_maturation_backlog = (
        derivative_clv_anchor_v4.load_corners_clv_maturation_backlog
    )
    price_resolver_v4._load_team_totals_maturation_backlog = (
        team_totals_clv_anchor_v4.load_team_totals_maturation_backlog
    )
    try:
        payload = await v128.run_tick()
    finally:
        price_resolver_v4._load_primary_clv_maturation_backlog = original_primary_loader
        price_resolver_v4._load_one_h_clv_maturation_backlog = original_one_h_loader
        price_resolver_v4._load_two_h_clv_maturation_backlog = original_two_h_loader
        price_resolver_v4._load_corners_clv_maturation_backlog = original_corners_loader
        price_resolver_v4._load_team_totals_maturation_backlog = original_team_totals_loader

    events = payload.get("events")
    if not isinstance(events, list):
        events = []

    payload["v212_dynamic_strength_challenger"] = dynamic_strength_challenger_v4.build_report(events)
    payload["v212_checkpoint"] = (
        "DYNAMIC STRENGTH CHALLENGER ACTIVE IN RESEARCH ONLY: the shadow compares the existing "
        "blended raw projection with a recent-only counterfactual built from the same verified "
        "sporting snapshot and the same goal-rate Poisson math. At least three recent matches per "
        "team are required. It adds no provider calls, carries decision_weight=0, and cannot alter "
        "thresholds, gates, strict-close semantics, canonical BET logic, or production promotion."
    )
    payload["v213_primary_clv_anchor_repair"] = {
        "schema_version": "1.0.0",
        "status": "ACTIVE_OLDEST_UNRESOLVED_POINT_IN_TIME_SIGNAL",
        "signal_anchor_policy": primary_clv_anchor_v4.ANCHOR_POLICY,
        "scope": ["1X2", "BTTS", "FT_TOTALS"],
        "strict_close_semantics_changed": False,
        "requires_captured_at_after_signal": True,
        "requires_provider_update_after_signal": True,
        "historical_rows_mutated": False,
        "historical_probabilities_recomputed": False,
        "retroactive_signal_created": False,
        "provider_budget_changed": False,
        "provider_requests_added_by_anchor_loader": 0,
        "decision_weight": 0.0,
        "production_promotion_allowed": False,
        "model_weights_changed": False,
        "thresholds_changed": False,
        "gates_changed": False,
        "canonical_bet_logic_changed": False,
        "frontend_changed": False,
        "policy": (
            "SELECT_OLDEST_UNRESOLVED_POINT_IN_TIME_SIGNAL_PER_FIXTURE_FAMILY; "
            "DO_NOT_ADVANCE_ANCHOR_ON_RECYCLED_PIPELINE_TICKS; "
            "STRICTLY_LATER_REAL_PROVIDER_QUOTE_STILL_REQUIRED"
        ),
    }
    payload["v213_checkpoint"] = (
        "PRIMARY TRUE-CLV MATURATION ANCHOR REPAIRED: use the oldest still-unresolved point-in-time "
        "signal for 1X2/BTTS/FT Totals instead of the newest recycled pipeline timestamp. A TRUE_CLV "
        "still requires a real pre-kickoff snapshot with captured_at and provider_update strictly "
        "later than that authentic signal. No history rewrite, synthetic close, budget increase, "
        "threshold/gate/model change, promotion, or frontend change."
    )
    payload["v215_6_derivative_clv_anchor_repair"] = {
        "schema_version": "1.0.0",
        "status": "ACTIVE_OLDEST_UNRESOLVED_EXACT_DERIVATIVE_SIGNAL",
        "signal_anchor_policy": derivative_clv_anchor_v4.ANCHOR_POLICY,
        "scope": ["1H", "2H", "FT_CORNERS", "TEAM_CORNERS"],
        "strict_close_semantics_changed": False,
        "requires_captured_at_after_signal": True,
        "requires_provider_update_after_signal": True,
        "requires_same_selection_and_line": True,
        "historical_rows_mutated": False,
        "historical_probabilities_recomputed": False,
        "retroactive_signal_created": False,
        "provider_budget_changed": False,
        "provider_requests_added_by_anchor_loader": 0,
        "decision_weight": 0.0,
        "production_promotion_allowed": False,
        "model_weights_changed": False,
        "thresholds_changed": False,
        "gates_changed": False,
        "canonical_bet_logic_changed": False,
        "team_totals_anchor_pending_separate_block": True,
        "policy": (
            "SELECT_OLDEST_UNRESOLVED_POINT_IN_TIME_SIGNAL_PER_EXACT_DERIVATIVE; "
            "DO_NOT_ADVANCE_ANCHOR_ON_RECYCLED_SCHEDULER_TICKS; "
            "STRICTLY_LATER_REAL_PROVIDER_QUOTE_AND_EXACT_SELECTION_LINE_STILL_REQUIRED"
        ),
    }
    payload["v215_6_checkpoint"] = (
        "DERIVATIVE TRUE-CLV ANCHOR REPAIRED FOR 1H/2H/CORNERS: repeated scheduler ticks no "
        "longer move the unresolved signal anchor forward. Strict captured_at/provider_update and "
        "same-selection/same-line requirements remain unchanged; no provider budget increase or "
        "history rewrite. Team Totals anchor remains a separate follow-up block."
    )
    payload["v215_7_team_totals_clv_anchor_repair"] = {
        "schema_version": "1.0.0",
        "status": "ACTIVE_OLDEST_UNRESOLVED_EXACT_TEAM_TOTAL_SIGNAL",
        "signal_anchor_policy": team_totals_clv_anchor_v4.ANCHOR_POLICY,
        "lookback_hours": team_totals_clv_anchor_v4.DEFAULT_LOOKBACK_HOURS,
        "strict_close_semantics_changed": False,
        "requires_captured_at_after_signal": True,
        "requires_provider_update_after_signal": True,
        "requires_same_selection_and_line": True,
        "historical_rows_mutated": False,
        "historical_probabilities_recomputed": False,
        "provider_budget_changed": False,
        "team_totals_maturation_max_calls_per_tick": price_resolver_v4.TEAM_TOTALS_MATURATION_MAX_CALLS_PER_TICK,
        "decision_weight": 0.0,
        "production_promotion_allowed": False,
        "model_weights_changed": False,
        "thresholds_changed": False,
        "gates_changed": False,
        "canonical_bet_logic_changed": False,
        "policy": (
            "UPCOMING_T55_FIXTURES_FIRST; BOUNDED_48H_OR_DIVERSITY_HORIZON_PLUS_MARGIN; "
            "OLDEST_UNRESOLVED_EXACT_MARKET_SIDE_LINE_SIGNAL; STRICTLY_LATER_REAL_PROVIDER_QUOTE; "
            "SAME_EXISTING_LEFTOVER_BUDGET_MAX12; NO_HISTORY_REWRITE"
        ),
    }
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
