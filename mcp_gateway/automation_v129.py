from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v128 as v128
from mcp_gateway import dynamic_strength_challenger_v4
from mcp_gateway import price_resolver_v4
from mcp_gateway import primary_clv_anchor_v4
from mcp_gateway import derivative_clv_anchor_v4
from mcp_gateway import team_totals_clv_anchor_v4
from mcp_gateway import team_totals_close_provenance_v4

MODEL_VERSION = v128.MODEL_VERSION
AUTOMATION_VERSION = "4.38.5-team-totals-close-provenance-audit"


async def run_tick() -> dict[str, Any]:
    original_primary_loader = price_resolver_v4._load_primary_clv_maturation_backlog
    original_one_h_loader = price_resolver_v4._load_one_h_clv_maturation_backlog
    original_two_h_loader = price_resolver_v4._load_two_h_clv_maturation_backlog
    original_corners_loader = price_resolver_v4._load_corners_clv_maturation_backlog
    original_team_totals_loader = price_resolver_v4._load_team_totals_maturation_backlog
    primary_loader_observation: dict[str, Any] = {}
    team_totals_candidate_events: list[dict[str, Any]] = []

    def observed_primary_loader(*args: Any, **kwargs: Any) -> dict[str, Any]:
        report = primary_clv_anchor_v4.load_primary_clv_maturation_backlog(*args, **kwargs)
        primary_loader_observation.clear()
        primary_loader_observation.update(
            {
                "source": report.get("source"),
                "signal_anchor_policy": report.get("signal_anchor_policy"),
                "candidate_count": int(report.get("candidate_count") or 0),
                "candidate_family_counts": dict(report.get("candidate_family_counts") or {}),
                "candidate_source_counts": dict(report.get("candidate_source_counts") or {}),
                "diagnostic_schema_version": report.get("diagnostic_schema_version"),
                "diagnostic_family_counts": dict(report.get("diagnostic_family_counts") or {}),
                "diagnostic_window": dict(report.get("diagnostic_window") or {}),
                "provider_requests_added": int(report.get("provider_requests_added") or 0),
                "selection_logic_changed": bool(report.get("selection_logic_changed", False)),
            }
        )
        return report

    def observed_team_totals_loader(*args: Any, **kwargs: Any) -> dict[str, Any]:
        report = team_totals_clv_anchor_v4.load_team_totals_maturation_backlog(*args, **kwargs)
        team_totals_candidate_events.clear()
        team_totals_candidate_events.extend(
            event for event in (report.get("candidate_events") or []) if isinstance(event, dict)
        )
        return report

    price_resolver_v4._load_primary_clv_maturation_backlog = observed_primary_loader
    price_resolver_v4._load_one_h_clv_maturation_backlog = (
        derivative_clv_anchor_v4.load_one_h_clv_maturation_backlog
    )
    price_resolver_v4._load_two_h_clv_maturation_backlog = (
        derivative_clv_anchor_v4.load_two_h_clv_maturation_backlog
    )
    price_resolver_v4._load_corners_clv_maturation_backlog = (
        derivative_clv_anchor_v4.load_corners_clv_maturation_backlog
    )
    price_resolver_v4._load_team_totals_maturation_backlog = observed_team_totals_loader
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

    current_tick_team_totals_close_audit = team_totals_close_provenance_v4.build_report(
        candidate_events=team_totals_candidate_events,
        resolved_events=events,
        captured_at=payload.get("generated_at_utc"),
    )
    historical_team_totals_close_audit = team_totals_close_provenance_v4.load_historical_report()
    if historical_team_totals_close_audit.get("status") == "NO_DATABASE":
        team_totals_close_audit = current_tick_team_totals_close_audit
    else:
        historical_team_totals_close_audit["current_tick_candidate_audit"] = (
            current_tick_team_totals_close_audit
        )
        team_totals_close_audit = historical_team_totals_close_audit

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
    payload["v216_7_primary_clv_maturation_exclusion_audit"] = {
        "schema_version": primary_clv_anchor_v4.DIAGNOSTIC_SCHEMA_VERSION,
        "status": "OBSERVABILITY_ONLY",
        "scope": ["1X2", "BTTS", "FT_TOTALS"],
        "loader_observation": primary_loader_observation,
        "provider_requests_added": 0,
        "selection_logic_changed": False,
        "strict_close_semantics_changed": False,
        "historical_rows_mutated": False,
        "model_weights_changed": False,
        "thresholds_changed": False,
        "gates_changed": False,
        "production_promotion_allowed": False,
        "purpose": (
            "COUNT PRICED/FUTURE/T55/STRICT-LATER-QUOTE/UNRESOLVED FIXTURES PER PRIMARY FAMILY "
            "WITHOUT CHANGING WHICH FIXTURES ENTER MATURATION"
        ),
    }
    payload["v216_8_team_totals_close_provenance_audit"] = team_totals_close_audit
    price_resolution = payload.get("price_resolution_v4")
    if isinstance(price_resolution, dict):
        price_resolution["team_totals_close_provenance_audit"] = team_totals_close_audit
    payload["v216_8_checkpoint"] = (
        "TEAM TOTALS CLOSE PROVENANCE AUDIT ACTIVE: recent kicked-off modeled Team Totals are "
        "audited directly from Postgres refresh events and market snapshots, independent of the "
        "Phase17 2,000-signal cap. Exact market, side and line plus captured_at/provider_update "
        "strictly after the authentic signal and before kickoff are still required. The current-tick "
        "candidate view is retained as a nested diagnostic. Observability only: no provider calls, "
        "selection changes, history rewrite, cap change, gate/threshold changes, decision weight, "
        "or production promotion."
    )
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
