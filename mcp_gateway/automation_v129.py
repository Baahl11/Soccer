from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v128 as v128
from mcp_gateway import dynamic_strength_challenger_v4
from mcp_gateway import price_resolver_v4
from mcp_gateway import primary_clv_anchor_v4
from mcp_gateway import primary_clv_anchor_normalized_v4
from mcp_gateway import derivative_clv_anchor_v4
from mcp_gateway import team_totals_clv_anchor_v4
from mcp_gateway import team_totals_close_provenance_v4

MODEL_VERSION = v128.MODEL_VERSION
AUTOMATION_VERSION = "4.38.8-normalized-primary-clv-anchor-fallback"


def _int_map(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    output: dict[str, int] = {}
    for key, raw in value.items():
        try:
            output[str(key).upper()] = int(raw or 0)
        except (TypeError, ValueError):
            output[str(key).upper()] = 0
    return output


def _true_clv_view(payload: dict[str, Any], family: str) -> dict[str, Any]:
    family = family.upper()
    validation_key = {
        "FT_TOTALS": "v4_016_ft_totals_production_validation",
        "1X2": "v4_017_1x2_calibration_validation",
        "BTTS": "v4_018_btts_calibration_validation",
        "TEAM_TOTALS": "v4_019_team_totals_oos_validation",
        "1H": "v4_020_1h_oos_validation",
        "2H": "v4_021_2h_oos_validation",
        "CARDS": "phase14_cards_referee_validation",
        "PLAYER_PROPS": "phase15_player_props_validation",
    }.get(family)
    if family in {"FT_CORNERS", "TEAM_CORNERS"}:
        validation = payload.get("v4_022_corners_oos_validation")
        validation = validation if isinstance(validation, dict) else {}
        family_views = validation.get("family_views")
        family_views = family_views if isinstance(family_views, dict) else {}
        view = family_views.get(family)
        view = view if isinstance(view, dict) else {}
        true_clv = view.get("true_clv")
        true_clv = true_clv if isinstance(true_clv, dict) else {}
        return {
            "validation_status": view.get("status") or validation.get("status"),
            "rows": int(true_clv.get("rows") or 0),
            "unique_fixtures": int(true_clv.get("unique_fixtures") or 0),
            "minimum_rows": int(true_clv.get("minimum_rows") or 50),
        }
    validation = payload.get(validation_key) if validation_key else {}
    validation = validation if isinstance(validation, dict) else {}
    true_clv = validation.get("true_clv")
    true_clv = true_clv if isinstance(true_clv, dict) else {}
    return {
        "validation_status": validation.get("status") or validation.get("validation_status"),
        "rows": int(true_clv.get("rows") or 0),
        "unique_fixtures": int(true_clv.get("unique_fixtures") or 0),
        "minimum_rows": int(true_clv.get("minimum_rows") or 50),
    }


def _market_maturation_health(payload: dict[str, Any]) -> dict[str, Any]:
    price = payload.get("price_resolution_v4")
    price = price if isinstance(price, dict) else {}
    candidates = _int_map(price.get("primary_clv_maturation_candidate_family_counts"))
    evaluated = _int_map(price.get("primary_clv_maturation_family_evaluation_counts"))
    refreshed = _int_map(price.get("primary_clv_maturation_family_refresh_counts"))
    not_matured = _int_map(price.get("primary_clv_maturation_not_matured_family_counts"))

    families: dict[str, dict[str, Any]] = {}
    for family in ("1X2", "FT_TOTALS", "BTTS", "1H", "2H", "FT_CORNERS", "TEAM_CORNERS"):
        clv = _true_clv_view(payload, family)
        row = {
            **clv,
            "candidate_signal_rows_this_tick": candidates.get(family, 0),
            "evaluated_signal_rows_this_tick": evaluated.get(family, 0),
            "strict_later_quote_rows_this_tick": refreshed.get(family, 0),
            "not_matured_rows_this_tick": not_matured.get(family, 0),
        }
        if row["rows"] >= row["minimum_rows"]:
            state = "CLV_GATE_MET"
        elif row["strict_later_quote_rows_this_tick"] > 0:
            state = "ADVANCING_THIS_TICK"
        elif row["evaluated_signal_rows_this_tick"] > 0 and row["not_matured_rows_this_tick"] > 0:
            state = "WAITING_STRICT_LATER_QUOTE"
        elif row["candidate_signal_rows_this_tick"] > 0:
            state = "CANDIDATES_QUEUED"
        elif row["rows"] > 0:
            state = "MATURATING"
        else:
            state = "NO_CURRENT_CLV_PROGRESS"
        row["maturation_state"] = state
        families[family] = row

    team_clv = _true_clv_view(payload, "TEAM_TOTALS")
    team_candidates = int(price.get("research_spillover_maturation_candidates") or 0)
    team_refreshes = int(price.get("research_spillover_maturation_later_real_quote_refreshes") or 0)
    team_unchanged = int(price.get("research_spillover_maturation_unchanged_provider_updates") or 0)
    families["TEAM_TOTALS"] = {
        **team_clv,
        "candidate_fixtures_this_tick": team_candidates,
        "strict_later_quote_refreshes_this_tick": team_refreshes,
        "unchanged_provider_updates_this_tick": team_unchanged,
        "maturation_state": (
            "CLV_GATE_MET"
            if team_clv["rows"] >= team_clv["minimum_rows"]
            else "ADVANCING_THIS_TICK"
            if team_refreshes > 0
            else "WAITING_STRICT_LATER_QUOTE"
            if team_candidates > 0 and team_unchanged > 0
            else "CANDIDATES_QUEUED"
            if team_candidates > 0
            else "MATURATING"
            if team_clv["rows"] > 0
            else "NO_CURRENT_CLV_PROGRESS"
        ),
    }

    cards_clv = _true_clv_view(payload, "CARDS")
    families["CARDS"] = {
        **cards_clv,
        "maturation_state": (
            "CLV_GATE_MET"
            if cards_clv["rows"] >= cards_clv["minimum_rows"]
            else "MATURATING"
            if cards_clv["rows"] > 0
            else "NO_CURRENT_CLV_PROGRESS"
        ),
        "runtime_capture_telemetry": "NOT_EXPOSED_BY_PRICE_RESOLVER",
    }

    props_validation = payload.get("phase15_player_props_validation")
    props_validation = props_validation if isinstance(props_validation, dict) else {}
    props_true_clv = props_validation.get("true_clv")
    props_true_clv = props_true_clv if isinstance(props_true_clv, dict) else {}
    props_by_family = props_true_clv.get("by_family")
    props_by_family = props_by_family if isinstance(props_by_family, dict) else {}
    prop_candidates = _int_map(price.get("player_props_clv_maturation_candidate_family_counts"))
    prop_evaluated = _int_map(price.get("player_props_clv_maturation_family_evaluation_counts"))
    prop_refreshed = _int_map(price.get("player_props_clv_maturation_family_refresh_counts"))
    prop_not_matured = _int_map(price.get("player_props_clv_maturation_not_matured_family_counts"))
    for source_key, family in (
        ("shots", "PROP_SHOTS"),
        ("sot", "PROP_SOT"),
        ("goalscorer", "PROP_GOALSCORER"),
        ("assists", "PROP_ASSISTS"),
        ("cards", "PROP_CARDS"),
        ("gk_saves", "PROP_GK_SAVES"),
    ):
        view = props_by_family.get(source_key)
        view = view if isinstance(view, dict) else {}
        lookup_keys = {source_key.upper(), family, family.replace("PROP_", "")}
        candidate_count = max((prop_candidates.get(key, 0) for key in lookup_keys), default=0)
        evaluation_count = max((prop_evaluated.get(key, 0) for key in lookup_keys), default=0)
        refresh_count = max((prop_refreshed.get(key, 0) for key in lookup_keys), default=0)
        not_matured_count = max((prop_not_matured.get(key, 0) for key in lookup_keys), default=0)
        rows = int(view.get("rows") or 0)
        minimum_rows = int(view.get("minimum_rows") or 50)
        families[family] = {
            "validation_status": props_validation.get("status") or props_validation.get("validation_status"),
            "rows": rows,
            "unique_fixtures": int(view.get("unique_fixtures") or 0),
            "minimum_rows": minimum_rows,
            "candidate_signal_rows_this_tick": candidate_count,
            "evaluated_signal_rows_this_tick": evaluation_count,
            "strict_later_quote_rows_this_tick": refresh_count,
            "not_matured_rows_this_tick": not_matured_count,
            "maturation_state": (
                "CLV_GATE_MET"
                if rows >= minimum_rows
                else "ADVANCING_THIS_TICK"
                if refresh_count > 0
                else "WAITING_STRICT_LATER_QUOTE"
                if evaluation_count > 0 and not_matured_count > 0
                else "CANDIDATES_QUEUED"
                if candidate_count > 0
                else "MATURATING"
                if rows > 0
                else "NO_CURRENT_CLV_PROGRESS"
            ),
        }

    return {
        "schema_version": "1.0.0",
        "status": "OBSERVABILITY_ONLY",
        "generated_at_utc": payload.get("generated_at_utc"),
        "strict_close_semantics_changed": False,
        "model_weights_changed": False,
        "thresholds_changed": False,
        "gates_changed": False,
        "decision_weight": 0.0,
        "runtime_heavy_history_policy": "DEFER_TO_OFFLINE_VALIDATION",
        "families": families,
    }



async def run_tick() -> dict[str, Any]:
    original_primary_loader = price_resolver_v4._load_primary_clv_maturation_backlog
    original_one_h_loader = price_resolver_v4._load_one_h_clv_maturation_backlog
    original_two_h_loader = price_resolver_v4._load_two_h_clv_maturation_backlog
    original_corners_loader = price_resolver_v4._load_corners_clv_maturation_backlog
    original_team_totals_loader = price_resolver_v4._load_team_totals_maturation_backlog
    primary_loader_observation: dict[str, Any] = {}
    primary_loader_fallback: dict[str, Any] = {
        "used": False,
        "reason": None,
        "normalized_error_type": None,
        "normalized_error": None,
        "fallback_source": None,
    }
    team_totals_candidate_events: list[dict[str, Any]] = []

    def observed_primary_loader(*args: Any, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("include_diagnostics", False)
        try:
            report = primary_clv_anchor_normalized_v4.load_primary_clv_maturation_backlog(
                *args, **kwargs
            )
        except Exception as exc:
            # Emergency-only compatibility fallback. The normalized relational
            # loader is the normal hot path; legacy JSON expansion is used only
            # if that loader itself fails, preserving scheduler continuity.
            report = primary_clv_anchor_v4.load_primary_clv_maturation_backlog(
                *args, **kwargs
            )
            primary_loader_fallback.update(
                {
                    "used": True,
                    "reason": "NORMALIZED_PRIMARY_CLV_LOADER_EXCEPTION",
                    "normalized_error_type": type(exc).__name__,
                    "normalized_error": str(exc)[:500],
                    "fallback_source": report.get("source"),
                }
            )
        primary_loader_observation.clear()
        primary_loader_observation.update(
            {
                "source": report.get("source"),
                "signal_anchor_policy": report.get("signal_anchor_policy"),
                "candidate_count": int(report.get("candidate_count") or 0),
                "candidate_family_counts": dict(report.get("candidate_family_counts") or {}),
                "candidate_source_counts": dict(report.get("candidate_source_counts") or {}),
                "diagnostic_schema_version": report.get("diagnostic_schema_version"),
                "diagnostic_status": report.get("diagnostic_status"),
                "diagnostic_family_counts": dict(report.get("diagnostic_family_counts") or {}),
                "diagnostic_window": dict(report.get("diagnostic_window") or {}),
                "provider_requests_added": int(report.get("provider_requests_added") or 0),
                "selection_logic_changed": bool(report.get("selection_logic_changed", False)),
                "normalized_storage": bool(report.get("normalized_storage", False)),
                "legacy_json_expansion_used": bool(
                    primary_loader_fallback.get("used")
                    or report.get("legacy_json_expansion_used", False)
                ),
                "fallback_used": bool(primary_loader_fallback.get("used")),
                "fallback_reason": primary_loader_fallback.get("reason"),
                "fallback_source": primary_loader_fallback.get("fallback_source"),
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
    team_totals_close_audit = dict(current_tick_team_totals_close_audit)
    team_totals_close_audit.update(
        {
            "runtime_scope": "CURRENT_TICK_ONLY",
            "historical_audit_status": "DEFERRED_TO_OFFLINE_VALIDATION",
            "historical_audit_reason": "KEEP_LARGE_SNAPSHOT_HISTORY_OUT_OF_512MB_LIVE_RUNTIME",
        }
    )

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
        "normalized_signal_storage": True,
        "legacy_pipeline_json_expansion_in_live_loader": bool(
            primary_loader_fallback.get("used")
        ),
        "normalized_loader_fallback": dict(primary_loader_fallback),
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
            "LIVE RUNTIME RECORDS THE PRIMARY ANCHOR/CANDIDATE COUNTS ONLY; THE HEAVY HISTORICAL "
            "EXCLUSION QUERY IS DEFERRED TO OFFLINE VALIDATION TO PROTECT SCHEDULER MEMORY/LATENCY"
        ),
    }
    payload["v216_8_team_totals_close_provenance_audit"] = team_totals_close_audit
    price_resolution = payload.get("price_resolution_v4")
    if isinstance(price_resolution, dict):
        price_resolution["team_totals_close_provenance_audit"] = team_totals_close_audit
    payload["v216_8_checkpoint"] = (
        "TEAM TOTALS LIVE CLOSE PROVENANCE IS CURRENT-TICK ONLY. Historical Team Totals provenance "
        "remains available to offline validation jobs but is intentionally excluded from the live "
        "scheduler process so tens of thousands of snapshot JSON rows cannot exhaust the 512 MiB "
        "runtime. Strict captured_at/provider_update chronology is unchanged; no model, threshold, "
        "gate, budget, history, or production-decision semantics changed."
    )
    payload["market_maturation_health"] = _market_maturation_health(payload)
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
