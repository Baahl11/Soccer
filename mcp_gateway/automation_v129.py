from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v128 as v128
from mcp_gateway import dynamic_strength_challenger_v4
from mcp_gateway import price_resolver_v4
from mcp_gateway import primary_clv_anchor_v4

MODEL_VERSION = v128.MODEL_VERSION
AUTOMATION_VERSION = "4.38.1-primary-clv-anchor-repair"


async def run_tick() -> dict[str, Any]:
    original_primary_loader = price_resolver_v4._load_primary_clv_maturation_backlog
    price_resolver_v4._load_primary_clv_maturation_backlog = (
        primary_clv_anchor_v4.load_primary_clv_maturation_backlog
    )
    try:
        payload = await v128.run_tick()
    finally:
        price_resolver_v4._load_primary_clv_maturation_backlog = original_primary_loader

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
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
