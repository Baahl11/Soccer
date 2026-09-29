from __future__ import annotations

from copy import deepcopy
from typing import Any

from mcp_gateway import automation_v125 as v125

MODEL_VERSION = v125.MODEL_VERSION
AUTOMATION_VERSION = "4.35.0-post-reserve-maturation-verification"


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _build_post_reserve_verification(payload: dict[str, Any]) -> dict[str, Any]:
    """Verify v208 reserve behavior from the final tick payload only.

    This is observability, not orchestration. It does not make provider calls,
    change budgets, create maturation candidates, relax strict-close semantics,
    or alter any model/decision/promotion input.
    """
    price = _dict(payload.get("price_resolution_v4"))
    paid_btts = _dict(payload.get("v207_btts_paid_odds_entry_capture"))

    calls = max(0, _as_int(payload.get("api_calls_this_tick")))
    configured_cap = max(0, _as_int(payload.get("max_api_calls_per_tick")))
    effective_cap = max(
        0,
        _as_int(payload.get("effective_max_api_calls_per_tick"), configured_cap),
    )
    cap_basis = effective_cap or configured_cap

    resolver_calls = max(0, _as_int(price.get("api_calls_added")))
    primary_calls = max(0, _as_int(price.get("primary_clv_maturation_api_calls_added")))
    resolver_max = max(0, _as_int(price.get("max_api_calls")))
    primary_max = max(0, _as_int(price.get("primary_clv_maturation_max_calls_per_tick")))
    candidate_families = deepcopy(_dict(price.get("primary_clv_maturation_candidate_family_counts")))

    headroom = max(0, cap_basis - calls) if cap_basis else None
    cap_respected = bool(cap_basis) and calls <= cap_basis
    price_budget_exercised = resolver_calls > 0
    primary_budget_exercised = primary_calls > 0

    if not cap_basis:
        status = "NOT_VERIFIED_NO_CAP_TELEMETRY"
    elif not cap_respected:
        status = "CAP_VIOLATION_DETECTED"
    elif price_budget_exercised:
        status = "LIVE_VERIFIED"
    else:
        status = "WAITING_PRICE_ACTIVITY"

    priority_targets_present = {
        family: max(0, _as_int(candidate_families.get(family)))
        for family in ("1X2", "BTTS", "FT_TOTALS")
    }

    return {
        "schema_version": "1.0.0",
        "status": status,
        "api_calls_this_tick": calls,
        "configured_cap": configured_cap,
        "effective_cap": effective_cap,
        "cap_basis": cap_basis,
        "cap_respected": cap_respected,
        "provider_headroom_after_tick": headroom,
        "price_resolver_api_calls_added": resolver_calls,
        "price_resolver_max_api_calls": resolver_max,
        "price_budget_exercised": price_budget_exercised,
        "primary_clv_maturation_api_calls_added": primary_calls,
        "primary_clv_maturation_max_calls_per_tick": primary_max,
        "primary_budget_exercised": primary_budget_exercised,
        "primary_clv_candidate_family_counts": candidate_families,
        "roadmap_priority_candidate_counts": priority_targets_present,
        "btts_paid_entry_rows_added": max(0, _as_int(paid_btts.get("entry_rows_added"))),
        "btts_paid_entry_captured_fixtures": max(0, _as_int(paid_btts.get("captured_fixtures"))),
        "provider_requests_added": 0,
        "provider_budget_changed": False,
        "decision_weight": 0.0,
        "production_promotion_allowed": False,
        "model_weights_changed": False,
        "thresholds_changed": False,
        "gates_changed": False,
        "canonical_bet_logic_changed": False,
        "strict_close_semantics_changed": False,
        "policy": (
            "FINAL_TICK_TELEMETRY_ONLY; VERIFY_MONOTONIC_HARD_CAP_AND_RESERVED_PRICE_ACTIVITY; "
            "NO_PROVIDER_CALLS; NO_CANDIDATE_CREATION; NO_THRESHOLD_OR_GATE_CHANGES; "
            "NO_SYNTHETIC_CLOSE"
        ),
    }


async def run_tick() -> dict[str, Any]:
    payload = await v125.run_tick()
    verification = _build_post_reserve_verification(payload)

    # Nest inside price_resolution_v4 so the existing compact scheduler-state
    # projection persists v209 automatically without changing the main workflow.
    price = payload.get("price_resolution_v4")
    if not isinstance(price, dict):
        price = {}
        payload["price_resolution_v4"] = price
    price["v209_post_reserve_verification"] = deepcopy(verification)

    payload["v209_post_reserve_maturation_verification"] = verification
    payload["v209_checkpoint"] = (
        "POST-v208 MATURATION VERIFICATION: observe the final tick and prove whether the tightened "
        "provider cap remained respected while the existing price-resolution/CLV reserve was usable. "
        "Observability only: no added provider requests, no candidate creation, no synthetic close, "
        "and no model, threshold, gate, stake or production-promotion change."
    )
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
