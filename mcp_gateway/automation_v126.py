from __future__ import annotations

from copy import deepcopy
from typing import Any, Awaitable, Callable

from mcp_gateway import automation_v2 as v2
from mcp_gateway import automation_v4 as v4
from mcp_gateway import automation_v125 as v125
from mcp_gateway import price_resolver_v4

MODEL_VERSION = v125.MODEL_VERSION
AUTOMATION_VERSION = "4.35.1-post-reserve-phase-release"


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _release_reserved_price_phase_cap() -> dict[str, Any]:
    """Release the v208 low-water cap only at the reserved price phase boundary.

    v208 correctly prevented later upstream layers from silently reopening the
    reserved cap, but that low-water mark must not survive into the *explicitly
    reserved* post-upstream price resolver phase. The global elastic cap has
    already been restored by v123 before `_fetch_fixture_odds` is called.

    This function never raises the declared global cap and never adds budget;
    it only lets the reserved portion of that same cap become usable by the
    component it was reserved for.
    """
    try:
        declared_global_cap = max(1, int(v2.MAX_API_CALLS_PER_TICK))
    except (TypeError, ValueError):
        declared_global_cap = 1
    try:
        calls_at_release = max(0, int(v2._API_CALLS_THIS_TICK or 0))
    except (TypeError, ValueError):
        calls_at_release = 0

    previous = v4._MONOTONIC_TICK_CAP
    previous_cap = _as_int(previous, declared_global_cap) if previous is not None else None
    released = previous is not None and declared_global_cap > int(previous)
    if released:
        # The global elastic cap is authoritative at this explicit phase
        # boundary. It must already be >= used calls; clamp defensively.
        v4._MONOTONIC_TICK_CAP = max(calls_at_release, declared_global_cap)

    return {
        "released": bool(released),
        "previous_low_water_cap": previous_cap,
        "declared_global_cap": declared_global_cap,
        "effective_cap_after_release": _as_int(v4._MONOTONIC_TICK_CAP, declared_global_cap),
        "provider_calls_at_release": calls_at_release,
        "provider_budget_changed": False,
        "provider_requests_added": 0,
        "phase": "POST_UPSTREAM_RESERVED_PRICE_RESOLUTION",
    }


def _build_post_reserve_verification(
    payload: dict[str, Any],
    phase_release: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify v208/v209 reserve behavior from final tick telemetry."""
    price = _dict(payload.get("price_resolution_v4"))
    paid_btts = _dict(payload.get("v207_btts_paid_odds_entry_capture"))
    phase_release = _dict(phase_release)

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
        "schema_version": "1.1.0",
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
        "reserved_price_phase_release": deepcopy(phase_release),
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
            "VERIFY_MONOTONIC_UPSTREAM_RESERVE; EXPLICITLY_RELEASE_LOW_WATER_ONLY_AT_RESERVED_PRICE_PHASE; "
            "SAME_GLOBAL_ELASTIC_CAP; NO_PROVIDER_BUDGET_INCREASE; NO_CANDIDATE_CREATION; "
            "NO_THRESHOLD_OR_GATE_CHANGES; NO_SYNTHETIC_CLOSE"
        ),
    }


async def run_tick() -> dict[str, Any]:
    original_fetch: Callable[..., Awaitable[Any]] = price_resolver_v4._fetch_fixture_odds
    phase_release: dict[str, Any] = {
        "released": False,
        "previous_low_water_cap": None,
        "declared_global_cap": None,
        "effective_cap_after_release": None,
        "provider_calls_at_release": None,
        "provider_budget_changed": False,
        "provider_requests_added": 0,
        "phase": "POST_UPSTREAM_RESERVED_PRICE_RESOLUTION",
    }

    async def phase_boundary_fetch(*args: Any, **kwargs: Any) -> Any:
        nonlocal phase_release
        current = _release_reserved_price_phase_cap()
        if current.get("released") or phase_release.get("declared_global_cap") is None:
            phase_release = current
        return await original_fetch(*args, **kwargs)

    # v125 installs its BTTS observer on top of this wrapper. Its observer then
    # delegates here, so the release happens only when the price resolver is
    # actually about to use the capacity explicitly reserved for it.
    price_resolver_v4._fetch_fixture_odds = phase_boundary_fetch
    try:
        payload = await v125.run_tick()
    finally:
        price_resolver_v4._fetch_fixture_odds = original_fetch

    verification = _build_post_reserve_verification(payload, phase_release)

    # Nest inside price_resolution_v4 so the existing compact scheduler-state
    # projection persists v209 automatically without changing the main workflow.
    price = payload.get("price_resolution_v4")
    if not isinstance(price, dict):
        price = {}
        payload["price_resolution_v4"] = price
    price["v209_post_reserve_verification"] = deepcopy(verification)

    payload["v209_post_reserve_maturation_verification"] = verification
    payload["v209_checkpoint"] = (
        "POST-v208 MATURATION VERIFICATION + RESERVED-PHASE RELEASE: keep the v208 low-water hard cap "
        "through all upstream work, then explicitly release only to the already-declared global elastic cap "
        "when the reserved price resolver begins. No added provider budget, no candidate creation, no synthetic "
        "close, and no model, threshold, gate, stake or production-promotion change."
    )
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
