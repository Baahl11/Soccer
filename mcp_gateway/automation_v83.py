from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v2 as v2
from mcp_gateway import automation_v82 as v82

MODEL_VERSION = v82.MODEL_VERSION
AUTOMATION_VERSION = "3.58.1"
TARGET_EDGE_PP = 3.5

_QUOTE_ONLY_BLOCKS = {
    "FINAL_PARLAY_QUOTE_NOT_VERIFIED",
    "EXACT_CORRELATION_ADJUSTED_SGP_QUOTE_NOT_EXPOSED",
    "EXACT_CORRELATION_ADJUSTED_SGP_QUOTE_NOT_EXPOSED",
}


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if out > 0 else None
    except (TypeError, ValueError):
        return None


def _minimum_decimal(probability: float | None) -> float | None:
    if probability is None:
        return None
    max_market_probability = probability - TARGET_EDGE_PP / 100.0
    if max_market_probability <= 0:
        return None
    return 1.0 / max_market_probability


def _manual_threshold(row: dict[str, Any]) -> float | None:
    if row.get("type") == "SAME_GAME_PARLAY":
        explicit = _num(row.get("minimum_sgp_decimal_for_target_edge"))
        if explicit is not None:
            return explicit
        return _minimum_decimal(_num(row.get("joint_model_probability")))
    explicit = _num(row.get("minimum_parlay_decimal_for_target_edge"))
    if explicit is not None:
        return explicit
    return _minimum_decimal(_num(row.get("conservative_joint_probability")))


def _annotate_candidate(row: dict[str, Any]) -> dict[str, Any]:
    original_blocks = [str(x) for x in (row.get("block_reasons") or [])]
    real_blocks = [x for x in original_blocks if x not in _QUOTE_ONLY_BLOCKS]
    threshold = _manual_threshold(row)
    manual_ready = threshold is not None and not real_blocks

    row["classification"] = "WATCH"
    row["surface_to_user"] = True
    row["manual_price_check"] = True
    row["manual_price_check_eligible"] = manual_ready
    row["manual_minimum_decimal"] = round(threshold, 4) if threshold is not None else None
    row["manual_minimum_american"] = v2._american(threshold) if threshold is not None else None
    row["pending_market_inputs"] = ["EXECUTABLE_COMBINED_DECIMAL_QUOTE"]
    row["legacy_block_reasons"] = original_blocks
    row["block_reasons"] = real_blocks
    row["display_status"] = (
        "MODEL PLAY — MANUAL PRICE CHECK"
        if manual_ready
        else "GALAXY WATCH — MANUAL PRICE CHECK + OTHER GATES"
    )
    row["bet_eligibility_reason"] = (
        "PENDING_MANUAL_COMBINED_PRICE_ONLY"
        if manual_ready
        else "ADDITIONAL_MODEL_OR_AVAILABILITY_GATES_REMAIN"
    )
    row["manual_decision_rule"] = (
        "IF YOUR BOOK COMBINED DECIMAL >= manual_minimum_decimal AND ALL NON-PRICE GATES REMAIN CLEAR, "
        "RETURN THE QUOTE FOR FINAL BET/EV CLASSIFICATION"
    )
    row["manual_price_policy"] = (
        "MISSING FINAL QUOTE MUST NOT HIDE THE MODEL PLAY; IT REMAINS WATCH UNTIL THE USER OR PROVIDER "
        "SUPPLIES AN EXECUTABLE COMBINED PRICE"
    )
    row["bet_eligible"] = False
    return row


def _surface(payload: dict[str, Any]) -> None:
    builder = payload.get("galaxy_builder") if isinstance(payload.get("galaxy_builder"), dict) else {}
    builder = dict(builder)
    surfaced: list[dict[str, Any]] = []

    for key in ("same_game_candidates", "multi_match_candidates"):
        rows = []
        for candidate in builder.get(key) or []:
            if not isinstance(candidate, dict):
                continue
            row = _annotate_candidate(dict(candidate))
            rows.append(row)
            if row.get("surface_to_user"):
                surfaced.append(row)
        builder[key] = rows

    surfaced.sort(
        key=lambda row: (
            1 if row.get("manual_price_check_eligible") else 0,
            1 if row.get("all_leg_models_actionable") else 0,
            float(row.get("joint_model_probability") or row.get("conservative_joint_probability") or 0.0),
        ),
        reverse=True,
    )
    builder["manual_price_check_count"] = len(surfaced)
    builder["manual_price_check_ready_count"] = sum(
        1 for row in surfaced if row.get("manual_price_check_eligible")
    )
    builder["manual_price_check_policy"] = (
        "SURFACE QUALIFYING GALAXY WATCHES EVEN WITHOUT A FINAL COMBINED QUOTE; SHOW THE MODEL-DERIVED "
        "MINIMUM ACCEPTABLE DECIMAL PRICE; FINAL BET/EV CLASSIFICATION STILL REQUIRES THE USER OR PROVIDER "
        "TO SUPPLY AN EXECUTABLE QUOTE"
    )
    payload["galaxy_builder"] = builder
    payload["manual_price_check_candidates"] = surfaced[:6]
    payload["manual_price_check_count"] = len(surfaced)
    payload["manual_price_check_ready_count"] = builder["manual_price_check_ready_count"]


async def run_tick() -> dict[str, Any]:
    payload = await v82.run_tick()
    _surface(payload)
    payload["v3581_provider_requests_added"] = 0
    payload["v3581_model_weights_changed"] = False
    payload["v3581_canonical_bet_logic_changed"] = False
    payload["v3581_manual_price_checkpoint"] = (
        "MISSING FINAL PARLAY/SGP QUOTE NO LONGER HIDES A MODEL PLAY; GALAXY SURFACES WATCH + "
        "MODEL-DERIVED MINIMUM ACCEPTABLE PRICE FOR MANUAL BOOK CHECK"
    )
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
