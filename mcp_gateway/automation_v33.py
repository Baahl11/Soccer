from __future__ import annotations

from typing import Any

from mcp_gateway import automation as base
from mcp_gateway import automation_v29 as v29
from mcp_gateway.galaxy_builder_v3 import build as build_galaxy_builder_v3
from mcp_gateway import match_intelligence_v1 as intelligence

MODEL_VERSION = v29.MODEL_VERSION
AUTOMATION_VERSION = "3.9.0"

_ORIGINAL_COMPACT_FIXTURE = base._compact_fixture
_ORIGINAL_WANTED_MARKET = base._wanted_market
_ORIGINAL_COMPACT_ODDS = base._compact_odds


def _compact_fixture_with_referee(item: dict[str, Any]) -> dict[str, Any]:
    out = _ORIGINAL_COMPACT_FIXTURE(item)
    fixture = item.get("fixture") or {}
    out["referee"] = fixture.get("referee")
    out["fixture_timezone"] = fixture.get("timezone")
    return out


def _wanted_market_complete(name: str) -> bool:
    if _ORIGINAL_WANTED_MARKET(name):
        return True
    n = str(name or "").lower()
    return any(token in n for token in ("card", "booking", "bookings"))


def _compact_odds_complete(payload: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for fixture_row in payload.get("response", []):
        update = fixture_row.get("update")
        for book in fixture_row.get("bookmakers") or []:
            for bet in book.get("bets") or []:
                name = bet.get("name") or ""
                if not _wanted_market_complete(name):
                    continue
                values = [
                    {"selection": value.get("value"), "price": value.get("odd")}
                    for value in (bet.get("values") or [])
                ]
                rows.append(
                    {
                        "bookmaker_id": book.get("id"),
                        "bookmaker": book.get("name"),
                        "market_id": bet.get("id"),
                        "market": name,
                        "values": values,
                        "provider_update": update,
                    }
                )
    # v3.9 widens the retained research market surface only. It adds no request.
    # Exact prices remain observations, never sport probabilities.
    max_rows = 120
    return {"markets": rows[:max_rows], "market_count": len(rows), "truncated": len(rows) > max_rows}


async def run_tick() -> dict[str, Any]:
    previous_fixture = base._compact_fixture
    previous_wanted = base._wanted_market
    previous_odds = base._compact_odds
    base._compact_fixture = _compact_fixture_with_referee
    base._wanted_market = _wanted_market_complete
    base._compact_odds = _compact_odds_complete
    try:
        payload = await v29.run_tick()
    finally:
        base._compact_fixture = previous_fixture
        base._wanted_market = previous_wanted
        base._compact_odds = previous_odds

    card_groups = 0
    card_quotes = 0
    for event in payload.get("events") or []:
        if not isinstance(event, dict):
            continue
        groups, quotes = intelligence.attach_card_inventory(event, payload.get("generated_at_local"))
        card_groups += groups
        card_quotes += quotes

    intelligence_metrics = intelligence.attach(payload)

    builder = build_galaxy_builder_v3(payload)
    payload["galaxy_builder"] = builder
    payload["galaxy_builder_candidate_count_this_tick"] = int(builder.get("candidate_count") or 0)
    payload["galaxy_builder_same_game_count_this_tick"] = int(builder.get("same_game_candidate_count") or 0)
    payload["galaxy_builder_multi_match_count_this_tick"] = int(builder.get("multi_match_candidate_count") or 0)
    payload["galaxy_builder_explicit_derivative_multi_count_this_tick"] = int(
        builder.get("explicit_derivative_multi_candidate_count") or 0
    )
    payload["galaxy_builder_derivative_model_contract"] = builder.get("derivative_model_contract")
    payload["galaxy_builder_policy"] = builder.get("policy")
    payload["galaxy_builder_actionable_scope"] = (
        "V0.3 SHADOW/RESEARCH; SGP AND MULTI-MATCH PARLAYS; TARGET +110 OR BETTER; "
        "DERIVATIVES REQUIRE EXPLICIT SPORT-FIRST MODEL CONTRACT; SAME-GAME DERIVATIVES "
        "REQUIRE CORRELATION-AWARE JOINT MODEL; NO BET PROMOTION WITHOUT FINAL VERIFIED "
        "SPORTSBOOK PARLAY QUOTE AND ALL MODEL GATES"
    )

    payload["match_intelligence_schema_version"] = intelligence.SCHEMA_VERSION
    payload["match_intelligence_event_count_this_tick"] = intelligence_metrics["attached"]
    payload["referee_assignment_verified_count_this_tick"] = intelligence_metrics["referee_verified"]
    payload["coach_pair_verified_count_this_tick"] = intelligence_metrics["coaches_verified"]
    payload["card_market_groups_this_tick"] = card_groups
    payload["card_market_quotes_this_tick"] = card_quotes
    payload["cards_probability_model_status"] = "RESEARCH_BASELINE_BUILDING; NOT_ACTIONABLE"
    payload["match_intelligence_policy"] = (
        "FULL_MATCH_ANALYSIS_SURFACE: SIDE+FT_GOALS+BTTS+HALVES+CORNERS+CARDS+PLAYERS+XI+FORMATION+GK+COACH+REFEREE+"
        "INJURIES+TACTICS+ADVANCED_METRICS+SET_PIECES+REST_TRAVEL+WEATHER+COMPETITION_CONTEXT+MARKET_PROVENANCE; "
        "MISSING INPUTS ARE EXPLICIT; ZERO INVENTED PROBABILITIES; CANONICAL BET LOGIC UNCHANGED"
    )
    payload["v39_provider_requests_added"] = 0
    payload["v39_model_weights_changed"] = False
    payload["v39_canonical_bet_logic_changed"] = False
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
