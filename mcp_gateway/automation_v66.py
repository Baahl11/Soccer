from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v65 as v65
from mcp_gateway import player_cards_intelligence

MODEL_VERSION = v65.MODEL_VERSION
AUTOMATION_VERSION = "3.42.0"


async def run_tick() -> dict[str, Any]:
    payload = await v65.run_tick()
    metrics = player_cards_intelligence.attach(payload)
    modeled_players = int(metrics.get("modeled_players") or 0)
    production_status = (
        "LIVE_RESEARCH_NOT_ACTIONABLE"
        if modeled_players > 0
        else "DORMANT_DATA_BLOCKED_UNTIL_FINALIZED_PLAYER_CARD_SAMPLES"
    )
    payload["player_cards_intelligence"] = {
        "schema_version": player_cards_intelligence.SCHEMA_VERSION,
        **metrics,
        "provider_requests_added": 0,
        "canonical_model_weights_changed": False,
        "canonical_thresholds_changed": False,
        "canonical_bet_logic_changed": False,
        "player_prop_bet_eligibility_changed": False,
        "production_status": production_status,
        "decision_weight": 0.0,
        "policy": "PLAYER BOOKED/YELLOW CARD RESEARCH ONLY; RED CARDS SEPARATE; NO BOOKMAKER CARD-POINT ASSUMPTION; NO VERIFIED PRICE/RULE/EV/PROMOTION",
    }
    payload["v342_provider_requests_added"] = 0
    payload["v342_model_weights_changed"] = False
    payload["v342_canonical_bet_logic_changed"] = False
    payload["v342_player_cards_checkpoint"] = (
        "PLAYER_YELLOW_CARD_PROBABILITIES_RESEARCH_ONLY"
        if modeled_players > 0
        else "PLAYER_CARD_MODEL_BUILT_BUT_DATA_BLOCKED; NATURAL_FINALIZED_PLAYER_YELLOW_CARD_SAMPLES_REQUIRED"
    )
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
