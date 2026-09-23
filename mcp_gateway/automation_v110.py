from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v109 as v109
from mcp_gateway import cards_referee_phase14_v4

MODEL_VERSION = v109.MODEL_VERSION
AUTOMATION_VERSION = "4.19.0-phase14.cards_referee"


def _annotate_phase14(payload: dict[str, Any]) -> None:
    payload["phase14_cards_referee_validation"] = {
        "schema_version": "1.0.0",
        "status": "VALIDATION_GATE_IMPLEMENTED",
        "model_version": cards_referee_phase14_v4.MODEL_VERSION,
        "minimum_yellow_oos": cards_referee_phase14_v4.MIN_YELLOW_OOS,
        "minimum_referee_adjusted": cards_referee_phase14_v4.MIN_REFEREE_ADJUSTED,
        "minimum_red_market_review": cards_referee_phase14_v4.MIN_RED_MARKET_REVIEW,
        "minimum_red_actionable": cards_referee_phase14_v4.MIN_RED_ACTIONABLE,
        "minimum_card_true_clv_rows": cards_referee_phase14_v4.MIN_CARD_TRUE_CLV,
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
        "runtime_prediction_weight": 0.0,
        "model_weights_changed": False,
        "canonical_bet_logic_changed": False,
        "note": (
            "Phase 14 validates yellow-card, red-card, referee-adjusted and player-card evidence "
            "separately. Referee-adjusted OOS, explicit sportsbook settlement mapping and "
            "card-family true CLV remain mandatory before any production review."
        ),
    }
    payload["phase14_provider_requests_added"] = 0
    payload["phase14_model_weights_changed"] = False
    payload["phase14_canonical_bet_logic_changed"] = False
    payload["phase14_checkpoint"] = (
        "CARDS/REFEREE VALIDATION GATE IMPLEMENTED. Yellow cards, red cards and player cards "
        "remain separate targets; verified referee assignments, OOS evidence, sportsbook rule "
        "mapping and card-family true CLV are required. No production promotion or BET logic change."
    )


async def run_tick() -> dict[str, Any]:
    payload = await v109.run_tick()
    _annotate_phase14(payload)
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
