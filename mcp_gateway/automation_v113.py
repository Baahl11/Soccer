from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v112 as v112
from mcp_gateway import clv_engine_v4

MODEL_VERSION = v112.MODEL_VERSION
AUTOMATION_VERSION = "4.22.0-phase17.clv_engine"


def _annotate_phase17(payload: dict[str, Any]) -> None:
    payload["phase17_clv_engine"] = {
        "schema_version": "1.0.0",
        "status": "ENGINE_IMPLEMENTED",
        "model_version": clv_engine_v4.MODEL_VERSION,
        "required_storage_fields": list(clv_engine_v4.REQUIRED_STORAGE_FIELDS),
        "analysis_dimensions": [
            "market",
            "league",
            "tier",
            "confidence",
            "stage",
            "model_version",
            "bookmaker",
        ],
        "minimum_true_close_rows": clv_engine_v4.MIN_TRUE_CLOSE_ROWS,
        "price_clv_formula": "entry_decimal/closing_decimal - 1",
        "explicit_closing_line_required_for_line_movement": True,
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
        "runtime_prediction_weight": 0.0,
        "model_weights_changed": False,
        "canonical_bet_logic_changed": False,
        "note": (
            "Phase 17 formalizes entry/close line and price, probability CLV, price CLV, line "
            "movement and bookmaker storage plus CLV analysis by market/league/tier/confidence/"
            "stage/model version. Legacy rows lacking closing_line/model_version remain incomplete."
        ),
    }
    payload["phase17_provider_requests_added"] = 0
    payload["phase17_model_weights_changed"] = False
    payload["phase17_canonical_bet_logic_changed"] = False
    payload["phase17_checkpoint"] = (
        "CLV ENGINE IMPLEMENTED. True-close probability CLV, price CLV and line movement are "
        "tracked explicitly, with segmented analysis by market, league, tier, confidence, stage "
        "and model version. Missing closing line/model version are surfaced as blockers."
    )


async def run_tick() -> dict[str, Any]:
    payload = await v112.run_tick()
    _annotate_phase17(payload)
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
