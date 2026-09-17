from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v44 as v44
from mcp_gateway import asian_handicap_intelligence

MODEL_VERSION = v44.MODEL_VERSION
AUTOMATION_VERSION = "3.21.0"


async def run_tick() -> dict[str, Any]:
    payload = await v44.run_tick()
    metrics = asian_handicap_intelligence.attach(payload)
    payload["asian_handicap_intelligence"] = {
        "schema_version": asian_handicap_intelligence.SCHEMA_VERSION,
        **metrics,
        "provider_requests_added": 0,
        "canonical_model_weights_changed": False,
        "canonical_thresholds_changed": False,
        "canonical_bet_logic_changed": False,
        "galaxy_promotion_allowed": False,
        "production_status": "RESEARCH_ONLY_NOT_ACTIONABLE",
        "model": "CANONICAL_SCORE_MARGIN_ASIAN_HANDICAP_v0.1",
        "settlement_policy": (
            "INTEGER/HALF/QUARTER LINES EXPLICIT; QUARTERS SPLIT TO ADJACENT HALF-LINES; "
            "FAIR PRICE SOLVES SETTLEMENT-AWARE EXPECTED RETURN=1; NO FAKE BINARY NO-VIG"
        ),
        "promotion_policy": (
            "REQUIRES OOS SETTLEMENT CALIBRATION, EXACT-LINE PRICE HISTORY, TRUE CLV, SHRINKAGE AND ADEQUATE PARENT MARGIN MODEL"
        ),
    }
    payload["v321_provider_requests_added"] = 0
    payload["v321_model_weights_changed"] = False
    payload["v321_canonical_bet_logic_changed"] = False
    payload["v321_asian_handicap_checkpoint"] = "LIVE_RESEARCH_SETTLEMENT_MODEL_ADDED; PRODUCTION_UNCHANGED"
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
