from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v42 as v42
from mcp_gateway import double_chance_intelligence

MODEL_VERSION = v42.MODEL_VERSION
AUTOMATION_VERSION = "3.19.0"


async def run_tick() -> dict[str, Any]:
    payload = await v42.run_tick()
    metrics = double_chance_intelligence.attach(payload)
    payload["double_chance_intelligence"] = {
        "schema_version": double_chance_intelligence.SCHEMA_VERSION,
        **metrics,
        "provider_requests_added": 0,
        "canonical_model_weights_changed": False,
        "canonical_thresholds_changed": False,
        "canonical_bet_logic_changed": False,
        "galaxy_promotion_allowed": False,
        "production_status": "RESEARCH_ONLY_PARENT_1X2_NOT_APPROVED",
        "model": "CANONICAL_1X2_DERIVED_DOUBLE_CHANCE_v0.1",
        "market_policy": (
            "DOUBLE-CHANCE OUTCOMES OVERLAP; DO NOT NORMALIZE DC PRICES AS A THREE-WAY NO-VIG MARKET; "
            "WHEN POSSIBLE DERIVE MARKET-FAIR DC FROM SAME-BOOK COMPLETE 1X2 NO-VIG PROBABILITIES"
        ),
        "promotion_policy": (
            "PARENT 1X2 MUST BE PRODUCTION-APPROVED FIRST; THEN DC REQUIRES OWN OOS/CALIBRATION/CLV/SHRINKAGE REVIEW"
        ),
    }
    payload["v319_provider_requests_added"] = 0
    payload["v319_model_weights_changed"] = False
    payload["v319_canonical_bet_logic_changed"] = False
    payload["v319_double_chance_checkpoint"] = "LIVE_RESEARCH_DERIVATION_ADDED; PRODUCTION_UNCHANGED"
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
