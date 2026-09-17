from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v37 as v37
from mcp_gateway import ft_goals_intelligence

MODEL_VERSION = v37.MODEL_VERSION
AUTOMATION_VERSION = "3.14.0"


async def run_tick() -> dict[str, Any]:
    payload = await v37.run_tick()
    metrics = ft_goals_intelligence.attach(payload)
    payload["ft_goals_intelligence"] = {
        "schema_version": ft_goals_intelligence.SCHEMA_VERSION,
        **metrics,
        "canonical_model_changed": False,
        "canonical_thresholds_changed": False,
        "canonical_staking_changed": False,
        "provider_requests_added": 0,
        "shadow_policy": "RELATIVE_STRENGTH_CHALLENGER_VISIBLE_LIVE_BUT_ZERO_DECISION_WEIGHT",
        "line_pricing_policy": "ONLY_EXACT_OBSERVED_FT_TOTAL_LINES_ALREADY_PRICED_BY_CANONICAL_MARKET_EVALUATOR",
        "promotion_policy": "NO_SHADOW_PROMOTION_UNTIL_OOS_VALIDATION_SUPPORTS_IT",
    }
    payload["v314_provider_requests_added"] = 0
    payload["v314_model_weights_changed"] = False
    payload["v314_canonical_bet_logic_changed"] = False
    payload["v314_ft_goals_checkpoint"] = "IMPROVED_OBSERVABILITY_PLUS_LIVE_SHADOW; CANONICAL_PRODUCTION_UNCHANGED"
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
