from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v87 as v87
from mcp_gateway import runtime_bet_demotion_guard

MODEL_VERSION = v87.MODEL_VERSION
AUTOMATION_VERSION = "3.61.0"


async def run_tick() -> dict[str, Any]:
    payload = await v87.run_tick()
    metrics = runtime_bet_demotion_guard.apply(payload)

    payload["v361_provider_requests_added"] = 0
    payload["v361_model_weights_changed"] = False
    payload["v361_runtime_promotion_added"] = False
    payload["v361_runtime_bet_demotion_guard"] = {
        **metrics,
        "settlement_basis": {
            "FT_TOTALS_BET": "7 settled, 3-4, -0.5652u",
            "2H_TOTALS_BET": "2 settled, 0-2, -0.72u",
            "2H_BTTS_BET": "1 settled, 0-1, -0.36u",
            "1H_OTHER_BET": "0 settled, 1 ungraded",
            "FT_TOTALS_LEAN": "8 settled, 8-0, +3.91u; preserved as LEAN/research accumulation, not promoted",
        },
        "runtime_effect": "TARGETED_BET_TO_WATCH_ONLY",
        "promotion_policy": "NO_TIER_B_A_S_OR_STAKE_INCREASE_UNTIL_PROMOTION_REVIEW_PASSES",
    }
    payload["v361_checkpoint"] = (
        "SETTLEMENT-AWARE RUNTIME DEMOTION GUARD: NEGATIVE/INSUFFICIENT-SAMPLE BET ROWS IN "
        "FT_TOTALS, 2H_TOTALS, 2H_BTTS AND 1H_OTHER ARE DOWNGRADED TO WATCH. LEAN ROWS ARE "
        "PRESERVED. NO PROVIDER CALLS, MODEL WEIGHTS, PROMOTIONS, TIERS OR STAKE INCREASES."
    )
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
