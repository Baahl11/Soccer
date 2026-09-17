from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v57 as v57
from mcp_gateway import goalkeeper_intelligence, postgame_player_capture

MODEL_VERSION = v57.MODEL_VERSION
AUTOMATION_VERSION = "3.34.0"


async def run_tick() -> dict[str, Any]:
    payload = await v57.run_tick()

    # Low-priority finalized player capture runs only after all existing pregame
    # work and only when the shared budget layer reports NORMAL mode + spare calls.
    capture_metrics = await postgame_player_capture.attach(payload)
    gk_metrics = goalkeeper_intelligence.attach(payload)

    payload["goalkeeper_intelligence"] = {
        "schema_version": goalkeeper_intelligence.SCHEMA_VERSION,
        **gk_metrics,
        "postgame_player_capture": capture_metrics,
        "canonical_goal_lambda_adjustment": 0.0,
        "canonical_model_weights_changed": False,
        "canonical_thresholds_changed": False,
        "canonical_bet_logic_changed": False,
        "production_status": "LIVE_RESEARCH_CONTEXT_ONLY",
        "decision_weight": 0.0,
        "model_status": "DESCRIPTIVE_PROFILE_ONLY; TRUE_SHOT_STOPPING_IMPACT_MODEL_PENDING",
        "policy": "CONFIRMED STARTER + FINALIZED SAVES/CONCEDED HISTORY WHEN AVAILABLE; NO PSxG FABRICATION; NO CANONICAL LAMBDA CHANGE",
    }
    payload["v334_provider_requests_added_max"] = postgame_player_capture.MAX_POSTGAME_PLAYER_FIXTURES_PER_TICK
    payload["v334_provider_requests_added_this_tick"] = int(capture_metrics.get("provider_requests_added") or 0)
    payload["v334_postgame_capture_budget_policy"] = "NORMAL_ONLY_LOW_PRIORITY_KEEP_3_CALLS_FREE"
    payload["v334_model_weights_changed"] = False
    payload["v334_canonical_bet_logic_changed"] = False
    payload["v334_goalkeeper_checkpoint"] = "GK_DATA_AND_LIVE_PROFILE_IMPROVED; TRUE_IMPACT_MODEL_NOT_CLAIMED"
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
