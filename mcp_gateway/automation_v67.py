from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v66 as v66
from mcp_gateway import xg_intelligence

MODEL_VERSION = v66.MODEL_VERSION
AUTOMATION_VERSION = "3.43.0"


async def run_tick() -> dict[str, Any]:
    payload = await v66.run_tick()
    metrics = xg_intelligence.attach(payload)
    ready = metrics.get("registry_status") == "RESEARCH_XG_TEAM_REGISTRY"
    payload["xg_xga_intelligence"] = {
        "schema_version": xg_intelligence.SCHEMA_VERSION,
        **metrics,
        "provider_requests_added": 0,
        "canonical_model_weights_changed": False,
        "canonical_thresholds_changed": False,
        "canonical_bet_logic_changed": False,
        "production_status": "LIVE_RESEARCH_NOT_ACTIONABLE" if ready else "DORMANT_EXTERNAL_DATA_BLOCKED",
        "decision_weight": 0.0,
        "policy": "SPORTMONKS TYPE_ID_5304 EXTERNAL xG/xGA ONLY; INTERNAL GOALS/LAMBDAS NEVER RELABELED xG; ZERO CANONICAL WEIGHT UNTIL OOS FEATURE-LIFT PROMOTION",
    }
    payload["v343_provider_requests_added"] = 0
    payload["v343_model_weights_changed"] = False
    payload["v343_canonical_bet_logic_changed"] = False
    payload["v343_xg_xga_checkpoint"] = (
        "EXTERNAL_XG_REGISTRY_LIVE_RESEARCH_ONLY"
        if ready
        else "XG_XGA_ADAPTER_AND_REGISTRY_BUILT; EXTERNAL SPORTMONKS DATA/TOKEN_IMPORT_REQUIRED"
    )
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
