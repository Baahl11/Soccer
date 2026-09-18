from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v66 as v66
from mcp_gateway import xg_intelligence

MODEL_VERSION = v66.MODEL_VERSION
AUTOMATION_VERSION = "3.43.1"


async def run_tick() -> dict[str, Any]:
    payload = await v66.run_tick()
    metrics = xg_intelligence.attach(payload)
    ready = int(metrics.get("live_research_events") or 0) > 0
    payload["xg_xga_intelligence"] = {
        "schema_version": xg_intelligence.SCHEMA_VERSION,
        **metrics,
        "provider_requests_added": 0,
        "canonical_model_weights_changed": False,
        "canonical_thresholds_changed": False,
        "canonical_bet_logic_changed": False,
        "production_status": "LIVE_RESEARCH_NOT_ACTIONABLE" if ready else "DORMANT_APPROVED_SOURCE_DATA_BLOCKED",
        "decision_weight": 0.0,
        "policy": (
            "API-FOOTBALL + GALAXYPARLAY ONLY; VERIFIED EXPLICIT xG REQUIRED; "
            "INTERNAL GOAL LAMBDAS/GOALS/SHOTS NEVER RELABELED xG; "
            "NO THIRD PROVIDER WITHOUT EXPLICIT PROJECT APPROVAL"
        ),
    }
    payload["v343_provider_requests_added"] = 0
    payload["v343_model_weights_changed"] = False
    payload["v343_canonical_bet_logic_changed"] = False
    payload["v343_xg_xga_checkpoint"] = (
        "APPROVED_SOURCE_VERIFIED_XG_LIVE_RESEARCH_ONLY"
        if ready
        else "XG_XGA_GUARD_BUILT; API_FOOTBALL_AND_CURRENT_GALAXYPARLAY_CONTRACT_DO_NOT_EXPOSE_VERIFIED_XG"
    )
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
