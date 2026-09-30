from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v128 as v128
from mcp_gateway import dynamic_strength_challenger_v4

MODEL_VERSION = v128.MODEL_VERSION
AUTOMATION_VERSION = "4.38.0-dynamic-strength-challenger"


async def run_tick() -> dict[str, Any]:
    payload = await v128.run_tick()
    events = payload.get("events")
    if not isinstance(events, list):
        events = []

    payload["v212_dynamic_strength_challenger"] = dynamic_strength_challenger_v4.build_report(events)
    payload["v212_checkpoint"] = (
        "DYNAMIC STRENGTH CHALLENGER ACTIVE IN RESEARCH ONLY: the shadow compares the existing "
        "blended raw projection with a recent-only counterfactual built from the same verified "
        "sporting snapshot and the same goal-rate Poisson math. At least three recent matches per "
        "team are required. It adds no provider calls, carries decision_weight=0, and cannot alter "
        "thresholds, gates, strict-close semantics, canonical BET logic, or production promotion."
    )
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
