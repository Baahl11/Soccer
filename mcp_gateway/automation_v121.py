from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v120 as v120
from mcp_gateway import product_dashboard_v4

MODEL_VERSION = v120.MODEL_VERSION
AUTOMATION_VERSION = "4.30.0-phase24.dashboard"


def _annotate_phase24_dashboard(payload: dict[str, Any]) -> None:
    current = payload.get("phase24_final_product_experience")
    if not isinstance(current, dict):
        current = {}
    current.update({
        "schema_version": "1.1.0",
        "status": "READ_ONLY_DASHBOARD_IMPLEMENTED",
        "dashboard_route": "/dashboard",
        "product_views_route": "/product/views",
        "dashboard_renderer_model_version": product_dashboard_v4.MODEL_VERSION,
        "ui_implementation_pending": False,
        "interactive_actions_enabled": False,
        "read_only": True,
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
        "runtime_prediction_weight": 0.0,
        "model_weights_changed": False,
        "canonical_bet_logic_changed": False,
        "note": (
            "Phase 24 now includes a live read-only dashboard route backed by the latest persisted "
            "Postgres pipeline payload plus the structured /product/views API. The UI remains "
            "non-executing and cannot place/promote bets or mutate model state."
        ),
    })
    payload["phase24_final_product_experience"] = current
    payload["phase24_checkpoint"] = (
        "FINAL PRODUCT READ-ONLY DASHBOARD IMPLEMENTED. /dashboard and /product/views expose the "
        "latest persisted operational views without new provider calls. No execution actions, "
        "model weights or canonical BET logic changed."
    )


async def run_tick() -> dict[str, Any]:
    payload = await v120.run_tick()
    _annotate_phase24_dashboard(payload)
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
