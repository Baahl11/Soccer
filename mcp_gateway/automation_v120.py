from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v119 as v119
from mcp_gateway import product_views_v4

MODEL_VERSION = v119.MODEL_VERSION
AUTOMATION_VERSION = "4.29.0-phase24.product_views"


def _annotate_phase24(payload: dict[str, Any]) -> None:
    product = product_views_v4.build_views(payload, limit=25)
    payload["phase24_final_product_experience"] = {
        "schema_version": "1.0.0",
        "status": product["status"],
        "model_version": product_views_v4.MODEL_VERSION,
        "dashboard_views": list(product_views_v4.VIEW_NAMES),
        "row_limit_per_view": product["row_limit_per_view"],
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
        "runtime_prediction_weight": 0.0,
        "model_weights_changed": False,
        "canonical_bet_logic_changed": False,
        "ui_implementation_pending": True,
        "note": (
            "Phase 24 now exposes an API-ready dashboard contract for Today's Slate, Strong Sport Signals, "
            "Value Plays, Waiting for Price/XI, Team Totals, 1H, 2H, Corners, Player Props, Closing Line, "
            "Performance, Model Health and Data Health. A visual frontend remains separate work."
        ),
    }
    payload["dashboard_views"] = product["views"]
    payload["phase24_provider_requests_added"] = 0
    payload["phase24_model_weights_changed"] = False
    payload["phase24_canonical_bet_logic_changed"] = False
    payload["phase24_checkpoint"] = (
        "FINAL PRODUCT API VIEW CONTRACT IMPLEMENTED. All master dashboard views have bounded structured "
        "payloads; visual UI/UX is still pending. No provider calls, model weights or canonical BET logic changed."
    )


async def run_tick() -> dict[str, Any]:
    payload = await v119.run_tick()
    _annotate_phase24(payload)
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
