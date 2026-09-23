from __future__ import annotations

from typing import Any

from mcp_gateway import alert_engine_v4
from mcp_gateway import automation_v116 as v116

MODEL_VERSION = v116.MODEL_VERSION
AUTOMATION_VERSION = "4.26.0-phase21.alert_engine"


def _annotate_phase21(payload: dict[str, Any]) -> None:
    payload["phase21_alert_engine"] = {
        "schema_version": "1.0.0",
        "status": "ENGINE_IMPLEMENTED_AUDIT_ONLY",
        "model_version": alert_engine_v4.MODEL_VERSION,
        "change_types": list(alert_engine_v4.CHANGE_TYPES),
        "expected_sections": [
            "fixture",
            "sport",
            "model",
            "best_market",
            "execution",
            "final",
        ],
        "github_role": "AUDIT_LOG",
        "webhook_app_role": "REAL_ALERT_DELIVERY",
        "change_detection_requires_previous_snapshot": True,
        "edge_threshold_must_be_explicit": True,
        "webhook_delivery_configured": False,
        "production_action_enabled": False,
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
        "runtime_prediction_weight": 0.0,
        "model_weights_changed": False,
        "canonical_bet_logic_changed": False,
        "note": (
            "Phase 21 implements the master alert format and change detectors for signal upgrade, "
            "price improvement, XI/GK confirmation, fresh quote, explicit edge threshold crossing, "
            "stale market, model disagreement and pick demotion. Delivery remains audit-only until "
            "a reviewed webhook/app destination and prior-snapshot store are connected."
        ),
    }
    payload["phase21_alert_audit_rows"] = []
    payload["phase21_provider_requests_added"] = 0
    payload["phase21_model_weights_changed"] = False
    payload["phase21_canonical_bet_logic_changed"] = False
    payload["phase21_checkpoint"] = (
        "ALERT ENGINE v4 IMPLEMENTED IN AUDIT-ONLY MODE. Change detection and the complete master "
        "alert payload/format exist, but no webhook is sent and no execution action is triggered "
        "until a reviewed destination plus previous-snapshot persistence are configured."
    )


async def run_tick() -> dict[str, Any]:
    payload = await v116.run_tick()
    _annotate_phase21(payload)
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
