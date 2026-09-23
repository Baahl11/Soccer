from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v118 as v118
from mcp_gateway import model_registry_v4

MODEL_VERSION = v118.MODEL_VERSION
AUTOMATION_VERSION = "4.28.0-phase23.model_registry"


def _annotate_phase23(payload: dict[str, Any]) -> None:
    payload["phase23_mlops_model_registry"] = {
        "schema_version": "1.0.0",
        "status": "FRAMEWORK_IMPLEMENTED_REGISTRY_POPULATION_PENDING",
        "model_version": model_registry_v4.MODEL_VERSION,
        "required_model_fields": list(model_registry_v4.REQUIRED_MODEL_FIELDS),
        "lifecycle_states": list(model_registry_v4.LIFECYCLE_STATES),
        "artifact_hash_algorithm": "SHA256_CANONICAL_JSON",
        "single_active_model_required": True,
        "challenger_supported": True,
        "rollback_plan_supported": True,
        "manual_promotion_approval_required": True,
        "automatic_activation_enabled": False,
        "automatic_rollback_execution_enabled": False,
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
        "runtime_prediction_weight": 0.0,
        "model_weights_changed": False,
        "canonical_bet_logic_changed": False,
        "note": (
            "Phase 23 defines model version, feature-schema version, training/validation windows, "
            "metrics, calibration version, artifact hash, active/challenger lifecycle and rollback. "
            "The generic registry is implemented but production model records must still be populated "
            "from validated artifacts; activation remains manual."
        ),
    }
    payload["phase23_provider_requests_added"] = 0
    payload["phase23_model_weights_changed"] = False
    payload["phase23_canonical_bet_logic_changed"] = False
    payload["phase23_checkpoint"] = (
        "ML OPS/MODEL REGISTRY FRAMEWORK IMPLEMENTED. Canonical SHA256 artifact identity, "
        "ACTIVE/CHALLENGER/ARCHIVED lifecycle, manual promotion plan and rollback plan are defined. "
        "No runtime activation or model-weight change occurs automatically."
    )


async def run_tick() -> dict[str, Any]:
    payload = await v118.run_tick()
    _annotate_phase23(payload)
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
