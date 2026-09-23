from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_MLOPS_REGISTRY_V4_1.0.0"

REQUIRED_MODEL_FIELDS = (
    "model_id",
    "model_version",
    "feature_schema_version",
    "training_window",
    "validation_window",
    "metrics",
    "calibration_version",
    "artifact_hash",
    "lifecycle_state",
)

LIFECYCLE_STATES = ("ACTIVE", "CHALLENGER", "ARCHIVED")


def canonical_artifact_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_model_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_MODEL_FIELDS:
        value = record.get(field)
        if value is None or value == "" or (field == "metrics" and not isinstance(value, dict)):
            errors.append(f"MISSING_OR_INVALID_{field.upper()}")

    state = str(record.get("lifecycle_state") or "").upper()
    if state and state not in LIFECYCLE_STATES:
        errors.append("INVALID_LIFECYCLE_STATE")

    artifact_hash = str(record.get("artifact_hash") or "")
    if artifact_hash and (len(artifact_hash) != 64 or any(char not in "0123456789abcdef" for char in artifact_hash.lower())):
        errors.append("INVALID_ARTIFACT_HASH")

    training = record.get("training_window")
    validation = record.get("validation_window")
    for name, window in (("TRAINING", training), ("VALIDATION", validation)):
        if not isinstance(window, dict) or not window.get("start") or not window.get("end"):
            errors.append(f"INVALID_{name}_WINDOW")

    return sorted(set(errors))


def build_registry(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    source = [dict(record) for record in records if isinstance(record, dict)]
    validation: dict[str, list[str]] = {}
    active: list[str] = []
    challengers: list[str] = []
    archived: list[str] = []

    for record in source:
        model_id = str(record.get("model_id") or "UNKNOWN")
        errors = validate_model_record(record)
        validation[model_id] = errors
        state = str(record.get("lifecycle_state") or "").upper()
        if not errors:
            if state == "ACTIVE":
                active.append(model_id)
            elif state == "CHALLENGER":
                challengers.append(model_id)
            elif state == "ARCHIVED":
                archived.append(model_id)

    blockers: list[str] = []
    if any(validation.values()):
        blockers.append("INVALID_MODEL_RECORDS_PRESENT")
    if len(active) > 1:
        blockers.append("MULTIPLE_ACTIVE_MODELS")
    if not active:
        blockers.append("NO_ACTIVE_MODEL_REGISTERED")

    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": "REGISTRY_VALID" if not blockers else "REGISTRY_INCOMPLETE",
        "records": source,
        "validation": validation,
        "active_models": active,
        "challenger_models": challengers,
        "archived_models": archived,
        "blockers": blockers,
        "production_mutation_enabled": False,
    }


def promotion_plan(
    registry: dict[str, Any],
    *,
    challenger_model_id: str,
    manual_approval: bool,
    promotion_framework_state: str,
) -> dict[str, Any]:
    records = registry.get("records") if isinstance(registry.get("records"), list) else []
    challenger = next(
        (record for record in records if isinstance(record, dict) and str(record.get("model_id")) == challenger_model_id),
        None,
    )
    blockers: list[str] = []

    if challenger is None:
        blockers.append("CHALLENGER_NOT_FOUND")
    elif str(challenger.get("lifecycle_state") or "").upper() != "CHALLENGER":
        blockers.append("MODEL_NOT_REGISTERED_AS_CHALLENGER")
    elif validate_model_record(challenger):
        blockers.append("CHALLENGER_RECORD_INVALID")

    if not manual_approval:
        blockers.append("MANUAL_APPROVAL_REQUIRED")

    eligible_states = {"TIER_B", "TIER_A", "TIER_S"}
    if str(promotion_framework_state or "").upper() not in eligible_states:
        blockers.append("PHASE19_PRODUCTION_ELIGIBILITY_REQUIRED")

    active = registry.get("active_models") if isinstance(registry.get("active_models"), list) else []
    previous_active = active[0] if len(active) == 1 else None
    if previous_active is None:
        blockers.append("SINGLE_ACTIVE_MODEL_REQUIRED_FOR_ROLLBACK_POINTER")

    return {
        "status": "PROMOTION_PLAN_READY" if not blockers else "PROMOTION_BLOCKED",
        "challenger_model_id": challenger_model_id,
        "previous_active_model_id": previous_active,
        "manual_approval": bool(manual_approval),
        "phase19_state": promotion_framework_state,
        "blockers": blockers,
        "automatic_activation": False,
        "rollback_pointer_preserved": previous_active is not None,
    }


def rollback_plan(registry: dict[str, Any], *, target_model_id: str) -> dict[str, Any]:
    records = registry.get("records") if isinstance(registry.get("records"), list) else []
    target = next(
        (record for record in records if isinstance(record, dict) and str(record.get("model_id")) == target_model_id),
        None,
    )
    blockers: list[str] = []
    if target is None:
        blockers.append("ROLLBACK_TARGET_NOT_FOUND")
    elif validate_model_record(target):
        blockers.append("ROLLBACK_TARGET_INVALID")
    if target is not None and str(target.get("lifecycle_state") or "").upper() not in {"ACTIVE", "ARCHIVED"}:
        blockers.append("ROLLBACK_TARGET_MUST_BE_ACTIVE_OR_ARCHIVED")

    return {
        "status": "ROLLBACK_PLAN_READY" if not blockers else "ROLLBACK_BLOCKED",
        "target_model_id": target_model_id,
        "blockers": blockers,
        "automatic_rollback_execution": False,
        "manual_execution_required": True,
    }
