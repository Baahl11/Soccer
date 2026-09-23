from mcp_gateway import model_registry_v4 as v


def _record(model_id: str, state: str):
    artifact = {"model_id": model_id, "weights": [1, 2, 3]}
    return {
        "model_id": model_id,
        "model_version": "1.0.0",
        "feature_schema_version": "4.0.0",
        "training_window": {"start": "2026-01-01", "end": "2026-06-30"},
        "validation_window": {"start": "2026-07-01", "end": "2026-08-31"},
        "metrics": {"brier": 0.21, "log_loss": 0.61},
        "calibration_version": "platt-v1",
        "artifact_hash": v.canonical_artifact_hash(artifact),
        "lifecycle_state": state,
    }


def test_phase23_artifact_hash_is_deterministic():
    a = v.canonical_artifact_hash({"b": 2, "a": 1})
    b = v.canonical_artifact_hash({"a": 1, "b": 2})
    assert a == b
    assert len(a) == 64


def test_phase23_registry_requires_single_active_model():
    registry = v.build_registry([
        _record("active-1", "ACTIVE"),
        _record("challenger-1", "CHALLENGER"),
    ])
    assert registry["status"] == "REGISTRY_VALID"
    assert registry["active_models"] == ["active-1"]
    assert registry["challenger_models"] == ["challenger-1"]
    assert registry["production_mutation_enabled"] is False


def test_phase23_promotion_requires_manual_approval_and_phase19_state():
    registry = v.build_registry([
        _record("active-1", "ACTIVE"),
        _record("challenger-1", "CHALLENGER"),
    ])
    blocked = v.promotion_plan(
        registry,
        challenger_model_id="challenger-1",
        manual_approval=False,
        promotion_framework_state="LEAN_ELIGIBLE",
    )
    assert blocked["status"] == "PROMOTION_BLOCKED"
    assert "MANUAL_APPROVAL_REQUIRED" in blocked["blockers"]
    assert "PHASE19_PRODUCTION_ELIGIBILITY_REQUIRED" in blocked["blockers"]

    ready = v.promotion_plan(
        registry,
        challenger_model_id="challenger-1",
        manual_approval=True,
        promotion_framework_state="TIER_B",
    )
    assert ready["status"] == "PROMOTION_PLAN_READY"
    assert ready["previous_active_model_id"] == "active-1"
    assert ready["automatic_activation"] is False


def test_phase23_rollback_requires_registered_valid_target():
    registry = v.build_registry([
        _record("active-1", "ACTIVE"),
        _record("old-1", "ARCHIVED"),
    ])
    plan = v.rollback_plan(registry, target_model_id="old-1")
    assert plan["status"] == "ROLLBACK_PLAN_READY"
    assert plan["manual_execution_required"] is True
    assert plan["automatic_rollback_execution"] is False


def test_phase23_invalid_artifact_hash_is_blocked():
    row = _record("bad", "ACTIVE")
    row["artifact_hash"] = "not-a-sha"
    errors = v.validate_model_record(row)
    assert "INVALID_ARTIFACT_HASH" in errors
