from __future__ import annotations

from mcp_gateway import automation_v127 as v


def _binary_state():
    return {
        "binary": {
            "current_source_model_version": "M1",
            "current_model_deployment_calibrators": {
                "btts": {
                    "calibrator": {
                        "status": "RESEARCH_CALIBRATOR_FITTED",
                        "parameters": {"intercept": -0.2, "slope": 1.1},
                    }
                }
            },
        }
    }


def _one_x_two_state():
    return {
        "multiclass_1x2": {
            "source_model_version": "M2",
            "research_deployment_calibrator": {
                "status": "RESEARCH_DEPLOYMENT_CALIBRATOR_FITTED",
                "temperature": 1.5,
            },
        }
    }


def test_binary_artifact_fingerprint_is_deterministic_and_model_bound():
    first = v._calibrator_artifact(
        family="BTTS",
        calibration_state=_binary_state(),
        model_version="M1",
    )
    second = v._calibrator_artifact(
        family="BTTS",
        calibration_state=_binary_state(),
        model_version="M1",
    )

    assert first is not None
    assert first == second
    assert first["kind"] == "BINARY_PLATT"
    assert first["target"] == "btts"
    assert first["source_model_version"] == "M1"
    assert first["source_model_version_matches"] is True
    assert len(first["fingerprint_sha256"]) == 64
    assert first["exact_calibrator_payload"]["calibrator"]["parameters"]["slope"] == 1.1


def test_exact_payload_is_detached_from_mutable_calibration_state():
    state = _binary_state()
    artifact = v._calibrator_artifact(family="BTTS", calibration_state=state, model_version="M1")
    state["binary"]["current_model_deployment_calibrators"]["btts"]["calibrator"]["parameters"]["slope"] = 9.9

    assert artifact is not None
    assert artifact["exact_calibrator_payload"]["calibrator"]["parameters"]["slope"] == 1.1


def test_fingerprint_changes_when_exact_calibrator_changes():
    state_a = _binary_state()
    state_b = _binary_state()
    state_b["binary"]["current_model_deployment_calibrators"]["btts"]["calibrator"]["parameters"]["slope"] = 1.2

    first = v._calibrator_artifact(family="BTTS", calibration_state=state_a, model_version="M1")
    second = v._calibrator_artifact(family="BTTS", calibration_state=state_b, model_version="M1")

    assert first["fingerprint_sha256"] != second["fingerprint_sha256"]
    assert first["exact_calibrator_payload"] != second["exact_calibrator_payload"]


def test_multiclass_artifact_uses_exact_temperature_payload():
    artifact = v._calibrator_artifact(
        family="1X2",
        calibration_state=_one_x_two_state(),
        model_version="M2",
    )

    assert artifact is not None
    assert artifact["kind"] == "MULTICLASS_TEMPERATURE"
    assert artifact["target"] == "1X2"
    assert artifact["source_model_version_matches"] is True
    assert len(artifact["fingerprint_sha256"]) == 64
    assert artifact["exact_calibrator_payload"]["calibrator"]["temperature"] == 1.5


def test_attach_requires_calibration_to_have_actually_been_applied():
    row = {"phase16_calibration_status": "CALIBRATION_NOT_AVAILABLE_FOR_CURRENT_MODEL"}
    attached = v._attach_artifact(
        row,
        family="BTTS",
        calibration_state=_binary_state(),
        model_version="M1",
    )

    assert attached is False
    assert "phase16_calibrator_artifact" not in row


def test_attach_marks_artifact_as_frozen_at_decision_time():
    row = {"phase16_calibration_status": "RESEARCH_CALIBRATION_APPLIED"}
    attached = v._attach_artifact(
        row,
        family="BTTS",
        calibration_state=_binary_state(),
        model_version="M1",
    )

    assert attached is True
    assert row["phase16_calibrator_artifact_frozen_at_decision_time"] is True
    assert row["phase16_calibrator_artifact"]["fingerprint_basis"] == "CANONICAL_JSON_EXACT_CALIBRATOR_PAYLOAD"
    assert row["phase16_calibrator_artifact"]["exact_calibrator_payload"]["target"] == "btts"
