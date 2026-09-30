from __future__ import annotations

from mcp_gateway import frozen_calibration_audit_v4 as v


def _row(*, calibrated=True, artifact=True, match=True):
    provenance = {
        "calibration_status": (
            "RESEARCH_CALIBRATION_APPLIED" if calibrated else "BINARY_DISCRIMINATION_NOT_READY"
        ),
        "calibration_source": "CURRENT_MODEL_OOS_PLATT:BTTS",
        "calibration_policy": "BINARY_PLATT+BrierLogLossImprovement+AUC_L95_GT_0_50",
        "calibrated_probability_fields": {"p_model_calibrated": 0.61} if calibrated else {},
        "binary_calibration_diagnostics": {
            "source_model_version": "M1",
            "requested_model_version": "M1" if match else "M2",
            "source_model_version_matches": match,
        },
    }
    if artifact:
        provenance["phase16_calibrator_artifact"] = {
            "kind": "BINARY_PLATT",
            "target": "btts",
            "source_model_version": "M1",
            "source_model_version_matches": match,
            "fingerprint_sha256": "abc123",
        }
    return {"phase16_calibration_provenance": provenance}


def test_complete_frozen_artifact_identity_is_ok():
    report = v.audit_rows([_row()])

    assert report["status"] == "OK_FROZEN_ARTIFACT_IDENTITY_COMPLETE"
    assert report["calibrated_provenance_rows"] == 1
    assert report["calibrated_rows_with_immutable_artifact_fingerprint"] == 1
    assert report["immutable_artifact_coverage"] == 1.0
    assert report["source_model_version_match_counts"] == {"MATCH": 1}


def test_calibrated_row_without_artifact_is_watch():
    report = v.audit_rows([_row(artifact=False)])

    assert report["status"] == "WATCH_FROZEN_ARTIFACT_IDENTITY_GAPS"
    assert report["calibrated_rows_with_immutable_artifact_fingerprint"] == 0
    assert report["gap_counts"]["MISSING_CALIBRATOR_ARTIFACT"] == 1
    assert report["gap_counts"]["MISSING_IMMUTABLE_ARTIFACT_FINGERPRINT"] == 1


def test_non_calibrated_provenance_does_not_require_artifact():
    report = v.audit_rows([_row(calibrated=False, artifact=False)])

    assert report["status"] == "NOT_VERIFIED_NO_CALIBRATED_ROWS"
    assert report["rows_with_phase16_calibration_provenance"] == 1
    assert report["calibrated_provenance_rows"] == 0
    assert report["gap_counts"] == {}


def test_source_model_mismatch_is_counted_without_recomputing_history():
    report = v.audit_rows([_row(match=False)])

    assert report["source_model_version_match_counts"] == {"MISMATCH": 1}
    assert report["historical_rows_mutated"] is False
    assert report["historical_probabilities_recomputed"] is False


def test_safety_invariants_are_zero_call_research_only():
    report = v.audit_rows([_row()])

    assert report["provider_requests_added"] == 0
    assert report["decision_weight"] == 0.0
    assert report["production_promotion_allowed"] is False
    assert report["model_weights_changed"] is False
    assert report["thresholds_changed"] is False
    assert report["gates_changed"] is False
    assert report["strict_close_semantics_changed"] is False
