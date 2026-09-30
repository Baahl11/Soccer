from __future__ import annotations

import json
from pathlib import Path

from mcp_gateway import frozen_calibration_audit_v4 as v


def _row(*, calibrated=True, provider_update="2026-09-30T01:00:00Z", exact_payload=True):
    artifact = {
        "kind": "BINARY_PLATT",
        "target": "btts",
        "source_model_version": "M1",
        "source_model_version_matches": True,
        "fingerprint_sha256": "abc123",
    }
    if exact_payload:
        artifact["exact_calibrator_payload"] = {
            "kind": "BINARY_PLATT",
            "target": "btts",
            "source_model_version": "M1",
            "calibrator": {"status": "RESEARCH_CALIBRATOR_FITTED", "parameters": {"slope": 1.1}},
        }
    return {
        "model_version": "M1",
        "stage": "T-20",
        "p_raw": 0.60,
        "p_market_fair": 0.55,
        "provider_update": provider_update,
        "phase16_calibration_status": "RESEARCH_CALIBRATION_APPLIED" if calibrated else "BINARY_DISCRIMINATION_NOT_READY",
        "p_model_calibrated": 0.61 if calibrated else None,
        "phase16_calibrator_artifact": artifact if calibrated else None,
        "phase16_calibration_provenance": {
            "calibration_status": "RESEARCH_CALIBRATION_APPLIED" if calibrated else "BINARY_DISCRIMINATION_NOT_READY",
            "source_model_version": "M1" if calibrated else None,
            "phase16_calibrator_artifact": artifact if calibrated else None,
        },
    }


def test_complete_point_in_time_signal_is_ok():
    report = v.audit_rows([_row()])

    assert report["status"] == "OK_FROZEN_POINT_IN_TIME_COMPLETE"
    assert report["historical_signal_rows"] == 1
    assert report["point_in_time_complete_rows"] == 1
    assert report["point_in_time_complete_coverage"] == 1.0
    assert report["calibrated_rows_with_exact_calibrator_payload"] == 1
    assert report["source_model_version_match_counts"] == {"MATCH": 1}


def test_legacy_calibrated_signal_without_exact_payload_is_reported_not_rebuilt():
    report = v.audit_rows([_row(exact_payload=False)])

    assert report["status"] == "WATCH_FROZEN_POINT_IN_TIME_GAPS"
    assert report["gap_counts"]["MISSING_EXACT_CALIBRATOR_PAYLOAD"] == 1
    assert report["historical_probabilities_recomputed"] is False
    assert report["historical_rows_recalibrated"] is False


def test_missing_provider_update_is_an_explicit_point_in_time_gap():
    report = v.audit_rows([_row(provider_update=None)])

    assert report["status"] == "WATCH_FROZEN_POINT_IN_TIME_GAPS"
    assert report["gap_counts"]["MISSING_PROVIDER_UPDATE"] == 1


def test_non_calibrated_signal_does_not_require_calibrator_artifact():
    report = v.audit_rows([_row(calibrated=False)])

    assert report["calibrated_signal_rows"] == 0
    assert "MISSING_CALIBRATOR_ARTIFACT" not in report["gap_counts"]
    assert report["status"] == "OK_FROZEN_POINT_IN_TIME_COMPLETE"


def test_history_reader_uses_only_same_tick_persisted_values(tmp_path: Path):
    history = tmp_path / "history"
    history.mkdir()
    artifact = {
        "kind": "BINARY_PLATT",
        "target": "btts",
        "source_model_version": "M1",
        "source_model_version_matches": True,
        "fingerprint_sha256": "f" * 64,
        "exact_calibrator_payload": {
            "kind": "BINARY_PLATT",
            "target": "btts",
            "source_model_version": "M1",
            "calibrator": {"parameters": {"slope": 1.1}},
        },
    }
    tick = {
        "generated_at_local": "2026-09-29T19:00:00-06:00",
        "model_version": "M1",
        "match_table_rows": [
            {
                "fixture_id": 10,
                "market_family": "BTTS",
                "market": "Both Teams To Score",
                "selection": "Yes",
                "price": 1.90,
                "p_raw": 0.60,
                "p_model_calibrated": 0.61,
                "p_market_fair": 0.55,
                "provider_update": "2026-09-30T00:59:00Z",
                "phase16_calibration_status": "RESEARCH_CALIBRATION_APPLIED",
                "phase16_calibrator_artifact": artifact,
            }
        ],
        "events": [
            {
                "fixture": {"fixture_id": 10},
                "stage": "T-20",
                "best_market": {
                    "family": "BTTS",
                    "market": "Both Teams To Score",
                    "selection": "Yes",
                    "decimal_price": 1.90,
                    "p_raw": 0.60,
                    "p_market_fair": 0.55,
                },
            }
        ],
    }
    (history / "sample.jsonl").write_text(json.dumps(tick) + "\n", encoding="utf-8")

    rows = list(v.iter_history_signal_rows(str(history)))
    report = v.audit_rows(rows)

    assert len(rows) == 1
    assert rows[0]["model_version"] == "M1"
    assert rows[0]["stage"] == "T-20"
    assert rows[0]["p_raw"] == 0.60
    assert rows[0]["p_model_calibrated"] == 0.61
    assert rows[0]["p_market_fair"] == 0.55
    assert rows[0]["provider_update"] == "2026-09-30T00:59:00Z"
    assert rows[0]["phase16_calibrator_artifact"] == artifact
    assert report["status"] == "OK_FROZEN_POINT_IN_TIME_COMPLETE"


def test_safety_invariants_are_zero_call_research_only():
    report = v.audit_rows([_row()])

    assert report["provider_requests_added"] == 0
    assert report["provider_budget_changed"] is False
    assert report["decision_weight"] == 0.0
    assert report["production_promotion_allowed"] is False
    assert report["model_weights_changed"] is False
    assert report["thresholds_changed"] is False
    assert report["gates_changed"] is False
    assert report["strict_close_semantics_changed"] is False
