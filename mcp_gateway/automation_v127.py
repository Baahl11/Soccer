from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

from mcp_gateway import automation_v126 as v126
from mcp_gateway import price_resolver_v4

MODEL_VERSION = v126.MODEL_VERSION
AUTOMATION_VERSION = "4.36.1-frozen-point-in-time-calibration"


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _canonical_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _freeze_exact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a detached JSON-safe copy of the exact calibrator payload.

    This is provenance only. It prevents later in-memory mutation of calibration
    state from changing the artifact that was actually used at decision time.
    """
    return json.loads(json.dumps(payload, sort_keys=True, ensure_ascii=True))


def _calibrator_artifact(
    *,
    family: str,
    calibration_state: dict[str, Any],
    model_version: str | None,
) -> dict[str, Any] | None:
    family = str(family or "").upper()

    if family in {"BTTS", "FT_TOTALS"}:
        target = "btts" if family == "BTTS" else "over_2_5"
        report = _dict(calibration_state.get("binary"))
        source_model_version = str(report.get("current_source_model_version") or "") or None
        targets = _dict(report.get("current_model_deployment_calibrators"))
        target_report = _dict(targets.get(target))
        calibrator = _dict(target_report.get("calibrator"))
        if not calibrator:
            return None
        exact_payload = {
            "kind": "BINARY_PLATT",
            "target": target,
            "source_model_version": source_model_version,
            "calibrator": calibrator,
        }
        frozen_payload = _freeze_exact_payload(exact_payload)
        return {
            "kind": "BINARY_PLATT",
            "target": target,
            "source_model_version": source_model_version,
            "source_model_version_matches": source_model_version == str(model_version or ""),
            "calibrator_status": calibrator.get("status"),
            "parameter_keys": sorted(str(key) for key in calibrator.keys()),
            "fingerprint_sha256": _canonical_fingerprint(exact_payload),
            "fingerprint_basis": "CANONICAL_JSON_EXACT_CALIBRATOR_PAYLOAD",
            "exact_calibrator_payload": frozen_payload,
        }

    if family == "1X2":
        report = _dict(calibration_state.get("multiclass_1x2"))
        source_model_version = str(report.get("source_model_version") or "") or None
        calibrator = _dict(report.get("research_deployment_calibrator"))
        if not calibrator:
            return None
        exact_payload = {
            "kind": "MULTICLASS_TEMPERATURE",
            "target": "1X2",
            "source_model_version": source_model_version,
            "calibrator": calibrator,
        }
        frozen_payload = _freeze_exact_payload(exact_payload)
        return {
            "kind": "MULTICLASS_TEMPERATURE",
            "target": "1X2",
            "source_model_version": source_model_version,
            "source_model_version_matches": source_model_version == str(model_version or ""),
            "calibrator_status": calibrator.get("status"),
            "parameter_keys": sorted(str(key) for key in calibrator.keys()),
            "fingerprint_sha256": _canonical_fingerprint(exact_payload),
            "fingerprint_basis": "CANONICAL_JSON_EXACT_CALIBRATOR_PAYLOAD",
            "exact_calibrator_payload": frozen_payload,
        }

    return None


def _attach_artifact(
    row: dict[str, Any],
    *,
    family: str,
    calibration_state: dict[str, Any],
    model_version: str | None,
) -> bool:
    if str(row.get("phase16_calibration_status") or "") != "RESEARCH_CALIBRATION_APPLIED":
        return False
    artifact = _calibrator_artifact(
        family=family,
        calibration_state=calibration_state,
        model_version=model_version,
    )
    if not artifact:
        return False
    row["phase16_calibrator_artifact"] = artifact
    row["phase16_calibrator_artifact_frozen_at_decision_time"] = True
    return True


async def run_tick() -> dict[str, Any]:
    original_apply: Callable[..., float | None] = price_resolver_v4._apply_phase16_calibration
    counters = {
        "calibration_apply_calls": 0,
        "calibrated_rows_seen": 0,
        "artifact_rows_frozen": 0,
        "artifact_missing_after_calibration": 0,
    }

    def apply_with_frozen_artifact(
        row: dict[str, Any],
        event: dict[str, Any],
        *,
        family: str,
        selection: str,
        p_raw: float | None,
        calibration_state: dict[str, Any],
        model_version: str | None,
    ) -> float | None:
        counters["calibration_apply_calls"] += 1
        value = original_apply(
            row,
            event,
            family=family,
            selection=selection,
            p_raw=p_raw,
            calibration_state=calibration_state,
            model_version=model_version,
        )
        if value is not None and str(row.get("phase16_calibration_status") or "") == "RESEARCH_CALIBRATION_APPLIED":
            counters["calibrated_rows_seen"] += 1
            if _attach_artifact(
                row,
                family=family,
                calibration_state=calibration_state,
                model_version=model_version,
            ):
                counters["artifact_rows_frozen"] += 1
            else:
                counters["artifact_missing_after_calibration"] += 1
        return value

    price_resolver_v4._apply_phase16_calibration = apply_with_frozen_artifact
    try:
        payload = await v126.run_tick()
    finally:
        price_resolver_v4._apply_phase16_calibration = original_apply

    payload["v210_frozen_calibration_provenance"] = {
        "schema_version": "1.1.0",
        "status": (
            "FROZEN_POINT_IN_TIME_ARTIFACT_ACTIVE"
            if counters["artifact_missing_after_calibration"] == 0
            else "WATCH_CALIBRATED_ROW_WITHOUT_ARTIFACT_IDENTITY"
        ),
        **counters,
        "fingerprint_algorithm": "SHA256",
        "fingerprint_basis": "CANONICAL_JSON_EXACT_CALIBRATOR_PAYLOAD",
        "exact_calibrator_payload_persisted": True,
        "historical_rows_mutated": False,
        "historical_probabilities_recomputed": False,
        "provider_requests_added": 0,
        "provider_budget_changed": False,
        "decision_weight": 0.0,
        "production_promotion_allowed": False,
        "model_weights_changed": False,
        "thresholds_changed": False,
        "gates_changed": False,
        "canonical_bet_logic_changed": False,
        "strict_close_semantics_changed": False,
        "policy": (
            "FREEZE_EXACT_CALIBRATOR_PAYLOAD_AND_IDENTITY_AT_DECISION_TIME; "
            "NO_RETROACTIVE_RECALIBRATION; NO_PROVIDER_CALLS; NO_DECISION_WEIGHT; NO_PROMOTION"
        ),
    }
    payload["v210_checkpoint"] = (
        "FROZEN POINT-IN-TIME CALIBRATION PROVENANCE ACTIVE: whenever Phase16 applies a research "
        "calibrator, the exact calibrator payload and canonical fingerprint used in that call are "
        "frozen at decision time. Historical rows are not rewritten and probabilities are not recomputed."
    )
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
