from __future__ import annotations

from typing import Any

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "FORMATION_FM7_PROMOTION_REVIEW_V1.0.0"
MIN_GRADED_BETS = 100


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def build_review(
    fm5_readiness: dict[str, Any],
    fm6_summary: dict[str, Any],
    prospective_summary: dict[str, Any],
) -> dict[str, Any]:
    readiness = _dict(fm5_readiness)
    market_validation = _dict(fm6_summary)
    prospective = _dict(prospective_summary)
    blockers: list[str] = []

    ready_components = [
        str(value) for value in (readiness.get("ready_components") or []) if value
    ]
    if readiness.get("integration_review_allowed") is not True or not ready_components:
        blockers.append("FM5_COMPONENT_NOT_REVIEW_READY")
    if readiness.get("market_prices_consumed") is not False:
        blockers.append("FM5_MARKET_LEAKAGE_DETECTED")

    if market_validation.get("exact_market_validation_ready") is not True:
        blockers.append("FM6_EXACT_MARKET_VALIDATION_NOT_READY")
    if market_validation.get("strict_close_semantics_clean") is not True:
        blockers.append("FM6_STRICT_CLOSE_NOT_CLEAN")
    if _int(market_validation.get("leakage_violations")) != 0:
        blockers.append("FM6_LEAKAGE_VIOLATIONS_PRESENT")
    if market_validation.get("true_clv_gate_passed") is not True:
        blockers.append("FM6_TRUE_CLV_GATE_NOT_PASSED")
    if market_validation.get("calibration_gate_passed") is not True:
        blockers.append("FM6_CALIBRATION_GATE_NOT_PASSED")
    if market_validation.get("multi_league_stability_passed") is not True:
        blockers.append("FM6_MULTI_LEAGUE_STABILITY_NOT_PASSED")
    if market_validation.get("concentration_gate_passed") is not True:
        blockers.append("FM6_CONCENTRATION_GATE_NOT_PASSED")
    if market_validation.get("stable_oos_lift") is not True:
        blockers.append("FM6_STABLE_OOS_LIFT_NOT_PASSED")

    endpoint = _dict(prospective.get("endpoint_summary"))
    graded = _dict(endpoint.get("graded_bet_sample"))
    graded_rows = _int(graded.get("rows"))
    graded_target = max(
        _int(endpoint.get("graded_bet_target")),
        MIN_GRADED_BETS,
    )
    if graded_rows < graded_target:
        blockers.append(
            f"PROSPECTIVE_GRADED_BETS_{graded_rows}_LT_{graded_target}"
        )
    if endpoint.get("material_recalibration_allowed") is not True:
        blockers.append("MATERIAL_RECALIBRATION_NOT_ALLOWED")
    if prospective.get("append_only_selection_policy") is not True:
        blockers.append("PROSPECTIVE_SELECTIONS_NOT_APPEND_ONLY")
    if prospective.get("outcomes_used_for_selection") is not False:
        blockers.append("PROSPECTIVE_OUTCOME_LEAKAGE_DETECTED")

    review_ready = not blockers
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": (
            "PRODUCTION_REVIEW_ELIGIBLE"
            if review_ready
            else "RESEARCH_ONLY"
        ),
        "ready_components": sorted(ready_components),
        "graded_bet_rows": graded_rows,
        "graded_bet_target": graded_target,
        "blockers": sorted(set(blockers)),
        "production_review_eligible": review_ready,
        "manual_approval_required": True,
        "automatic_activation_allowed": False,
        "production_enabled": False,
        "decision_weight": 0.0,
        "canonical_bet_logic_changed": False,
        "model_weights_changed": False,
        "historical_predictions_rewritten": False,
        "production_promotion_allowed": False,
    }


def build_activation_plan(
    review: dict[str, Any],
    *,
    manual_approval: bool,
    approved_model_id: str | None,
    rollback_model_id: str | None,
) -> dict[str, Any]:
    row = _dict(review)
    blockers: list[str] = []
    if row.get("model_version") != MODEL_VERSION:
        blockers.append("FM7_REVIEW_CONTRACT_REQUIRED")
    if row.get("production_review_eligible") is not True:
        blockers.append("FM7_PRODUCTION_REVIEW_NOT_ELIGIBLE")
    if not manual_approval:
        blockers.append("MANUAL_APPROVAL_REQUIRED")
    if not approved_model_id:
        blockers.append("APPROVED_MODEL_ID_REQUIRED")
    if not rollback_model_id:
        blockers.append("ROLLBACK_MODEL_ID_REQUIRED")

    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": (
            "PRODUCTION_ACTIVATION_PLAN_READY"
            if not blockers
            else "PRODUCTION_ACTIVATION_BLOCKED"
        ),
        "approved_model_id": approved_model_id,
        "rollback_model_id": rollback_model_id,
        "manual_approval": bool(manual_approval),
        "blockers": sorted(set(blockers)),
        "automatic_activation_allowed": False,
        "execution_performed": False,
        "rollback_pointer_preserved": bool(rollback_model_id),
        "production_enabled": False,
    }
