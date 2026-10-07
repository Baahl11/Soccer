from mcp_gateway import formation_fm7_promotion_review_v1 as fm7


def _readiness(ready=True):
    return {
        "ready_components": ["FM4_STYLE_GOALS"] if ready else [],
        "integration_review_allowed": ready,
        "market_prices_consumed": False,
    }


def _fm6(ready=True):
    return {
        "exact_market_validation_ready": ready,
        "strict_close_semantics_clean": True,
        "leakage_violations": 0,
        "true_clv_gate_passed": True,
        "calibration_gate_passed": True,
        "multi_league_stability_passed": True,
        "concentration_gate_passed": True,
        "stable_oos_lift": True,
    }


def _prospective(rows=100, allowed=True):
    return {
        "append_only_selection_policy": True,
        "outcomes_used_for_selection": False,
        "endpoint_summary": {
            "graded_bet_sample": {"rows": rows},
            "graded_bet_target": 100,
            "material_recalibration_allowed": allowed,
        },
    }


def test_current_insufficient_sample_blocks_production_review():
    report = fm7.build_review(
        _readiness(),
        _fm6(),
        _prospective(rows=12, allowed=False),
    )
    assert report["status"] == "RESEARCH_ONLY"
    assert report["production_review_eligible"] is False
    assert report["production_enabled"] is False
    assert report["decision_weight"] == 0.0
    assert report["automatic_activation_allowed"] is False
    assert "PROSPECTIVE_GRADED_BETS_12_LT_100" in report["blockers"]
    assert "MATERIAL_RECALIBRATION_NOT_ALLOWED" in report["blockers"]


def test_all_evidence_gates_only_reach_manual_production_review():
    report = fm7.build_review(
        _readiness(),
        _fm6(),
        _prospective(rows=120, allowed=True),
    )
    assert report["status"] == "PRODUCTION_REVIEW_ELIGIBLE"
    assert report["production_review_eligible"] is True
    assert report["blockers"] == []
    assert report["manual_approval_required"] is True
    assert report["automatic_activation_allowed"] is False
    assert report["production_enabled"] is False
    assert report["production_promotion_allowed"] is False


def test_market_validation_failure_blocks_review_even_with_sample():
    fm6 = _fm6()
    fm6["true_clv_gate_passed"] = False
    fm6["multi_league_stability_passed"] = False

    report = fm7.build_review(
        _readiness(),
        fm6,
        _prospective(rows=150, allowed=True),
    )

    assert report["production_review_eligible"] is False
    assert "FM6_TRUE_CLV_GATE_NOT_PASSED" in report["blockers"]
    assert "FM6_MULTI_LEAGUE_STABILITY_NOT_PASSED" in report["blockers"]


def test_append_only_and_outcome_isolation_are_mandatory():
    prospective = _prospective(rows=150, allowed=True)
    prospective["append_only_selection_policy"] = False
    prospective["outcomes_used_for_selection"] = True

    report = fm7.build_review(_readiness(), _fm6(), prospective)

    assert "PROSPECTIVE_SELECTIONS_NOT_APPEND_ONLY" in report["blockers"]
    assert "PROSPECTIVE_OUTCOME_LEAKAGE_DETECTED" in report["blockers"]


def test_activation_plan_requires_manual_approval_and_rollback_pointer():
    review = fm7.build_review(
        _readiness(),
        _fm6(),
        _prospective(rows=150, allowed=True),
    )

    blocked = fm7.build_activation_plan(
        review,
        manual_approval=False,
        approved_model_id="fm5-goals-v1",
        rollback_model_id=None,
    )
    assert blocked["status"] == "PRODUCTION_ACTIVATION_BLOCKED"
    assert "MANUAL_APPROVAL_REQUIRED" in blocked["blockers"]
    assert "ROLLBACK_MODEL_ID_REQUIRED" in blocked["blockers"]
    assert blocked["execution_performed"] is False
    assert blocked["production_enabled"] is False

    ready = fm7.build_activation_plan(
        review,
        manual_approval=True,
        approved_model_id="fm5-goals-v1",
        rollback_model_id="soccer-edge-v1.7",
    )
    assert ready["status"] == "PRODUCTION_ACTIVATION_PLAN_READY"
    assert ready["blockers"] == []
    assert ready["automatic_activation_allowed"] is False
    assert ready["execution_performed"] is False
    assert ready["rollback_pointer_preserved"] is True
    assert ready["production_enabled"] is False
