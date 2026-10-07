from copy import deepcopy

from mcp_gateway import formation_fm5_raw_projection_v1 as fm5


def _baseline():
    return {
        "status": "MODELED_LIMITED",
        "model_version": "SOCCER EDGE ENGINE v1.6",
        "projection_model": "SOCCER_VERIFIED_GOAL_RATE_BASELINE",
        "raw_home_goal_rate": 1.5,
        "raw_away_goal_rate": 1.1,
        "raw_total_goals": 2.6,
        "raw_home_win_prob": 0.46,
        "raw_draw_prob": 0.27,
        "raw_away_win_prob": 0.27,
        "raw_btts_yes_prob": 0.55,
        "raw_over_1_5_prob": 0.74,
        "raw_over_2_5_prob": 0.49,
        "raw_over_3_5_prob": 0.27,
        "sample": {"minimum_split_sample": 12},
    }


def _blocked_gate():
    return {
        "model_version": "FORMATION_FM5_READINESS_GATE_V1.0.0",
        "status": "FM5_BLOCKED_EVIDENCE_GATES",
        "ready_components": [],
        "integration_review_allowed": False,
        "market_prices_consumed": False,
        "production_enabled": False,
    }


def _ready_gate():
    return {
        "model_version": "FORMATION_FM5_READINESS_GATE_V1.0.0",
        "status": "FM5_COMPONENT_REVIEW_READY",
        "ready_components": ["FM4_STYLE_GOALS"],
        "integration_review_allowed": True,
        "market_prices_consumed": False,
        "production_enabled": False,
    }


def _adjustment():
    return {
        "source_component": "FM4_STYLE_GOALS",
        "source_model_version": "FORMATION_MATCHUP_FM4_STYLE_ABLATION_V1.0.0",
        "source_artifact_id": "fixture:123:fm4-style-goals-v1",
        "feature_as_of": "2026-10-07T17:00:00+00:00",
        "fixture_kickoff": "2026-10-07T18:00:00+00:00",
        "prior_only": True,
        "oos_gate_passed": True,
        "market_fields_used": False,
        "manual_research_approval": True,
        "adjusted_home_goal_rate": 1.62,
        "adjusted_away_goal_rate": 1.04,
    }


def test_blocked_gate_preserves_canonical_baseline_and_zero_weight():
    baseline = _baseline()
    report = fm5.build_research_candidate(
        baseline,
        _blocked_gate(),
        None,
        availability_confidence=0.92,
    )

    assert report["status"] == "FM5_BLOCKED"
    assert report["research_candidate_available"] is False
    assert report["fm5_raw_projection_research"] is None
    assert report["baseline_raw_projection"] == baseline
    assert report["canonical_raw_projection_changed"] is False
    assert report["decision_weight"] == 0.0
    assert report["production_enabled"] is False
    assert report["market_evaluation_enabled"] is False
    assert report["production_promotion_allowed"] is False


def test_valid_prior_only_adjustment_builds_research_candidate_not_production():
    report = fm5.build_research_candidate(
        _baseline(),
        _ready_gate(),
        _adjustment(),
        availability_confidence=0.92,
    )

    candidate = report["fm5_raw_projection_research"]
    assert report["status"] == "FM5_RESEARCH_CANDIDATE_AVAILABLE"
    assert report["research_candidate_available"] is True
    assert report["ready_component"] == "FM4_STYLE_GOALS"
    assert report["decision_weight"] == 0.0
    assert report["production_enabled"] is False
    assert report["automatic_production_activation_allowed"] is False
    assert report["market_evaluation_enabled"] is False
    assert report["canonical_raw_projection_changed"] is False
    assert report["production_promotion_allowed"] is False
    assert candidate["market_independent"] is True
    assert candidate["prior_only"] is True
    assert candidate["oos_gate_passed"] is True
    assert candidate["raw_home_goal_rate"] == 1.62
    assert candidate["raw_away_goal_rate"] == 1.04
    assert candidate["raw_total_goals"] == 2.66
    assert 0.0 < candidate["raw_home_win_prob"] < 1.0
    assert 0.0 < candidate["raw_draw_prob"] < 1.0
    assert 0.0 < candidate["raw_away_win_prob"] < 1.0
    assert round(
        candidate["raw_home_win_prob"]
        + candidate["raw_draw_prob"]
        + candidate["raw_away_win_prob"],
        5,
    ) == 1.0


def test_adjustment_from_not_ready_component_is_blocked():
    adjustment = _adjustment()
    adjustment["source_component"] = "FM4_STYLE_SHOTS"

    report = fm5.build_research_candidate(
        _baseline(),
        _ready_gate(),
        adjustment,
    )

    assert report["status"] == "FM5_BLOCKED"
    assert "SOURCE_COMPONENT_NOT_READINESS_APPROVED" in report["blockers"]


def test_market_or_price_fields_in_adjustment_are_rejected():
    adjustment = _adjustment()
    adjustment["market_price"] = 1.91

    report = fm5.build_research_candidate(
        _baseline(),
        _ready_gate(),
        adjustment,
    )

    assert report["status"] == "FM5_BLOCKED"
    assert any(
        blocker.startswith("FORBIDDEN_MARKET_FIELD_KEYS:")
        for blocker in report["blockers"]
    )


def test_feature_timestamp_must_be_strictly_before_kickoff():
    adjustment = _adjustment()
    adjustment["feature_as_of"] = adjustment["fixture_kickoff"]

    report = fm5.build_research_candidate(
        _baseline(),
        _ready_gate(),
        adjustment,
    )

    assert report["status"] == "FM5_BLOCKED"
    assert "FEATURE_TIMESTAMP_NOT_STRICTLY_BEFORE_KICKOFF" in report["blockers"]


def test_manual_research_approval_is_required_even_when_evidence_gate_passes():
    adjustment = _adjustment()
    adjustment["manual_research_approval"] = False

    report = fm5.build_research_candidate(
        _baseline(),
        _ready_gate(),
        adjustment,
    )

    assert report["status"] == "FM5_BLOCKED"
    assert "MANUAL_RESEARCH_APPROVAL_REQUIRED" in report["blockers"]


def test_inputs_are_not_mutated():
    baseline = _baseline()
    readiness = _ready_gate()
    adjustment = _adjustment()
    before_baseline = deepcopy(baseline)
    before_readiness = deepcopy(readiness)
    before_adjustment = deepcopy(adjustment)

    fm5.build_research_candidate(
        baseline,
        readiness,
        adjustment,
        availability_confidence=0.9,
    )

    assert baseline == before_baseline
    assert readiness == before_readiness
    assert adjustment == before_adjustment


def test_public_view_removes_private_distribution_only():
    report = fm5.build_research_candidate(
        _baseline(),
        _ready_gate(),
        _adjustment(),
    )
    public = fm5.public_view(report)

    assert "_total_dist" in report["fm5_raw_projection_research"]
    assert "_total_dist" not in public["fm5_raw_projection_research"]
    assert public["decision_weight"] == 0.0
    assert public["production_enabled"] is False
