from mcp_gateway import formation_lifecycle_health_v1 as health


def _fm4(style_n=2, prior_xi=0):
    return {
        "model_version": "FORMATION_MATCHUP_FM4_STYLE_ABLATION_V1.1.0",
        "status": "RESEARCH_HOLD_FM4_STYLE_PERSONNEL_ABLATION",
        "style_profile": {
            "prior_density": {"rows_with_both_min_field_n_ge_3": style_n}
        },
        "personnel_overlay": {
            "coverage": {
                "current_both_xi_confirmed_rows": 18,
                "rows_with_both_prior_confirmed_xi": prior_xi,
            },
            "outcome_ablation_ready": False,
        },
    }


def _fm5(ready=False):
    return {
        "model_version": "FORMATION_FM5_READINESS_GATE_V1.0.0",
        "status": (
            "FM5_COMPONENT_REVIEW_READY"
            if ready
            else "FM5_BLOCKED_EVIDENCE_GATES"
        ),
        "ready_components": ["FM4_STYLE_GOALS"] if ready else [],
        "integration_review_allowed": ready,
        "fm4": {"minimum_prior_style_rows": 100},
        "blockers": [] if ready else ["FM4_STYLE_HISTORY_2_LT_100"],
    }


def _prospective(rows=0, allowed=False):
    return {
        "endpoint_summary": {
            "graded_bet_sample": {"rows": rows},
            "graded_bet_target": 100,
            "material_recalibration_allowed": allowed,
        }
    }


def test_lifecycle_distinguishes_software_complete_from_evidence_blocked():
    report = health.build_health(
        fm4_report=_fm4(),
        fm5_readiness=_fm5(),
        prospective=_prospective(),
        personnel_backfill=None,
    )

    assert report["status"] == "SOFTWARE_CONTRACTS_COMPLETE_EVIDENCE_ACCUMULATING"
    assert report["software_contracts_complete"] is True
    assert report["phases"]["FM5"]["status"] == "FM5_BLOCKED_EVIDENCE_GATES"
    assert report["phases"]["FM6"]["status"] == "BLOCKED_WAITING_FM5_SPORTING_COMPONENT"
    assert report["phases"]["FM7"]["status"] == "BLOCKED_UPSTREAM_FM5"
    assert "FM4_STYLE_HISTORY_2_LT_100" in report["evidence_blockers"]
    assert "FM4_PERSONNEL_BACKFILL_ARTIFACT_NOT_MATERIALIZED" in report["evidence_blockers"]
    assert "PROSPECTIVE_GRADED_BETS_0_LT_100" in report["evidence_blockers"]
    assert report["production_promotion_allowed"] is False
    assert report["provider_requests_added"] == 0


def test_ready_fm5_only_moves_fm6_to_research_not_production():
    report = health.build_health(
        fm4_report=_fm4(style_n=120, prior_xi=100),
        fm5_readiness=_fm5(ready=True),
        prospective=_prospective(rows=40, allowed=False),
        personnel_backfill={
            "model_version": "SOCCER_FM4_PERSONNEL_HISTORY_BACKFILL_V1.0.0",
            "status": "RESEARCH_BACKFILL_COMPLETE",
            "captured": 8,
            "materialized_personnel_history_fixture_count": 120,
        },
    )

    assert report["phases"]["FM6"]["status"] == "READY_FOR_EXACT_MARKET_RESEARCH"
    assert report["phases"]["FM6"]["production_enabled"] is False
    assert report["phases"]["FM6"]["decision_weight"] == 0.0
    assert report["phases"]["FM7"]["status"] == "AWAITING_FM6_VALIDATION_AND_PROSPECTIVE_SAMPLE"
    assert report["phases"]["FM7"]["automatic_activation_allowed"] is False


def test_personnel_artifact_is_reported_without_changing_decision_weight():
    report = health.build_health(
        fm4_report=_fm4(),
        fm5_readiness=_fm5(),
        prospective=_prospective(),
        personnel_backfill={
            "model_version": "SOCCER_FM4_PERSONNEL_HISTORY_BACKFILL_V1.0.0",
            "status": "RESEARCH_BACKFILL_COMPLETE",
            "captured": 4,
            "materialized_personnel_history_fixture_count": 55,
        },
    )

    fm4 = report["phases"]["FM4"]
    assert fm4["personnel_backfill_artifact_materialized"] is True
    assert fm4["personnel_backfill_last_run_captured"] == 4
    assert fm4["personnel_history_fixture_count"] == 55
    assert fm4["decision_weight"] == 0.0
    assert "FM4_PERSONNEL_BACKFILL_ARTIFACT_NOT_MATERIALIZED" not in report["evidence_blockers"]


def test_lifecycle_never_claims_production_from_sample_count_alone():
    report = health.build_health(
        fm4_report=_fm4(style_n=150, prior_xi=120),
        fm5_readiness=_fm5(ready=True),
        prospective=_prospective(rows=150, allowed=True),
        personnel_backfill={
            "model_version": "SOCCER_FM4_PERSONNEL_HISTORY_BACKFILL_V1.0.0",
            "status": "RESEARCH_BACKFILL_COMPLETE",
            "captured": 8,
            "materialized_personnel_history_fixture_count": 160,
        },
    )

    assert report["software_contracts_complete"] is True
    assert report["phases"]["FM5"]["production_enabled"] is False
    assert report["phases"]["FM6"]["production_enabled"] is False
    assert report["phases"]["FM7"]["production_enabled"] is False
    assert report["production_promotion_allowed"] is False
    assert report["model_weights_changed"] is False
    assert report["canonical_bet_logic_changed"] is False


def test_stale_render_report_does_not_count_as_personnel_history_artifact():
    report = health.build_health(
        fm4_report=_fm4(),
        fm5_readiness=_fm5(),
        prospective=_prospective(),
        personnel_backfill={
            "model_version": "FM4_PERSONNEL_DEPLOYMENT_BLOCK_V1.0.0",
            "status": "BLOCKED_STALE_RENDER_RUNTIME",
            "captured": 0,
            "materialized_personnel_history_fixture_count": 0,
        },
    )

    assert report["phases"]["FM4"]["personnel_backfill_artifact_materialized"] is False
    assert "FM4_PERSONNEL_BACKFILL_ARTIFACT_NOT_MATERIALIZED" in report["evidence_blockers"]
