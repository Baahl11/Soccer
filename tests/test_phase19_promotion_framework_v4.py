from mcp_gateway import promotion_framework_v4 as v


def test_phase19_sample_policy_matches_master_roadmap():
    assert v.DIRECTIONAL_READ_MIN == 20
    assert v.TIER_B_REVIEW_MIN == 50
    assert v.TIER_A_REVIEW_MIN == 100
    assert v.TIER_S_REVIEW_MIN == 200
    assert v.MODEL_WEIGHT_CHANGE_MIN == 200


def test_raw_clv_rows_cannot_inflate_promotion_sample():
    review = v.review_market(
        market_family="1H",
        unique_fixtures=17,
        settled=100,
        roi_per_settled_unit=0.10,
        clv_rows=992,
        avg_clv_pp=0.5,
        stability_status="DATA_BLOCKED",
        validation_blockers=[],
    )
    assert review["recommended_state"] == "RESEARCH"
    assert review["tier_review_eligibility"]["directional_read"] is False


def test_phase19_never_auto_promotes_without_manual_approval():
    review = v.review_market(
        market_family="FT_TOTALS",
        unique_fixtures=80,
        settled=80,
        roi_per_settled_unit=0.12,
        clv_rows=160,
        avg_clv_pp=1.0,
        stability_status="STABILITY_REVIEW_READY",
        validation_blockers=[],
        manual_approval=False,
    )
    assert review["recommended_state"] == "LEAN_ELIGIBLE"
    assert "MANUAL_APPROVAL_REQUIRED_FOR_PRODUCTION_TIER" in review["warnings"]


def test_phase19_manual_approval_can_recommend_tier_by_unique_fixture_and_settlement_sample():
    review = v.review_market(
        market_family="FT_TOTALS",
        unique_fixtures=120,
        settled=120,
        roi_per_settled_unit=0.08,
        clv_rows=240,
        avg_clv_pp=0.5,
        stability_status="STABILITY_REVIEW_READY",
        validation_blockers=[],
        manual_approval=True,
    )
    assert review["recommended_state"] == "TIER_A"


def test_phase19_validation_blocker_holds_market_in_shadow_after_directional_sample():
    review = v.review_market(
        market_family="FT_CORNERS",
        unique_fixtures=60,
        settled=60,
        roi_per_settled_unit=0.05,
        clv_rows=200,
        avg_clv_pp=0.8,
        stability_status="STABILITY_REVIEW_READY",
        validation_blockers=["FORMATION_ADJUSTED_39_LT_100"],
    )
    assert review["recommended_state"] == "SHADOW"
    assert "VALIDATION:FORMATION_ADJUSTED_39_LT_100" in review["blockers"]


def test_phase19_flags_safety_demotion_only_for_production_collapse():
    review = v.review_market(
        market_family="FT_TOTALS",
        unique_fixtures=60,
        settled=60,
        roi_per_settled_unit=-0.05,
        clv_rows=120,
        avg_clv_pp=-0.2,
        stability_status="STABILITY_REVIEW_READY",
        validation_blockers=[],
        current_state="TIER_B",
    )
    assert review["automatic_demotion_candidate"] is True
    assert review["recommended_state"] == "DEMOTED"


def test_build_report_uses_g5_unique_fixtures():
    report = v.build_report(
        {"by_market_family": {"FT_1X2": {"settled": 10, "roi_per_decision_units": 1.12}}},
        {
            "families": {
                "1X2": {
                    "status": "DATA_BLOCKED",
                    "overall": {
                        "rows": 27,
                        "unique_fixtures": 16,
                        "fixture_weighted_avg_probability_clv_pp": 0.159329,
                    },
                }
            }
        },
        {"1X2": {"status": "RESEARCH_HOLD", "blockers": ["ACTIONABLE_SAMPLE_LT_20"]}},
    )
    reviews = {row["market_family"]: row for row in report["market_family_reviews"]}
    assert reviews["1X2"]["unique_fixtures"] == 16
    assert reviews["1X2"]["true_clv_rows"] == 27
    assert reviews["1X2"]["recommended_state"] == "RESEARCH"
    assert report["automatic_promotion_allowed"] is False
    assert report["runtime_state_mutation_enabled"] is False
