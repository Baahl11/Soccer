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
        shadow_settled=100,
        shadow_roi_per_settled_unit=0.05,
        shadow_sample_status="SHADOW_REVIEW_READY",
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
        shadow_settled=80,
        shadow_roi_per_settled_unit=0.05,
        shadow_sample_status="SHADOW_REVIEW_READY",
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
        shadow_settled=120,
        shadow_roi_per_settled_unit=0.04,
        shadow_sample_status="SHADOW_REVIEW_READY",
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
        shadow_settled=60,
        shadow_roi_per_settled_unit=0.03,
        shadow_sample_status="SHADOW_REVIEW_READY",
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
        shadow_settled=60,
        shadow_roi_per_settled_unit=-0.03,
        shadow_sample_status="SHADOW_REVIEW_READY",
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
        {
            "by_market_family": {
                "FT_1X2": {
                    "rows": 30,
                    "settled": 30,
                    "shadow_roi_hypothetical_units": 1.5,
                }
            }
        },
    )
    reviews = {row["market_family"]: row for row in report["market_family_reviews"]}
    assert reviews["1X2"]["unique_fixtures"] == 16
    assert reviews["1X2"]["true_clv_rows"] == 27
    assert reviews["1X2"]["recommended_state"] == "RESEARCH"
    assert report["automatic_promotion_allowed"] is False
    assert report["runtime_state_mutation_enabled"] is False


def test_negative_shadow_roi_blocks_lean_eligibility():
    review = v.review_market(
        market_family="1X2",
        unique_fixtures=80,
        settled=80,
        roi_per_settled_unit=0.10,
        clv_rows=100,
        avg_clv_pp=0.4,
        stability_status="STABILITY_REVIEW_READY",
        shadow_settled=170,
        shadow_roi_per_settled_unit=-0.16,
        shadow_sample_status="SHADOW_REVIEW_READY",
        validation_blockers=[],
    )
    assert review["recommended_state"] == "SHADOW"
    assert "PROMOTION_SHADOW_ROI_NOT_POSITIVE" in review["blockers"]


def test_combined_family_performance_aggregates_aliases():
    perf = v._performance_for_family(
        {
            "by_market_family": {
                "2H_BTTS": {"n": 1, "settled": 1, "roi_units": -0.36},
                "2H_TOTALS": {"n": 2, "settled": 2, "roi_units": -0.72},
            }
        },
        ("2H", "2H_TOTALS", "2H_BTTS"),
    )
    assert perf["n"] == 3
    assert perf["settled"] == 3
    assert perf["roi_units"] == -1.08


def test_watch_alert_roi_does_not_count_as_promotion_shadow():
    report = v.build_report(
        {"by_market_family": {"FT_1X2": {"n": 80, "settled": 80, "roi_units": 8.0}}},
        {
            "families": {
                "1X2": {
                    "status": "STABILITY_REVIEW_READY",
                    "overall": {
                        "rows": 100,
                        "unique_fixtures": 80,
                        "fixture_weighted_avg_probability_clv_pp": 0.4,
                    },
                }
            }
        },
        {"1X2": {"status": "CALIBRATION_REVIEW_ELIGIBLE", "blockers": []}},
        {
            "by_market_family": {
                "FT_1X2": {
                    "rows": 170,
                    "settled": 170,
                    "shadow_roi_hypothetical_units": -28.22,
                    "shadow_roi_per_settled_unit": -0.166,
                    "sample_status": "SHADOW_REVIEW_READY",
                }
            }
        },
        {
            "market_family": "FT_1X2",
            "promotion_evaluable": {
                "settled": 0,
                "roi_per_settled_unit": None,
                "sample_status": "DATA_BLOCKED",
            },
        },
    )
    review = next(row for row in report["market_family_reviews"] if row["market_family"] == "1X2")
    assert review["recommended_state"] == "SHADOW"
    assert "PROMOTION_SHADOW_SETTLED_0_LT_DIRECTIONAL_20" in review["blockers"]
    assert "PROMOTION_SHADOW_ROI_NOT_POSITIVE" not in review["blockers"]
    assert review["watch_shadow_settled"] == 170
    assert review["watch_shadow_roi_per_settled_unit"] == -0.166


def test_clean_promotion_shadow_can_support_lean_eligibility():
    report = v.build_report(
        {"by_market_family": {"FT_1X2": {"n": 80, "settled": 80, "roi_units": 8.0}}},
        {
            "families": {
                "1X2": {
                    "status": "STABILITY_REVIEW_READY",
                    "overall": {
                        "rows": 100,
                        "unique_fixtures": 80,
                        "fixture_weighted_avg_probability_clv_pp": 0.4,
                    },
                }
            }
        },
        {"1X2": {"status": "CALIBRATION_REVIEW_ELIGIBLE", "blockers": []}},
        {},
        {
            "market_family": "FT_1X2",
            "promotion_evaluable": {
                "settled": 60,
                "roi_per_settled_unit": 0.08,
                "sample_status": "SHADOW_REVIEW_READY",
            },
        },
    )
    review = next(row for row in report["market_family_reviews"] if row["market_family"] == "1X2")
    assert review["recommended_state"] == "LEAN_ELIGIBLE"
    assert review["promotion_shadow_settled"] == 60
