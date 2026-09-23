from mcp_gateway import promotion_framework_v4 as v


def test_phase19_sample_policy_matches_master_roadmap():
    assert v.DIRECTIONAL_READ_MIN == 20
    assert v.TIER_B_REVIEW_MIN == 50
    assert v.TIER_A_REVIEW_MIN == 100
    assert v.TIER_S_REVIEW_MIN == 200
    assert v.MODEL_WEIGHT_CHANGE_MIN == 200


def test_phase19_never_auto_promotes_without_manual_approval():
    review = v.review_market(
        market_family="FT_TOTALS",
        settled=250,
        roi_per_settled_unit=0.12,
        clv_rows=250,
        avg_clv_pp=1.0,
        oos_framework_ready=True,
        current_state="RESEARCH",
        manual_approval=False,
    )
    assert review["recommended_state"] == "LEAN_ELIGIBLE"
    assert "MANUAL_APPROVAL_REQUIRED_FOR_PRODUCTION_TIER" in review["warnings"]


def test_phase19_manual_approval_can_recommend_tier_by_sample():
    review = v.review_market(
        market_family="FT_TOTALS",
        settled=120,
        roi_per_settled_unit=0.08,
        clv_rows=120,
        avg_clv_pp=0.5,
        oos_framework_ready=True,
        current_state="SHADOW",
        manual_approval=True,
    )
    assert review["recommended_state"] == "TIER_A"


def test_phase19_flags_safety_demotion_only_for_production_collapse():
    review = v.review_market(
        market_family="FT_TOTALS",
        settled=60,
        roi_per_settled_unit=-0.05,
        clv_rows=60,
        avg_clv_pp=-0.2,
        oos_framework_ready=True,
        current_state="TIER_B",
        manual_approval=False,
    )
    assert review["automatic_demotion_candidate"] is True
    assert review["recommended_state"] == "DEMOTED"


def test_phase19_current_small_samples_stay_research_or_shadow():
    report = v.build_report(
        {
            "by_market_family": {
                "FT_1X2": {"settled": 10, "roi_per_decision_units": 1.12},
                "FT_TOTALS": {"settled": 15, "roi_per_decision_units": 0.223},
            }
        },
        {
            "by_market": {
                "Match Winner": {"rows": 97, "avg_probability_clv_pp": 0.16},
            }
        },
        {"status": "OOS_FRAMEWORK_DATA_GAPS"},
    )
    states = {row["market_family"]: row["recommended_state"] for row in report["market_family_reviews"]}
    assert states["FT_1X2"] == "RESEARCH"
    assert states["FT_TOTALS"] == "RESEARCH"
    assert report["automatic_promotion_allowed"] is False
    assert report["runtime_state_mutation_enabled"] is False
