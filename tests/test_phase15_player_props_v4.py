from mcp_gateway import player_props_phase15_v4 as v


BASE = {
    "status": "PASS",
    "failures": [],
    "profiles_checked": 1000,
    "profiles_valid": 1000,
    "oos_validation_complete": False,
    "actionable": False,
    "decision_weight": 0,
    "validation_scope": "STRUCTURAL_ONLY_NOT_OOS_PERFORMANCE",
}


def test_phase15_blocks_structural_only_props():
    gk = dict(BASE)
    gk["profiles_checked"] = 22
    gk["profiles_valid"] = 22

    report = v.build_report(
        dict(BASE), dict(BASE), dict(BASE), dict(BASE), dict(BASE), gk, []
    )
    assert report["status"] == "RESEARCH_HOLD"
    assert "SHOTS_OOS_VALIDATION_INCOMPLETE" in report["blockers"]
    assert "GK_SAVES_VALID_PROFILES_22_LT_100" in report["blockers"]
    assert "PLAYER_PROP_TRUE_CLV_0_LT_50" in report["blockers"]
    assert "EXACT_OBSERVED_PROP_LINE_HISTORY_NOT_MATERIALIZED" in report["blockers"]
    assert report["production_promotion_allowed"] is False


def test_phase15_prop_clv_matcher_is_prop_specific():
    summary = v.summarize_true_clv([
        {"market": "Player Shots", "selection": "Player A Over 2.5", "fixture_id": 1, "clv_probability_pp": 0.02},
        {"market": "Goalkeeper Saves", "selection": "Player B Over 3.5", "fixture_id": 2, "clv_probability_pp": -0.01},
        {"market": "Goals Over/Under", "selection": "Over 2.5", "fixture_id": 3, "clv_probability_pp": 0.20},
    ])
    assert summary["rows"] == 2
    assert summary["unique_fixtures"] == 2
    assert summary["avg_probability_clv_pp"] == 0.005
