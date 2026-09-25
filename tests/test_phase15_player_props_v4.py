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
    assert "SHOTS_OBSERVED_MARKET_PRICE_HISTORY_MISSING" in report["blockers"]
    assert "SHOTS_EXACT_LINE_HISTORY_MISSING" in report["blockers"]
    assert "ASSISTS_CONFIRMED_XI_MARKET_OVERLAP_MISSING" in report["blockers"]
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



def test_phase15_market_audit_is_family_specific_and_line_aware():
    audit = {
        "model_version": "SOCCER_RESEARCH_DERIVATIVE_MARKET_AUDIT_V4_1.0.0",
        "families": {
            "SHOTS": {
                "market_snapshot_rows": 10,
                "priced_value_rows": 20,
                "exact_line_value_rows": 20,
                "unique_fixtures": 4,
                "pre_kickoff_unique_fixtures": 4,
                "provider_update_unique_fixtures": 4,
                "confirmed_xi_pre_kickoff_unique_fixtures": 2,
                "bookmaker_count": 2,
                "exact_observed_market_history_materialized": True,
                "confirmed_xi_overlap_materialized": True,
            },
            "ASSISTS": {
                "market_snapshot_rows": 35,
                "priced_value_rows": 57,
                "exact_line_value_rows": 0,
                "unique_fixtures": 4,
                "pre_kickoff_unique_fixtures": 4,
                "provider_update_unique_fixtures": 4,
                "confirmed_xi_pre_kickoff_unique_fixtures": 1,
                "bookmaker_count": 2,
                "exact_observed_market_history_materialized": True,
                "confirmed_xi_overlap_materialized": True,
            },
        },
    }
    report = v.build_report(
        dict(BASE), dict(BASE), dict(BASE), dict(BASE), dict(BASE), dict(BASE), [], audit
    )

    assert report["prop_families"]["shots"]["market_evidence"]["exact_line_value_rows"] == 20
    assert "SHOTS_OBSERVED_MARKET_PRICE_HISTORY_MISSING" not in report["blockers"]
    assert "SHOTS_CONFIRMED_XI_MARKET_OVERLAP_MISSING" not in report["blockers"]
    assert "SHOTS_EXACT_LINE_HISTORY_MISSING" not in report["blockers"]

    # Assists is a binary player-event market; lack of a numeric O/U line is not
    # itself a blocker when an exact priced market is observed.
    assert report["prop_families"]["assists"]["market_evidence"]["priced_value_rows"] == 57
    assert "ASSISTS_OBSERVED_MARKET_PRICE_HISTORY_MISSING" not in report["blockers"]
    assert "ASSISTS_CONFIRMED_XI_MARKET_OVERLAP_MISSING" not in report["blockers"]
    assert "ASSISTS_EXACT_LINE_HISTORY_MISSING" not in report["blockers"]

    # Player cards must use PLAYER_CARDS evidence, never generic match-card rows.
    assert "CARDS_OBSERVED_MARKET_PRICE_HISTORY_MISSING" in report["blockers"]
