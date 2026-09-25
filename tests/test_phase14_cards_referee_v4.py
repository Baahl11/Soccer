from mcp_gateway import cards_referee_phase14_v4 as v


def test_phase14_blocks_current_referee_and_market_evidence():
    report = v.build_report(
        {
            "metrics": {
                "n": 254,
                "referee_adjusted_n": 0,
                "mae_total_yellow": 1.587206,
                "lines": {"2.5": {"brier": 0.20, "log_loss": 0.59}},
            },
            "market_scope": "NO_ACTIONABLE_CARD_MARKET_MAPPING_YET",
            "promotion_gate": {"enabled": False},
        },
        {
            "walk_forward_evaluated": 184,
            "overall": {"n": 184, "brier": 0.16623, "log_loss": 0.517398},
            "referee_adjusted": {"n": 0},
            "promotion_gate": {"enabled": False, "minimum_referee_adjusted_oos": 200},
        },
        {
            "status": "PASS",
            "failures": [],
            "profiles_checked": 2514,
            "oos_validation_complete": False,
            "actionable": False,
            "decision_weight": 0,
            "bookmaker_card_scoring_rule_assumed": False,
        },
        [],
    )
    assert report["status"] == "RESEARCH_HOLD"
    assert "YELLOW_REFEREE_ADJUSTED_0_LT_100" in report["blockers"]
    assert "RED_CARD_OOS_184_LT_MARKET_REVIEW_500" in report["blockers"]
    assert "PLAYER_CARDS_OOS_VALIDATION_INCOMPLETE" in report["blockers"]
    assert "CARD_TRUE_CLV_0_LT_50" in report["blockers"]
    assert report["production_promotion_allowed"] is False


def test_phase14_card_clv_matcher_is_strict():
    summary = v.summarize_true_clv([
        {"market": "Total Yellow Cards", "fixture_id": 1, "clv_probability_pp": 0.02},
        {"market": "Red Card In Match", "fixture_id": 2, "clv_probability_pp": -0.01},
        {"market": "Goals Over/Under", "fixture_id": 3, "clv_probability_pp": 0.20},
    ])
    assert summary["rows"] == 2
    assert summary["unique_fixtures"] == 2
    assert summary["avg_probability_clv_pp"] == 0.005



def test_phase14_uses_match_cards_and_player_cards_as_separate_evidence():
    audit = {
        "model_version": "SOCCER_RESEARCH_DERIVATIVE_MARKET_AUDIT_V4_1.0.0",
        "families": {
            "CARDS": {
                "market_snapshot_rows": 675,
                "unique_fixtures": 32,
                "pre_kickoff_unique_fixtures": 29,
                "provider_update_unique_fixtures": 32,
                "confirmed_xi_pre_kickoff_unique_fixtures": 4,
                "bookmaker_count": 4,
                "priced_value_rows": 2387,
                "exact_line_value_rows": 1530,
                "exact_observed_market_history_materialized": True,
                "confirmed_xi_overlap_materialized": True,
            },
            "PLAYER_CARDS": {
                "market_snapshot_rows": 0,
                "unique_fixtures": 0,
                "pre_kickoff_unique_fixtures": 0,
                "provider_update_unique_fixtures": 0,
                "confirmed_xi_pre_kickoff_unique_fixtures": 0,
                "bookmaker_count": 0,
                "priced_value_rows": 0,
                "exact_line_value_rows": 0,
                "exact_observed_market_history_materialized": False,
                "confirmed_xi_overlap_materialized": False,
            },
        },
    }
    report = v.build_report(
        {
            "metrics": {"n": 254, "referee_adjusted_n": 0},
            "promotion_gate": {"enabled": False},
        },
        {
            "walk_forward_evaluated": 184,
            "overall": {"n": 184},
            "referee_adjusted": {"n": 0},
            "promotion_gate": {"enabled": False, "minimum_referee_adjusted_oos": 200},
        },
        {
            "status": "PASS",
            "failures": [],
            "profiles_checked": 2514,
            "oos_validation_complete": False,
            "actionable": False,
            "decision_weight": 0,
            "bookmaker_card_scoring_rule_assumed": False,
        },
        [],
        audit,
    )

    assert report["market_evidence"]["match_cards"]["market_snapshot_rows"] == 675
    assert report["market_evidence"]["match_cards"]["exact_line_value_rows"] == 1530
    assert "MATCH_CARD_OBSERVED_MARKET_PRICE_HISTORY_MISSING" not in report["blockers"]
    assert "PLAYER_CARD_OBSERVED_MARKET_PRICE_HISTORY_MISSING" in report["blockers"]
