from mcp_gateway import player_props_phase15_v4 as v


def _sanity(valid=200):
    return {
        "status": "PASS",
        "profiles_checked": valid,
        "profiles_valid": valid,
        "failures": [],
        "oos_validation_complete": False,
        "actionable": False,
        "decision_weight": 0.0,
        "validation_scope": "STRUCTURAL_ONLY_NOT_OOS_PERFORMANCE",
    }


def test_goal_scorer_market_name_is_recognized_as_player_prop():
    assert v.is_player_prop_market({"market": "Home Anytime Goal Scorer", "selection": "Player A"}) is True
    assert v.is_player_prop_market({"market": "Away First Goal Scorer", "selection": "Player B"}) is True
    assert v.is_player_prop_market({"market": "Home Last Goal Scorer", "selection": "Player C"}) is True


def test_phase15_goalscorer_evidence_uses_anytime_family_only():
    market_audit = {
        "model_version": "SOCCER_RESEARCH_DERIVATIVE_MARKET_AUDIT_V4_1.2.0",
        "families": {
            "GOALSCORER_ANYTIME": {
                "market_snapshot_rows": 159,
                "unique_fixtures": 12,
                "pre_kickoff_unique_fixtures": 12,
                "provider_update_unique_fixtures": 12,
                "confirmed_xi_pre_kickoff_unique_fixtures": 3,
                "bookmaker_count": 3,
                "priced_value_rows": 500,
                "exact_line_value_rows": 0,
                "exact_observed_market_history_materialized": True,
                "confirmed_xi_overlap_materialized": True,
            },
            "GOALSCORER_FIRST": {
                "market_snapshot_rows": 159,
                "unique_fixtures": 12,
                "pre_kickoff_unique_fixtures": 12,
                "provider_update_unique_fixtures": 12,
                "confirmed_xi_pre_kickoff_unique_fixtures": 5,
                "bookmaker_count": 3,
                "priced_value_rows": 500,
                "exact_line_value_rows": 0,
                "exact_observed_market_history_materialized": True,
                "confirmed_xi_overlap_materialized": True,
            },
            "GOALSCORER_LAST": {
                "market_snapshot_rows": 159,
                "unique_fixtures": 12,
                "pre_kickoff_unique_fixtures": 12,
                "provider_update_unique_fixtures": 12,
                "confirmed_xi_pre_kickoff_unique_fixtures": 5,
                "bookmaker_count": 3,
                "priced_value_rows": 500,
                "exact_line_value_rows": 0,
                "exact_observed_market_history_materialized": True,
                "confirmed_xi_overlap_materialized": True,
            },
        },
    }

    report = v.build_report(
        _sanity(),
        _sanity(),
        _sanity(),
        _sanity(),
        _sanity(),
        _sanity(120),
        [],
        market_audit,
    )

    evidence = report["prop_families"]["goalscorer"]["market_evidence"]
    assert evidence["audit_family"] == "GOALSCORER_ANYTIME"
    assert evidence["market_snapshot_rows"] == 159
    assert evidence["confirmed_xi_pre_kickoff_unique_fixtures"] == 3
    assert "GOALSCORER_OBSERVED_MARKET_PRICE_HISTORY_MISSING" not in report["blockers"]
    assert "GOALSCORER_CONFIRMED_XI_MARKET_OVERLAP_MISSING" not in report["blockers"]
