from mcp_gateway import automation_v92 as v92


def test_model_signal_is_independent_of_price():
    row = {
        "side_score": 72.0,
        "goals_score": 64.0,
        "two_way_score": 58.0,
        "price": 1.01,
        "prob_edge_pp": -99.0,
        "estimated_ev": -1.0,
    }
    assert v92._derive_model_signal(row) == "STRONG"

    row["price"] = 9.99
    row["prob_edge_pp"] = 99.0
    row["estimated_ev"] = 4.0
    assert v92._derive_model_signal(row) == "STRONG"


def test_execution_status_can_block_strong_model_signal():
    payload = {
        "match_table_rows": [
            {
                "classification": "WATCH",
                "event_classification": "WATCH",
                "side_score": 82.0,
                "market": "Goals Over/Under",
                "selection": "Over",
                "line": 2.5,
                "price": None,
                "reason": "NO_VERIFIED_MARKET_PRICE",
                "bet_eligible": False,
            }
        ]
    }

    v92._annotate_decision_separation(payload)
    row = payload["match_table_rows"][0]
    assert row["model_signal"] == "VERY_STRONG"
    assert row["execution_status"] == "WAIT_PRICE"
    assert row["blockers"][0] == "WAIT_PRICE"


def test_ready_requires_bet_eligible_bet_with_price():
    row = {
        "classification": "BET",
        "event_classification": "BET",
        "side_score": 75.0,
        "market": "Match Winner",
        "selection": "Home",
        "price": 2.05,
        "bet_eligible": True,
        "reason": "QUALIFIED",
    }
    assert v92._derive_execution_status(row) == "READY"


def test_pass_research_row_keeps_model_signal_but_is_not_ready():
    row = {
        "classification": "WATCH",
        "event_classification": "PASS",
        "goals_score": 71.0,
        "market": "Goals Over/Under",
        "selection": "Over research",
        "price": 2.10,
        "bet_eligible": False,
        "reason": "SPORTING_SCREEN_BELOW_SHORTLIST",
    }
    assert v92._derive_model_signal(row) == "STRONG"
    assert v92._derive_execution_status(row) == "RESEARCH_ONLY"


def test_shortlist_rank_does_not_inflate_model_strength():
    row = {
        "side_score": 52.6,
        "goals_score": 10.0,
        "two_way_score": 48.3,
        "shortlist_rank": 100.0,
    }
    assert v92._derive_model_signal(row) == "WEAK"


def test_decision_separation_exposes_actual_model_signal_score():
    payload = {"match_table_rows": [{
        "classification": "WATCH",
        "event_classification": "PASS",
        "side_score": 52.6,
        "goals_score": 10.0,
        "two_way_score": 48.3,
        "shortlist_rank": 100.0,
        "market": "Research screen",
        "price": None,
        "reason": "SPORTING_SCREEN_PASS",
    }]}
    v92._annotate_decision_separation(payload)
    row = payload["match_table_rows"][0]
    assert row["model_signal"] == "WEAK"
    assert row["model_signal_score"] == 52.6
    assert payload["decision_separation"]["shortlist_rank_is_not_model_strength"] is True


def test_v4_005_score_matrix_observability_preserves_research_only_gate():
    payload = {
        "match_table_rows": [],
        "galaxy_builder": {
            "score_matrix_expansion": {
                "schema_version": "0.6.0",
                "status": "LIVE_RESEARCH",
                "events_modeled": 3,
                "family_leg_counts": {"TEAM_TOTAL_HOME": 2, "CORRECT_SCORE": 1},
                "research_candidate_count": 2,
                "settlement_diagnostic_count": 4,
                "binary_joint_method": "DIRECT_SCORE_MATRIX_INTERSECTION",
                "settlement_joint_method": "DIRECT_SCORE_MATRIX_SETTLEMENT_INTEGRATION",
                "production_promotion_allowed": False,
                "candidate_merge_into_primary_galaxy_feed": False,
            }
        },
    }
    v92._annotate_decision_separation(payload)
    summary = payload["v4_005_score_matrix_observability"]
    assert summary["events_modeled"] == 3
    assert summary["research_candidate_count"] == 2
    assert summary["production_promotion_allowed"] is False
    assert summary["candidate_merge_into_primary_galaxy_feed"] is False
