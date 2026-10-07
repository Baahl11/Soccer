from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone

from mcp_gateway import prospective_evaluation_registry_v1 as v


def _base_row():
    return {
        "event_id": 101,
        "fixture_id": 9001,
        "generated_at": datetime(2026, 10, 7, 14, 35, tzinfo=timezone.utc),
        "classification": "BET",
        "availability_confidence": 0.90,
        "data_tier": "A",
    }


def _base_event():
    return {
        "model_version": "SOCCER EDGE ENGINE v1.7",
        "stage": "T-10",
        "classification": "BET",
        "tier": "B",
        "availability_confidence": 0.90,
        "coverage": {"data_tier": "A"},
        "fixture": {
            "fixture_id": 9001,
            "kickoff": "2026-10-07T16:00:00+00:00",
            "league_id": 39,
            "league": "Test League",
            "country": "Test",
            "season": 2026,
            "home_team_id": 1,
            "home_team": "Home FC",
            "away_team_id": 2,
            "away_team": "Away FC",
        },
        "lineups": {
            "both_xi_confirmed": True,
            "both_goalkeepers_confirmed": True,
            "teams": [],
        },
        "sporting_shortlist": {
            "shortlisted": True,
            "rank": 80.0,
            "tracks": ["SIDE"],
            "reason": "SPORTING_SCREEN_PASS",
        },
        "best_market": {
            "market": "Match Winner",
            "selection": "Home",
            "decimal_price": 2.10,
            "bookmaker": "Book",
            "p_raw": 0.56,
            "p_market_fair": 0.50,
            "p_shrunk": 0.53,
            "prob_edge_pp": 3.0,
            "estimated_ev": 0.113,
            "tier": "B",
        },
    }


def test_policy_is_predeclared_and_production_disabled():
    assert v.COHORT_START_UTC == "2026-10-07T14:30:00+00:00"
    assert v.STAGE == "T-10"
    assert v.POLICY["outcomes_never_used_for_selection"] is True
    assert v.POLICY["market_prices_never_used_to_create_raw_sport_projection"] is True
    assert v.POLICY["production_promotion_allowed"] is False
    assert len(v.POLICY_HASH) == 64


def test_1x2_tier_b_runtime_bet_is_frozen_and_counts_toward_graded_sample():
    candidate = v._best_1x2(_base_row(), _base_event())

    assert candidate is not None
    assert candidate["track"] == "1X2"
    assert candidate["candidate_type"] == "RUNTIME_TIER_B_BET"
    assert candidate["counts_toward_100_graded_bets"] is True
    assert candidate["selection_frozen"] is True
    assert candidate["outcome_known_at_selection"] is False
    assert candidate["policy_hash"] == v.POLICY_HASH


def test_1x2_advanced_cap_shadow_is_kept_but_does_not_count():
    row = _base_row()
    event = _base_event()
    row["classification"] = "WATCH"
    event["classification"] = "WATCH"
    event["tier"] = "A"
    event["best_market"]["tier"] = "A"
    event["best_market"]["prob_edge_pp"] = 5.5
    event["best_market"]["p_raw"] = 0.58
    event["best_market"]["p_market_fair"] = 0.50

    candidate = v._best_1x2(row, event)

    assert candidate is not None
    assert candidate["candidate_type"] == "ADVANCED_METRIC_CAP_SHADOW"
    assert candidate["counts_toward_100_graded_bets"] is False


def test_1x2_discrepancy_ge_12pp_is_excluded():
    row = _base_row()
    event = _base_event()
    row["classification"] = "WATCH"
    event["classification"] = "WATCH"
    event["tier"] = "S"
    event["best_market"]["tier"] = "S"
    event["best_market"]["p_raw"] = 0.63
    event["best_market"]["p_market_fair"] = 0.50

    assert v._best_1x2(row, event) is None


def test_common_gate_requires_tier_availability_xi_gk_and_sport_shortlist():
    for mutate in (
        lambda row, event: event["coverage"].update({"data_tier": "C"}),
        lambda row, event: event.update({"availability_confidence": 0.70}),
        lambda row, event: event["lineups"].update({"both_xi_confirmed": False}),
        lambda row, event: event["lineups"].update({"both_goalkeepers_confirmed": False}),
        lambda row, event: event["sporting_shortlist"].update({"shortlisted": False}),
    ):
        row = _base_row()
        event = _base_event()
        mutate(row, event)
        assert v._best_1x2(row, event) is None


def test_team_totals_selects_highest_positive_fresh_half_goal_edge():
    row = _base_row()
    event = _base_event()
    event["team_totals_intelligence"] = {
        "observed_exact_market_rows": [
            {
                "team_role": "HOME",
                "team_id": 1,
                "team": "Home FC",
                "market": "Total - Home",
                "selection": "OVER",
                "line": 1.5,
                "probability_model": 0.61,
                "decimal_price": 1.95,
                "p_market_fair": 0.53,
                "raw_edge_vs_market_fair_pp": 8.0,
                "market_fresh": True,
                "bookmaker": "A",
                "bookmaker_id": 1,
                "market_id": 16,
                "provider_update": "2026-10-07T14:33:00+00:00",
            },
            {
                "team_role": "AWAY",
                "team_id": 2,
                "team": "Away FC",
                "market": "Total - Away",
                "selection": "UNDER",
                "line": 0.5,
                "probability_model": 0.70,
                "decimal_price": 1.80,
                "p_market_fair": 0.58,
                "raw_edge_vs_market_fair_pp": 12.0,
                "market_fresh": True,
                "bookmaker": "B",
                "bookmaker_id": 2,
                "market_id": 17,
                "provider_update": "2026-10-07T14:34:00+00:00",
            },
            {
                "team_role": "HOME",
                "team_id": 1,
                "team": "Home FC",
                "market": "Total - Home",
                "selection": "OVER",
                "line": 2.0,
                "probability_model": 0.50,
                "decimal_price": 2.00,
                "p_market_fair": 0.45,
                "raw_edge_vs_market_fair_pp": 20.0,
                "market_fresh": True,
            },
            {
                "team_role": "HOME",
                "team_id": 1,
                "team": "Home FC",
                "market": "Total - Home",
                "selection": "UNDER",
                "line": 0.5,
                "probability_model": 0.60,
                "decimal_price": 1.90,
                "p_market_fair": 0.55,
                "raw_edge_vs_market_fair_pp": 30.0,
                "market_fresh": False,
            },
        ]
    }

    candidate = v._best_team_total(row, event)

    assert candidate is not None
    assert candidate["track"] == "TEAM_TOTALS"
    assert candidate["team_role"] == "AWAY"
    assert candidate["selection"] == "UNDER"
    assert candidate["line"] == 0.5
    assert candidate["raw_edge_vs_market_fair_pp"] == 12.0
    assert candidate["counts_toward_100_graded_bets"] is False


def test_selection_builder_does_not_read_result_payload():
    row = _base_row()
    event = _base_event()
    before = v._best_1x2(row, event)

    event_with_result = deepcopy(event)
    event_with_result["result"] = {
        "goals": {"home": 0, "away": 5},
        "status": "FT",
    }
    after = v._best_1x2(row, event_with_result)

    assert before == after


def test_settlement_is_separate_from_selection_and_grades_supported_tracks():
    one = v._best_1x2(_base_row(), _base_event())
    assert one is not None
    assert v._grade_1x2(one, 2, 1) == "WIN"
    assert v._grade_1x2(one, 0, 1) == "LOSS"

    tt = {
        "track": "TEAM_TOTALS",
        "team_role": "HOME",
        "selection": "OVER",
        "line": 1.5,
    }
    assert v._grade_team_total(tt, 2, 0) == "WIN"
    assert v._grade_team_total(tt, 1, 0) == "LOSS"


def test_only_runtime_tier_b_rows_can_satisfy_100_graded_bet_gate():
    rows = [
        {
            "fixture_id": i,
            "outcome": "WIN",
            "settled": True,
            "roi_units": 0.9,
            "counts_toward_100_graded_bets": i < 100,
        }
        for i in range(120)
    ]
    graded = [r for r in rows if r["counts_toward_100_graded_bets"]]
    summary = v._summarize(graded)
    assert summary["settled"] == 100
