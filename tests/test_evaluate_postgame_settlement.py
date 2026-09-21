from __future__ import annotations

from mcp_gateway.evaluate_postgame import grade_market, market_family, performance_summary, settlement_row, tier_gate


FINAL_2_1 = {
    "goals": {"home": 2, "away": 1},
    "score": {"halftime": {"home": 1, "away": 0}, "fulltime": {"home": 2, "away": 1}},
}

TACTICAL_2_1 = {
    **FINAL_2_1,
    "tactical_stats": {
        "teams": [
            {"team_id": 10, "team": "Home", "corners": 6, "yellow_cards": 3, "red_cards": 0},
            {"team_id": 20, "team": "Away", "corners": 4, "yellow_cards": 2, "red_cards": 1},
        ]
    },
}


def test_ft_total_grades_against_match_total() -> None:
    best = {"market": "Goals Over/Under", "selection": "Over 2.5", "line": 2.5, "decimal_price": 1.91}
    assert market_family(best) == "FT_TOTALS"
    assert grade_market(best, FINAL_2_1, "Home", "Away") == "WIN"


def test_1h_total_grades_against_halftime_total() -> None:
    best = {"market": "First Half Goals Over/Under", "selection": "Under 1.5", "line": 1.5, "decimal_price": 1.75}
    assert market_family(best) == "1H_TOTALS"
    assert grade_market(best, FINAL_2_1, "Home", "Away") == "WIN"


def test_2h_total_grades_against_second_half_total() -> None:
    best = {"market": "Second Half Goals Over/Under", "selection": "Over 1.5", "line": 1.5, "decimal_price": 2.05}
    assert market_family(best) == "2H_TOTALS"
    assert grade_market(best, FINAL_2_1, "Home", "Away") == "WIN"


def test_period_btts_grades_against_period_score() -> None:
    first_half = {"market": "First Half Both Teams To Score", "selection": "No", "decimal_price": 1.7}
    second_half = {"market": "Second Half Both Teams To Score", "selection": "Yes", "decimal_price": 2.2}
    assert market_family(first_half) == "1H_BTTS"
    assert market_family(second_half) == "2H_BTTS"
    assert grade_market(first_half, FINAL_2_1, "Home", "Away") == "WIN"
    assert grade_market(second_half, FINAL_2_1, "Home", "Away") == "WIN"


def test_team_total_grades_when_side_is_identified() -> None:
    home_over = {"market": "Team Total Goals", "selection": "Home Over 1.5", "line": 1.5, "decimal_price": 1.83}
    away_under = {"market": "Team Total Goals", "selection": "Away Under 1.5", "line": 1.5, "decimal_price": 1.66}
    assert market_family(home_over) == "FT_TEAM_TOTAL"
    assert grade_market(home_over, FINAL_2_1, "Home", "Away") == "WIN"
    assert grade_market(away_under, FINAL_2_1, "Home", "Away") == "WIN"


def test_team_total_can_use_team_name() -> None:
    best = {"market": "Team Total Goals", "selection": "Home Over 2.5", "line": 2.5, "decimal_price": 2.1}
    assert market_family(best) == "FT_TEAM_TOTAL"
    assert grade_market(best, FINAL_2_1, "Home", "Away") == "LOSS"


def test_team_total_without_side_remains_ungradable() -> None:
    best = {"market": "Team Total Goals", "selection": "Over 1.5", "line": 1.5, "decimal_price": 1.9}
    assert grade_market(best, FINAL_2_1, "Home", "Away") == "UNGRADABLE_TEAM_TOTAL_SIDE"


def test_corners_total_grades_when_tactical_stats_exist() -> None:
    best = {"market": "Corners Over/Under", "selection": "Over 9.5", "line": 9.5, "decimal_price": 1.95}
    assert market_family(best) == "FT_CORNERS"
    assert grade_market(best, TACTICAL_2_1, "Home", "Away", 10, 20) == "WIN"


def test_team_corners_grade_when_side_is_identified() -> None:
    best = {"market": "Corners Over/Under", "selection": "Home Over 5.5", "line": 5.5, "decimal_price": 2.05}
    assert market_family(best) == "FT_CORNERS"
    assert grade_market(best, TACTICAL_2_1, "Home", "Away", 10, 20) == "WIN"


def test_corners_without_tactical_stats_are_not_graded() -> None:
    best = {"market": "Corners Over/Under", "selection": "Over 9.5", "line": 9.5, "decimal_price": 1.95}
    assert grade_market(best, FINAL_2_1, "Home", "Away") == "NO_TACTICAL_STATS"


def test_yellow_cards_grade_only_when_market_is_explicit_yellow() -> None:
    best = {"market": "Yellow Cards Over/Under", "selection": "Over 4.5", "line": 4.5, "decimal_price": 1.9}
    assert market_family(best) == "FT_CARDS"
    assert grade_market(best, TACTICAL_2_1, "Home", "Away", 10, 20) == "WIN"


def test_generic_cards_remain_unsupported_without_book_rule() -> None:
    best = {"market": "Cards Over/Under", "selection": "Over 4.5", "line": 4.5, "decimal_price": 1.9}
    assert market_family(best) == "FT_CARDS"
    assert grade_market(best, TACTICAL_2_1, "Home", "Away", 10, 20) == "UNSUPPORTED_CARD_RULES"


def test_settlement_row_has_market_family_and_roi() -> None:
    ev = {
        "event_key": "fx-1",
        "fixture_id": 1,
        "kickoff_local": "2026-09-21T12:00:00-06:00",
        "generated_at_local": "2026-09-21T10:30:00-06:00",
        "stage": "T-90",
        "league": "Test League",
        "home_team": "Home",
        "away_team": "Away",
        "classification": "LEAN",
        "tier": "B",
        "data_tier": "A",
        "availability_confidence": 0.9,
        "bet_eligible": True,
        "canonical_ft_market": True,
        "best_market": {"market": "Both Teams To Score", "selection": "Yes", "decimal_price": 1.8},
        "market_outcome": "WIN",
        "roi_units": 0.8,
        "result": FINAL_2_1,
    }
    row = settlement_row(ev)
    assert row["market_family"] == "FT_BTTS"
    assert row["settled"] is True
    assert row["roi_units"] == 0.8


def test_performance_summary_counts_ungraded_derivatives() -> None:
    rows = [
        {"settlement_status": "WIN", "roi_units": 0.9},
        {"settlement_status": "LOSS", "roi_units": -1.0},
        {"settlement_status": "UNSUPPORTED_CARD_RULES", "roi_units": None},
    ]
    summary = performance_summary(rows)
    assert summary["n"] == 3
    assert summary["settled"] == 2
    assert summary["ungraded"] == 1
    assert summary["unsupported_derivative"] == 1
    assert summary["hit_rate_ex_push"] == 0.5


def test_tier_gate_keeps_small_samples_in_research() -> None:
    status = tier_gate({"n": 3, "settled": 2, "roi_units": 2.0, "hit_rate_ex_push": 1.0})
    assert status["status"] == "RESEARCH_ONLY_SAMPLE_TOO_SMALL"


def test_tier_gate_marks_positive_sample_as_tier_b_candidate() -> None:
    status = tier_gate({"n": 24, "settled": 24, "roi_units": 5.0, "hit_rate_ex_push": 0.625})
    assert status["status"] == "TIER_B_CANDIDATE"
