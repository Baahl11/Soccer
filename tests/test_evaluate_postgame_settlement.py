from __future__ import annotations

from mcp_gateway.evaluate_postgame import grade_market, market_family, performance_summary, settlement_row


FINAL_2_1 = {
    "goals": {"home": 2, "away": 1},
    "score": {"halftime": {"home": 1, "away": 0}, "fulltime": {"home": 2, "away": 1}},
}


def test_ft_total_grades_against_match_total() -> None:
    best = {"market": "Goals Over/Under", "selection": "Over 2.5", "line": 2.5, "decimal_price": 1.91}
    assert market_family(best) == "FT_TOTALS"
    assert grade_market(best, FINAL_2_1, "Home", "Away") == "WIN"


def test_team_total_is_bucketed_but_not_misgraded_as_match_total() -> None:
    best = {"market": "Team Total Goals", "selection": "Home Over 2.5", "line": 2.5, "decimal_price": 2.1}
    assert market_family(best) == "FT_TEAM_TOTAL"
    assert grade_market(best, FINAL_2_1, "Home", "Away") == "UNSUPPORTED_DERIVATIVE"


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
        {"settlement_status": "UNSUPPORTED_DERIVATIVE", "roi_units": None},
    ]
    summary = performance_summary(rows)
    assert summary["n"] == 3
    assert summary["settled"] == 2
    assert summary["ungraded"] == 1
    assert summary["unsupported_derivative"] == 1
    assert summary["hit_rate_ex_push"] == 0.5
