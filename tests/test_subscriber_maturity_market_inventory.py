"""Frontend maturity contract regression: no synthesized CLV, no production promotion."""
from mcp_gateway import subscriber_maturity_v232 as maturity


def test_maturity_market_inventory_is_complete_but_not_promoted():
    families = [
        {
            "label": "1X2", "source": "1x2_report.json",
            "model_evidence": {"current": 670, "target": None, "ready": True, "unit": "OOS"},
            "true_clv_target": 50, "stage": "MODEL REVIEW + CLV COLLECTION",
            "blockers": ["TRUE_CLV_GATE_PENDING"], "next_gate": "TRUE_CLV_GATE_PENDING",
        },
        {
            "label": "Team Totals", "source": "team_totals.json",
            "model_evidence": {"current": 500, "target": 500, "ready": True, "unit": "OOS"},
            "true_clv_target": 50, "stage": "REVIEW READY",
        },
    ]
    clv = {
        "family_counts": {"1X2": 53, "HOME_TT": 248, "AWAY_TT": 266},
        "priced_entry_family_counts": {"1X2": 79, "HOME_TT": 310},
        "mapped_family_counts": {"1X2": 100, "HOME_TT": 330},
    }
    rows = maturity._build_market_inventory(families, clv, {})
    assert len(rows) == 21
    by_key = {row["key"]: row for row in rows}
    assert by_key["1X2"]["true_clv_rows"] == 53
    assert by_key["1X2"]["model_evidence"]["current"] == 670
    assert by_key["HOME_TT"]["true_clv_rows"] == 248
    assert by_key["AWAY_TT"]["true_clv_rows"] == 266
    # Parent's 500 OOS fixtures are NOT copied to each team-total side.
    assert by_key["HOME_TT"]["model_evidence"]["current"] is None
    assert by_key["AWAY_TT"]["model_evidence"]["current"] is None
    assert by_key["CORRECT_SCORE"]["true_clv_rows"] is None
    assert by_key["PLAYER_CARDS"]["model_evidence"]["current"] is None
    assert all(row["production_promotion_allowed"] is False for row in rows)
    assert all(row["classification"] == "RESEARCH_ONLY" for row in rows)


def test_clv_unknown_vs_actual_zero_and_card_oos_provenance():
    families = [{"label": "Cards", "source": "cards.json", "blockers": ["REFEREE_ADJUSTED_MISSING"]}]
    reports = {"Cards": {"yellow_cards": {"oos_n": 254, "minimum_oos": 100, "referee_adjusted_n": 0}, "red_cards": {"oos_n": 184, "minimum_oos": 500}}}
    missing = maturity._build_market_inventory(families, {}, reports)
    by_key = {row["key"]: row for row in missing}
    assert by_key["1X2"]["true_clv_rows"] is None
    assert by_key["YELLOW_CARDS"]["model_evidence"]["current"] == 254
    assert by_key["RED_CARDS"]["model_evidence"]["current"] == 184
    assert by_key["RED_CARDS"]["model_evidence"]["ready"] is False
    # A generic cards CLV 0 must not be attributed to yellow and red twice.
    assert by_key["YELLOW_CARDS"]["true_clv_rows"] is None
    assert by_key["RED_CARDS"]["true_clv_rows"] is None
