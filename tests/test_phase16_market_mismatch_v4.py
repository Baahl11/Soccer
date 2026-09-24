from mcp_gateway import market_mismatch_v4 as v


def _row(**overrides):
    row = {
        "fixture_id": 1,
        "league": "Test League",
        "home": "A",
        "away": "B",
        "stage": "T-20",
        "market_family": "TOTAL",
        "market": "Goals Over/Under",
        "selection": "Over",
        "line": 2.5,
        "price": 2.10,
        "bookmaker": "Book",
        "model_signal_score": 82.0,
        "data_tier": "A",
        "p_market_fair": 0.46,
        "p_model_calibrated": 0.58,
        "prob_edge_pp": 14.0,
        "blockers": [],
        "model_disagreement": "LOW",
    }
    row.update(overrides)
    return row


def test_phase16_requires_calibrated_probability_for_rank():
    result = v.find_mismatches([
        _row(p_model_calibrated=None),
    ])
    assert result["rows_analyzed"] == 1
    assert result["rankable_rows"] == 0
    assert result["primary_candidates"] == []
    analyzed = v.analyze_row(_row(p_model_calibrated=None))
    assert "CALIBRATED_MODEL_PROBABILITY_MISSING" in analyzed["blockers"]
    assert analyzed["raw_diagnostic_score"] is not None
    assert analyzed["mismatch_score"] is None
    assert analyzed["rankability_reasons"] == ["CALIBRATED_MODEL_PROBABILITY_MISSING"]


def test_phase16_ranks_positive_calibrated_edge():
    result = v.find_mismatches([_row()])
    assert result["rankable_rows"] == 1
    candidate = result["primary_candidates"][0]
    assert candidate["market_family"] == "FT_TOTALS"
    assert candidate["calibrated_edge_pp"] == 12.0
    assert candidate["mismatch_score"] > 0
    assert candidate["production_promotion_allowed"] is False
    assert candidate["bet_eligible"] is False


def test_phase16_suppresses_correlated_goal_markets_per_fixture():
    rows = [
        _row(p_model_calibrated=0.60, p_market_fair=0.45),
        _row(
            market_family="BTTS",
            market="Both Teams To Score",
            selection="Yes",
            p_model_calibrated=0.61,
            p_market_fair=0.50,
            price=2.0,
        ),
    ]
    result = v.find_mismatches(rows)
    assert result["rankable_rows"] == 2
    assert len(result["primary_candidates"]) == 1
    assert len(result["correlated_candidates_suppressed"]) == 1
    assert result["correlated_candidates_suppressed"][0]["suppression_reason"] == "CORRELATED_MARKET_GROUP_ALREADY_REPRESENTED"


def test_phase16_keeps_independent_market_groups():
    rows = [
        _row(),
        _row(
            market_family="CORNERS",
            market="Corners Over/Under",
            selection="Over",
            line=9.5,
            p_model_calibrated=0.57,
            p_market_fair=0.48,
        ),
    ]
    result = v.find_mismatches(rows)
    assert len(result["primary_candidates"]) == 2
    assert {row["correlation_group"] for row in result["primary_candidates"]} == {"MATCH_GOALS", "CORNERS"}


def test_phase16_stale_quote_cannot_rank():
    analyzed = v.analyze_row(_row(blockers=["STALE_QUOTE"]))
    assert analyzed["rankable"] is False
    assert "STALE_QUOTE" in analyzed["blockers"]
    assert analyzed["price_quality_score"] == 0.0


def test_phase16_negative_calibrated_edge_is_explicitly_non_rankable():
    analyzed = v.analyze_row(_row(p_model_calibrated=0.40, p_market_fair=0.55))
    assert analyzed["rankable"] is False
    assert analyzed["calibrated_edge_pp"] == -15.0
    assert analyzed["rankability_reasons"] == ["CALIBRATED_EDGE_NOT_POSITIVE"]


def test_phase16_summary_counts_non_rankable_reasons_by_family():
    result = v.find_mismatches([
        _row(p_model_calibrated=0.40, p_market_fair=0.55),
        _row(fixture_id=2, p_model_calibrated=None),
        _row(fixture_id=3, price=None, p_market_fair=None),
    ])
    assert result["non_rankable_rows"] == 3
    assert result["non_rankable_reason_counts"]["CALIBRATED_EDGE_NOT_POSITIVE"] == 1
    assert result["non_rankable_reason_counts"]["CALIBRATED_MODEL_PROBABILITY_MISSING"] >= 1
    assert result["non_rankable_reason_counts_by_family"]["FT_TOTALS"]["CALIBRATED_EDGE_NOT_POSITIVE"] == 1
