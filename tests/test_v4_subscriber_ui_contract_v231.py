from mcp_gateway.subscriber_ui_contract_v231 import adapt_market_row


def test_adapts_real_probability_and_derives_display_edge_only_when_missing():
    row = {
        "fixture_id": 123,
        "home_team": "Alpha",
        "away_team": "Beta",
        "market_family": "FT_TOTALS",
        "selection": "Over 2.5",
        "p_model_calibrated": 0.577,
        "p_market_fair": 0.429,
        "execution_status": "RESEARCH_ONLY",
    }
    out = adapt_market_row(row)
    assert out["match"]["label"] == "Alpha vs Beta"
    assert out["model"]["probability"] == 0.577
    assert out["pricing"]["market_probability"] == 0.429
    assert round(out["pricing"]["edge_pp"], 1) == 14.8
    assert out["pricing"]["edge_source"] == "presentation_derived"


def test_preserves_real_zero_without_turning_missing_into_zero():
    zero = adapt_market_row({"p_raw": 0.0, "p_market_devig": 0.0})
    missing = adapt_market_row({})
    assert zero["model"]["probability"] == 0.0
    assert zero["pricing"]["market_probability"] == 0.0
    assert missing["model"]["probability"] is None
    assert missing["pricing"]["market_probability"] is None
    assert missing["pricing"]["edge_pp"] is None


def test_resolves_nested_team_shapes():
    out = adapt_market_row({
        "fixture_id": 777,
        "teams": {"home": {"name": "Home FC"}, "away": {"name": "Away FC"}},
    })
    assert out["match"]["label"] == "Home FC vs Away FC"


def test_persisted_edge_wins_over_presentation_derived_edge():
    out = adapt_market_row({
        "p_model_calibrated": 0.70,
        "p_market_fair": 0.50,
        "edge_pp": 7.5,
    })
    assert out["pricing"]["edge_pp"] == 7.5
    assert out["pricing"]["edge_source"] == "persisted"
