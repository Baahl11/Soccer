from __future__ import annotations

from mcp_gateway import subscriber_product_v234


def _row(family: str, *, price=None, market_probability=None, edge_pp=None, probability=None):
    return {
        "market": {"family": family, "selection": family},
        "pricing": {"price": price, "market_probability": market_probability, "edge_pp": edge_pp},
        "model": {"probability": probability, "version": "test" if probability is not None else None},
        "state": {"status": "WATCH"},
    }


def test_v234_groups_persisted_fixture_markets_by_match_center_tab():
    groups = subscriber_product_v234._group_fixture_rows([
        _row("FT_TOTALS", price=1.9, market_probability=.52, edge_pp=4.0, probability=.56),
        _row("FT_CORNERS", price=1.8, probability=.58),
        _row("CARDS", market_probability=.50, probability=.54),
        _row("SHOTS", price=2.1, probability=.49),
    ])
    assert len(groups["Goals"]) == 1
    assert len(groups["Corners"]) == 1
    assert len(groups["Cards"]) == 1
    assert len(groups["Players"]) == 1
    assert len(groups["Market"]) == 4
    assert len(groups["Model"]) == 4


def test_v234_product_html_has_real_filter_controls_and_modes():
    html = subscriber_product_v234.product_html()
    for marker in ("v234League", "v234Market", "v234Kickoff", "v234Edge", "v234Conf", "v234Status"):
        assert marker in html
    for label in ("Strong Only", "Ready Only", "Next 3 Hours", "Confirmed XI"):
        assert label in html
    assert "applyFilters" in html
    assert "openFixture" in html


def test_v234_match_tabs_render_persisted_rows_not_synthetic_defaults():
    html = subscriber_product_v234.product_html()
    assert "markets_by_tab" in html
    assert "No persisted" in html
    assert "Missing market data is not synthesized" in html
    for label in ("Goals", "Corners", "Cards", "Players", "Market", "Model"):
        assert label in html


def test_v234_contract_preserves_engine_firewall():
    c = subscriber_product_v234.contract()
    assert c["edge_feed_filters"] == ["league", "market", "kickoff", "edge_pp", "confidence", "status"]
    assert c["missing_market_rows_are_synthesized"] is False
    assert c["provider_requests_added"] == 0
    assert c["canonical_bet_logic_changed"] is False
    assert c["model_weights_changed"] is False
    assert c["production_promotion_allowed"] is False
