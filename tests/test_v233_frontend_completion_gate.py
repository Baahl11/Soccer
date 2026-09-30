from __future__ import annotations

from mcp_gateway import subscriber_product_v233


def test_v233_product_shell_matches_approved_information_architecture():
    html = subscriber_product_v233.product_html()
    for label in (
        "Today",
        "Edge Feed",
        "Matches",
        "Markets",
        "Performance",
        "My Edge",
        "Control Tower",
        "Research Lab",
    ):
        assert label in html
    for tab in ("Overview", "Goals", "Corners", "Cards", "Players", "Market", "Model"):
        assert tab in html
    assert "SOCCER_V233_PRODUCT_STYLE" in html
    assert "SOCCER_V233_PRODUCT_SCRIPT" in html
    assert "Soccer Edge · Product Preview" not in html


def test_v233_embeds_account_and_regional_billing_without_exposing_price_ids():
    html = subscriber_product_v233.product_html()
    assert "v233AccountDock" in html
    assert "v233Auth" in html
    assert "US$14.99 / month" in html
    assert "MX$249 / month" in html
    assert "billing_market" in html
    assert "price_1UL" not in html
    assert "create-checkout-session" in html
    assert "create-customer-portal" in html


def test_v233_dynamic_match_and_filter_interactions_are_present():
    html = subscriber_product_v233.product_html()
    assert "/app/match?fixture_id=" in html
    assert "selectFixture" in html
    assert "wireRows" in html
    assert "wireFilters" in html
    assert "wireTabs" in html
    assert "No persisted" in html
    assert "does not synthesize missing family data" in html


def test_v233_contract_preserves_runtime_invariants():
    contract = subscriber_product_v233.contract()
    assert contract["source_of_truth"] == "PANEL_DE_ANALISIS_SOCCER_EDGE_MOCKUP"
    assert contract["dynamic_match_selection"] is True
    assert contract["explorer_missing_premium_values_are_redacted"] is True
    assert contract["browser_sends_price_id"] is False
    assert contract["billing_market_selection"] == "EXPLICIT_USER_CHOICE"
    assert contract["provider_requests_added"] == 0
    assert contract["canonical_bet_logic_changed"] is False
    assert contract["model_weights_changed"] is False
    assert contract["production_promotion_allowed"] is False
