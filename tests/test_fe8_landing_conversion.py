from __future__ import annotations

from mcp_gateway import landing_page_v2


def test_fe8_landing_has_no_fake_probabilities_or_fixed_accuracy_claims():
    html = landing_page_v2.render()
    contract = landing_page_v2.contract()

    for forbidden in ("53.5%", "61.4%", "87% confidence", "90% accuracy"):
        assert forbidden not in html

    assert contract["mock_probabilities"] is False
    assert contract["fixed_accuracy_claims"] is False
    assert contract["public_subscription_price_claims"] is False


def test_fe8_landing_sells_canonical_decision_states_not_forced_picks():
    html = landing_page_v2.render()

    assert "BET" in html
    assert "LEAN" in html
    assert "WATCH" in html
    assert "PASS" in html
    assert "ZERO BETS IS VALID" in html
    assert "No fabricated picks" in html


def test_fe8_landing_preserves_sport_first_market_second():
    html = landing_page_v2.render()
    contract = landing_page_v2.contract()

    assert "Sport first. Market second." in html
    assert "RAW SPORT ≠ MARKET SHRUNK" in html
    assert contract["primary_message"] == "SPORT_FIRST_MARKET_SECOND"


def test_fe8_landing_does_not_publish_unapproved_subscription_price():
    html = landing_page_v2.render()

    assert "US$14.99" not in html
    assert "MX$249" not in html
    assert "public paid launch is gated" in html.lower()
    assert "pricing strategy are explicitly approved" in html


def test_fe8_landing_analytics_are_product_only_and_allowed_events():
    html = landing_page_v2.render()

    assert "track-product-event" in html
    assert "landing_view" in html
    assert "explorer_cta" in html
    assert "language_change" in html
    assert "model_weight" not in html
    assert "canonical_bet_logic_changed" not in html.split("<script>")[-1]


def test_fe8_landing_routes_to_v2_app_and_has_responsible_risk_copy():
    html = landing_page_v2.render()
    contract = landing_page_v2.contract()

    assert 'href="/app-v2"' in html
    assert "Betting involves risk" in html
    assert "Apostar implica riesgo" in html
    assert contract["app_destination"] == "/app-v2"
    assert contract["responsible_risk_copy_present"] is True


def test_fe8_route_is_installed():
    import mcp_gateway
    from mcp.server.fastmcp import FastMCP

    app = FastMCP(
        "fe8-route-test",
        stateless_http=True,
        json_response=True,
    ).streamable_http_app()
    paths = {getattr(route, "path", None) for route in app.router.routes}
    assert "/landing-v2" in paths


def test_fe8_contract_preserves_engine_firewall():
    contract = landing_page_v2.contract()

    assert contract["provider_requests_added"] == 0
    assert contract["canonical_bet_logic_changed"] is False
    assert contract["model_weights_changed"] is False
    assert contract["production_promotion_allowed"] is False
