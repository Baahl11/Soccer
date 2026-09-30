from __future__ import annotations

from mcp_gateway import subscriber_app_v4, subscriber_billing_market_v226


def test_v226_contract_is_explicit_and_price_id_never_browser_input():
    contract = subscriber_billing_market_v226.contract()
    assert contract["billing_market_selection"] == "EXPLICIT_USER_CHOICE"
    assert contract["browser_sends_price_id"] is False
    assert contract["billing_markets"]["US"]["display_price"] == "US$14.99 / month"
    assert contract["billing_markets"]["MX_LATAM"]["display_price"] == "MX$249 / month"
    assert contract["provider_requests_added"] == 0
    assert contract["canonical_bet_logic_changed"] is False
    assert contract["model_weights_changed"] is False
    assert contract["production_promotion_allowed"] is False


def test_v226_injected_subscriber_html_has_region_enum_not_stripe_price_ids():
    html = subscriber_app_v4._app_html()
    assert "SOCCER_V226_REGIONAL_BILLING" in html
    assert 'value="US"' in html
    assert 'value="MX_LATAM"' in html
    assert "US$14.99 / month" in html
    assert "MX$249 / month" in html
    assert "billing_market" in html
    assert "price_1ULUp1BE8LWeAYkQBNijFI03" not in html
    assert "price_1ULUp7BE8LWeAYkQbanCnbG7" not in html
    assert "Language and location are never used to choose a price" in html


def test_v226_checkout_requires_auth_token_and_server_function():
    html = subscriber_app_v4._app_html()
    assert "soccer_edge_access_token" in html
    assert "/functions/v1/create-checkout-session" in html
    assert "Authorization:`Bearer ${token}`" in html
    assert "Choose billing market" in html


def test_v226_layer_is_idempotent():
    base = "<html><body><main>ok</main></body></html>"
    once = subscriber_billing_market_v226.inject(base)
    twice = subscriber_billing_market_v226.inject(once)
    assert once == twice
    assert once.count("SOCCER_V226_REGIONAL_BILLING") >= 1
