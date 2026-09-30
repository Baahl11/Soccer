from __future__ import annotations

from mcp_gateway import subscriber_billing_v226


def test_billing_ui_exposes_only_market_enums_not_stripe_price_ids():
    rendered = subscriber_billing_v226.inject_billing("<html><body><main>app</main></body></html>")
    assert 'data-billing-market="MX_LATAM"' in rendered
    assert 'data-billing-market="US"' in rendered
    assert "MX$249 / mes" in rendered
    assert "US$14.99 / month" in rendered
    assert "price_" not in rendered
    assert "stripe_price_id" not in rendered
    assert "billing_market:market" in rendered


def test_billing_ui_requires_existing_supabase_access_token():
    rendered = subscriber_billing_v226.inject_billing("<html><body></body></html>")
    assert "soccer_edge_access_token" in rendered
    assert "Sign in first" in rendered
    assert "Authorization" in rendered
    assert "Bearer" in rendered


def test_billing_ui_uses_only_supabase_edge_functions_and_stripe_hosted_redirects():
    rendered = subscriber_billing_v226.inject_billing("<html><body></body></html>")
    assert "create-checkout-session" in rendered
    assert "create-customer-portal" in rendered
    assert "checkout\\.stripe\\.com" in rendered
    assert "billing\\.stripe\\.com" in rendered
    assert "STRIPE_SECRET_KEY" not in rendered
    assert "whsec_" not in rendered
    assert "sk_live_" not in rendered


def test_billing_injection_is_idempotent():
    once = subscriber_billing_v226.inject_billing("<html><body></body></html>")
    twice = subscriber_billing_v226.inject_billing(once)
    assert twice == once
    assert twice.count('id="v226-billing-panel"') == 1
