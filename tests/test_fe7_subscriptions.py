from __future__ import annotations

from mcp_gateway import subscriber_billing_v2
from mcp_gateway import subscriber_contract_v2
from mcp_gateway import subscriber_frontend_v2
from mcp_gateway import subscription_entitlements_v4


def _entitlement(**overrides):
    row = {
        "authenticated": True,
        "effective_plan": "FREE",
        "effective_plan_reason": "DEFAULT_FREE_NO_ENTITLEMENT_ROW",
        "feature_access": subscription_entitlements_v4.feature_access("FREE"),
        "user": {"id": "user-1", "email": "user@example.com", "role": "USER"},
        "persisted_entitlement": None,
    }
    row.update(overrides)
    return row


def test_fe7_verified_live_prices_are_public_labels_only(monkeypatch):
    monkeypatch.setenv("SOCCER_SUPABASE_URL", "https://project.supabase.co")
    monkeypatch.setenv("SOCCER_SUPABASE_PUBLISHABLE_KEY", "sb_publishable")
    monkeypatch.delenv("SOCCER_PUBLIC_BILLING_ENABLED", raising=False)

    contract = subscriber_billing_v2.public_billing_contract()

    assert contract["markets"]["US"]["display_price"] == "US$14.99 / month"
    assert contract["markets"]["MX_LATAM"]["display_price"] == "MX$249 / month"
    assert contract["price_ids_exposed_to_browser"] is False
    assert contract["browser_sends_price_id"] is False
    assert contract["public_launch_enabled"] is False
    assert contract["checkout_enabled"] is False
    assert contract["public_price_strategy_approved"] is False


def test_fe7_public_checkout_requires_explicit_launch_gate(monkeypatch):
    monkeypatch.setenv("SOCCER_SUPABASE_URL", "https://project.supabase.co")
    monkeypatch.setenv("SOCCER_SUPABASE_PUBLISHABLE_KEY", "sb_publishable")
    monkeypatch.setenv("SOCCER_PUBLIC_BILLING_ENABLED", "true")

    contract = subscriber_billing_v2.public_billing_contract()

    assert contract["infrastructure_ready"] is True
    assert contract["public_launch_enabled"] is True
    assert contract["checkout_enabled"] is True
    assert contract["billing_market_selection"] == "EXPLICIT_USER_CHOICE"
    assert contract["checkout_surface"] == "STRIPE_HOSTED_CHECKOUT"


def test_fe7_free_user_can_upgrade_only_after_launch_gate(monkeypatch):
    monkeypatch.setenv("SOCCER_SUPABASE_URL", "https://project.supabase.co")
    monkeypatch.setenv("SOCCER_SUPABASE_PUBLISHABLE_KEY", "sb_publishable")

    monkeypatch.delenv("SOCCER_PUBLIC_BILLING_ENABLED", raising=False)
    blocked = subscriber_billing_v2.account_billing_state(_entitlement())
    assert blocked["upgrade_eligible"] is False

    monkeypatch.setenv("SOCCER_PUBLIC_BILLING_ENABLED", "true")
    ready = subscriber_billing_v2.account_billing_state(_entitlement())
    assert ready["upgrade_eligible"] is True
    assert ready["manage_subscription_eligible"] is False


def test_fe7_owner_admin_is_never_prompted_to_purchase(monkeypatch):
    monkeypatch.setenv("SOCCER_SUPABASE_URL", "https://project.supabase.co")
    monkeypatch.setenv("SOCCER_SUPABASE_PUBLISHABLE_KEY", "sb_publishable")
    monkeypatch.setenv("SOCCER_PUBLIC_BILLING_ENABLED", "true")

    owner = _entitlement(
        effective_plan="PRO",
        owner=True,
        admin=True,
        user={"id": "owner", "email": "owner@example.com", "role": "OWNER"},
        feature_access=subscription_entitlements_v4.feature_access("PRO"),
    )
    state = subscriber_billing_v2.account_billing_state(owner)

    assert state["owner_or_admin_override"] is True
    assert state["upgrade_eligible"] is False
    assert state["manage_subscription_eligible"] is False


def test_fe7_stripe_pro_can_manage_subscription_without_purchase_prompt(monkeypatch):
    monkeypatch.setenv("SOCCER_SUPABASE_URL", "https://project.supabase.co")
    monkeypatch.setenv("SOCCER_SUPABASE_PUBLISHABLE_KEY", "sb_publishable")
    monkeypatch.delenv("SOCCER_PUBLIC_BILLING_ENABLED", raising=False)

    pro = _entitlement(
        effective_plan="PRO",
        effective_plan_reason="PRO_ACTIVE",
        feature_access=subscription_entitlements_v4.feature_access("PRO"),
        persisted_entitlement={
            "plan": "PRO",
            "status": "ACTIVE",
            "source": "STRIPE",
            "current_period_end": "2026-11-07T00:00:00+00:00",
            "cancel_at_period_end": False,
        },
    )
    state = subscriber_billing_v2.account_billing_state(pro)

    assert state["upgrade_eligible"] is False
    assert state["manage_subscription_eligible"] is True
    assert state["current_period_end"] == "2026-11-07T00:00:00+00:00"
    assert state["cancel_at_period_end"] is False


def test_fe7_entitlement_lookup_requests_subscription_lifecycle_fields():
    source = open("mcp_gateway/subscription_entitlements_v4.py", encoding="utf-8").read()
    assert "current_period_end" in source
    assert "cancel_at_period_end" in source


def test_fe7_frontend_never_contains_stripe_price_ids():
    html = subscriber_frontend_v2.render()

    assert "price_1ULUp1BE8LWeAYkQBNijFI03" not in html
    assert "price_1ULUp7BE8LWeAYkQbanCnbG7" not in html
    assert "billing_market:market" in html
    assert "create-checkout-session" in html
    assert "create-customer-portal" in html
    assert "The browser never sends a Stripe Price ID." in html


def test_fe7_frontend_checkout_is_webhook_authoritative():
    html = subscriber_frontend_v2.render()

    assert "updates only after the verified Stripe webhook persists the entitlement" in html
    assert "No subscription change was assumed." in html
    assert "public launch disabled" in html


def test_fe7_contract_firewall_remains_intact():
    billing = subscriber_billing_v2.contract()
    subscriber = subscriber_contract_v2.contract()
    frontend = subscriber_frontend_v2.contract()

    assert billing["stripe_price_ids_client_side"] is False
    assert billing["explicit_billing_market_required"] is True
    assert billing["public_launch_default"] is False
    assert billing["owner_admin_purchase_prompt_allowed"] is False
    assert billing["webhook_entitlement_authoritative"] is True
    assert billing["model_input_allowed"] is False
    assert subscriber["billing_model_input_allowed"] is False
    assert frontend["browser_sends_stripe_price_id"] is False
    assert frontend["public_billing_launch_default"] is False
    assert billing["provider_requests_added"] == 0
    assert billing["canonical_bet_logic_changed"] is False
    assert billing["model_weights_changed"] is False
