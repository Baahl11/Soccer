from __future__ import annotations

import os
from typing import Any

from mcp_gateway import subscriber_billing_market_v226
from mcp_gateway import supabase_auth_v4

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_SUBSCRIBER_BILLING_V2_1.0.0"

CHECKOUT_FUNCTION = "create-checkout-session"
PORTAL_FUNCTION = "create-customer-portal"

# Current live Stripe Price objects were independently verified on 2026-10-07.
# Price IDs remain server-side in the Supabase Checkout Edge Function and are
# intentionally excluded from this customer-facing contract.
VERIFIED_PRICE_SNAPSHOT = {
    "verified_at": "2026-10-07",
    "product": "Soccer Edge Pro",
    "markets": {
        "US": {
            "display_price": "US$14.99 / month",
            "currency": "usd",
            "unit_amount_minor": 1499,
            "interval": "month",
            "active": True,
        },
        "MX_LATAM": {
            "display_price": "MX$249 / month",
            "currency": "mxn",
            "unit_amount_minor": 24900,
            "interval": "month",
            "active": True,
        },
    },
}


def _flag(name: str, default: bool = False) -> bool:
    raw = str(os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def public_billing_contract() -> dict[str, Any]:
    auth = supabase_auth_v4.public_auth_config()
    legacy = subscriber_billing_market_v226.contract()
    markets = {
        key: {
            **value,
            "display_price": VERIFIED_PRICE_SNAPSHOT["markets"][key]["display_price"],
            "active": VERIFIED_PRICE_SNAPSHOT["markets"][key]["active"],
            "interval": VERIFIED_PRICE_SNAPSHOT["markets"][key]["interval"],
        }
        for key, value in legacy["billing_markets"].items()
        if key in VERIFIED_PRICE_SNAPSHOT["markets"]
    }

    infrastructure_ready = bool(
        auth.get("configured")
        and auth.get("project_url")
        and auth.get("publishable_key")
    )
    public_launch_enabled = _flag("SOCCER_PUBLIC_BILLING_ENABLED", False)

    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": (
            "PUBLIC_BILLING_ENABLED"
            if infrastructure_ready and public_launch_enabled
            else "BILLING_INFRASTRUCTURE_READY_LAUNCH_DISABLED"
            if infrastructure_ready
            else "BILLING_NOT_CONFIGURED"
        ),
        "infrastructure_ready": infrastructure_ready,
        "public_launch_enabled": public_launch_enabled,
        "checkout_enabled": infrastructure_ready and public_launch_enabled,
        "portal_enabled": infrastructure_ready,
        "checkout_function": CHECKOUT_FUNCTION,
        "portal_function": PORTAL_FUNCTION,
        "billing_market_selection": "EXPLICIT_USER_CHOICE",
        "browser_sends_price_id": False,
        "price_ids_exposed_to_browser": False,
        "markets": markets,
        "price_snapshot_verified_at": VERIFIED_PRICE_SNAPSHOT["verified_at"],
        "product_name": VERIFIED_PRICE_SNAPSHOT["product"],
        "checkout_surface": "STRIPE_HOSTED_CHECKOUT",
        "subscription_model": "FLAT_RATE_MONTHLY_PRO",
        "free_access_model": "FREEMIUM_NO_CARD_REQUIRED",
        "self_management": "STRIPE_CUSTOMER_PORTAL",
        "public_price_strategy_approved": False,
        "tax_managed_payments_eligibility_assumed": False,
        "provider_requests_added": 0,
        "canonical_bet_logic_changed": False,
        "model_weights_changed": False,
        "production_promotion_allowed": False,
    }


def account_billing_state(entitlement: dict[str, Any]) -> dict[str, Any]:
    public = public_billing_contract()
    row = entitlement.get("persisted_entitlement")
    persisted = row if isinstance(row, dict) else {}
    user = entitlement.get("user") if isinstance(entitlement.get("user"), dict) else {}
    owner = entitlement.get("owner") is True or str(user.get("role") or "").upper() == "OWNER"
    admin = entitlement.get("admin") is True or owner or str(user.get("role") or "").upper() == "ADMIN"
    effective_plan = str(entitlement.get("effective_plan") or "FREE").upper()
    status = str(persisted.get("status") or "").upper() or None

    upgrade_eligible = (
        bool(entitlement.get("authenticated"))
        and effective_plan != "PRO"
        and not owner
        and not admin
        and public["checkout_enabled"]
    )
    manage_eligible = (
        bool(entitlement.get("authenticated"))
        and effective_plan == "PRO"
        and not owner
        and not admin
        and str(persisted.get("source") or "").upper() == "STRIPE"
        and public["portal_enabled"]
    )

    return {
        "status": status,
        "effective_plan": effective_plan,
        "effective_plan_reason": entitlement.get("effective_plan_reason"),
        "source": persisted.get("source"),
        "current_period_end": persisted.get("current_period_end"),
        "cancel_at_period_end": (
            persisted.get("cancel_at_period_end")
            if isinstance(persisted.get("cancel_at_period_end"), bool)
            else None
        ),
        "valid_until": persisted.get("valid_until"),
        "owner_or_admin_override": owner or admin,
        "upgrade_eligible": upgrade_eligible,
        "manage_subscription_eligible": manage_eligible,
        "billing": public,
    }


def contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "stripe_price_ids_client_side": False,
        "explicit_billing_market_required": True,
        "public_launch_default": False,
        "owner_admin_purchase_prompt_allowed": False,
        "webhook_entitlement_authoritative": True,
        "model_input_allowed": False,
        "provider_requests_added": 0,
        "canonical_bet_logic_changed": False,
        "model_weights_changed": False,
        "production_promotion_allowed": False,
    }
