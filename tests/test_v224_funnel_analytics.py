from __future__ import annotations

from mcp_gateway import landing_page_v4, product_analytics_v4, subscriber_app_v4


def test_landing_includes_privacy_conscious_funnel_tracking(monkeypatch):
    monkeypatch.setenv("SOCCER_SUPABASE_URL", "https://project.supabase.co")
    monkeypatch.setenv("SOCCER_SUPABASE_PUBLISHABLE_KEY", "sb_publishable_test")
    rendered = landing_page_v4.render_landing()
    assert "SOCCER_PRODUCT_ANALYTICS_V4_LANDING" in rendered
    assert "/functions/v1/track-product-event" in rendered
    assert "landing_view" in rendered
    assert "explorer_cta" in rendered
    assert "utm_campaign" in rendered
    assert "cohort" in rendered
    assert "soccer_edge_anon_id" in rendered
    assert "soccer_edge_session_id" in rendered
    assert "IP" not in rendered


def test_app_tracks_auth_and_checkout_without_backend_secrets(monkeypatch):
    monkeypatch.setenv("SOCCER_SUPABASE_URL", "https://project.supabase.co")
    monkeypatch.setenv("SOCCER_SUPABASE_PUBLISHABLE_KEY", "sb_publishable_test")
    rendered = subscriber_app_v4._app_html()
    assert "SOCCER_PRODUCT_ANALYTICS_V4_APP" in rendered
    for event in ("app_view", "signup_click", "signin_click", "authenticated_view", "pro_checkout_click", "customer_portal_click"):
        assert event in rendered
    assert "analytics:{anonymous_id" in rendered
    assert "STRIPE_SECRET_KEY" not in rendered
    assert "STRIPE_WEBHOOK_SECRET" not in rendered
    assert "service_role" not in rendered.lower()


def test_analytics_layer_does_not_modify_canonical_product_payload():
    product = {
        "views": {
            "todays_slate": {"total": 1, "rows": [{"fixture_id": 7, "market_family": "BTTS", "status": "WAIT_XI", "price": 1.91}]},
            "waiting_for_price": {"total": 0, "rows": []},
            "waiting_for_xi": {"total": 1, "rows": []},
            "control_tower": {},
        }
    }
    payload = subscriber_app_v4.build_subscriber_payload(product, subscriber_app_v4.anonymous_entitlement())
    row = payload["public"]["verified_slate"]["rows"][0]
    assert row["market_family"] == "BTTS"
    assert row["status"] == "WAIT_XI"
    assert "price" not in row
    assert payload["provider_requests_added"] == 0


def test_analytics_module_has_no_provider_or_model_mutation_path():
    source = open(product_analytics_v4.__file__, "r", encoding="utf-8").read().lower()
    for forbidden in ("api-football", "model_weights", "decision_weight", "production_promotion_allowed", "threshold"):
        assert forbidden not in source
