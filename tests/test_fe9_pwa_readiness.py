from __future__ import annotations

from mcp_gateway import subscriber_frontend_v2
from mcp_gateway import subscriber_pwa_v2


def test_fe9_manifest_targets_v2_customer_app():
    manifest = subscriber_pwa_v2.manifest_payload()

    assert manifest["start_url"] == "/app-v2"
    assert manifest["scope"] == "/"
    assert manifest["display"] == "standalone"
    assert manifest["theme_color"] == "#06111a"
    assert manifest["icons"][0]["src"] == "/pwa/icon.svg"


def test_fe9_service_worker_never_caches_live_customer_api_or_prices():
    script = subscriber_pwa_v2.service_worker_script()
    contract = subscriber_pwa_v2.contract()

    assert "/app/api/v2/" in script
    assert "NETWORK_ONLY_PREFIXES" in script
    assert "cache:'no-store'" in script
    assert contract["api_cache_policy"] == "NETWORK_ONLY"
    assert contract["live_price_cache_allowed"] is False
    assert contract["live_decision_cache_allowed"] is False


def test_fe9_offline_copy_never_shows_stale_pick_or_price():
    script = subscriber_pwa_v2.service_worker_script()

    assert "The live snapshot is unavailable offline." in script
    assert "Reconnect to load current picks, prices and verification state." in script
    assert "BET" not in script.split("offline")[1]


def test_fe9_push_is_dormant_until_separately_approved():
    contract = subscriber_pwa_v2.contract()
    script = subscriber_pwa_v2.service_worker_script()

    assert contract["push_subscription_enabled"] is False
    assert contract["outbound_push_delivery_enabled"] is False
    assert contract["push_requires_explicit_user_consent"] is True
    assert "if(payload.enabled!==true)return;" in script


def test_fe9_frontend_registers_manifest_and_service_worker():
    html = subscriber_frontend_v2.render()
    contract = subscriber_frontend_v2.contract()

    assert 'rel="manifest" href="/app.webmanifest"' in html
    assert "navigator.serviceWorker.register('/sw.js'" in html
    assert contract["live_data_cache_allowed"] is False
    assert contract["push_notifications_enabled"] is False


def test_fe9_routes_are_installed():
    import mcp_gateway
    from mcp.server.fastmcp import FastMCP

    app = FastMCP(
        "fe9-route-test",
        stateless_http=True,
        json_response=True,
    ).streamable_http_app()
    paths = {getattr(route, "path", None) for route in app.router.routes}

    assert "/app.webmanifest" in paths
    assert "/sw.js" in paths
    assert "/pwa/icon.svg" in paths


def test_fe9_contract_preserves_engine_firewall():
    contract = subscriber_pwa_v2.contract()

    assert contract["model_input_allowed"] is False
    assert contract["provider_requests_added"] == 0
    assert contract["canonical_bet_logic_changed"] is False
    assert contract["model_weights_changed"] is False
    assert contract["production_promotion_allowed"] is False
