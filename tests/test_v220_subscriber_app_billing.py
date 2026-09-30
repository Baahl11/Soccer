from __future__ import annotations

from mcp_gateway import subscriber_app_v4, subscription_entitlements_v4


def _product():
    premium_row = {
        "fixture_id": 101,
        "kickoff": "2026-09-30T20:00:00Z",
        "league": "Test League",
        "home_team": "Home",
        "away_team": "Away",
        "market_family": "1X2",
        "selection": "HOME",
        "price": -110,
        "bookmaker": "Book",
        "p_market_fair": 0.50,
        "p_model_calibrated": 0.58,
        "edge": 0.08,
        "status": "READY",
    }
    return {
        "generated_at_utc": "2026-09-30T04:00:00Z",
        "pipeline_version": "4.38.0-test",
        "views": {
            "todays_slate": {"total": 1, "rows": [premium_row]},
            "waiting_for_price": {"total": 2, "rows": [premium_row]},
            "waiting_for_xi": {"total": 3, "rows": []},
            "strong_sport_signals": {"total": 4, "rows": [premium_row]},
            "value_plays": {"total": 5, "rows": [premium_row]},
            "team_totals": {"total": 1, "rows": [premium_row]},
            "first_half": {"total": 1, "rows": []},
            "second_half": {"total": 1, "rows": []},
            "corners": {"total": 1, "rows": []},
            "player_props": {"total": 1, "rows": []},
            "performance": {"total": 1, "rows": []},
            "control_tower": {
                "maturity_snapshot": {
                    "maturation_control_tower": {
                        "families": [
                            {"id": "one_x_two", "label": "1X2", "status": "MATURING", "current": 47, "target": 50}
                        ]
                    }
                }
            },
        },
    }


def test_free_payload_redacts_premium_fields_and_rows():
    result = subscriber_app_v4.build_subscriber_payload(_product(), subscriber_app_v4.anonymous_entitlement())
    assert result["effective_plan"] == "FREE"
    assert result["premium_unlocked"] is False
    assert result["pro"] is None
    row = result["public"]["verified_slate"]["rows"][0]
    assert row["fixture_id"] == 101
    assert "price" not in row
    assert "bookmaker" not in row
    assert "selection" not in row
    assert "edge" not in row
    assert "p_model_calibrated" not in row
    assert result["locked_counts"]["strong_sport_signals"] == 4
    assert result["public"]["maturity_overview"][0]["current"] == 47
    assert result["provider_requests_added"] == 0


def test_pro_payload_unlocks_advanced_views():
    entitlement = {
        "authenticated": True,
        "effective_plan": "PRO",
        "effective_plan_reason": "PRO_ACTIVE",
        "feature_access": subscription_entitlements_v4.feature_access("PRO"),
        "user": {"id": "user-1", "email": "member@example.com"},
    }
    result = subscriber_app_v4.build_subscriber_payload(_product(), entitlement)
    assert result["premium_unlocked"] is True
    assert result["pro"]["strong_sport_signals"]["rows"][0]["price"] == -110
    assert result["pro"]["value_plays"]["rows"][0]["edge"] == 0.08
    assert "control_tower" not in result["pro"]
    assert result["user"]["email"] == "member@example.com"


def test_app_html_has_auth_checkout_and_no_backend_secret(monkeypatch):
    monkeypatch.setenv("SOCCER_SUPABASE_URL", "https://project.supabase.co")
    monkeypatch.setenv("SOCCER_SUPABASE_PUBLISHABLE_KEY", "sb_publishable_test")
    rendered = subscriber_app_v4._app_html()
    assert "SUBSCRIBER APP · V220" in rendered
    assert "/auth/v1/token?grant_type=password" in rendered
    assert "/auth/v1/signup" in rendered
    assert "/functions/v1/create-checkout-session" in rendered
    assert "/functions/v1/create-customer-portal" in rendered
    assert "sessionStorage" in rendered
    assert "STRIPE_SECRET_KEY" not in rendered
    assert "STRIPE_WEBHOOK_SECRET" not in rendered
    assert "sb_secret_" not in rendered


def test_server_router_receives_v220_routes():
    from mcp_gateway import server

    paths = {getattr(route, "path", None) for route in server.app.router.routes}
    assert "/app" in paths
    assert "/app/data" in paths
