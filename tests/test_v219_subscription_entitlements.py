from datetime import datetime, timezone

import httpx

from mcp_gateway import subscription_entitlements_v4 as v


class DummyClient:
    def __init__(self, response):
        self.response = response
        self.calls = []
        self.closed = False

    def get(self, url, headers=None, params=None):
        self.calls.append((url, dict(headers or {}), dict(params or {})))
        return self.response

    def close(self):
        self.closed = True


def _auth_ok(monkeypatch):
    monkeypatch.setattr(
        v.supabase_auth_v4,
        "verify_access_token",
        lambda token, client=None: {
            "ok": True,
            "status": "AUTHENTICATED",
            "user": {"id": "user-123", "email": "user@example.com"},
        },
    )
    monkeypatch.setattr(
        v.supabase_auth_v4,
        "auth_config",
        lambda: {
            "configured": True,
            "project_url": "https://soccer-edge.supabase.co",
            "publishable_key": "sb_publishable_example",
        },
    )


def test_v219_contract_preserves_v215_free_and_pro_product_split():
    contract = v.plan_contract()
    assert contract["default_plan"] == "FREE"
    assert contract["billing_enabled"] is False
    assert contract["entitlements_enforced"] is True
    assert contract["authorization_source"] == "SUPABASE_RLS_SUBSCRIPTION_ENTITLEMENTS"
    assert contract["provider_requests_added"] == 0
    assert "Today's verified slate" in contract["free_features"]
    assert "Strong signal desk" in contract["pro_features"]
    assert "Calibrated value plays" in contract["pro_features"]


def test_v219_missing_entitlement_row_is_free(monkeypatch):
    _auth_ok(monkeypatch)
    response = httpx.Response(
        200,
        request=httpx.Request("GET", "https://soccer-edge.supabase.co/rest/v1/subscription_entitlements"),
        json=[],
    )
    client = DummyClient(response)
    result = v.resolve_entitlement("jwt-token", client=client)
    assert result["ok"] is True
    assert result["effective_plan"] == "FREE"
    assert result["effective_plan_reason"] == "DEFAULT_FREE_NO_ENTITLEMENT_ROW"
    assert result["feature_access"]["verified_slate"] is True
    assert result["feature_access"]["strong_signal_desk"] is False
    assert result["billing_enabled"] is False
    assert result["entitlements_enforced"] is True
    assert client.calls[0][2]["user_id"] == "eq.user-123"
    assert client.calls[0][1]["Authorization"] == "Bearer jwt-token"


def test_v219_active_pro_unlocks_pro_features(monkeypatch):
    _auth_ok(monkeypatch)
    response = httpx.Response(
        200,
        request=httpx.Request("GET", "https://soccer-edge.supabase.co/rest/v1/subscription_entitlements"),
        json=[{
            "user_id": "user-123",
            "plan": "PRO",
            "status": "ACTIVE",
            "source": "MANUAL",
            "starts_at": "2026-09-01T00:00:00+00:00",
            "valid_until": "2026-10-31T00:00:00+00:00",
            "updated_at": "2026-09-30T00:00:00+00:00",
        }],
    )
    result = v.resolve_entitlement(
        "jwt-token",
        client=DummyClient(response),
        now=datetime(2026, 9, 30, tzinfo=timezone.utc),
    )
    assert result["effective_plan"] == "PRO"
    assert result["effective_plan_reason"] == "PRO_ACTIVE"
    assert result["feature_access"]["strong_signal_desk"] is True
    assert result["feature_access"]["premium_match_detail"] is True
    assert result["feature_access"]["favorites_alerts"] is True


def test_v219_inactive_or_expired_pro_fails_closed_to_free():
    now = datetime(2026, 9, 30, tzinfo=timezone.utc)
    assert v.effective_plan({"plan": "PRO", "status": "PAST_DUE"}, now=now) == ("FREE", "PRO_STATUS_PAST_DUE")
    assert v.effective_plan({"plan": "PRO", "status": "CANCELED"}, now=now) == ("FREE", "PRO_STATUS_CANCELED")
    assert v.effective_plan({
        "plan": "PRO",
        "status": "ACTIVE",
        "valid_until": "2026-09-29T23:59:59+00:00",
    }, now=now) == ("FREE", "PRO_EXPIRED")


def test_v219_entitlement_lookup_failure_never_promotes_to_pro(monkeypatch):
    _auth_ok(monkeypatch)
    response = httpx.Response(
        403,
        request=httpx.Request("GET", "https://soccer-edge.supabase.co/rest/v1/subscription_entitlements"),
        json={"message": "forbidden"},
    )
    result = v.resolve_entitlement("jwt-token", client=DummyClient(response))
    assert result["ok"] is False
    assert result["effective_plan"] == "FREE"
    assert result["feature_access"]["strong_signal_desk"] is False


def test_v219_requires_valid_auth_before_entitlement_lookup(monkeypatch):
    monkeypatch.setattr(
        v.supabase_auth_v4,
        "verify_access_token",
        lambda token, client=None: {"ok": False, "status": "AUTH_REJECTED_401", "user": None},
    )
    result = v.resolve_entitlement("bad-token")
    assert result["ok"] is False
    assert result["authenticated"] is False
    assert result["effective_plan"] is None
    assert result["feature_access"] == {}


def test_v219_source_does_not_trust_user_metadata_or_frontend_secrets():
    source = open("mcp_gateway/subscription_entitlements_v4.py", encoding="utf-8").read()
    assert "user_metadata" not in source
    assert "service_role" not in source
    assert "SOCCER_SUPABASE_SECRET" not in source
    assert "SOCCER_SUPABASE_SERVICE_ROLE" not in source
    assert "SUPABASE_RLS_SUBSCRIPTION_ENTITLEMENTS" in source
