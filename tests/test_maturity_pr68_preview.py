"""Contract tests for an isolated read-only PR #68 Render preview.

Mocks only the production account lookup; no network, DB, provider or bet calls.
"""
from __future__ import annotations

import json

from starlette.testclient import TestClient

from mcp_gateway import maturity_pr68_preview as p


def _account(plan="FREE", *, owner=False):
    return {
        "status": "ACCOUNT_ACCESS_READY",
        "access": {
            "authenticated": True,
            "effective_plan": plan,
            "display_role": "OWNER" if owner else plan,
            "owner": owner,
            "premium_unlocked": owner or plan == "PRO",
        },
        "user": {"email": "member@example.test"},
    }


def test_preview_health_is_isolated_and_never_runs_scheduler():
    client = TestClient(p.app)
    payload = client.get("/health").json()
    assert payload["status"] == "ok"
    assert payload["preview_only"] is True
    assert payload["scheduler_enabled"] is False
    assert payload["database_writes_enabled"] is False
    assert payload["betting_mutations_enabled"] is False
    assert client.post("/internal/tick").status_code == 404
    assert client.get("/internal/tick").status_code == 404
    assert client.post("/app/api/v2/maturity").status_code == 405
    assert client.get("/app/api/v2/performance").status_code == 404
    assert client.get("/app/api/v2/bets").status_code == 404


def test_preview_boots_maturity_and_marks_match_center_as_sample():
    client = TestClient(p.app)
    location = client.get("/", follow_redirects=False).headers["location"]
    assert location == "/app-v3-react/?sample=1&maturity-preview=1"
    assert client.get("/app/api/v2/today").json()["slate"]["rows"] == []


def test_maturity_rejects_unauthenticated_before_loading_research(monkeypatch):
    async def cannot_call(*args, **kwargs):
        raise AssertionError("Unauthorized request must not hit production")
    monkeypatch.setattr(p, "_read_upstream", cannot_call)
    assert TestClient(p.app).get("/app/api/v2/maturity").status_code == 401


def test_maturity_propagates_upstream_outage_without_leaking_research(monkeypatch):
    async def offline(*args, **kwargs):
        return 503, {"error": "UPSTREAM_AUTH_UNAVAILABLE"}
    monkeypatch.setattr(p, "_read_upstream", offline)
    r = TestClient(p.app).get(
        "/app/api/v2/maturity", headers={"authorization": "Bearer test-access"}
    )
    assert r.status_code == 503
    assert "market_rows" not in r.json()


def test_maturity_free_is_403_and_has_no_research(monkeypatch):
    async def free(*args, **kwargs):
        return 200, _account("FREE")
    monkeypatch.setattr(p, "_read_upstream", free)
    def block_data():
        raise AssertionError("FREE cannot load research")
    monkeypatch.setattr(p.subscriber_maturity_v232, "load_maturity_evidence", block_data)
    r = TestClient(p.app).get(
        "/app/api/v2/maturity", headers={"authorization": "Bearer test-access"}
    )
    assert r.status_code == 403
    assert "market_rows" not in r.json()


def test_pro_and_owner_reuse_same_research_without_mutating_shared_cache(monkeypatch):
    cache = {
        "status": "PARTIAL",
        "market_rows": [{"key": "1X2", "true_clv_rows": 53, "production_promotion_allowed": False}],
        "production_promotion_allowed": False,
    }
    monkeypatch.setattr(p.subscriber_maturity_v232, "load_maturity_evidence", lambda: cache)
    role = {"name": "PRO"}
    async def authorized(*args, **kwargs):
        return 200, _account(role["name"], owner=role["name"] == "OWNER")
    monkeypatch.setattr(p, "_read_upstream", authorized)
    client = TestClient(p.app)
    role["name"] = "PRO"
    pro = client.get("/app/api/v2/maturity", headers={"authorization": "Bearer token"})
    assert pro.status_code == 200
    assert pro.json()["owner"] is False
    assert pro.json()["source_scope"] == "PUBLIC_GITHUB_STATE_READ_ONLY"
    assert pro.json()["production_promotion_allowed"] is False
    role["name"] = "OWNER"
    owner = client.get("/app/api/v2/maturity", headers={"authorization": "Bearer token"})
    assert owner.status_code == 200
    assert owner.json()["owner"] is True
    assert owner.json()["effective_plan"] == "OWNER"
    assert "owner" not in cache and "effective_plan" not in cache
    assert cache["market_rows"][0]["true_clv_rows"] == 53


def test_account_contract_exposes_no_upstream_subscription_payload(monkeypatch):
    async def authorized(*args, **kwargs):
        payload = _account("PRO")
        payload["user"]["id"] = "private-id"
        payload["billing_state"] = {"sensitive": "do-not-forward"}
        payload["persisted_entitlement"] = {"sensitive": "do-not-forward"}
        return 200, payload
    monkeypatch.setattr(p, "_read_upstream", authorized)
    r = TestClient(p.app).get(
        "/app/api/v2/account", headers={"authorization": "Bearer access"}
    )
    assert r.status_code == 200
    assert r.json()["access"]["premium_unlocked"] is True
    assert "private-id" not in r.text and "do-not-forward" not in r.text
    assert "no-store" in r.headers["cache-control"]


def test_auth_config_only_republishes_public_fields(monkeypatch):
    async def configured(*args, **kwargs):
        return 200, {
            "supabase_url": "https://example.supabase.co",
            "publishable_key": "public-example",
            "auth_configured": True,
            "service_role_key": "not-to-be-forwarded",
        }
    monkeypatch.setattr(p, "_read_upstream", configured)
    r = TestClient(p.app).get("/app-v3-react/auth-config")
    assert r.status_code == 200
    assert r.json()["auth_configured"] is True
    assert r.json()["api_base"] == "/app/api/v2"
    assert "service_role_key" not in r.json()
