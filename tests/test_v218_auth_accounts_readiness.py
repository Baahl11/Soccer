import httpx

from mcp_gateway import supabase_auth_v4 as v


class DummyClient:
    def __init__(self, response):
        self.response = response
        self.calls = []
        self.closed = False

    def get(self, url, headers=None):
        self.calls.append((url, dict(headers or {})))
        return self.response

    def close(self):
        self.closed = True


def test_v218_is_disabled_without_dedicated_project(monkeypatch):
    monkeypatch.delenv("SOCCER_SUPABASE_URL", raising=False)
    monkeypatch.delenv("SOCCER_SUPABASE_PUBLISHABLE_KEY", raising=False)
    config = v.auth_config()
    assert config["status"] == "AUTH_NOT_CONFIGURED"
    assert config["configured"] is False
    assert config["secret_key_required"] is False
    assert config["service_role_required"] is False
    assert config["billing_enabled"] is False
    assert config["entitlements_enforced"] is False
    assert config["provider_requests_added"] == 0
    assert v.verify_access_token("token")["status"] == "AUTH_NOT_CONFIGURED"


def test_v218_public_config_contains_only_publishable_auth_material(monkeypatch):
    monkeypatch.setenv("SOCCER_SUPABASE_URL", "https://soccer-edge.supabase.co/")
    monkeypatch.setenv("SOCCER_SUPABASE_PUBLISHABLE_KEY", "sb_publishable_example")
    config = v.public_auth_config()
    assert config["status"] == "AUTH_READY"
    assert config["project_url"] == "https://soccer-edge.supabase.co"
    assert config["publishable_key"] == "sb_publishable_example"
    assert "secret" not in " ".join(config.keys()).lower()
    assert "service_role" not in " ".join(config.keys()).lower()


def test_v218_verifies_user_token_against_supabase_auth_user_endpoint(monkeypatch):
    monkeypatch.setenv("SOCCER_SUPABASE_URL", "https://soccer-edge.supabase.co")
    monkeypatch.setenv("SOCCER_SUPABASE_PUBLISHABLE_KEY", "sb_publishable_example")
    response = httpx.Response(
        200,
        request=httpx.Request("GET", "https://soccer-edge.supabase.co/auth/v1/user"),
        json={"id": "user-123", "email": "user@example.com", "aud": "authenticated", "role": "authenticated"},
    )
    client = DummyClient(response)
    result = v.verify_access_token("jwt-user-token", client=client)
    assert result["ok"] is True
    assert result["status"] == "AUTHENTICATED"
    assert result["user"]["id"] == "user-123"
    assert result["user"]["email"] == "user@example.com"
    assert result["authorization_metadata_ignored"] is True
    assert client.calls == [(
        "https://soccer-edge.supabase.co/auth/v1/user",
        {
            "apikey": "sb_publishable_example",
            "Authorization": "Bearer jwt-user-token",
            "Accept": "application/json",
        },
    )]


def test_v218_rejects_invalid_token_and_parses_bearer(monkeypatch):
    monkeypatch.setenv("SOCCER_SUPABASE_URL", "https://soccer-edge.supabase.co")
    monkeypatch.setenv("SOCCER_SUPABASE_PUBLISHABLE_KEY", "sb_publishable_example")
    response = httpx.Response(
        401,
        request=httpx.Request("GET", "https://soccer-edge.supabase.co/auth/v1/user"),
        json={"message": "invalid token"},
    )
    result = v.verify_access_token("bad-token", client=DummyClient(response))
    assert result == {"ok": False, "status": "AUTH_REJECTED_401", "user": None}
    assert v.bearer_token("Bearer abc.def.ghi") == "abc.def.ghi"
    assert v.bearer_token("bearer token") == "token"
    assert v.bearer_token("Basic abc") == ""


def test_v218_product_fragment_never_claims_auth_when_unconfigured(monkeypatch):
    monkeypatch.delenv("SOCCER_SUPABASE_URL", raising=False)
    monkeypatch.delenv("SOCCER_SUPABASE_PUBLISHABLE_KEY", raising=False)
    rendered = v.render_fragment()
    assert "AUTH_NOT_CONFIGURED" in rendered
    assert "Sign-in stays disabled" in rendered
    assert "existing projects are not reused" in rendered


def test_v218_source_does_not_use_unsafe_authorization_metadata():
    source = open("mcp_gateway/supabase_auth_v4.py", encoding="utf-8").read()
    assert "user_metadata" not in source
    assert "service_role" not in source
    assert "SOCCER_SUPABASE_SECRET" not in source
    assert "SOCCER_SUPABASE_SERVICE_ROLE" not in source
