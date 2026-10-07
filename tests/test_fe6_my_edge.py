from __future__ import annotations

from mcp_gateway import subscriber_contract_v2
from mcp_gateway import subscriber_frontend_v2
from mcp_gateway import subscriber_saved_items_v4


class _Response:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else []

    def json(self):
        return self._payload


class _Client:
    def __init__(self):
        self.calls = []
        self.responses = []

    def queue(self, response):
        self.responses.append(response)

    def _next(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self.responses.pop(0) if self.responses else _Response()

    def get(self, url, **kwargs):
        return self._next("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self._next("POST", url, **kwargs)

    def delete(self, url, **kwargs):
        return self._next("DELETE", url, **kwargs)


def _config(monkeypatch):
    monkeypatch.setattr(
        subscriber_saved_items_v4.supabase_auth_v4,
        "auth_config",
        lambda: {
            "configured": True,
            "project_url": "https://example.supabase.co",
            "publishable_key": "public-key",
        },
    )


def test_fe6_saved_item_key_is_stable_and_market_specific():
    a = subscriber_saved_items_v4.sanitize_item(
        {
            "item_type": "BET",
            "fixture_id": 123,
            "market_family": "FT_TOTALS",
            "market_name": "Goals Over/Under",
            "selection": "Over",
            "line": 2.5,
            "payload": {"home_team": "A", "away_team": "B"},
        }
    )
    b = subscriber_saved_items_v4.sanitize_item(
        {
            "item_type": "BET",
            "fixture_id": 123,
            "market_family": "FT_TOTALS",
            "market_name": "Goals Over/Under",
            "selection": "Over",
            "line": 3.5,
            "payload": {"home_team": "A", "away_team": "B"},
        }
    )

    assert a["item_key"] == subscriber_saved_items_v4.sanitize_item(dict(a))["item_key"]
    assert a["item_key"] != b["item_key"]


def test_fe6_saved_item_payload_is_allowlisted_product_state_only():
    row = subscriber_saved_items_v4.sanitize_item(
        {
            "item_type": "LEAN",
            "fixture_id": 9,
            "market_family": "1X2",
            "selection": "Home",
            "payload": {
                "home_team": "Home",
                "away_team": "Away",
                "classification": "LEAN",
                "raw_secret": "must-not-persist",
                "model_weight": 0.88,
            },
        }
    )

    assert row["payload"]["classification"] == "LEAN"
    assert "raw_secret" not in row["payload"]
    assert "model_weight" not in row["payload"]


def test_fe6_invalid_type_cannot_be_persisted():
    try:
        subscriber_saved_items_v4.sanitize_item(
            {"item_type": "PASS", "fixture_id": 1}
        )
    except ValueError as exc:
        assert str(exc) == "ITEM_TYPE_NOT_ALLOWED"
    else:
        raise AssertionError("PASS should not be accepted as a saved action type")


def test_fe6_list_saved_uses_verified_user_and_rls_rest_query(monkeypatch):
    _config(monkeypatch)
    client = _Client()
    client.queue(
        _Response(
            200,
            [
                {
                    "item_key": "bet:1:key",
                    "item_type": "BET",
                    "fixture_id": 1,
                }
            ],
        )
    )

    result = subscriber_saved_items_v4.list_saved(
        "jwt",
        verified_user_id="11111111-1111-1111-1111-111111111111",
        client=client,
    )

    assert result["ok"] is True
    assert result["rows"][0]["item_type"] == "BET"
    method, url, kwargs = client.calls[0]
    assert method == "GET"
    assert url.endswith("/rest/v1/subscriber_saved_items")
    assert kwargs["params"]["user_id"] == (
        "eq.11111111-1111-1111-1111-111111111111"
    )
    assert kwargs["headers"]["Authorization"] == "Bearer jwt"


def test_fe6_save_item_upserts_own_user_only(monkeypatch):
    _config(monkeypatch)
    client = _Client()
    client.queue(_Response(201, [{"item_key": "saved", "item_type": "MATCH"}]))

    result = subscriber_saved_items_v4.save_item(
        "jwt",
        {
            "item_type": "MATCH",
            "fixture_id": 22,
            "payload": {"home_team": "A", "away_team": "B"},
        },
        verified_user_id="22222222-2222-2222-2222-222222222222",
        client=client,
    )

    assert result["ok"] is True
    method, _, kwargs = client.calls[0]
    assert method == "POST"
    assert kwargs["params"]["on_conflict"] == "user_id,item_key"
    assert kwargs["json"]["user_id"] == "22222222-2222-2222-2222-222222222222"
    assert kwargs["json"]["item_type"] == "MATCH"


def test_fe6_delete_item_filters_by_verified_user_and_item_key(monkeypatch):
    _config(monkeypatch)
    client = _Client()
    client.queue(_Response(204, None))

    result = subscriber_saved_items_v4.delete_item(
        "jwt",
        "match:22:key",
        verified_user_id="33333333-3333-3333-3333-333333333333",
        client=client,
    )

    assert result["ok"] is True
    method, _, kwargs = client.calls[0]
    assert method == "DELETE"
    assert kwargs["params"]["user_id"] == (
        "eq.33333333-3333-3333-3333-333333333333"
    )
    assert kwargs["params"]["item_key"] == "eq.match:22:key"


def test_fe6_contract_keeps_my_edge_out_of_model_inputs():
    contract = subscriber_saved_items_v4.contract()
    subscriber_contract = subscriber_contract_v2.contract()

    assert contract["rls_required"] is True
    assert contract["auth_uid_owns_rows"] is True
    assert contract["model_input_allowed"] is False
    assert contract["provider_requests_added"] == 0
    assert contract["canonical_bet_logic_changed"] is False
    assert contract["model_weights_changed"] is False
    assert subscriber_contract["my_edge_model_input_allowed"] is False
    assert "my_edge" in subscriber_contract["resources"]


def test_fe6_frontend_has_real_my_edge_save_remove_flow():
    html = subscriber_frontend_v2.render()
    contract = subscriber_frontend_v2.contract()

    assert "Saved decisions and tracked matches, persisted to your authenticated account." in html
    assert "data-save-key" in html
    assert "data-save-match" in html
    assert "apiWrite('/my-edge','POST'" in html
    assert "apiWrite('/my-edge?item_key='" in html
    assert "Account-persisted product state only; never a model input." in html
    assert 'data-page="myedge">My Edge</button>' in html
    assert contract["my_edge_model_input_allowed"] is False


def test_fe6_routes_install_my_edge_with_get_post_delete():
    import mcp_gateway
    from mcp.server.fastmcp import FastMCP

    app = FastMCP(
        "fe6-route-test",
        stateless_http=True,
        json_response=True,
    ).streamable_http_app()

    routes = [
        route
        for route in app.router.routes
        if getattr(route, "path", None) == "/app/api/v2/my-edge"
    ]
    assert len(routes) == 1
    methods = set(getattr(routes[0], "methods", set()) or set())
    assert {"GET", "POST", "DELETE"}.issubset(methods)
