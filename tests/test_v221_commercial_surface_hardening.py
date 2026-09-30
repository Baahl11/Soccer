from __future__ import annotations

import asyncio

from starlette.requests import Request
from starlette.responses import JSONResponse

from mcp_gateway import commercial_surface_guard_v4


def _request(path: str = "/") -> Request:
    return Request({"type": "http", "http_version": "1.1", "method": "GET", "scheme": "https", "path": path, "raw_path": path.encode(), "query_string": b"", "headers": [], "client": ("127.0.0.1", 1234), "server": ("testserver", 443)})


def test_dashboard_guard_redirects_public_operator_surface_to_app():
    response = asyncio.run(commercial_surface_guard_v4.dashboard_guard(_request("/dashboard")))
    assert response.status_code == 307
    assert response.headers["location"] == "/app"


def test_product_views_guard_delegates_to_entitlement_filtered_app_data(monkeypatch):
    async def fake_app_data(request):
        return JSONResponse({"effective_plan": "FREE", "premium_unlocked": False, "provider_requests_added": 0})

    monkeypatch.setattr(commercial_surface_guard_v4.subscriber_app_v4, "app_data", fake_app_data)
    response = asyncio.run(commercial_surface_guard_v4.product_views_guard(_request("/product/views")))
    assert response.status_code == 200
    assert b'"effective_plan":"FREE"' in response.body
    assert b'"premium_unlocked":false' in response.body
    assert b'"provider_requests_added":0' in response.body


def test_server_prepends_v221_guards_ahead_of_legacy_routes():
    from mcp_gateway import server

    dashboard_routes = [route for route in server.app.router.routes if getattr(route, "path", None) == "/dashboard"]
    product_routes = [route for route in server.app.router.routes if getattr(route, "path", None) == "/product/views"]

    assert dashboard_routes
    assert product_routes
    assert getattr(dashboard_routes[0], "name", None) == "v221_dashboard_guard"
    assert getattr(product_routes[0], "name", None) == "v221_product_views_guard"


def test_v221_guard_layer_has_no_provider_or_model_mutation_contract():
    source = open(commercial_surface_guard_v4.__file__, "r", encoding="utf-8").read()
    assert "api-football" not in source.lower()
    assert "provider_requests" not in source.lower()
    assert "threshold" not in source.lower()
    assert "model_weights" not in source.lower()
