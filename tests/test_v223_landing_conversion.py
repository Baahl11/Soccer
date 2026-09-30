from __future__ import annotations

from mcp_gateway import landing_page_v4


def test_landing_is_bilingual_and_points_to_explorer():
    rendered = landing_page_v4.render_landing()
    assert "Stop betting blind." in rendered
    assert "Deja de apostar a ciegas." in rendered
    assert 'href="/app"' in rendered
    assert 'id="enBtn"' in rendered
    assert 'id="esBtn"' in rendered
    assert "soccer_edge_locale" in rendered


def test_landing_positions_model_vs_market_not_guaranteed_picks():
    rendered = landing_page_v4.render_landing()
    assert "MODEL VS MARKET" in rendered
    assert "MODELO VS MERCADO" in rendered
    assert "NO BET" in rendered
    assert "not guaranteed outcomes" in rendered
    assert "no resultados garantizados" in rendered


def test_landing_has_conversion_and_compliance_copy():
    rendered = landing_page_v4.render_landing()
    assert "Create free account" in rendered
    assert "Crear cuenta gratis" in rendered
    assert "Betting involves risk" in rendered
    assert "Apostar implica riesgo" in rendered
    assert 'name="description"' in rendered
    assert 'property="og:title"' in rendered


def test_server_exposes_v223_root_without_removing_existing_surfaces():
    from mcp_gateway import server

    paths = [getattr(route, "path", None) for route in server.app.router.routes]
    names = [getattr(route, "name", None) for route in server.app.router.routes]
    assert "/" in paths
    assert "/app" in paths
    assert "/app/data" in paths
    assert "/dashboard" in paths
    assert "v223_landing_page" in names


def test_landing_layer_does_not_touch_betting_runtime():
    source = open(landing_page_v4.__file__, "r", encoding="utf-8").read().lower()
    for forbidden in ("api-football", "threshold", "decision_weight", "provider_requests", "model_weights"):
        assert forbidden not in source
