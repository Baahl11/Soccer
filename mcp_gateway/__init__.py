"""Soccer Edge API MCP gateway package."""

from __future__ import annotations


def _install_commercial_product_layers() -> None:
    from mcp_gateway import commercial_shell_v4, match_detail_v4, product_dashboard_v4, public_performance_v4, subscription_entitlements_v4, supabase_auth_v4
    if hasattr(product_dashboard_v4, "_v215_operator_render_dashboard"):
        return
    operator_renderer = product_dashboard_v4.render_dashboard
    product_dashboard_v4._v215_operator_render_dashboard = operator_renderer
    def render_dashboard_with_product_layers(product_payload):
        rendered = operator_renderer(product_payload)
        fragments = commercial_shell_v4.render_membership_fragment(product_payload) + public_performance_v4.render_fragment() + match_detail_v4.render_fragment(product_payload) + supabase_auth_v4.render_fragment() + subscription_entitlements_v4.render_fragment()
        marker = "</main>"
        return rendered.replace(marker, fragments + marker, 1) if marker in rendered else rendered + fragments
    product_dashboard_v4.render_dashboard = render_dashboard_with_product_layers


def _install_subscriber_i18n_layer() -> None:
    from mcp_gateway import subscriber_app_v4, subscriber_i18n_v4
    if hasattr(subscriber_app_v4, "_v222_base_app_html"):
        return
    base_app_html = subscriber_app_v4._app_html
    subscriber_app_v4._v222_base_app_html = base_app_html
    def bilingual_app_html() -> str:
        return subscriber_i18n_v4.inject_i18n(base_app_html())
    subscriber_app_v4._app_html = bilingual_app_html


def _install_product_analytics_layer() -> None:
    from mcp_gateway import landing_page_v4, product_analytics_v4, subscriber_app_v4
    if not hasattr(landing_page_v4, "_v224_base_render_landing"):
        base_landing = landing_page_v4.render_landing
        landing_page_v4._v224_base_render_landing = base_landing
        def tracked_landing() -> str:
            return product_analytics_v4.inject_analytics(base_landing(), surface="landing")
        landing_page_v4.render_landing = tracked_landing
    if not hasattr(subscriber_app_v4, "_v224_base_app_html"):
        base_app_html = subscriber_app_v4._app_html
        subscriber_app_v4._v224_base_app_html = base_app_html
        def tracked_app_html() -> str:
            return product_analytics_v4.inject_analytics(base_app_html(), surface="app")
        subscriber_app_v4._app_html = tracked_app_html


def _install_subscriber_frontend_hotfix() -> None:
    from mcp_gateway import subscriber_app_v4, subscriber_frontend_hotfix_v4
    subscriber_frontend_hotfix_v4.install(subscriber_app_v4)


def _install_v226_regional_billing_layer() -> None:
    """Keep legacy subscriber_app billing compatible while the consolidated product owns /app."""
    from mcp_gateway import subscriber_app_v4, subscriber_billing_market_v226
    if hasattr(subscriber_app_v4, "_v226_base_app_html"):
        return
    base_app_html = subscriber_app_v4._app_html
    subscriber_app_v4._v226_base_app_html = base_app_html
    def regional_billing_app_html() -> str:
        return subscriber_billing_market_v226.inject(base_app_html())
    subscriber_app_v4._app_html = regional_billing_app_html


def _install_v236_today_layer() -> None:
    """Render the subscriber Today surface from the approved mockup hierarchy."""
    from mcp_gateway import subscriber_product_v235, subscriber_today_v236
    subscriber_today_v236.install(subscriber_product_v235)


def _install_v237_visual_layer() -> None:
    """Match the approved mockup crest scale and faceoff composition."""
    from mcp_gateway import subscriber_product_v235, subscriber_visual_v237
    subscriber_visual_v237.install(subscriber_product_v235)


def _install_subscriber_app_routes() -> None:
    """Add customer routes while preserving the FastMCP ASGI lifespan."""
    from mcp.server.fastmcp import FastMCP
    from starlette.responses import RedirectResponse
    from starlette.routing import Mount, Route
    if hasattr(FastMCP, "_v220_streamable_http_app"):
        return
    original = FastMCP.streamable_http_app
    FastMCP._v220_streamable_http_app = original

    async def auth_landing(request):
        from mcp_gateway import landing_page_v2
        return await landing_page_v2.landing_page(request)

    async def legacy_preview_landing(request):
        return RedirectResponse(url="/app", status_code=307)

    def streamable_http_app_with_subscriber_routes(self, *args, **kwargs):
        app = original(self, *args, **kwargs)
        from mcp_gateway import (
            content_factory_http_v4,
            landing_page_v2,
            subscriber_product_v235,
            subscriber_contract_v2,
            subscriber_frontend_v2,
            subscriber_frontend_v3,
            subscriber_react_v3,
            subscriber_pwa_v2,
            ui_golden_master_v1,
            subscriber_preview_data_v231,
            subscriber_preview_maturity_v232,
            subscriber_preview_performance_v231,
            subscriber_today_v236,
        )
        existing_paths = {getattr(route, "path", None) for route in app.router.routes}
        existing_names = {getattr(route, "name", None) for route in app.router.routes}
        additions = []
        if "v226_auth_landing" not in existing_names:
            additions.append(Route("/", auth_landing, methods=["GET"], name="v226_auth_landing"))
        if "/internal/content-packages" not in existing_paths:
            additions.append(Route("/internal/content-packages", content_factory_http_v4.content_packages, methods=["POST"], name="v225_content_packages"))
        if "/landing-v2" not in existing_paths:
            additions.append(Route("/landing-v2", landing_page_v2.landing_page, methods=["GET"], name="landing_v2_preview"))
        if "/app" not in existing_paths:
            additions.append(Route("/app", subscriber_frontend_v2.app_page, methods=["GET"], name="subscriber_frontend_v2_primary"))
        if "/app-v2" not in existing_paths:
            additions.append(Route("/app-v2", subscriber_frontend_v2.app_page, methods=["GET"], name="subscriber_frontend_v2_preview"))
        if "/app-v3" not in existing_paths:
            additions.append(Route("/app-v3", subscriber_frontend_v3.app_page, methods=["GET"], name="subscriber_frontend_v3_preview"))
        if "/app-v3/match/{fixture_id:int}" not in existing_paths:
            additions.append(Route("/app-v3/match/{fixture_id:int}", subscriber_frontend_v3.app_page, methods=["GET"], name="subscriber_frontend_v3_match_page"))
        if "/app-v3-react" not in existing_paths:
            additions.append(Route("/app-v3-react", subscriber_react_v3.app_page, methods=["GET"], name="subscriber_frontend_v3_react_preview"))
        if "/app-v3-react/match/{fixture_id:int}" not in existing_paths:
            additions.append(Route("/app-v3-react/match/{fixture_id:int}", subscriber_react_v3.app_page, methods=["GET"], name="subscriber_frontend_v3_react_match"))
        if "/app-v3-react/auth-config" not in existing_paths:
            additions.append(Route("/app-v3-react/auth-config", subscriber_react_v3.auth_config, methods=["GET"], name="subscriber_frontend_v3_react_auth_config"))
        if "/app-v3-react/assets" not in existing_paths:
            additions.append(Mount("/app-v3-react/assets", app=subscriber_react_v3.assets, name="subscriber_frontend_v3_react_assets"))
        if "/design-lab/match-center" not in existing_paths:
            additions.append(Route("/design-lab/match-center", ui_golden_master_v1.design_match_center, methods=["GET"], name="soccer_edge_match_center_golden_master"))
        if "/app/match/{fixture_id:int}" not in existing_paths:
            additions.append(Route("/app/match/{fixture_id:int}", subscriber_frontend_v2.app_page, methods=["GET"], name="subscriber_frontend_v2_match_page"))
        if "/app.webmanifest" not in existing_paths:
            additions.append(Route("/app.webmanifest", subscriber_pwa_v2.manifest, methods=["GET"], name="subscriber_pwa_manifest"))
        if "/sw.js" not in existing_paths:
            additions.append(Route("/sw.js", subscriber_pwa_v2.service_worker, methods=["GET"], name="subscriber_pwa_service_worker"))
        if "/pwa/icon.svg" not in existing_paths:
            additions.append(Route("/pwa/icon.svg", subscriber_pwa_v2.icon, methods=["GET"], name="subscriber_pwa_icon"))
        if "/app/data" not in existing_paths:
            additions.append(Route("/app/data", subscriber_product_v235.app_data, methods=["GET"], name="v235_subscriber_data"))
        if "/app/match" not in existing_paths:
            additions.append(Route("/app/match", subscriber_product_v235.match_data, methods=["GET"], name="v235_subscriber_match_data"))
        if "/app/fixture-identities" not in existing_paths:
            additions.append(Route("/app/fixture-identities", subscriber_today_v236.fixture_identities, methods=["GET"], name="v236_fixture_identities"))
        if "/app/api/v2/today" not in existing_paths:
            additions.append(Route("/app/api/v2/today", subscriber_contract_v2.today, methods=["GET"], name="subscriber_v2_today"))
        if "/app/api/v2/picks" not in existing_paths:
            additions.append(Route("/app/api/v2/picks", subscriber_contract_v2.picks, methods=["GET"], name="subscriber_v2_picks"))
        if "/app/api/v2/leans" not in existing_paths:
            additions.append(Route("/app/api/v2/leans", subscriber_contract_v2.leans, methods=["GET"], name="subscriber_v2_leans"))
        if "/app/api/v2/watches" not in existing_paths:
            additions.append(Route("/app/api/v2/watches", subscriber_contract_v2.watches, methods=["GET"], name="subscriber_v2_watches"))
        if "/app/api/v2/maturity" not in existing_paths:
            additions.append(Route("/app/api/v2/maturity", subscriber_contract_v2.maturity, methods=["GET"], name="subscriber_v2_maturity"))
        if "/app/api/v2/performance" not in existing_paths:
            additions.append(Route("/app/api/v2/performance", subscriber_contract_v2.performance, methods=["GET"], name="subscriber_v2_performance"))
        if "/app/api/v2/my-edge" not in existing_paths:
            additions.append(Route("/app/api/v2/my-edge", subscriber_contract_v2.my_edge, methods=["GET", "POST", "DELETE"], name="subscriber_v2_my_edge"))
        if "/app/api/v2/account" not in existing_paths:
            additions.append(Route("/app/api/v2/account", subscriber_contract_v2.account, methods=["GET"], name="subscriber_v2_account"))
        if "/app/api/v2/match/{fixture_id:int}" not in existing_paths:
            additions.append(Route("/app/api/v2/match/{fixture_id:int}", subscriber_contract_v2.match_detail, methods=["GET"], name="subscriber_v2_match"))
        if "/app-preview" not in existing_paths:
            additions.append(Route("/app-preview", legacy_preview_landing, methods=["GET"], name="v235_legacy_preview_redirect"))
        if "/app-preview/data" not in existing_paths:
            additions.append(Route("/app-preview/data", subscriber_preview_data_v231.preview_data, methods=["GET"], name="v231_subscriber_preview_data"))
        if "/app-preview/performance" not in existing_paths:
            additions.append(Route("/app-preview/performance", subscriber_preview_performance_v231.preview_performance, methods=["GET"], name="v231_subscriber_preview_performance"))
        if "/app-preview/maturity" not in existing_paths:
            additions.append(Route("/app-preview/maturity", subscriber_preview_maturity_v232.preview_maturity, methods=["GET"], name="v232_subscriber_preview_maturity"))
        app.router.routes[0:0] = additions
        return app
    FastMCP.streamable_http_app = streamable_http_app_with_subscriber_routes


_install_commercial_product_layers()
_install_subscriber_i18n_layer()
_install_product_analytics_layer()
_install_subscriber_frontend_hotfix()
_install_v226_regional_billing_layer()
_install_v236_today_layer()
_install_v237_visual_layer()
_install_subscriber_app_routes()
