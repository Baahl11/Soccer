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


def _install_subscriber_app_routes() -> None:
    """Add customer routes while preserving the FastMCP ASGI lifespan."""
    from mcp.server.fastmcp import FastMCP
    from starlette.responses import RedirectResponse
    from starlette.routing import Route
    if hasattr(FastMCP, "_v220_streamable_http_app"):
        return
    original = FastMCP.streamable_http_app
    FastMCP._v220_streamable_http_app = original

    async def auth_landing(request):
        return RedirectResponse(url="/app", status_code=307)

    async def legacy_preview_landing(request):
        return RedirectResponse(url="/app", status_code=307)

    def streamable_http_app_with_subscriber_routes(self, *args, **kwargs):
        app = original(self, *args, **kwargs)
        from mcp_gateway import (
            content_factory_http_v4,
            subscriber_product_v235,
            subscriber_preview_data_v231,
            subscriber_preview_maturity_v232,
            subscriber_preview_performance_v231,
        )
        existing_paths = {getattr(route, "path", None) for route in app.router.routes}
        existing_names = {getattr(route, "name", None) for route in app.router.routes}
        additions = []
        if "v226_auth_landing" not in existing_names:
            additions.append(Route("/", auth_landing, methods=["GET"], name="v226_auth_landing"))
        if "/internal/content-packages" not in existing_paths:
            additions.append(Route("/internal/content-packages", content_factory_http_v4.content_packages, methods=["POST"], name="v225_content_packages"))
        if "/app" not in existing_paths:
            additions.append(Route("/app", subscriber_product_v235.app_page, methods=["GET"], name="v235_subscriber_product"))
        if "/app/data" not in existing_paths:
            additions.append(Route("/app/data", subscriber_product_v235.app_data, methods=["GET"], name="v235_subscriber_data"))
        if "/app/match" not in existing_paths:
            additions.append(Route("/app/match", subscriber_product_v235.match_data, methods=["GET"], name="v235_subscriber_match_data"))
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
_install_subscriber_app_routes()
