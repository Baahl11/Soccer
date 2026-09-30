"""Soccer Edge API MCP gateway package."""

from __future__ import annotations


def _install_commercial_product_layers() -> None:
    # Keep the V214 operator renderer authoritative. V215-V219 only inject
    # read-only product/account fragments before </main>; runtime model logic is untouched.
    from mcp_gateway import commercial_shell_v4, match_detail_v4, product_dashboard_v4, public_performance_v4, subscription_entitlements_v4, supabase_auth_v4

    if hasattr(product_dashboard_v4, "_v215_operator_render_dashboard"):
        return

    operator_renderer = product_dashboard_v4.render_dashboard
    product_dashboard_v4._v215_operator_render_dashboard = operator_renderer

    def render_dashboard_with_product_layers(product_payload):
        rendered = operator_renderer(product_payload)
        fragments = (
            commercial_shell_v4.render_membership_fragment(product_payload)
            + public_performance_v4.render_fragment()
            + match_detail_v4.render_fragment(product_payload)
            + supabase_auth_v4.render_fragment()
            + subscription_entitlements_v4.render_fragment()
        )
        marker = "</main>"
        if marker in rendered:
            return rendered.replace(marker, fragments + marker, 1)
        return rendered + fragments

    product_dashboard_v4.render_dashboard = render_dashboard_with_product_layers


def _install_subscriber_i18n_layer() -> None:
    """Localize only the subscriber presentation; canonical product data stays unchanged."""
    from mcp_gateway import subscriber_app_v4, subscriber_i18n_v4

    if hasattr(subscriber_app_v4, "_v222_base_app_html"):
        return

    base_app_html = subscriber_app_v4._app_html
    subscriber_app_v4._v222_base_app_html = base_app_html

    def bilingual_app_html() -> str:
        return subscriber_i18n_v4.inject_i18n(base_app_html())

    subscriber_app_v4._app_html = bilingual_app_html


def _install_product_analytics_layer() -> None:
    """Instrument commercial surfaces without feeding analytics back into betting logic."""
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


def _install_subscriber_app_routes() -> None:
    """Add commercial routes without replacing the FastMCP ASGI lifespan."""
    from mcp.server.fastmcp import FastMCP
    from starlette.routing import Route

    if hasattr(FastMCP, "_v220_streamable_http_app"):
        return

    original = FastMCP.streamable_http_app
    FastMCP._v220_streamable_http_app = original

    def streamable_http_app_with_subscriber_routes(self, *args, **kwargs):
        app = original(self, *args, **kwargs)
        from mcp_gateway import commercial_surface_guard_v4, content_factory_http_v4, landing_page_v4, subscriber_app_v4

        existing_paths = {getattr(route, "path", None) for route in app.router.routes}
        existing_names = {getattr(route, "name", None) for route in app.router.routes}
        additions = []

        # V221: prepend entitlement-safe shadow routes ahead of legacy public
        # /dashboard and /product/views routes. Internal MCP tools remain untouched.
        if "v221_dashboard_guard" not in existing_names:
            additions.append(
                Route(
                    "/dashboard",
                    commercial_surface_guard_v4.dashboard_guard,
                    methods=["GET"],
                    name="v221_dashboard_guard",
                )
            )
        if "v221_product_views_guard" not in existing_names:
            additions.append(
                Route(
                    "/product/views",
                    commercial_surface_guard_v4.product_views_guard,
                    methods=["GET"],
                    name="v221_product_views_guard",
                )
            )
        # V225: premium content export is never public. The handler validates
        # GitHub OIDC against the dedicated content-factory workflow.
        if "/internal/content-packages" not in existing_paths:
            additions.append(
                Route(
                    "/internal/content-packages",
                    content_factory_http_v4.content_packages,
                    methods=["POST"],
                    name="v225_content_packages",
                )
            )
        if "/" not in existing_paths:
            additions.append(Route("/", landing_page_v4.landing_page, methods=["GET"], name="v223_landing_page"))
        if "/app" not in existing_paths:
            additions.append(Route("/app", subscriber_app_v4.app_page, methods=["GET"], name="v220_subscriber_app"))
        if "/app/data" not in existing_paths:
            additions.append(Route("/app/data", subscriber_app_v4.app_data, methods=["GET"], name="v220_subscriber_data"))

        app.router.routes[0:0] = additions
        return app

    FastMCP.streamable_http_app = streamable_http_app_with_subscriber_routes


_install_commercial_product_layers()
_install_subscriber_i18n_layer()
_install_product_analytics_layer()
_install_subscriber_app_routes()
