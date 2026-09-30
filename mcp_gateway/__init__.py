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


def _install_subscriber_app_routes() -> None:
    """Add V220 subscriber routes without replacing the FastMCP ASGI lifespan."""
    from mcp.server.fastmcp import FastMCP
    from starlette.routing import Route

    if hasattr(FastMCP, "_v220_streamable_http_app"):
        return

    original = FastMCP.streamable_http_app
    FastMCP._v220_streamable_http_app = original

    def streamable_http_app_with_subscriber_routes(self, *args, **kwargs):
        app = original(self, *args, **kwargs)
        from mcp_gateway import subscriber_app_v4

        existing_paths = {getattr(route, "path", None) for route in app.router.routes}
        additions = []
        if "/app" not in existing_paths:
            additions.append(Route("/app", subscriber_app_v4.app_page, methods=["GET"]))
        if "/app/data" not in existing_paths:
            additions.append(Route("/app/data", subscriber_app_v4.app_data, methods=["GET"]))
        app.router.routes[0:0] = additions
        return app

    FastMCP.streamable_http_app = streamable_http_app_with_subscriber_routes


_install_commercial_product_layers()
_install_subscriber_app_routes()
