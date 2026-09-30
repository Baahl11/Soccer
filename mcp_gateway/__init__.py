"""Soccer Edge API MCP gateway package."""

from __future__ import annotations


def _install_commercial_product_layers() -> None:
    # Keep the V214 operator renderer authoritative. V215/V216 only inject
    # read-only product fragments before </main> and never alter runtime logic.
    from mcp_gateway import commercial_shell_v4, product_dashboard_v4, public_performance_v4

    if hasattr(product_dashboard_v4, "_v215_operator_render_dashboard"):
        return

    operator_renderer = product_dashboard_v4.render_dashboard
    product_dashboard_v4._v215_operator_render_dashboard = operator_renderer

    def render_dashboard_with_product_layers(product_payload):
        rendered = operator_renderer(product_payload)
        fragments = (
            commercial_shell_v4.render_membership_fragment(product_payload)
            + public_performance_v4.render_fragment()
        )
        marker = "</main>"
        if marker in rendered:
            return rendered.replace(marker, fragments + marker, 1)
        return rendered + fragments

    product_dashboard_v4.render_dashboard = render_dashboard_with_product_layers


_install_commercial_product_layers()
