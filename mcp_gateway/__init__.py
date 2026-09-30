"""Soccer Edge API MCP gateway package."""

from __future__ import annotations


def _install_v215_commercial_shell() -> None:
    # Keep the V214 operator renderer authoritative and only inject a commercial
    # product fragment before </main>. This preserves the existing dashboard
    # contract/tests while Auth/Billing/Entitlements remain explicitly disabled.
    from mcp_gateway import commercial_shell_v4, product_dashboard_v4

    if hasattr(product_dashboard_v4, "_v215_operator_render_dashboard"):
        return

    operator_renderer = product_dashboard_v4.render_dashboard
    product_dashboard_v4._v215_operator_render_dashboard = operator_renderer

    def render_dashboard_with_shell(product_payload):
        rendered = operator_renderer(product_payload)
        fragment = commercial_shell_v4.render_membership_fragment(product_payload)
        marker = "</main>"
        if marker in rendered:
            return rendered.replace(marker, fragment + marker, 1)
        return rendered + fragment

    product_dashboard_v4.render_dashboard = render_dashboard_with_shell


_install_v215_commercial_shell()
