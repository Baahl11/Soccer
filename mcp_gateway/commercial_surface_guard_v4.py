from __future__ import annotations

from starlette.requests import Request
from starlette.responses import RedirectResponse, Response

from mcp_gateway import subscriber_app_v4

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_COMMERCIAL_SURFACE_GUARD_V4_1.0.0"


async def dashboard_guard(request: Request) -> Response:
    """Remove the public operator dashboard from the customer-facing HTTP surface."""
    return RedirectResponse(url="/app", status_code=307)


async def product_views_guard(request: Request) -> Response:
    """Expose only entitlement-filtered product data over the legacy HTTP URL."""
    return await subscriber_app_v4.app_data(request)
