from __future__ import annotations

import asyncio

from starlette.requests import Request
from starlette.responses import JSONResponse

from mcp_gateway import subscriber_maturity_v232
from mcp_gateway import subscription_entitlements_v4
from mcp_gateway import supabase_auth_v4


async def preview_maturity(request: Request) -> JSONResponse:
    token = supabase_auth_v4.bearer_token(request.headers.get("authorization"))
    if not token:
        return JSONResponse({"error": "AUTH_REQUIRED"}, status_code=401)

    entitlement = await asyncio.to_thread(subscription_entitlements_v4.resolve_entitlement, token)
    if not entitlement.get("ok") or not entitlement.get("authenticated"):
        return JSONResponse({"error": entitlement.get("status") or "AUTH_REQUIRED"}, status_code=401)

    is_owner = bool(entitlement.get("owner")) or (entitlement.get("user") or {}).get("role") == "OWNER"
    is_pro = str(entitlement.get("effective_plan") or "").upper() == subscription_entitlements_v4.PRO_PLAN
    if not (is_owner or is_pro):
        return JSONResponse({"error": "PREVIEW_REQUIRES_PRO"}, status_code=403)

    result = await asyncio.to_thread(subscriber_maturity_v232.load_maturity_evidence)
    result["owner"] = is_owner
    result["effective_plan"] = entitlement.get("effective_plan")
    return JSONResponse(result)
