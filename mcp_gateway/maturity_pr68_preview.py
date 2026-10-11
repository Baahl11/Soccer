"""Isolated, GET-only browser QA service for the PR #68 market-maturity frontend.

No scheduler, databases, sportsbook/provider credentials, OIDC tick routes,
or betting mutations are exposed. Real entitlement decisions are delegated
to the existing production /app/api/v2/account read endpoint, and scientific
reports are loaded exclusively from public, versioned research state.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, RedirectResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from mcp_gateway import subscriber_maturity_v232

PRODUCTION_ORIGIN = "https://soccer-edge-api.onrender.com"
DIST_DIR = Path(__file__).resolve().parents[1] / "web_v3" / "dist"
ASSETS_DIR = DIST_DIR / "assets"
NO_STORE = {"Cache-Control": "no-store", "X-Robots-Tag": "noindex, nofollow"}


def _json(data: dict[str, Any], status_code: int = 200) -> JSONResponse:
    return JSONResponse(data, status_code=status_code, headers=NO_STORE)


def _bearer(request: Request) -> str:
    auth = request.headers.get("authorization", "").strip()
    if not auth.lower().startswith("bearer "):
        return ""
    return auth[7:].strip()


async def _read_upstream(path: str, token: str = "") -> tuple[int, dict[str, Any]]:
    """Only fixed, GET-only product-auth endpoints may be called upstream."""
    assert path in ("/app/api/v2/account", "/app-v3-react/auth-config")
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    try:
        async with httpx.AsyncClient(timeout=8.0, follow_redirects=False) as http:
            response = await http.get(PRODUCTION_ORIGIN + path, headers=headers)
        if len(response.content) > 65536:
            return 503, {"error": "UPSTREAM_RESPONSE_TOO_LARGE"}
        if response.status_code != 200:
            if response.status_code == 401:
                return 401, {"error": "AUTH_REQUIRED"}
            if response.status_code == 403:
                return 403, {"error": "ACCESS_DENIED"}
            return 503, {"error": "UPSTREAM_AUTH_UNAVAILABLE"}
        payload = response.json()
        if not isinstance(payload, dict):
            return 503, {"error": "UPSTREAM_AUTH_INVALID"}
        return 200, payload
    except (httpx.HTTPError, ValueError):
        return 503, {"error": "UPSTREAM_AUTH_UNAVAILABLE"}


async def _authorized_account(request: Request) -> tuple[int, dict[str, Any]]:
    token = _bearer(request)
    if not token:
        return 401, {"error": "AUTH_REQUIRED"}
    status, payload = await _read_upstream("/app/api/v2/account", token)
    if status != 200:
        return status, payload
    access = payload.get("access")
    if not isinstance(access, dict) or access.get("authenticated") is not True:
        return 401, {"error": "AUTH_REQUIRED"}
    return 200, payload


def _safe_access(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "authenticated": raw.get("authenticated") is True,
        "effective_plan": str(raw.get("effective_plan") or "FREE"),
        "display_role": str(raw.get("display_role") or "Member"),
        "owner": raw.get("owner") is True,
        "premium_unlocked": raw.get("premium_unlocked") is True,
    }


async def health(_: Request) -> JSONResponse:
    return _json({
        "status": "ok",
        "service": "soccer-maturity-pr68-preview",
        "preview_only": True,
        "render_git_commit": os.environ.get("RENDER_GIT_COMMIT") or "NOT VERIFIED",
        "scheduler_enabled": False,
        "provider_requests_added": 0,
        "database_writes_enabled": False,
        "betting_mutations_enabled": False,
    })


async def preview_index(_: Request) -> FileResponse:
    return FileResponse(DIST_DIR / "index.html", media_type="text/html", headers=NO_STORE)


async def preview_home(_: Request) -> RedirectResponse:
    return RedirectResponse("/app-v3-react/?sample=1&maturity-preview=1", status_code=307, headers=NO_STORE)


async def auth_config(_: Request) -> JSONResponse:
    status, payload = await _read_upstream("/app-v3-react/auth-config")
    if status != 200:
        return _json({"error": "AUTH_CONFIG_UNAVAILABLE", "auth_configured": False}, 503)
    # The upstream auth-config exposes only publicly publishable Supabase config.
    return _json({
        "supabase_url": payload.get("supabase_url"),
        "publishable_key": payload.get("publishable_key"),
        "auth_configured": payload.get("auth_configured") is True,
        "api_base": "/app/api/v2",
    })


async def account(request: Request) -> JSONResponse:
    status, payload = await _authorized_account(request)
    if status != 200:
        return _json(payload, status)
    user = payload.get("user") if isinstance(payload.get("user"), dict) else {}
    return _json({
        "status": "PREVIEW_ACCOUNT_READ_ONLY",
        "access": _safe_access(payload["access"]),
        "user": {"email": user.get("email")} if user else {},
        "preview_only": True,
    })


async def empty_slate(_: Request) -> JSONResponse:
    """The preview is not a sporting feed. Never substitute demo prices."""
    return _json({
        "status": "PREVIEW_ONLY_NO_LIVE_SLATE",
        "slate": {"rows": []},
        "preview_only": True,
        "provider_requests_added": 0,
    })


async def maturity(request: Request) -> JSONResponse:
    status, payload = await _authorized_account(request)
    if status != 200:
        return _json(payload, status)

    access = _safe_access(payload["access"])
    if not (access["owner"] or access["effective_plan"].upper() == "PRO"):
        return _json({"error": "PREVIEW_REQUIRES_PRO"}, 403)

    # Existing module has a shared cache. Return a copy, never mutate it by role.
    state = dict(await asyncio.to_thread(subscriber_maturity_v232.load_maturity_evidence))
    state.update({
        "owner": access["owner"],
        "effective_plan": access["effective_plan"],
        "preview_only": True,
        "source_scope": "PUBLIC_GITHUB_STATE_READ_ONLY",
        "production_promotion_allowed": False,
        "provider_requests_added": 0,
        "database_writes_enabled": False,
    })
    return _json(state)


# Deliberately no /internal/tick, no /app/api/v2/performance, no payment or
# betting endpoints, no mount of the production FastMCP app.
app = Starlette(
    debug=False,
    routes=[
        Route("/", preview_home, methods=["GET"]),
        Route("/health", health, methods=["GET"]),
        Route("/app-v3-react", preview_home, methods=["GET"]),
        Route("/app-v3-react/", preview_index, methods=["GET"]),
        Route("/app-v3-react/auth-config", auth_config, methods=["GET"]),
        Mount("/app-v3-react/assets", app=StaticFiles(directory=str(ASSETS_DIR), check_dir=False)),
        Route("/app/api/v2/account", account, methods=["GET"]),
        Route("/app/api/v2/today", empty_slate, methods=["GET"]),
        Route("/app/api/v2/maturity", maturity, methods=["GET"]),
    ],
)
