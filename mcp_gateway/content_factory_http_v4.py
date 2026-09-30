from __future__ import annotations

import asyncio
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from mcp_gateway import content_factory_v4, persistence as persistence_base, product_views_v4

ALLOWED_WORKFLOW = ".github/workflows/soccer-edge-content-factory.yml"
ALLOWED_REF = "refs/heads/soccer-edge-mcp-v1"


def _content_oidc_claims(request: Request) -> dict[str, Any]:
    """Verify GitHub OIDC for only the V225 production content workflow."""
    # Lazy import avoids a package import cycle while server.py is booting.
    from mcp_gateway import server

    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer "):
        raise PermissionError("Missing bearer token")
    token = auth[7:].strip()
    signing_key = server._JWK_CLIENT.get_signing_key_from_jwt(token)
    claims = server.jwt.decode(
        token,
        signing_key.key,
        algorithms=["RS256"],
        audience=server.GITHUB_OIDC_AUDIENCE,
        issuer=server.GITHUB_ISSUER,
        options={"require": ["exp", "iat", "iss", "aud", "sub"]},
    )
    if claims.get("repository") != server.GITHUB_REPOSITORY:
        raise PermissionError("Repository not allowed")
    workflow_ref = str(claims.get("workflow_ref") or "")
    expected_prefix = f"{server.GITHUB_REPOSITORY}/{ALLOWED_WORKFLOW}@"
    if not workflow_ref.startswith(expected_prefix):
        raise PermissionError("Workflow not allowed")
    if claims.get("ref") != ALLOWED_REF:
        raise PermissionError("Production branch content workflow required")
    return claims


async def content_packages(request: Request) -> Response:
    """Export evidence-locked content packages to the dedicated GitHub OIDC workflow only."""
    try:
        _content_oidc_claims(request)
    except Exception as exc:
        return JSONResponse({"error": "unauthorized", "detail": str(exc)[:200]}, status_code=401)

    try:
        body: Any = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    try:
        limit = max(1, min(int(body.get("limit", 6)), content_factory_v4.MAX_PACKAGES))
    except (TypeError, ValueError):
        return JSONResponse({"error": "invalid_limit"}, status_code=400)

    try:
        payload = await asyncio.to_thread(persistence_base.load_latest_pipeline_payload)
    except Exception as exc:
        return JSONResponse({"error": "content_source_unavailable", "detail": str(exc)[:300]}, status_code=503)
    if not isinstance(payload, dict):
        return JSONResponse({"error": "NO_PERSISTED_PIPELINE_RUN"}, status_code=503)

    payload.setdefault("status", "ok")
    payload["database_persisted"] = True
    payload["database_error"] = None
    product = product_views_v4.build_views(payload, limit=product_views_v4.MAX_ROWS_PER_VIEW)
    product["generated_at_utc"] = payload.get("generated_at_utc")
    product["pipeline_version"] = payload.get("version")
    result = content_factory_v4.build_content_packages(product, limit=limit)
    result["generated_at_utc"] = payload.get("generated_at_utc")
    result["pipeline_version"] = payload.get("version")
    result["source"] = "POSTGRES_LATEST_PIPELINE_RUN"
    return JSONResponse(result)
