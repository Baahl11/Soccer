from __future__ import annotations

import asyncio
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

from mcp_gateway import server_base as _base_server
from mcp_gateway import signal_ledger_postgres_delta_v4
from mcp_gateway import team_totals_phase17_anchor_patch_v4

# Keep mcp_gateway.server as the canonical compatibility surface.  The entire
# pre-v215 server implementation is preserved byte-for-byte in server_base;
# this module only adds one isolated ASGI route and delegates everything else.
for _name, _value in vars(_base_server).items():
    if not _name.startswith("__") and _name != "app":
        globals()[_name] = _value

# v216.9 research-only compatibility patch: Phase17 Team Totals must anchor to
# the oldest exact fixture/market/side/line signal, matching the live maturation
# contract. This changes neither the global signal cap nor strict-close rules.
V216_9_PHASE17_TEAM_TOTALS_ANCHOR_PATCH = team_totals_phase17_anchor_patch_v4.install()

V215_SIGNAL_LEDGER_ROUTE = "/internal/signal-ledger-postgres-v4/build"
V215_SIGNAL_LEDGER_WORKFLOW = ".github/workflows/v215-signal-ledger-postgres-materialization.yml"
V215_SIGNAL_LEDGER_REF = "refs/heads/soccer-edge-mcp-v1"


def _v215_github_oidc_claims(request: Request) -> dict[str, Any]:
    """Validate only the isolated v215 materializer without relaxing global OIDC rules."""
    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer "):
        raise PermissionError("Missing bearer token")
    token = auth[7:].strip()
    signing_key = _base_server._JWK_CLIENT.get_signing_key_from_jwt(token)
    claims = _base_server.jwt.decode(
        token,
        signing_key.key,
        algorithms=["RS256"],
        audience=_base_server.GITHUB_OIDC_AUDIENCE,
        issuer=_base_server.GITHUB_ISSUER,
        options={"require": ["exp", "iat", "iss", "aud", "sub"]},
    )
    if claims.get("repository") != _base_server.GITHUB_REPOSITORY:
        raise PermissionError("Repository not allowed")
    workflow_ref = str(claims.get("workflow_ref") or "")
    expected_prefix = f"{_base_server.GITHUB_REPOSITORY}/{V215_SIGNAL_LEDGER_WORKFLOW}@"
    if not workflow_ref.startswith(expected_prefix):
        raise PermissionError("Workflow not allowed")
    if claims.get("ref") != V215_SIGNAL_LEDGER_REF:
        raise PermissionError("Only soccer-edge-mcp-v1 v215 materializer is allowed")
    return claims


async def _handle_v215_signal_ledger(scope, receive, send) -> None:
    request = Request(scope, receive=receive)
    try:
        _v215_github_oidc_claims(request)
    except Exception as exc:
        response = JSONResponse({"error": "unauthorized", "detail": str(exc)[:200]}, status_code=401)
        await response(scope, receive, send)
        return

    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}

    since = str(body.get("since") or "").strip()
    if not since:
        response = JSONResponse({"error": "since_required"}, status_code=400)
        await response(scope, receive, send)
        return

    try:
        after_event_id = max(0, int(body.get("after_event_id", 0)))
        max_rows = max(
            1,
            min(
                int(body.get("max_rows", 500)),
                signal_ledger_postgres_delta_v4.MAX_PAGE_ROWS,
            ),
        )
    except (TypeError, ValueError):
        response = JSONResponse({"error": "invalid_signal_ledger_parameters"}, status_code=400)
        await response(scope, receive, send)
        return

    try:
        payload: dict[str, Any] = await asyncio.to_thread(
            signal_ledger_postgres_delta_v4.build_delta,
            since=since,
            after_event_id=after_event_id,
            max_rows=max_rows,
        )
        response = JSONResponse(payload)
    except Exception as exc:
        response = JSONResponse(
            {"error": "signal_ledger_postgres_build_failed", "detail": str(exc)[:500]},
            status_code=500,
        )
    await response(scope, receive, send)


class V215SignalLedgerRouter:
    """Intercept the v215 materialization route and delegate every other request."""

    def __init__(self, downstream) -> None:
        self.downstream = downstream

    async def __call__(self, scope, receive, send) -> None:
        if (
            scope.get("type") == "http"
            and scope.get("path") == V215_SIGNAL_LEDGER_ROUTE
            and str(scope.get("method") or "").upper() == "POST"
        ):
            await _handle_v215_signal_ledger(scope, receive, send)
            return
        await self.downstream(scope, receive, send)


app = V215SignalLedgerRouter(_base_server.app)
