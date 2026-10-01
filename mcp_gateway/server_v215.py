from __future__ import annotations

import asyncio
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

from mcp_gateway import server as base_server
from mcp_gateway import signal_ledger_postgres_delta_v4

ROUTE = "/internal/signal-ledger-postgres-v4/build"
ALLOWED_WORKFLOW = ".github/workflows/v215-signal-ledger-postgres-materialization.yml"


async def _handle_signal_ledger(scope, receive, send) -> None:
    request = Request(scope, receive=receive)
    try:
        base_server._github_oidc_claims(request, {ALLOWED_WORKFLOW})
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
        max_rows = max(1, min(int(body.get("max_rows", 500)), signal_ledger_postgres_delta_v4.MAX_PAGE_ROWS))
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
    """Intercept one v215 internal route; delegate every other ASGI request unchanged."""

    def __init__(self, downstream) -> None:
        self.downstream = downstream

    async def __call__(self, scope, receive, send) -> None:
        if (
            scope.get("type") == "http"
            and scope.get("path") == ROUTE
            and str(scope.get("method") or "").upper() == "POST"
        ):
            await _handle_signal_ledger(scope, receive, send)
            return
        await self.downstream(scope, receive, send)


app = V215SignalLedgerRouter(base_server.app)
