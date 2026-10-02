from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from mcp_gateway import player_props_shots_anchor_patch_v4
from mcp_gateway import server_base as _base_server
from mcp_gateway import signal_ledger_postgres_delta_v4
from mcp_gateway import team_totals_phase17_anchor_patch_v4

# Keep mcp_gateway.server as the canonical compatibility surface.  The entire
# pre-v215 server implementation is preserved byte-for-byte in server_base;
# this module only adds isolated compatibility/diagnostic routes and delegates
# everything else.
for _name, _value in vars(_base_server).items():
    if not _name.startswith("__") and _name != "app":
        globals()[_name] = _value

# v216.9 research-only compatibility patch: Phase17 Team Totals must anchor to
# the oldest exact fixture/market/side/line signal, matching the live maturation
# contract. This changes neither the global signal cap nor strict-close rules.
V216_9_PHASE17_TEAM_TOTALS_ANCHOR_PATCH = team_totals_phase17_anchor_patch_v4.install()

# v217 research-only compatibility patch: repeated SHOTS signals from successive
# pre-kickoff ticks must anchor to the oldest exact fixture/player/side/line
# signal. The patch is DB-only, bounded, adds no provider requests, and leaves
# strict-close semantics unchanged.
V217_PLAYER_PROPS_SHOTS_ANCHOR_PATCH = player_props_shots_anchor_patch_v4.install()

V215_SIGNAL_LEDGER_ROUTE = "/internal/signal-ledger-postgres-v4/build"
V215_SIGNAL_LEDGER_WORKFLOW = ".github/workflows/v215-signal-ledger-postgres-materialization.yml"
V215_SIGNAL_LEDGER_REF = "refs/heads/soccer-edge-mcp-v1"
TICK_ROUTE = "/internal/tick"
TICK_TIMEOUT_SECONDS = 420


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


async def _handle_instrumented_tick(scope, receive, send) -> None:
    """Run the canonical tick while surfacing worker timing checkpoints.

    This is observability-only. It preserves the existing OIDC validation,
    subprocess isolation, 420-second timeout, JSON response bytes, model logic,
    gates, thresholds, provider budget and persistence behavior.
    """
    request = Request(scope, receive=receive)
    try:
        _base_server._github_oidc_claims(request)
    except Exception as exc:
        response = JSONResponse({"error": "unauthorized", "detail": str(exc)[:200]}, status_code=401)
        await response(scope, receive, send)
        return

    env = os.environ.copy()
    env.setdefault("MALLOC_ARENA_MAX", "2")
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "mcp_gateway.tick_worker",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stderr_lines: list[str] = []

        async def _pump_worker_stderr() -> None:
            assert proc.stderr is not None
            while True:
                raw = await proc.stderr.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").rstrip()
                stderr_lines.append(line)
                if len(stderr_lines) > 200:
                    del stderr_lines[:-200]
                if line.startswith("TICK_TIMING "):
                    print(line, file=sys.stderr, flush=True)

        assert proc.stdout is not None
        stderr_task = asyncio.create_task(_pump_worker_stderr())
        stdout_task = asyncio.create_task(proc.stdout.read())
        try:
            await asyncio.wait_for(proc.wait(), timeout=TICK_TIMEOUT_SECONDS)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            await stderr_task
            stdout_task.cancel()
            timing_tail = [
                line for line in stderr_lines if line.startswith("TICK_TIMING ")
            ][-20:]
            response = JSONResponse(
                {
                    "error": "tick_timeout",
                    "timeout_seconds": TICK_TIMEOUT_SECONDS,
                    "worker_timing_tail": timing_tail,
                },
                status_code=504,
            )
            await response(scope, receive, send)
            return

        await stderr_task
        stdout = await stdout_task
        stderr_text = "\n".join(stderr_lines)
        if proc.returncode != 0:
            detail = stderr_text[-1000:]
            response = JSONResponse({"error": "tick_failed", "detail": detail}, status_code=500)
            await response(scope, receive, send)
            return
        if not stdout:
            response = JSONResponse(
                {"error": "tick_failed", "detail": "worker returned empty output"},
                status_code=500,
            )
            await response(scope, receive, send)
            return

        response = Response(content=stdout, media_type="application/json", status_code=200)
        await response(scope, receive, send)
    except Exception as exc:
        response = JSONResponse(
            {"error": "tick_failed", "detail": str(exc)[:500]},
            status_code=500,
        )
        await response(scope, receive, send)


class V215SignalLedgerRouter:
    """Intercept isolated compatibility routes and delegate every other request."""

    def __init__(self, downstream) -> None:
        self.downstream = downstream

    async def __call__(self, scope, receive, send) -> None:
        if scope.get("type") == "http":
            path = scope.get("path")
            method = str(scope.get("method") or "").upper()
            if path == V215_SIGNAL_LEDGER_ROUTE and method == "POST":
                await _handle_v215_signal_ledger(scope, receive, send)
                return
            if path == TICK_ROUTE and method == "POST":
                await _handle_instrumented_tick(scope, receive, send)
                return
        await self.downstream(scope, receive, send)


app = V215SignalLedgerRouter(_base_server.app)
