"""Non-provider regression tests for the instrumented tick OOM admission guard."""
from __future__ import annotations

import asyncio

import pytest
from starlette.responses import JSONResponse

from mcp_gateway import server


def _scope():
    return {
        "type": "http",
        "method": "POST",
        "path": "/internal/tick",
        "headers": [],
        "query_string": b"",
        "http_version": "1.1",
        "scheme": "http",
        "server": ("localhost", 80),
        "client": ("127.0.0.1", 1000),
    }


async def _request(handler):
    messages = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    await handler(_scope(), receive, send)
    return messages


def _status(messages):
    return next(msg["status"] for msg in messages if msg["type"] == "http.response.start")


def test_tick_anonymous_is_401_before_admission(monkeypatch):
    def reject(_):
        raise PermissionError("no token")
    monkeypatch.setattr(server._base_server, "_github_oidc_claims", reject)
    messages = asyncio.run(_request(server._handle_instrumented_tick))
    assert _status(messages) == 401


def test_instrumented_tick_serializes_with_db_heavy_and_releases_gate(monkeypatch):
    monkeypatch.setattr(server._base_server, "_github_oidc_claims", lambda _: {"ref": "test"})

    async def scenario():
        running, release = asyncio.Event(), asyncio.Event()

        async def simulated_worker(scope, receive, send):
            running.set()
            await release.wait()
            await JSONResponse({"ok": True})(scope, receive, send)

        monkeypatch.setattr(server, "_run_instrumented_tick", simulated_worker)
        first = asyncio.create_task(_request(server._handle_instrumented_tick))
        await asyncio.wait_for(running.wait(), timeout=3)
        second = await _request(server._handle_instrumented_tick)
        assert _status(second) == 409, "second tick must not spawn a worker"
        assert server._base_server._acquire_db_heavy_gate("simulated_heavy_job").status_code == 409
        release.set()
        assert _status(await asyncio.wait_for(first, timeout=3)) == 200
        free = server._base_server._acquire_db_heavy_gate("after_tick")
        assert free is None
        server._base_server._release_db_heavy_gate()

    asyncio.run(scenario())


def test_gate_released_when_tick_worker_raises(monkeypatch):
    monkeypatch.setattr(server._base_server, "_github_oidc_claims", lambda _: {"ref": "test"})

    async def boom(*_):
        raise RuntimeError("test abort")

    monkeypatch.setattr(server, "_run_instrumented_tick", boom)
    with pytest.raises(RuntimeError, match="test abort"):
        asyncio.run(_request(server._handle_instrumented_tick))
    assert server._base_server._acquire_db_heavy_gate("recovery") is None
    server._base_server._release_db_heavy_gate()


def test_guard_detects_cgroup_pressure_without_confusing_metrics(monkeypatch):
    limit = 512 * 1024 * 1024
    monkeypatch.setattr(server, "_cgroup_memory_snapshot", lambda: (300 * 1024 * 1024, limit))
    assert server._tick_memory_pressure() is None
    monkeypatch.setattr(server, "_cgroup_memory_snapshot", lambda: (451 * 1024 * 1024, limit))
    result = server._tick_memory_pressure()
    assert result is not None
    assert result["used_bytes"] == 451 * 1024 * 1024
    assert result["limit_bytes"] == limit
    assert result["threshold_bytes"] <= 448 * 1024 * 1024


def test_monitor_kills_only_worker_when_cgroup_exceeds_safe_threshold(monkeypatch):
    class FakeProc:
        returncode = None
        killed = False

        def kill(self):
            self.killed = True
            self.returncode = -9

    monkeypatch.setattr(server, "_tick_memory_pressure", lambda: {
        "used_bytes": 480,
        "limit_bytes": 512,
        "threshold_bytes": 448,
    })

    proc = FakeProc()
    state = {}
    asyncio.run(server._monitor_tick_memory(proc, state))
    assert proc.killed is True
    assert state["threshold_bytes"] == 448


def test_unbounded_cgroup_does_not_raise_or_abort(monkeypatch):
    monkeypatch.setattr(server, "_cgroup_memory_snapshot", lambda: None)
    assert server._tick_memory_pressure() is None


def test_cancelled_request_releases_tick_admission(monkeypatch):
    monkeypatch.setattr(server._base_server, "_github_oidc_claims", lambda _: {"ref": "test"})

    async def scenario():
        started = asyncio.Event()

        async def until_cancelled(*_):
            started.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(server, "_run_instrumented_tick", until_cancelled)
        task = asyncio.create_task(_request(server._handle_instrumented_tick))
        await asyncio.wait_for(started.wait(), timeout=3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert server._base_server._acquire_db_heavy_gate("after_cancel") is None
        server._base_server._release_db_heavy_gate()

    asyncio.run(scenario())
