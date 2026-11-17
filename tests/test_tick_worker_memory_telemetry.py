import json

from mcp_gateway import tick_worker


def test_timing_includes_guard_and_worker_bytes(monkeypatch, capsys):
    data = {
        "/sys/fs/cgroup/memory.current": "470175744\n",
        "/proc/self/status": "VmRSS:\t256 kB\nVmHWM:\t300 kB\n",
    }
    monkeypatch.setattr(tick_worker.Path, "read_text", lambda self, **_: data[str(self)])
    tick_worker._emit_timing("test_stage", call=1)
    record = json.loads(capsys.readouterr().err.removeprefix("TICK_TIMING "))
    assert record == {"stage": "test_stage", "call": 1, "cgroup_used_bytes": 470175744,
                      "worker_rss_bytes": 262144, "worker_peak_rss_bytes": 307200}


def test_missing_memory_files_never_block_timing(monkeypatch, capsys):
    def unavailable(*_, **__):
        raise FileNotFoundError
    monkeypatch.setattr(tick_worker.Path, "read_text", unavailable)
    tick_worker._emit_timing("test_stage")
    assert json.loads(capsys.readouterr().err.removeprefix("TICK_TIMING ")) == {"stage": "test_stage"}
