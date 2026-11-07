"""OOM regression: importing long-lived web persistence must not load the model."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_persistence_import_does_not_load_canonical_automation_or_model():
    run = subprocess.run(
        [sys.executable, "-c", (
            "import sys; import mcp_gateway.persistence; "
            "print(int('mcp_gateway.automation_v2' in sys.modules)); "
            "print(int('mcp_gateway.soccer_model' in sys.modules)); "
            "print(int('mcp_gateway.fe11_postgame_corners_v1' in sys.modules))"
        )],
        capture_output=True, text=True, timeout=15, check=True,
    )
    assert run.stdout.strip().splitlines() == ["0", "0", "0"]


def test_universal_capture_is_not_inside_memory_heavy_canonical_worker():
    root = Path(__file__).resolve().parents[1] / "mcp_gateway"
    scanner = (root/"automation_v7.py").read_text()
    worker = (root/"tick_worker.py").read_text()
    persistence = (root/"fe11_corners_persistence_v1.py").read_text()
    assert "fe11_universal.build_capture(" not in scanner
    assert "await fe11_postgame_corners_v1.collect(payload)" not in worker
    assert "from mcp_gateway import fe11_postgame_corners_v1 as corners" not in persistence
    assert 'CORNERS_SOURCE = "API_FOOTBALL_FIXTURE_STATISTICS"' in persistence
    assert 'CORNERS_VERSION = "SOCCER_FE11_POSTGAME_CORNERS_v1.0"' in persistence
