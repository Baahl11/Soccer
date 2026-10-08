from __future__ import annotations

import ast
from pathlib import Path


def test_automation_v7_run_tick_does_not_shadow_timedelta():
    source = Path("mcp_gateway/automation_v7.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    run_tick = next(
        node for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "run_tick"
    )

    local_timedelta_imports = []
    for node in ast.walk(run_tick):
        if isinstance(node, ast.ImportFrom) and node.module == "datetime":
            for alias in node.names:
                if alias.name == "timedelta":
                    local_timedelta_imports.append(node.lineno)

    assert local_timedelta_imports == []
    assert "early_horizon = now_utc + timedelta(hours=12)" in source
