from __future__ import annotations

import glob
import json
import os

from mcp_gateway import analyze_formation_intelligence as base

_original_load_history = base.load_history


def _enhanced_load_history(history_dir: str):
    fixtures = _original_load_history(history_dir)
    for path in sorted(glob.glob(os.path.join(history_dir, "*.jsonl"))):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    tick = json.loads(line)
                except Exception:
                    continue
                for event in tick.get("events") or []:
                    if not isinstance(event, dict):
                        continue
                    fx = event.get("fixture") or {}
                    fid = fx.get("fixture_id")
                    if not fid:
                        continue
                    rec = fixtures.get(int(fid))
                    if rec is None:
                        continue
                    result = event.get("result")
                    if isinstance(result, dict) and isinstance(result.get("tactical_stats"), dict):
                        rec["tactical_stats"] = result.get("tactical_stats")
    return fixtures


base.load_history = _enhanced_load_history

if __name__ == "__main__":
    base.main()
