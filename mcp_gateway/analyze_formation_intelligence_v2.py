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

                    historical_personnel = event.get("historical_personnel_fact")
                    if isinstance(historical_personnel, dict):
                        facts = rec.setdefault("historical_personnel_facts", [])
                        facts.append(
                            {
                                "retrieved_at": historical_personnel.get("retrieved_at"),
                                "historical_fact_available_at": historical_personnel.get(
                                    "historical_fact_available_at"
                                ),
                                "source": historical_personnel.get("source"),
                                "both_xi_confirmed": historical_personnel.get(
                                    "both_xi_confirmed"
                                )
                                is True,
                                "both_goalkeepers_confirmed": historical_personnel.get(
                                    "both_goalkeepers_confirmed"
                                )
                                is True,
                                "teams": historical_personnel.get("teams") or [],
                                "retroactive_current_fixture_allowed": False,
                            }
                        )
    return fixtures


base.load_history = _enhanced_load_history

if __name__ == "__main__":
    base.main()
