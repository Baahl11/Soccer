from __future__ import annotations

import argparse
import json
import os
from typing import Any


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load(path: str) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate API-Football finalized historical xG registry.")
    ap.add_argument("--registry", default="soccer_edge_state/analysis/xg_team_registry.json")
    ap.add_argument("--output", default="soccer_edge_state/analysis/xg_registry_sanity.json")
    args = ap.parse_args()

    registry = _load(args.registry)
    fixtures = int(registry.get("fixture_count") or 0)
    profiles = registry.get("profiles") if isinstance(registry.get("profiles"), dict) else {}
    source = registry.get("source") if isinstance(registry.get("source"), dict) else {}
    source_ok = (
        source.get("provider") == "API-Football v3"
        and source.get("endpoint") == "/fixtures/statistics"
        and source.get("stat_type") == "expected_goals"
        and source.get("same_fixture_pregame_use_allowed") is False
    )

    failures: list[dict[str, Any]] = []
    checked_windows = 0
    for team_id, profile in profiles.items():
        if not isinstance(profile, dict):
            continue
        windows = profile.get("windows") if isinstance(profile.get("windows"), dict) else {}
        for name in ("last_5", "last_10", "last_20"):
            block = windows.get(name) if isinstance(windows.get(name), dict) else {}
            n = int(block.get("n") or 0)
            if n <= 0:
                continue
            checked_windows += 1
            xgf = _num(block.get("avg_xg_for"))
            xga = _num(block.get("avg_xg_against"))
            diff = _num(block.get("avg_xg_diff"))
            valid = (
                xgf is not None
                and xga is not None
                and diff is not None
                and xgf >= 0
                and xga >= 0
                and abs((xgf - xga) - diff) <= 2e-5
            )
            if not valid:
                failures.append({
                    "team_id": team_id,
                    "window": name,
                    "n": n,
                    "avg_xg_for": xgf,
                    "avg_xg_against": xga,
                    "avg_xg_diff": diff,
                })

    if fixtures == 0:
        status = "BLOCKED_ACCUMULATING_API_FOOTBALL_XG"
    elif failures or not source_ok or checked_windows == 0:
        status = "FAIL"
    else:
        status = "PASS"

    report = {
        "schema_version": "1.1.0",
        "status": status,
        "fixture_count": fixtures,
        "team_profiles": len(profiles),
        "checked_windows": checked_windows,
        "source_provenance_valid": source_ok,
        "same_fixture_leakage_guard_valid": source.get("same_fixture_pregame_use_allowed") is False,
        "failures": failures[:50],
        "data_accumulating": fixtures == 0,
        "validation_scope": "STRUCTURAL_PROVENANCE_AND_LEAKAGE_GUARD_NOT_PREDICTIVE_OOS",
        "oos_feature_lift_complete": False,
        "actionable": False,
        "decision_weight": 0.0,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "failures"}, indent=2))
    if status == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
