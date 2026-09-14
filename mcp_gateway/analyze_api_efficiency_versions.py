from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter, defaultdict
from typing import Any


def fnum(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_ticks(history_dir: str) -> list[dict[str, Any]]:
    ticks: list[dict[str, Any]] = []
    for path in sorted(glob.glob(os.path.join(history_dir, "*.jsonl"))):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    ticks.append(row)
    ticks.sort(key=lambda t: str(t.get("generated_at_local") or t.get("generated_at_utc") or ""))
    return ticks


def summarize(ticks: list[dict[str, Any]]) -> dict[str, Any]:
    sums = Counter()
    for t in ticks:
        for key in (
            "api_calls_this_tick",
            "deep_dive_processed_count",
            "deferred_due_to_priority",
            "deferred_due_to_budget",
            "urgent_late_shortlist_due",
            "urgent_late_shortlist_processed",
            "market_requests_avoided_by_sport_screen",
        ):
            value = fnum(t.get(key))
            if value is not None:
                sums[key] += value
        gm = t.get("galaxy_first_metrics") or {}
        for key in (
            "provider_requests_avoided",
            "api_team_stats_calls_avoided",
            "api_recent_calls_avoided",
            "api_odds_calls_avoided",
        ):
            value = fnum(gm.get(key))
            if value is not None:
                sums[key] += value

    due = float(sums["urgent_late_shortlist_due"])
    processed = float(sums["urgent_late_shortlist_processed"])
    calls = float(sums["api_calls_this_tick"])
    dives = float(sums["deep_dive_processed_count"])
    return {
        "ticks": len(ticks),
        "urgent_due": int(due),
        "urgent_processed": int(processed),
        "urgent_missed": int(max(0.0, due - processed)),
        "urgent_completion_rate": round(processed / due, 4) if due else None,
        "api_calls": int(calls),
        "deep_dives": int(dives),
        "api_calls_per_deep_dive": round(calls / dives, 3) if dives else None,
        "deferred_due_to_priority": int(sums["deferred_due_to_priority"]),
        "deferred_due_to_budget": int(sums["deferred_due_to_budget"]),
        "provider_requests_avoided": int(sums["provider_requests_avoided"]),
        "market_requests_avoided_by_sport_screen": int(sums["market_requests_avoided_by_sport_screen"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Version-aware Soccer Edge API/urgent-window efficiency report.")
    ap.add_argument("--history-dir", default="soccer_edge_state/history")
    ap.add_argument("--output", default="soccer_edge_state/analysis/api_efficiency_versions.json")
    args = ap.parse_args()

    ticks = load_ticks(args.history_dir)
    by_version: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in ticks:
        version = str(t.get("version") or "UNKNOWN")
        by_version[version].append(t)

    v23_plus = [
        t for t in ticks
        if str(t.get("version") or "") in {"2.3.0", "2.4.0", "2.5.0", "3.0.0"}
        or bool(t.get("deep_dive_capacity_policy"))
    ]

    report = {
        "schema_version": "1.0.0",
        "status": "ACTIVE",
        "timezone_basis": "America/Mexico_City",
        "overall": summarize(ticks),
        "by_version": {version: summarize(group) for version, group in sorted(by_version.items())},
        "recent_25_ticks": summarize(ticks[-25:]),
        "recent_50_ticks": summarize(ticks[-50:]),
        "v2_3_plus": summarize(v23_plus),
        "validation_policy": {
            "target_urgent_completion_rate": 0.98,
            "minimum_ticks_before_claiming_stable_improvement": 25,
            "minimum_urgent_windows_before_claiming_stable_improvement": 20,
            "note": "Historical all-version completion is not used to judge v2.3 because it mixes earlier queue/cap behavior. Provider API hard caps remain unchanged.",
        },
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
