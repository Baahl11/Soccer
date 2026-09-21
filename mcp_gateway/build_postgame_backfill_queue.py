from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Any

PENDING_REASON = "PENDING_FINAL_RESULT"


def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def norm_list(values: set[Any]) -> list[str]:
    return sorted({str(v) for v in values if v not in (None, "")})


def priority_score(rows: list[dict[str, Any]]) -> int:
    score = 0
    classifications = {str(r.get("classification") or "").upper() for r in rows}
    market_families = {str(r.get("market_family") or "") for r in rows}
    if "BET" in classifications:
        score += 100
    if "LEAN" in classifications:
        score += 25
    score += min(len(rows), 10)
    if any(m.endswith("_TOTALS") or m.endswith("_BTTS") or m.endswith("_1X2") for m in market_families):
        score += 10
    if any(m.startswith("1H_") or m.startswith("2H_") for m in market_families):
        score += 5
    return score


def build_queue(backlog_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pending = [r for r in backlog_rows if r.get("coverage_reason") == PENDING_REASON]
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in pending:
        try:
            fid = int(row["fixture_id"])
        except (KeyError, TypeError, ValueError):
            continue
        grouped[fid].append(row)

    queue: list[dict[str, Any]] = []
    for fid, rows in grouped.items():
        latest = max(rows, key=lambda r: str(r.get("generated_at_local") or ""))
        classifications = {r.get("classification") for r in rows}
        market_families = {r.get("market_family") for r in rows}
        stages = {r.get("stage") for r in rows}
        lines = []
        for r in rows:
            lines.append({
                "event_key": r.get("event_key"),
                "classification": r.get("classification"),
                "stage": r.get("stage"),
                "market_family": r.get("market_family"),
                "market": r.get("market"),
                "selection": r.get("selection"),
                "line": r.get("line"),
                "decimal_price": r.get("decimal_price"),
                "generated_at_local": r.get("generated_at_local"),
            })
        queue.append({
            "schema_version": "1.0.0",
            "queue_reason": PENDING_REASON,
            "fixture_id": fid,
            "kickoff_local": latest.get("kickoff_local"),
            "league": latest.get("league"),
            "home_team": latest.get("home_team"),
            "away_team": latest.get("away_team"),
            "classifications": norm_list(classifications),
            "market_families": norm_list(market_families),
            "stages": norm_list(stages),
            "source_backlog_rows": len(rows),
            "priority_score": priority_score(rows),
            "provider_hint": {
                "provider": "api-football",
                "endpoint": "fixtures",
                "params": {"id": fid},
                "expected_fields": ["fixture.status.short", "goals", "score.fulltime", "score.halftime"],
            },
            "next_action": "fetch final fixture result by id and append/update postgame history before next settlement review",
            "safety": {
                "no_odds_request": True,
                "no_runtime_pick_change": True,
                "no_stake_or_tier_change": True,
                "requires_provider_budget_guard": True,
            },
            "decision_lines": lines,
        })

    queue.sort(key=lambda r: (-int(r.get("priority_score") or 0), str(r.get("kickoff_local") or ""), int(r.get("fixture_id") or 0)))

    by_classification: Counter[str] = Counter()
    by_market_family: Counter[str] = Counter()
    for row in pending:
        by_classification[str(row.get("classification") or "UNKNOWN")] += 1
        by_market_family[str(row.get("market_family") or "UNKNOWN")] += 1

    summary = {
        "schema_version": "1.0.0",
        "status": "POSTGAME_FINAL_BACKFILL_QUEUE",
        "source_reason": PENDING_REASON,
        "pending_backlog_rows": len(pending),
        "unique_fixtures": len(queue),
        "max_provider_calls_needed": len(queue),
        "by_classification": dict(sorted(by_classification.items())),
        "by_market_family": dict(sorted(by_market_family.items())),
        "priority_policy": {
            "BET": "+100",
            "LEAN": "+25",
            "per_source_row": "+1 up to +10",
            "canonical_market_family": "+10",
            "period_market": "+5",
        },
        "safety_note": "Queue generation is diagnostic/planning only. It does not call providers, trigger ticks, deploy Render, change runtime tiers, stakes, classifications, or model weights.",
        "next_step": "A separate guarded backfill runner may consume this queue with provider budget limits and write final results to history/state.",
    }
    return queue, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a fixture-level backfill queue for BET/LEAN rows missing final results.")
    ap.add_argument("--backlog", default="soccer_edge_state/analysis/settlement_backlog.jsonl")
    ap.add_argument("--queue-output", default="soccer_edge_state/analysis/postgame_final_backfill_queue.jsonl")
    ap.add_argument("--summary-output", default="soccer_edge_state/analysis/postgame_final_backfill_queue_summary.json")
    args = ap.parse_args()

    queue, summary = build_queue(read_jsonl(args.backlog))

    os.makedirs(os.path.dirname(args.queue_output), exist_ok=True)
    with open(args.queue_output, "w", encoding="utf-8") as fh:
        for row in queue:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    os.makedirs(os.path.dirname(args.summary_output), exist_ok=True)
    with open(args.summary_output, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
