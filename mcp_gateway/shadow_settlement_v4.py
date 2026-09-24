from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

from mcp_gateway import evaluate_postgame as ep

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_SHADOW_SETTLEMENT_V4_1.0.0"
DIRECTIONAL_MIN = 20
REVIEW_MIN = 50
PREGAME_STAGES = {"EARLY_RESEARCH", "T-90", "T-60", "T-40", "T-30", "T-20", "T-10", "CLOSE"}


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        out = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return out if out.tzinfo is not None else None


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path or not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _latest_watch_by_fixture_family(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    finals: dict[int, dict[str, Any]] = {}
    for row in rows:
        try:
            fixture_id = int(row.get("fixture_id"))
        except (TypeError, ValueError):
            continue
        result = row.get("result")
        if isinstance(result, dict) and result:
            finals[fixture_id] = result

    chosen: dict[tuple[int, str], tuple[datetime, dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("classification") or "").upper() != "WATCH":
            continue
        stage = str(row.get("stage") or "").upper()
        if stage not in PREGAME_STAGES:
            continue
        best = row.get("best_market")
        if not isinstance(best, dict) or not best:
            continue
        try:
            fixture_id = int(row.get("fixture_id"))
        except (TypeError, ValueError):
            continue
        if fixture_id not in finals:
            continue

        generated = _parse_dt(row.get("generated_at_local") or row.get("generated_at_utc"))
        kickoff = _parse_dt(row.get("kickoff_local"))
        if generated is None or kickoff is None or generated >= kickoff:
            continue

        family = ep.market_family(best)
        key = (fixture_id, family)
        prior = chosen.get(key)
        if prior is None or generated > prior[0]:
            chosen[key] = (generated, row)

    return [row for _, row in sorted(chosen.values(), key=lambda item: item[0])]


def build(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    finals: dict[int, dict[str, Any]] = {}
    for row in rows:
        try:
            fixture_id = int(row.get("fixture_id"))
        except (TypeError, ValueError):
            continue
        result = row.get("result")
        if isinstance(result, dict) and result:
            finals[fixture_id] = result

    selected = _latest_watch_by_fixture_family(rows)
    ledger: list[dict[str, Any]] = []

    for row in selected:
        fixture_id = int(row["fixture_id"])
        result = finals.get(fixture_id)
        best = row.get("best_market") or {}
        outcome = ep.grade_market(
            best,
            result,
            row.get("home_team") or "",
            row.get("away_team") or "",
            row.get("home_team_id"),
            row.get("away_team_id"),
        )
        price = ep.fnum(best.get("decimal_price"))
        hypothetical_roi = ep.roi_units(outcome, price, 1.0)

        ledger.append({
            "schema_version": SCHEMA_VERSION,
            "fixture_id": fixture_id,
            "generated_at_local": row.get("generated_at_local"),
            "kickoff_local": row.get("kickoff_local"),
            "stage": row.get("stage"),
            "league": row.get("league"),
            "home_team": row.get("home_team"),
            "away_team": row.get("away_team"),
            "classification": "WATCH",
            "market_family": ep.market_family(best),
            "market": best.get("market"),
            "selection": best.get("selection"),
            "line": ep.fnum(best.get("line")),
            "decimal_price": price,
            "bookmaker": best.get("bookmaker"),
            "shadow_settlement_status": outcome,
            "shadow_settled": outcome in {"WIN", "LOSS", "PUSH"},
            "shadow_roi_hypothetical_units": hypothetical_roi,
            "real_wager_assumed": False,
            "bankroll_impact": 0.0,
            "result": result,
        })

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ledger:
        by_family[str(row.get("market_family") or "UNKNOWN")].append(row)

    summaries: dict[str, Any] = {}
    for family, group in sorted(by_family.items()):
        statuses = Counter(str(row.get("shadow_settlement_status") or "UNKNOWN") for row in group)
        settled_rows = [row for row in group if row.get("shadow_settled")]
        decided = statuses["WIN"] + statuses["LOSS"]
        roi_values = [
            float(row["shadow_roi_hypothetical_units"])
            for row in settled_rows
            if row.get("shadow_roi_hypothetical_units") is not None
        ]
        n = len(group)
        settled = len(settled_rows)
        if settled >= REVIEW_MIN:
            sample_status = "SHADOW_REVIEW_READY"
        elif settled >= DIRECTIONAL_MIN:
            sample_status = "DIRECTIONAL_SHADOW"
        else:
            sample_status = "DATA_BLOCKED"

        summaries[family] = {
            "rows": n,
            "unique_fixtures": len({row["fixture_id"] for row in group}),
            "settled": settled,
            "win": statuses["WIN"],
            "loss": statuses["LOSS"],
            "push": statuses["PUSH"],
            "ungraded": n - settled,
            "hit_rate_ex_push": round(statuses["WIN"] / decided, 6) if decided else None,
            "shadow_roi_hypothetical_units": round(sum(roi_values), 6) if roi_values else 0.0,
            "shadow_roi_per_settled_unit": round(sum(roi_values) / settled, 6) if roi_values and settled else None,
            "sample_status": sample_status,
            "stage_counts": dict(sorted(Counter(str(row.get("stage") or "UNKNOWN") for row in group).items())),
            "league_count": len({str(row.get("league") or "UNKNOWN") for row in group}),
        }

    summary = {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": "SHADOW_PERFORMANCE_ACTIVE",
        "source_classification": "WATCH",
        "decision_policy": "LATEST_PREKICKOFF_WATCH_PER_FIXTURE_AND_MARKET_FAMILY",
        "shadow_rows": len(ledger),
        "shadow_settled_rows": sum(1 for row in ledger if row.get("shadow_settled")),
        "unique_fixtures": len({row["fixture_id"] for row in ledger}),
        "family_count": len(summaries),
        "by_market_family": summaries,
        "sample_policy": {
            "directional_minimum": DIRECTIONAL_MIN,
            "review_minimum": REVIEW_MIN,
            "primary_unit": "UNIQUE_FIXTURE_MARKET_FAMILY",
        },
        "provider_requests_added": 0,
        "real_wagers_assumed": False,
        "bankroll_impact": 0.0,
        "production_promotion_allowed": False,
        "notes": [
            "WATCH decisions are graded as shadow observations only and are never merged into the real bet settlement ledger.",
            "Hypothetical ROI uses the observed signal price at one unit solely for research comparison.",
            "Only the latest pre-kickoff WATCH decision per fixture and market family is retained to reduce correlated snapshot duplication.",
        ],
    }
    return ledger, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade WATCH decisions into a separate shadow-performance ledger.")
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    args = parser.parse_args()

    ledger, summary = build(_load_jsonl(args.ledger))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        for row in ledger:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    with open(args.summary_output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
