from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from statistics import median
from typing import Any, Iterable

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_CORNERS_CLV_CONCENTRATION_AUDIT_V4_1.0.0"


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _corner_family(row: dict[str, Any]) -> str | None:
    family = str(row.get("market_family") or "").strip().upper()
    if family in {"FT_CORNERS", "TEAM_CORNERS"}:
        return family
    market = _norm(row.get("market"))
    if "corner" not in market:
        return None
    if "team" in market or "home corners" in market or "away corners" in market:
        return "TEAM_CORNERS"
    return "FT_CORNERS"


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _family_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_fixture: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        fixture_id = row.get("fixture_id")
        if fixture_id is None:
            continue
        by_fixture[str(fixture_id)].append(row)

    fixture_rows = sorted(
        (
            {
                "fixture_id": fixture_id,
                "rows": len(items),
                "lines": sorted({
                    _num(item.get("line"))
                    for item in items
                    if _num(item.get("line")) is not None
                }),
                "bookmakers": sorted({
                    str(item.get("bookmaker") or "").strip()
                    for item in items
                    if str(item.get("bookmaker") or "").strip()
                }),
                "stages": sorted({
                    str(item.get("stage") or item.get("signal_stage") or "").strip()
                    for item in items
                    if str(item.get("stage") or item.get("signal_stage") or "").strip()
                }),
                "selections": sorted({
                    str(item.get("selection") or "").strip()
                    for item in items
                    if str(item.get("selection") or "").strip()
                }),
            }
            for fixture_id, items in by_fixture.items()
        ),
        key=lambda row: (-row["rows"], row["fixture_id"]),
    )

    counts = [row["rows"] for row in fixture_rows]
    return {
        "rows": len(rows),
        "unique_fixtures": len(fixture_rows),
        "rows_per_fixture": {
            "min": min(counts) if counts else 0,
            "median": median(counts) if counts else 0,
            "max": max(counts) if counts else 0,
            "mean": round(sum(counts) / len(counts), 4) if counts else 0,
        },
        "fixtures": fixture_rows,
        "top_fixture_concentration": fixture_rows[:20],
    }


def build_report(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {"FT_CORNERS": [], "TEAM_CORNERS": []}
    ignored = Counter()
    for row in rows:
        if not isinstance(row, dict):
            continue
        family = _corner_family(row)
        if family in grouped:
            grouped[family].append(row)
        else:
            ignored[str(row.get("market_family") or "OTHER")] += 1

    families = {family: _family_summary(items) for family, items in grouped.items()}
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": "CORNERS_CLV_CONCENTRATION_AUDIT_COMPLETE",
        "families": families,
        "ignored_rows_by_family": dict(ignored),
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
        "model_weights_changed": False,
        "canonical_bet_logic_changed": False,
        "notes": [
            "CLV row count is not a promotion sample unit; unique fixtures remain the primary promotion sample unit.",
            "Multiple line/side/bookmaker rows from one fixture are retained for market diagnostics but must not be mistaken for independent fixtures.",
            "This audit is diagnostic only and does not alter corners probability, tier, stake, promotion or canonical bet logic.",
        ],
    }


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build_report(_load_jsonl(args.tracking))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
