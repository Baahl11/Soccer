from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from typing import Any, Iterable

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_CLV_ENGINE_V4_1.0.0"

MIN_TRUE_CLOSE_ROWS = 50
REQUIRED_STORAGE_FIELDS = (
    "entry_line",
    "entry_price",
    "closing_line",
    "closing_price",
    "probability_clv",
    "price_clv",
    "line_movement",
    "bookmaker",
)


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _string(value: Any, default: str = "UNKNOWN") -> str:
    text = str(value or "").strip()
    return text or default


def _price_clv_pct(entry_price: Any, closing_price: Any) -> float | None:
    entry = _num(entry_price)
    close = _num(closing_price)
    if entry is None or close is None or entry <= 1.0 or close <= 1.0:
        return None
    # Positive when the bettor captured a larger decimal price than the close.
    return round((entry / close - 1.0) * 100.0, 6)


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    entry_line = _num(row.get("entry_line"))
    if entry_line is None:
        entry_line = _num(row.get("line"))

    closing_line = _num(row.get("closing_line"))
    entry_price = _num(row.get("entry_price"))
    if entry_price is None:
        entry_price = _num(row.get("signal_price"))
    closing_price = _num(row.get("closing_price"))
    if closing_price is None:
        closing_price = _num(row.get("close_price"))

    probability_clv = _num(row.get("probability_clv"))
    if probability_clv is None:
        probability_clv = _num(row.get("clv_probability_pp"))

    price_clv = _num(row.get("price_clv"))
    if price_clv is None:
        price_clv = _price_clv_pct(entry_price, closing_price)

    line_movement = _num(row.get("line_movement"))
    if line_movement is None and entry_line is not None and closing_line is not None:
        line_movement = round(closing_line - entry_line, 6)

    bookmaker = row.get("bookmaker")
    if not bookmaker:
        bookmaker = row.get("bookmaker_at_signal")

    confidence = row.get("confidence")
    if confidence is None:
        confidence = row.get("model_signal")
    if confidence is None:
        confidence = row.get("tier")

    model_version = row.get("model_version") or row.get("automation_version")

    normalized = {
        "fixture_id": row.get("fixture_id"),
        "league": _string(row.get("league")),
        "market": _string(row.get("market")),
        "selection": row.get("selection"),
        "tier": _string(row.get("tier")),
        "confidence": _string(confidence),
        "stage": _string(row.get("stage")),
        "model_version": _string(model_version),
        "entry_line": entry_line,
        "entry_price": entry_price,
        "closing_line": closing_line,
        "closing_price": closing_price,
        "probability_clv": probability_clv,
        "price_clv": price_clv,
        "line_movement": line_movement,
        "bookmaker": _string(bookmaker),
        "is_true_closing_line": bool(row.get("is_true_closing_line")),
        "closing_line_status": row.get("closing_line_status"),
    }

    normalized["field_completeness"] = {
        field: normalized.get(field) is not None and normalized.get(field) != "UNKNOWN"
        for field in REQUIRED_STORAGE_FIELDS
    }
    return normalized


def _aggregate(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_string(row.get(key))].append(row)

    out: dict[str, Any] = {}
    for group, items in sorted(groups.items()):
        probability = [_num(item.get("probability_clv")) for item in items]
        probability = [value for value in probability if value is not None]
        price = [_num(item.get("price_clv")) for item in items]
        price = [value for value in price if value is not None]
        line = [_num(item.get("line_movement")) for item in items]
        line = [value for value in line if value is not None]
        out[group] = {
            "rows": len(items),
            "unique_fixtures": len({item.get("fixture_id") for item in items if item.get("fixture_id") is not None}),
            "avg_probability_clv_pp": round(sum(probability) / len(probability), 6) if probability else None,
            "positive_probability_clv_rate": round(sum(1 for value in probability if value > 0) / len(probability), 6) if probability else None,
            "avg_price_clv_pct": round(sum(price) / len(price), 6) if price else None,
            "avg_line_movement": round(sum(line) / len(line), 6) if line else None,
        }
    return out


def build_report(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    normalized = [normalize_row(row) for row in rows if isinstance(row, dict)]
    true_close = [row for row in normalized if row["is_true_closing_line"]]

    field_coverage: dict[str, dict[str, Any]] = {}
    for field in REQUIRED_STORAGE_FIELDS:
        count = sum(1 for row in normalized if row["field_completeness"][field])
        field_coverage[field] = {
            "present": count,
            "rows": len(normalized),
            "coverage": round(count / len(normalized), 6) if normalized else 0.0,
        }

    blockers: list[str] = []
    warnings: list[str] = []

    if len(true_close) < MIN_TRUE_CLOSE_ROWS:
        blockers.append(f"TRUE_CLOSE_ROWS_{len(true_close)}_LT_{MIN_TRUE_CLOSE_ROWS}")
    if field_coverage["closing_line"]["coverage"] < 1.0:
        blockers.append("CLOSING_LINE_NOT_FULLY_CAPTURED")
    if field_coverage["line_movement"]["coverage"] < 1.0:
        blockers.append("LINE_MOVEMENT_NOT_FULLY_CAPTURED")
    if field_coverage["confidence"]["coverage"] < 1.0 if "confidence" in field_coverage else False:
        warnings.append("CONFIDENCE_COVERAGE_INCOMPLETE")
    if any(row["model_version"] == "UNKNOWN" for row in normalized):
        blockers.append("MODEL_VERSION_NOT_CAPTURED_PER_CLV_ROW")
    if any(row["confidence"] == "UNKNOWN" for row in normalized):
        blockers.append("CONFIDENCE_NOT_CAPTURED_PER_CLV_ROW")

    market_agg = _aggregate(true_close, "market")
    league_agg = _aggregate(true_close, "league")
    tier_agg = _aggregate(true_close, "tier")
    confidence_agg = _aggregate(true_close, "confidence")
    stage_agg = _aggregate(true_close, "stage")
    model_agg = _aggregate(true_close, "model_version")
    bookmaker_agg = _aggregate(true_close, "bookmaker")

    probability_values = [_num(row.get("probability_clv")) for row in true_close]
    probability_values = [value for value in probability_values if value is not None]
    price_values = [_num(row.get("price_clv")) for row in true_close]
    price_values = [value for value in price_values if value is not None]

    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "phase": "FASE_17_CLV_ENGINE",
        "status": "CLV_ANALYSIS_AVAILABLE" if not blockers else "CLV_TRACKING_INCOMPLETE",
        "production_promotion_allowed": False,
        "manual_review_required": True,
        "provider_requests_added": 0,
        "model_weights_changed": False,
        "canonical_bet_logic_changed": False,
        "rows": len(normalized),
        "true_closing_line_rows": len(true_close),
        "minimum_true_close_rows": MIN_TRUE_CLOSE_ROWS,
        "overall": {
            "avg_probability_clv_pp": round(sum(probability_values) / len(probability_values), 6) if probability_values else None,
            "positive_probability_clv_rate": round(sum(1 for value in probability_values if value > 0) / len(probability_values), 6) if probability_values else None,
            "avg_price_clv_pct": round(sum(price_values) / len(price_values), 6) if price_values else None,
        },
        "field_coverage": field_coverage,
        "by_market": market_agg,
        "by_league": league_agg,
        "by_tier": tier_agg,
        "by_confidence": confidence_agg,
        "by_stage": stage_agg,
        "by_model_version": model_agg,
        "by_bookmaker": bookmaker_agg,
        "blockers": blockers,
        "warnings": warnings,
        "notes": [
            "Probability CLV uses the existing de-vig true-close probability delta when available.",
            "Price CLV is derived as entry_decimal / closing_decimal - 1, positive when entry captured the larger price.",
            "Line movement requires an explicit closing_line field; entry line alone is not treated as line movement.",
            "Current legacy rows may contain tier but not a dedicated confidence field; new rows should capture both confidence and model_version explicitly.",
        ],
    }


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 17 CLV engine report.")
    parser.add_argument("--tracking", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build_report(_load_jsonl(args.tracking))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
