from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from typing import Any, Iterable

SCHEMA_VERSION = "1.1.0"
MODEL_VERSION = "SOCCER_CLV_ENGINE_V4_1.1.0"

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
    "model_version",
    "confidence",
)

# These markets have a price but no numerical handicap/total line by design.
LINELESS_FAMILIES = {"1X2", "BTTS"}

# Historical dedicated-close rows remain valid CLV evidence, but old rows did
# not persist model attribution/confidence. We expose that debt explicitly
# instead of fabricating metadata or letting legacy gaps block modern capture.
LEGACY_ATTRIBUTION_EXEMPT_SOURCES = {"HISTORICAL_DEDICATED_CLOSE"}


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _string(value: Any, default: str = "UNKNOWN") -> str:
    text = str(value or "").strip()
    return text or default


def _market_family(row: dict[str, Any]) -> str:
    existing = str(row.get("market_family") or row.get("family") or "").strip().upper()
    if existing:
        return existing
    market = str(row.get("market") or "").lower()
    if "winner" in market:
        return "1X2"
    if "both teams" in market or "btts" in market:
        return "BTTS"
    if "corner" in market:
        return "FT_CORNERS"
    if "first half" in market or "1h" in market:
        return "1H"
    if "team total" in market:
        return "TEAM_TOTALS"
    if "over/under" in market or "over under" in market:
        return "FT_TOTALS"
    return "UNKNOWN"


def _price_clv_pct(entry_price: Any, closing_price: Any) -> float | None:
    entry = _num(entry_price)
    close = _num(closing_price)
    if entry is None or close is None or entry <= 1.0 or close <= 1.0:
        return None
    return round((entry / close - 1.0) * 100.0, 6)


def _field_present(value: Any) -> bool:
    return value is not None and value != "" and value != "UNKNOWN"


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    family = _market_family(row)
    source = _string(row.get("signal_source"))
    line_applicable = family not in LINELESS_FAMILIES
    legacy_attribution_exempt = source in LEGACY_ATTRIBUTION_EXEMPT_SOURCES

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

    model_version = row.get("model_version") or row.get("automation_version")

    normalized = {
        "fixture_id": row.get("fixture_id"),
        "league": _string(row.get("league")),
        "market_family": family,
        "market": _string(row.get("market")),
        "selection": row.get("selection"),
        "tier": _string(row.get("tier")),
        "confidence": _string(confidence),
        "stage": _string(row.get("stage")),
        "model_version": _string(model_version),
        "signal_source": source,
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
        "line_applicable": line_applicable,
        "legacy_attribution_exempt": legacy_attribution_exempt,
    }

    field_status: dict[str, str] = {}
    for field in REQUIRED_STORAGE_FIELDS:
        if field in {"entry_line", "closing_line", "line_movement"} and not line_applicable:
            field_status[field] = "NOT_APPLICABLE"
        elif (
            field in {"model_version", "confidence"}
            and legacy_attribution_exempt
            and not _field_present(normalized.get(field))
        ):
            field_status[field] = "LEGACY_UNKNOWN"
        else:
            field_status[field] = "PRESENT" if _field_present(normalized.get(field)) else "MISSING"

    normalized["field_status"] = field_status
    normalized["field_completeness"] = {
        field: status in {"PRESENT", "NOT_APPLICABLE", "LEGACY_UNKNOWN"}
        for field, status in field_status.items()
    }
    normalized["capture_complete"] = all(status != "MISSING" for status in field_status.values())
    normalized["fully_attributed"] = all(
        status not in {"MISSING", "LEGACY_UNKNOWN"} for status in field_status.values()
    )
    return normalized


def _coverage(normalized: list[dict[str, Any]], field: str) -> dict[str, Any]:
    statuses = [row["field_status"][field] for row in normalized]
    present = sum(status == "PRESENT" for status in statuses)
    missing = sum(status == "MISSING" for status in statuses)
    not_applicable = sum(status == "NOT_APPLICABLE" for status in statuses)
    legacy_unknown = sum(status == "LEGACY_UNKNOWN" for status in statuses)
    required_rows = present + missing
    return {
        "present": present,
        "required_rows": required_rows,
        "rows": len(normalized),
        "missing": missing,
        "not_applicable": not_applicable,
        "legacy_unknown": legacy_unknown,
        "coverage": round(present / required_rows, 6) if required_rows else 1.0,
    }


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
        line = [_num(item.get("line_movement")) for item in items if item.get("line_applicable")]
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

    field_coverage = {
        field: _coverage(normalized, field)
        for field in REQUIRED_STORAGE_FIELDS
    }

    blockers: list[str] = []
    warnings: list[str] = []

    if len(true_close) < MIN_TRUE_CLOSE_ROWS:
        blockers.append(f"TRUE_CLOSE_ROWS_{len(true_close)}_LT_{MIN_TRUE_CLOSE_ROWS}")

    for field in REQUIRED_STORAGE_FIELDS:
        info = field_coverage[field]
        if info["required_rows"] > 0 and info["coverage"] < 1.0:
            blockers.append(f"{field.upper()}_NOT_FULLY_CAPTURED")

    legacy_model_unknown = field_coverage["model_version"]["legacy_unknown"]
    legacy_confidence_unknown = field_coverage["confidence"]["legacy_unknown"]
    if legacy_model_unknown:
        warnings.append(f"LEGACY_MODEL_VERSION_UNKNOWN_{legacy_model_unknown}")
    if legacy_confidence_unknown:
        warnings.append(f"LEGACY_CONFIDENCE_UNKNOWN_{legacy_confidence_unknown}")

    legacy_rows = sum(
        1 for row in normalized if row.get("legacy_attribution_exempt")
    )
    if legacy_rows:
        warnings.append(f"LEGACY_ATTRIBUTION_ROWS_{legacy_rows}")

    market_agg = _aggregate(true_close, "market")
    family_agg = _aggregate(true_close, "market_family")
    league_agg = _aggregate(true_close, "league")
    tier_agg = _aggregate(true_close, "tier")
    confidence_agg = _aggregate(true_close, "confidence")
    stage_agg = _aggregate(true_close, "stage")
    model_agg = _aggregate(true_close, "model_version")
    bookmaker_agg = _aggregate(true_close, "bookmaker")
    source_agg = _aggregate(true_close, "signal_source")

    probability_values = [_num(row.get("probability_clv")) for row in true_close]
    probability_values = [value for value in probability_values if value is not None]
    price_values = [_num(row.get("price_clv")) for row in true_close]
    price_values = [value for value in price_values if value is not None]

    capture_complete_rows = sum(1 for row in normalized if row["capture_complete"])
    fully_attributed_rows = sum(1 for row in normalized if row["fully_attributed"])
    line_applicable_rows = sum(1 for row in normalized if row["line_applicable"])

    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "phase": "FASE_17_CLV_ENGINE",
        "status": "CLV_CAPTURE_COMPLETE_ANALYSIS_AVAILABLE" if not blockers else "CLV_TRACKING_INCOMPLETE",
        "production_promotion_allowed": False,
        "manual_review_required": True,
        "provider_requests_added": 0,
        "model_weights_changed": False,
        "canonical_bet_logic_changed": False,
        "rows": len(normalized),
        "true_closing_line_rows": len(true_close),
        "minimum_true_close_rows": MIN_TRUE_CLOSE_ROWS,
        "capture_completeness": {
            "complete_rows": capture_complete_rows,
            "complete_rate": round(capture_complete_rows / len(normalized), 6) if normalized else 0.0,
            "fully_attributed_rows": fully_attributed_rows,
            "fully_attributed_rate": round(fully_attributed_rows / len(normalized), 6) if normalized else 0.0,
            "legacy_attribution_rows": legacy_rows,
            "line_applicable_rows": line_applicable_rows,
            "line_not_applicable_rows": len(normalized) - line_applicable_rows,
            "policy": (
                "Numerical line fields are required only for line-based markets. "
                "1X2/BTTS line fields are NOT_APPLICABLE. Historical dedicated-close rows "
                "may retain explicit LEGACY_UNKNOWN model/confidence metadata and never receive fabricated attribution."
            ),
        },
        "overall": {
            "avg_probability_clv_pp": round(sum(probability_values) / len(probability_values), 6) if probability_values else None,
            "positive_probability_clv_rate": round(sum(1 for value in probability_values if value > 0) / len(probability_values), 6) if probability_values else None,
            "avg_price_clv_pct": round(sum(price_values) / len(price_values), 6) if price_values else None,
        },
        "field_coverage": field_coverage,
        "by_market": market_agg,
        "by_market_family": family_agg,
        "by_league": league_agg,
        "by_tier": tier_agg,
        "by_confidence": confidence_agg,
        "by_stage": stage_agg,
        "by_model_version": model_agg,
        "by_bookmaker": bookmaker_agg,
        "by_signal_source": source_agg,
        "blockers": blockers,
        "warnings": warnings,
        "notes": [
            "Probability CLV uses the existing de-vig true-close probability delta when available.",
            "Price CLV is derived as entry_decimal / closing_decimal - 1 when not already stored.",
            "Line movement is required only for line-based markets; 1X2 and BTTS are explicitly line-less.",
            "Modern Postgres-native CLV rows must persist model_version and confidence. Historical dedicated-close rows with unavailable attribution remain explicitly LEGACY_UNKNOWN and are excluded from the modern-attribution requirement.",
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
    parser = argparse.ArgumentParser(description="Phase 17 CLV true-close completeness report.")
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
