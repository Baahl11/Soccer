from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from statistics import median
from typing import Any, Iterable

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_LEAGUE_STAGE_STABILITY_V4_1.0.0"
MIN_DIRECTIONAL_FIXTURES = 20
MIN_REVIEW_FIXTURES = 50


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _label(value: Any, fallback: str) -> str:
    text = " ".join(str(value or "").strip().split())
    return text or fallback


def _fixture_weighted_mean(rows: list[dict[str, Any]], field: str) -> float | None:
    by_fixture: dict[Any, list[float]] = defaultdict(list)
    for row in rows:
        fixture_id = row.get("fixture_id")
        value = _num(row.get(field))
        if fixture_id is None or value is None:
            continue
        by_fixture[fixture_id].append(value)
    if not by_fixture:
        return None
    fixture_means = [sum(values) / len(values) for values in by_fixture.values() if values]
    return round(sum(fixture_means) / len(fixture_means), 6) if fixture_means else None


def summarize(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if isinstance(row, dict)]
    probability_values = [_num(row.get("probability_clv")) for row in selected]
    probability_values = [value for value in probability_values if value is not None]
    price_values = [_num(row.get("price_clv")) for row in selected]
    price_values = [value for value in price_values if value is not None]
    fixtures = {row.get("fixture_id") for row in selected if row.get("fixture_id") is not None}
    fixture_n = len(fixtures)

    if fixture_n >= MIN_REVIEW_FIXTURES:
        sample_status = "REVIEW_SAMPLE"
    elif fixture_n >= MIN_DIRECTIONAL_FIXTURES:
        sample_status = "DIRECTIONAL_SAMPLE"
    else:
        sample_status = "DATA_BLOCKED"

    return {
        "rows": len(selected),
        "unique_fixtures": fixture_n,
        "sample_status": sample_status,
        "avg_probability_clv_pp": round(sum(probability_values) / len(probability_values), 6) if probability_values else None,
        "median_probability_clv_pp": round(float(median(probability_values)), 6) if probability_values else None,
        "fixture_weighted_avg_probability_clv_pp": _fixture_weighted_mean(selected, "probability_clv"),
        "avg_price_clv_pct": round(sum(price_values) / len(price_values), 6) if price_values else None,
        "fixture_weighted_avg_price_clv_pct": _fixture_weighted_mean(selected, "price_clv"),
        "positive_rows": sum(1 for value in probability_values if value > 0),
        "negative_rows": sum(1 for value in probability_values if value < 0),
        "flat_rows": sum(1 for value in probability_values if math.isclose(value, 0.0, abs_tol=1e-12)),
        "positive_row_rate": round(sum(1 for value in probability_values if value > 0) / len(probability_values), 6) if probability_values else None,
    }


def _segment_rows(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fallback = "<UNKNOWN_LEAGUE>" if key == "league" else "<UNKNOWN_STAGE>"
    for row in rows:
        grouped[_label(row.get(key), fallback)].append(row)
    return dict(grouped)


def _negative_directional_segments(segments: dict[str, dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for name, summary in segments.items():
        if int(summary.get("unique_fixtures") or 0) < MIN_DIRECTIONAL_FIXTURES:
            continue
        value = _num(summary.get("fixture_weighted_avg_probability_clv_pp"))
        if value is not None and value < 0:
            out.append(name)
    return sorted(out)


def build_report(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    clean = [
        row for row in rows
        if isinstance(row, dict)
        and row.get("market_family")
        and _num(row.get("probability_clv")) is not None
    ]

    by_family_raw: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in clean:
        by_family_raw[_label(row.get("market_family"), "<UNKNOWN_FAMILY>")].append(row)

    families: dict[str, Any] = {}
    for family, family_rows in sorted(by_family_raw.items()):
        league_raw = _segment_rows(family_rows, "league")
        stage_raw = _segment_rows(family_rows, "stage")
        by_league = {name: summarize(values) for name, values in sorted(league_raw.items())}
        by_stage = {name: summarize(values) for name, values in sorted(stage_raw.items())}
        overall = summarize(family_rows)

        total_fixtures = max(1, int(overall.get("unique_fixtures") or 0))
        max_league_fixture_share = max(
            (int(summary.get("unique_fixtures") or 0) / total_fixtures for summary in by_league.values()),
            default=0.0,
        )
        max_stage_fixture_share = max(
            (int(summary.get("unique_fixtures") or 0) / total_fixtures for summary in by_stage.values()),
            default=0.0,
        )

        negative_leagues = _negative_directional_segments(by_league)
        negative_stages = _negative_directional_segments(by_stage)
        fixture_n = int(overall.get("unique_fixtures") or 0)
        if fixture_n < MIN_DIRECTIONAL_FIXTURES:
            status = "DATA_BLOCKED"
        elif fixture_n < MIN_REVIEW_FIXTURES:
            status = "DIRECTIONAL_ONLY"
        elif negative_leagues or negative_stages:
            status = "SEGMENT_REVIEW"
        else:
            status = "STABILITY_REVIEW_READY"

        families[family] = {
            "status": status,
            "overall": overall,
            "league_count": len(by_league),
            "stage_count": len(by_stage),
            "directional_leagues": sorted(name for name, value in by_league.items() if int(value.get("unique_fixtures") or 0) >= MIN_DIRECTIONAL_FIXTURES),
            "review_sample_leagues": sorted(name for name, value in by_league.items() if int(value.get("unique_fixtures") or 0) >= MIN_REVIEW_FIXTURES),
            "directional_stages": sorted(name for name, value in by_stage.items() if int(value.get("unique_fixtures") or 0) >= MIN_DIRECTIONAL_FIXTURES),
            "review_sample_stages": sorted(name for name, value in by_stage.items() if int(value.get("unique_fixtures") or 0) >= MIN_REVIEW_FIXTURES),
            "negative_directional_leagues": negative_leagues,
            "negative_directional_stages": negative_stages,
            "max_league_fixture_share": round(max_league_fixture_share, 6),
            "max_stage_fixture_share": round(max_stage_fixture_share, 6),
            "by_league": by_league,
            "by_stage": by_stage,
        }

    family_status_counts: dict[str, int] = defaultdict(int)
    for value in families.values():
        family_status_counts[str(value.get("status"))] += 1

    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "validation_block": "G5_LEAGUE_STAGE_STABILITY",
        "status": "ACTIVE_RESEARCH_VALIDATION",
        "production_promotion_allowed": False,
        "manual_review_required": True,
        "provider_requests_added": 0,
        "sample_policy": {
            "primary_unit": "UNIQUE_FIXTURES",
            "directional_minimum": MIN_DIRECTIONAL_FIXTURES,
            "review_minimum": MIN_REVIEW_FIXTURES,
            "raw_clv_rows_are_not_treated_as_independent_samples": True,
        },
        "input_rows": len(clean),
        "unique_fixtures": len({row.get("fixture_id") for row in clean if row.get("fixture_id") is not None}),
        "market_family_count": len(families),
        "family_status_counts": dict(sorted(family_status_counts.items())),
        "families": families,
        "notes": [
            "League/stage stability gates use unique fixtures rather than raw CLV rows to avoid inflating evidence from multiple correlated lines or selections in one match.",
            "Both row-weighted and fixture-weighted CLV are reported; fixture-weighted CLV is preferred for stability review.",
            "This validator does not promote any market family automatically.",
            "Negative directional segments are surfaced only after at least 20 unique fixtures in that segment.",
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
    parser = argparse.ArgumentParser(description="G5 league/stage stability validator from canonical true-CLV rows.")
    parser.add_argument("--true-clv-tracking", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build_report(_load_jsonl(args.true_clv_tracking))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({
        "model_version": report["model_version"],
        "status": report["status"],
        "input_rows": report["input_rows"],
        "unique_fixtures": report["unique_fixtures"],
        "market_family_count": report["market_family_count"],
        "family_status_counts": report["family_status_counts"],
        "provider_requests_added": report["provider_requests_added"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
