from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from typing import Any, Iterable

SETTLED_OUTCOMES = {"WIN", "LOSS", "PUSH"}
DECIDED_OUTCOMES = {"WIN", "LOSS"}


def fnum(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def safe_key(value: Any, fallback: str = "UNKNOWN") -> str:
    s = norm(value)
    return s if s else fallback


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def odds_band(price: Any) -> str:
    p = fnum(price)
    if p is None or p <= 1.0:
        return "NO_PRICE"
    if p < 1.50:
        return "LT_1_50"
    if p < 1.80:
        return "1_50_1_79"
    if p < 2.20:
        return "1_80_2_19"
    if p < 3.00:
        return "2_20_2_99"
    return "GE_3_00"


def stage_bucket(stage: Any) -> str:
    s = safe_key(stage).upper().replace(" ", "_")
    if s in {"POSTGAME", "CLOSE", "EARLY_RESEARCH"}:
        return s
    m = re.match(r"T-(\d+)", s)
    if not m:
        return "UNKNOWN_STAGE"
    minutes = int(m.group(1))
    if minutes <= 30:
        return "T_MINUS_0_30"
    if minutes <= 60:
        return "T_MINUS_31_60"
    if minutes <= 90:
        return "T_MINUS_61_90"
    return "T_MINUS_91_PLUS"


def league_key(row: dict[str, Any]) -> str:
    lid = row.get("league_id")
    league = safe_key(row.get("league"), "UNKNOWN_LEAGUE")
    if lid is not None:
        return f"{lid}|{league}"
    return league


def segment_status(summary: dict[str, Any]) -> dict[str, Any]:
    settled = int(summary.get("settled") or 0)
    roi = float(summary.get("roi_units") or 0.0)
    hit_rate = summary.get("hit_rate_ex_push")

    if settled < 10:
        status = "INSUFFICIENT_SAMPLE"
        reason = "fewer than 10 settled decisions in this segment"
    elif settled < 20:
        status = "DIRECTIONAL_ONLY"
        reason = "10-19 settled decisions; useful for diagnosis but not promotion"
    elif hit_rate is None:
        status = "HOLD_NO_DECIDED_OUTCOMES"
        reason = "segment has no win/loss decisions after pushes"
    elif roi <= 0:
        status = "HOLD_OR_DEMOTE_CANDIDATE"
        reason = "20+ settled decisions and non-positive ROI"
    else:
        status = "PROMOTION_REVIEW_INPUT"
        reason = "20+ settled decisions and positive ROI; requires CLV and stability checks before promotion"

    return {
        "status": status,
        "reason": reason,
        "settled": settled,
        "hit_rate_ex_push": hit_rate,
        "roi_units": round(roi, 4),
    }


def performance_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    group = list(rows)
    c = Counter(str(row.get("settlement_status") or "UNKNOWN") for row in group)
    decided = c["WIN"] + c["LOSS"]
    settled = decided + c["PUSH"]
    roi = sum(float(row.get("roi_units") or 0.0) for row in group if row.get("settlement_status") in SETTLED_OUTCOMES)
    prices = [float(p) for p in (fnum(row.get("decimal_price")) for row in group) if p is not None]
    summary = {
        "n": len(group),
        "settled": settled,
        "win": c["WIN"],
        "loss": c["LOSS"],
        "push": c["PUSH"],
        "ungraded": len(group) - settled,
        "hit_rate_ex_push": round(c["WIN"] / decided, 4) if decided else None,
        "roi_units": round(roi, 4),
        "roi_per_settled_units": round(roi / settled, 4) if settled else None,
        "avg_decimal_price": round(sum(prices) / len(prices), 4) if prices else None,
        "outcome_counts": dict(sorted(c.items())),
    }
    summary["segment_review"] = segment_status(summary)
    return summary


def add_group(groups: dict[str, list[dict[str, Any]]], key: str, row: dict[str, Any]) -> None:
    groups[key].append(row)


def build_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_dimension: dict[str, dict[str, list[dict[str, Any]]]] = {
        "classification": defaultdict(list),
        "market_family": defaultdict(list),
        "market_family__classification": defaultdict(list),
        "market_family__stage_bucket": defaultdict(list),
        "market_family__odds_band": defaultdict(list),
        "market_family__data_tier": defaultdict(list),
        "market_family__league": defaultdict(list),
        "market_family__stage_bucket__odds_band": defaultdict(list),
        "market_family__classification__stage_bucket": defaultdict(list),
    }

    for row in rows:
        market = safe_key(row.get("market_family"), "UNKNOWN_MARKET")
        classification = safe_key(row.get("classification"), "UNKNOWN_CLASS")
        stage = stage_bucket(row.get("stage"))
        band = odds_band(row.get("decimal_price"))
        data_tier = safe_key(row.get("data_tier"), "UNKNOWN_DATA_TIER")
        league = league_key(row)

        add_group(by_dimension["classification"], classification, row)
        add_group(by_dimension["market_family"], market, row)
        add_group(by_dimension["market_family__classification"], f"{market}|{classification}", row)
        add_group(by_dimension["market_family__stage_bucket"], f"{market}|{stage}", row)
        add_group(by_dimension["market_family__odds_band"], f"{market}|{band}", row)
        add_group(by_dimension["market_family__data_tier"], f"{market}|{data_tier}", row)
        add_group(by_dimension["market_family__league"], f"{market}|{league}", row)
        add_group(by_dimension["market_family__stage_bucket__odds_band"], f"{market}|{stage}|{band}", row)
        add_group(by_dimension["market_family__classification__stage_bucket"], f"{market}|{classification}|{stage}", row)

    dimension_summaries = {
        dimension: {key: performance_summary(group) for key, group in sorted(groups.items())}
        for dimension, groups in by_dimension.items()
    }

    def eligible_items(dimension: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for key, summary in dimension_summaries.get(dimension, {}).items():
            settled = int(summary.get("settled") or 0)
            if settled < 10:
                continue
            item = {"key": key, **summary}
            items.append(item)
        items.sort(key=lambda x: (float(x.get("roi_per_settled_units") or -999), int(x.get("settled") or 0)), reverse=True)
        return items[:25]

    return {
        "schema_version": "1.0.0",
        "status": "SEGMENTATION_RESEARCH_ONLY",
        "input_rows": len(rows),
        "settled_rows": sum(1 for row in rows if row.get("settlement_status") in SETTLED_OUTCOMES),
        "minimum_sample_policy": {
            "insufficient_sample_lt": 10,
            "directional_only_lt": 20,
            "promotion_review_min_settled": 20,
            "requires_before_promotion": ["true CLV", "league stability", "stage stability", "odds-band stability", "manual sanity review"],
        },
        "overall": performance_summary(rows),
        "dimensions": dimension_summaries,
        "top_positive_segments": {
            "market_family__classification": eligible_items("market_family__classification"),
            "market_family__stage_bucket": eligible_items("market_family__stage_bucket"),
            "market_family__odds_band": eligible_items("market_family__odds_band"),
            "market_family__league": eligible_items("market_family__league"),
        },
        "safety_note": "This report is diagnostic only. It does not promote BET/LEAN tiers by itself; segments require CLV and stability confirmation before any production change.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment Soccer Edge settlement performance by market, stage, odds band, data tier and league.")
    parser.add_argument("--settlement-ledger", default="soccer_edge_state/analysis/bet_settlement_ledger.jsonl")
    parser.add_argument("--output", default="soccer_edge_state/analysis/settlement_segments_summary.json")
    args = parser.parse_args()

    rows = load_jsonl(args.settlement_ledger)
    report = build_report(rows)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps({
        "status": report["status"],
        "input_rows": report["input_rows"],
        "settled_rows": report["settled_rows"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
