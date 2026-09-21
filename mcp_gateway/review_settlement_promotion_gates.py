from __future__ import annotations

import argparse
import json
import os
from typing import Any


MIN_TIER_B_SETTLED = 20
MIN_TIER_A_SETTLED = 50
MIN_TIER_S_SETTLED = 100
MIN_TRUE_CLV_DIRECTIONAL = 50
MIN_TRUE_CLV_MODEL_CHANGE = 200


def fnum(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_json(path: str) -> dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload if isinstance(payload, dict) else {}


def segment_pass_rate(segment_summaries: dict[str, Any], family: str) -> dict[str, Any]:
    relevant: list[dict[str, Any]] = []
    for key, summary in segment_summaries.items():
        if not isinstance(summary, dict) or not str(key).startswith(f"{family}|"):
            continue
        settled = int(summary.get("settled") or 0)
        if settled < 10:
            continue
        review = summary.get("segment_review") or {}
        relevant.append({
            "key": key,
            "settled": settled,
            "roi_units": fnum(summary.get("roi_units")) or 0.0,
            "roi_per_settled_units": fnum(summary.get("roi_per_settled_units")),
            "hit_rate_ex_push": summary.get("hit_rate_ex_push"),
            "status": review.get("status"),
        })
    if not relevant:
        return {
            "reviewed_segments": 0,
            "positive_segments": 0,
            "negative_segments": 0,
            "positive_segment_rate": None,
            "sample_note": "no >=10-settled child segments",
        }
    positive = [s for s in relevant if (s.get("roi_units") or 0.0) > 0]
    negative = [s for s in relevant if (s.get("roi_units") or 0.0) <= 0]
    return {
        "reviewed_segments": len(relevant),
        "positive_segments": len(positive),
        "negative_segments": len(negative),
        "positive_segment_rate": round(len(positive) / len(relevant), 4),
        "strongest_segments": sorted(positive, key=lambda x: (x.get("roi_per_settled_units") or -999, x.get("settled") or 0), reverse=True)[:5],
        "weakest_segments": sorted(negative, key=lambda x: (x.get("roi_per_settled_units") or 999, -(x.get("settled") or 0)))[:5],
    }


def clv_review(true_clv_summary: dict[str, Any], proxy_clv_summary: dict[str, Any]) -> dict[str, Any]:
    true_rows = int(true_clv_summary.get("true_clv_rows") or 0)
    avg_true = fnum(true_clv_summary.get("avg_true_clv_probability_pp"))
    proxy_rows = int(proxy_clv_summary.get("tracked_signal_rows") or proxy_clv_summary.get("ledger_market_observations") or 0)
    proxy_positive_rate = fnum(proxy_clv_summary.get("positive_clv_rate"))

    if true_rows >= MIN_TRUE_CLV_MODEL_CHANGE and avg_true is not None and avg_true >= 0:
        status = "CLV_MODEL_CHANGE_READY"
        reason = "true CLV sample is large enough for model-change consideration and is non-negative"
    elif true_rows >= MIN_TRUE_CLV_DIRECTIONAL and avg_true is not None and avg_true >= 0:
        status = "CLV_DIRECTIONAL_OK"
        reason = "true CLV sample is directional only; enough to avoid negative-CLV warning but not enough for Tier S"
    elif true_rows > 0 and avg_true is not None and avg_true < 0:
        status = "CLV_NEGATIVE_WARNING"
        reason = "true CLV is negative; do not promote until this improves"
    elif proxy_rows > 0:
        status = "PROXY_CLV_ONLY"
        reason = "true CLV sample is missing or too small; proxy CLV can inform diagnostics only"
    else:
        status = "CLV_MISSING"
        reason = "no usable CLV evidence available"

    return {
        "status": status,
        "reason": reason,
        "true_clv_rows": true_rows,
        "avg_true_clv_probability_pp": avg_true,
        "proxy_rows": proxy_rows,
        "proxy_positive_clv_rate": proxy_positive_rate,
        "minimum_policy": {
            "directional_read": MIN_TRUE_CLV_DIRECTIONAL,
            "model_change": MIN_TRUE_CLV_MODEL_CHANGE,
        },
    }


def family_review(family: str, summary: dict[str, Any], segments: dict[str, Any], clv: dict[str, Any]) -> dict[str, Any]:
    settled = int(summary.get("settled") or 0)
    roi = fnum(summary.get("roi_units")) or 0.0
    hit_rate = summary.get("hit_rate_ex_push")
    roi_per = fnum(summary.get("roi_per_settled_units") or summary.get("roi_per_decision_units"))
    child_segments = segments.get("dimensions", {}).get("market_family__stage_bucket", {})
    stage_stability = segment_pass_rate(child_segments if isinstance(child_segments, dict) else {}, family)
    league_segments = segments.get("dimensions", {}).get("market_family__league", {})
    league_stability = segment_pass_rate(league_segments if isinstance(league_segments, dict) else {}, family)

    blockers: list[str] = []
    warnings: list[str] = []
    if settled < MIN_TIER_B_SETTLED:
        blockers.append("sample_below_tier_b_minimum")
    if hit_rate is None:
        blockers.append("no_decided_win_loss_sample")
    if roi <= 0:
        blockers.append("non_positive_roi")
    if clv.get("status") in {"CLV_NEGATIVE_WARNING"}:
        blockers.append("negative_true_clv")
    if clv.get("status") in {"CLV_MISSING", "PROXY_CLV_ONLY"}:
        warnings.append("clv_not_strong_enough_for_tier_a_or_s")
    if (stage_stability.get("reviewed_segments") or 0) and (stage_stability.get("positive_segment_rate") or 0.0) < 0.5:
        warnings.append("stage_stability_weak")
    if (league_stability.get("reviewed_segments") or 0) and (league_stability.get("positive_segment_rate") or 0.0) < 0.5:
        warnings.append("league_stability_weak")

    if blockers:
        recommendation = "HOLD_OR_DEMOTE_REVIEW"
        reason = "blocked by " + ", ".join(blockers)
    elif settled >= MIN_TIER_S_SETTLED and clv.get("status") == "CLV_MODEL_CHANGE_READY" and not warnings:
        recommendation = "TIER_S_CANDIDATE_MANUAL_REVIEW"
        reason = "100+ settled decisions, positive ROI, strong CLV and no stability warnings"
    elif settled >= MIN_TIER_A_SETTLED and clv.get("status") in {"CLV_DIRECTIONAL_OK", "CLV_MODEL_CHANGE_READY"}:
        recommendation = "TIER_A_CANDIDATE_MANUAL_REVIEW"
        reason = "50+ settled decisions, positive ROI and non-negative true CLV; still requires manual review"
    else:
        recommendation = "TIER_B_CANDIDATE_MANUAL_REVIEW"
        reason = "20+ settled decisions and positive ROI; keep low-confidence until CLV/stability improve"

    return {
        "market_family": family,
        "n": summary.get("n"),
        "settled": settled,
        "hit_rate_ex_push": hit_rate,
        "roi_units": round(roi, 4),
        "roi_per_settled_units": roi_per,
        "recommendation": recommendation,
        "reason": reason,
        "blockers": blockers,
        "warnings": warnings,
        "stage_stability": stage_stability,
        "league_stability": league_stability,
        "clv_review": clv,
    }


def build_report(
    segments: dict[str, Any],
    market_summary: dict[str, Any],
    true_clv_summary: dict[str, Any],
    proxy_clv_summary: dict[str, Any],
) -> dict[str, Any]:
    families = market_summary.get("by_market_family") or segments.get("dimensions", {}).get("market_family") or {}
    if not isinstance(families, dict):
        families = {}
    clv = clv_review(true_clv_summary, proxy_clv_summary)
    reviews = [family_review(family, summary, segments, clv) for family, summary in sorted(families.items()) if isinstance(summary, dict)]
    counts: dict[str, int] = {}
    for item in reviews:
        counts[item["recommendation"]] = counts.get(item["recommendation"], 0) + 1
    return {
        "schema_version": "1.0.0",
        "status": "PROMOTION_REVIEW_RESEARCH_ONLY",
        "policy": {
            "tier_b_min_settled": MIN_TIER_B_SETTLED,
            "tier_a_min_settled": MIN_TIER_A_SETTLED,
            "tier_s_min_settled": MIN_TIER_S_SETTLED,
            "true_clv_directional_min_rows": MIN_TRUE_CLV_DIRECTIONAL,
            "true_clv_model_change_min_rows": MIN_TRUE_CLV_MODEL_CHANGE,
            "never_auto_promote": True,
            "requires_manual_review": True,
        },
        "input_status": {
            "segments_status": segments.get("status"),
            "market_summary_schema": market_summary.get("schema_version"),
            "true_clv_status": true_clv_summary.get("status"),
            "proxy_clv_rows": proxy_clv_summary.get("tracked_signal_rows") or proxy_clv_summary.get("ledger_market_observations"),
        },
        "recommendation_counts": dict(sorted(counts.items())),
        "market_family_reviews": reviews,
        "safety_note": "This report creates review candidates only. It must not change runtime pick tiers, stakes, classifications or model weights without a separate reviewed implementation PR.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create conservative Tier B/A/S review gates from settlement segmentation and CLV summaries.")
    parser.add_argument("--segments-summary", default="soccer_edge_state/analysis/settlement_segments_summary.json")
    parser.add_argument("--market-summary", default="soccer_edge_state/analysis/market_performance_summary.json")
    parser.add_argument("--true-clv-summary", default="soccer_edge_state/analysis/true_clv_summary.json")
    parser.add_argument("--clv-summary", default="soccer_edge_state/analysis/clv_summary.json")
    parser.add_argument("--output", default="soccer_edge_state/analysis/settlement_promotion_review.json")
    args = parser.parse_args()

    report = build_report(
        load_json(args.segments_summary),
        load_json(args.market_summary),
        load_json(args.true_clv_summary),
        load_json(args.clv_summary),
    )
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps({
        "status": report["status"],
        "recommendation_counts": report["recommendation_counts"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
