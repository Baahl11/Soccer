from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_PROMOTION_FRAMEWORK_V4_1.0.0"

STATES = (
    "DORMANT",
    "RESEARCH",
    "SHADOW",
    "LEAN_ELIGIBLE",
    "TIER_B",
    "TIER_A",
    "TIER_S",
    "DEMOTED",
)

DIRECTIONAL_READ_MIN = 20
TIER_B_REVIEW_MIN = 50
TIER_A_REVIEW_MIN = 100
TIER_S_REVIEW_MIN = 200
MODEL_WEIGHT_CHANGE_MIN = 200

PRODUCTION_STATES = {"TIER_B", "TIER_A", "TIER_S"}


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).upper()


def canonical_clv_family(market_name: str) -> str | None:
    value = _norm(market_name)
    if "MATCH WINNER" in value or value == "WINNER":
        return "FT_1X2"
    if "BOTH TEAMS" in value or "BTTS" in value:
        return "BTTS"
    if any(token in value for token in ("FIRST HALF", "1ST HALF", "1H")):
        return "1H"
    if any(token in value for token in ("SECOND HALF", "2ND HALF", "2H")):
        return "2H"
    if "CORNER" in value:
        return "CORNERS"
    if "CARD" in value or "BOOKING" in value:
        return "CARDS"
    if "PLAYER" in value or "SHOTS ON TARGET" in value or "GOALKEEPER SAVES" in value:
        return "PROPS"
    if "TEAM TOTAL" in value or "TEAM GOALS" in value:
        return "TEAM_TOTALS"
    if "GOALS OVER/UNDER" in value or "OVER/UNDER" in value:
        return "FT_TOTALS"
    return None


def family_clv_from_phase17(clv_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = clv_report.get("by_market") if isinstance(clv_report.get("by_market"), dict) else {}
    accum: dict[str, dict[str, float]] = {}
    for market_name, metrics in raw.items():
        if not isinstance(metrics, dict):
            continue
        family = canonical_clv_family(str(market_name))
        if family is None:
            continue
        rows = int(metrics.get("rows") or 0)
        avg = _num(metrics.get("avg_probability_clv_pp"))
        if family not in accum:
            accum[family] = {"rows": 0.0, "weighted": 0.0, "known_rows": 0.0}
        accum[family]["rows"] += rows
        if avg is not None and rows > 0:
            accum[family]["weighted"] += avg * rows
            accum[family]["known_rows"] += rows

    out: dict[str, dict[str, Any]] = {}
    for family, values in accum.items():
        known_rows = int(values["known_rows"])
        out[family] = {
            "rows": int(values["rows"]),
            "avg_probability_clv_pp": round(values["weighted"] / known_rows, 6) if known_rows else None,
        }
    return out


def review_market(
    *,
    market_family: str,
    settled: int,
    roi_per_settled_unit: float | None,
    clv_rows: int,
    avg_clv_pp: float | None,
    oos_framework_ready: bool,
    current_state: str = "RESEARCH",
    manual_approval: bool = False,
) -> dict[str, Any]:
    current = _norm(current_state)
    if current not in STATES:
        current = "RESEARCH"

    blockers: list[str] = []
    warnings: list[str] = []

    if settled < DIRECTIONAL_READ_MIN:
        blockers.append(f"SETTLED_{settled}_LT_DIRECTIONAL_{DIRECTIONAL_READ_MIN}")
    if settled < TIER_B_REVIEW_MIN:
        blockers.append(f"SETTLED_{settled}_LT_TIER_B_REVIEW_{TIER_B_REVIEW_MIN}")
    if clv_rows < TIER_B_REVIEW_MIN:
        blockers.append(f"TRUE_CLV_{clv_rows}_LT_DIRECTIONAL_{TIER_B_REVIEW_MIN}")
    if roi_per_settled_unit is None:
        blockers.append("ROI_MISSING")
    elif roi_per_settled_unit <= 0:
        blockers.append("ROI_NOT_POSITIVE")
    if avg_clv_pp is None:
        blockers.append("FAMILY_CLV_MISSING")
    elif avg_clv_pp < 0:
        blockers.append("FAMILY_CLV_NEGATIVE")
    if not oos_framework_ready:
        blockers.append("OOS_FRAMEWORK_NOT_READY")

    tier_review_eligibility = {
        "directional_read": settled >= DIRECTIONAL_READ_MIN,
        "tier_b_review": settled >= TIER_B_REVIEW_MIN,
        "tier_a_review": settled >= TIER_A_REVIEW_MIN,
        "tier_s_review": settled >= TIER_S_REVIEW_MIN,
        "model_weight_change_review": settled >= MODEL_WEIGHT_CHANGE_MIN,
    }

    collapse = (
        current in PRODUCTION_STATES
        and settled >= TIER_B_REVIEW_MIN
        and roi_per_settled_unit is not None
        and roi_per_settled_unit < 0
        and avg_clv_pp is not None
        and avg_clv_pp < 0
    )

    if collapse:
        recommended_state = "DEMOTED"
        automatic_demotion_candidate = True
    elif settled < DIRECTIONAL_READ_MIN:
        recommended_state = "RESEARCH"
        automatic_demotion_candidate = False
    elif blockers:
        recommended_state = "SHADOW"
        automatic_demotion_candidate = False
    elif not manual_approval:
        recommended_state = "LEAN_ELIGIBLE"
        automatic_demotion_candidate = False
        warnings.append("MANUAL_APPROVAL_REQUIRED_FOR_PRODUCTION_TIER")
    else:
        if settled >= TIER_S_REVIEW_MIN:
            recommended_state = "TIER_S"
        elif settled >= TIER_A_REVIEW_MIN:
            recommended_state = "TIER_A"
        elif settled >= TIER_B_REVIEW_MIN:
            recommended_state = "TIER_B"
        else:
            recommended_state = "LEAN_ELIGIBLE"
        automatic_demotion_candidate = False

    return {
        "market_family": market_family,
        "current_state": current,
        "recommended_state": recommended_state,
        "settled": settled,
        "roi_per_settled_unit": roi_per_settled_unit,
        "true_clv_rows": clv_rows,
        "avg_true_clv_probability_pp": avg_clv_pp,
        "oos_framework_ready": oos_framework_ready,
        "tier_review_eligibility": tier_review_eligibility,
        "manual_approval_present": bool(manual_approval),
        "automatic_demotion_candidate": automatic_demotion_candidate,
        "blockers": blockers,
        "warnings": warnings,
    }


def build_report(
    market_performance: dict[str, Any],
    clv_report: dict[str, Any],
    oos_report: dict[str, Any],
    *,
    current_states: dict[str, str] | None = None,
    manual_approvals: dict[str, bool] | None = None,
) -> dict[str, Any]:
    current_states = current_states or {}
    manual_approvals = manual_approvals or {}
    by_family = market_performance.get("by_market_family") if isinstance(market_performance.get("by_market_family"), dict) else {}
    clv_by_family = family_clv_from_phase17(clv_report)
    oos_ready = oos_report.get("status") == "OOS_FRAMEWORK_READY_FOR_MODEL_PREDICTIONS"

    reviews: list[dict[str, Any]] = []
    for family, perf in sorted(by_family.items()):
        if not isinstance(perf, dict):
            continue
        settled = int(perf.get("settled") or 0)
        roi_units = _num(perf.get("roi_units"))
        roi_per = _num(perf.get("roi_per_decision_units"))
        if roi_per is None and roi_units is not None and settled > 0:
            roi_per = roi_units / settled
        clv = clv_by_family.get(family, {"rows": 0, "avg_probability_clv_pp": None})
        reviews.append(review_market(
            market_family=family,
            settled=settled,
            roi_per_settled_unit=roi_per,
            clv_rows=int(clv.get("rows") or 0),
            avg_clv_pp=_num(clv.get("avg_probability_clv_pp")),
            oos_framework_ready=oos_ready,
            current_state=current_states.get(family, "RESEARCH"),
            manual_approval=bool(manual_approvals.get(family, False)),
        ))

    counts: dict[str, int] = {}
    for review in reviews:
        state = review["recommended_state"]
        counts[state] = counts.get(state, 0) + 1

    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "phase": "FASE_19_PROMOTION_FRAMEWORK",
        "status": "PROMOTION_REVIEW_FRAMEWORK_ACTIVE",
        "states": list(STATES),
        "sample_policy": {
            "directional_read": DIRECTIONAL_READ_MIN,
            "tier_b_review_preferred": TIER_B_REVIEW_MIN,
            "tier_a_review": TIER_A_REVIEW_MIN,
            "tier_s_review": TIER_S_REVIEW_MIN,
            "model_weight_change": MODEL_WEIGHT_CHANGE_MIN,
        },
        "automatic_report": True,
        "manual_approval_required": True,
        "automatic_promotion_allowed": False,
        "automatic_demotion_allowed_under_safety_policy": True,
        "runtime_state_mutation_enabled": False,
        "production_promotion_allowed": False,
        "provider_requests_added": 0,
        "model_weights_changed": False,
        "canonical_bet_logic_changed": False,
        "oos_framework_ready": oos_ready,
        "market_family_reviews": reviews,
        "recommended_state_counts": counts,
        "notes": [
            "No market is automatically promoted. Tier B/A/S requires explicit manual approval after evidence gates pass.",
            "Automatic demotion may be flagged only for an already-production market with >=50 settled decisions and both negative ROI and negative family-specific CLV.",
            "Sample thresholds follow the V4 master roadmap: 20 directional, 50 Tier B review, 100 Tier A, 200 Tier S/model-weight review.",
        ],
    }


def _load_json(path: str) -> dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 19 promotion framework report.")
    parser.add_argument("--market-performance", required=True)
    parser.add_argument("--clv-report", required=True)
    parser.add_argument("--oos-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build_report(
        _load_json(args.market_performance),
        _load_json(args.clv_report),
        _load_json(args.oos_report),
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
