from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any

SCHEMA_VERSION = "1.3.0"
MODEL_VERSION = "SOCCER_PROMOTION_FRAMEWORK_V4_1.7.0"

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

FAMILY_SPECS = {
    "FT_TOTALS": {
        "validation_file": "v4_016_ft_totals_production_validation.json",
        "stability_keys": ("FT_TOTALS",),
        "performance_aliases": ("FT_TOTALS",),
    },
    "1X2": {
        "validation_file": "v4_017_1x2_calibration_validation.json",
        "stability_keys": ("1X2", "FT_1X2"),
        "performance_aliases": ("FT_1X2", "1X2"),
    },
    "BTTS": {
        "validation_file": "v4_018_btts_calibration_validation.json",
        "stability_keys": ("BTTS",),
        "performance_aliases": ("BTTS", "FT_BTTS"),
    },
    "TEAM_TOTALS": {
        "validation_file": "v4_019_team_totals_oos_validation.json",
        "stability_keys": ("TEAM_TOTALS", "HOME_TT", "AWAY_TT"),
        "performance_aliases": ("TEAM_TOTALS", "HOME_TT", "AWAY_TT"),
    },
    "1H": {
        "validation_file": "v4_020_1h_oos_validation.json",
        "stability_keys": ("1H",),
        "performance_aliases": ("1H", "1H_TOTALS", "1H_OTHER"),
    },
    "2H": {
        "validation_file": "v4_021_2h_oos_validation.json",
        "stability_keys": ("2H",),
        "performance_aliases": ("2H", "2H_TOTALS", "2H_BTTS"),
    },
    "FT_CORNERS": {
        "validation_file": "v4_022_corners_oos_validation.json",
        "stability_keys": ("FT_CORNERS",),
        "performance_aliases": ("FT_CORNERS", "CORNERS"),
    },
    "TEAM_CORNERS": {
        "validation_file": "v4_022_corners_oos_validation.json",
        "stability_keys": ("TEAM_CORNERS",),
        "performance_aliases": ("TEAM_CORNERS",),
    },
    "CARDS": {
        "validation_file": "phase14_cards_referee_validation.json",
        "stability_keys": ("CARDS",),
        "performance_aliases": ("CARDS", "YELLOW_CARDS", "RED_CARDS"),
    },
    "PLAYER_PROPS": {
        "validation_file": "phase15_player_props_validation.json",
        "stability_keys": ("PLAYER_PROPS", "PROPS"),
        "performance_aliases": ("PLAYER_PROPS", "PROPS"),
    },
}


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().upper().split())


def _load_json(path: str) -> dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def _validation_blockers(report: dict[str, Any]) -> list[str]:
    blockers = report.get("blockers")
    if not isinstance(blockers, list):
        return []
    return [str(value) for value in blockers if value]


def _performance_for_family(market_performance: dict[str, Any], aliases: tuple[str, ...]) -> dict[str, Any]:
    by_family = market_performance.get("by_market_family")
    if not isinstance(by_family, dict):
        return {}
    wanted = {_norm(value) for value in aliases}
    matches = [
        value for key, value in by_family.items()
        if _norm(key) in wanted and isinstance(value, dict)
    ]
    if not matches:
        return {}
    n = sum(int(value.get("n") or 0) for value in matches)
    settled = sum(int(value.get("settled") or 0) for value in matches)
    roi_units = sum(float(value.get("roi_units") or 0.0) for value in matches)
    return {
        "n": n,
        "settled": settled,
        "roi_units": round(roi_units, 6),
        "roi_per_decision_units": round(roi_units / n, 6) if n else None,
    }


def _shadow_for_family(shadow_performance: dict[str, Any], aliases: tuple[str, ...]) -> dict[str, Any]:
    by_family = shadow_performance.get("by_market_family")
    if not isinstance(by_family, dict):
        return {}
    wanted = {_norm(value) for value in aliases}
    matches = [
        value for key, value in by_family.items()
        if _norm(key) in wanted and isinstance(value, dict)
    ]
    if not matches:
        return {}
    rows = sum(int(value.get("rows") or 0) for value in matches)
    settled = sum(int(value.get("settled") or 0) for value in matches)
    roi_units = sum(float(value.get("shadow_roi_hypothetical_units") or 0.0) for value in matches)
    if settled >= TIER_B_REVIEW_MIN:
        sample_status = "SHADOW_REVIEW_READY"
    elif settled >= DIRECTIONAL_READ_MIN:
        sample_status = "DIRECTIONAL_SHADOW"
    else:
        sample_status = "DATA_BLOCKED"
    negative_stages = sorted({
        stage
        for value in matches
        for stage in (value.get("negative_directional_stages") or [])
    })
    positive_stages = sorted({
        stage
        for value in matches
        for stage in (value.get("positive_directional_stages") or [])
    })
    return {
        "rows": rows,
        "settled": settled,
        "shadow_roi_hypothetical_units": round(roi_units, 6),
        "shadow_roi_per_settled_unit": round(roi_units / settled, 6) if settled else None,
        "sample_status": sample_status,
        "negative_directional_stages": negative_stages,
        "positive_directional_stages": positive_stages,
    }


def _promotion_shadow_for_family(
    family: str,
    shadow_selection_diagnostics: dict[str, Any],
    promotion_shadow_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    promotion_shadow_report = promotion_shadow_report if isinstance(promotion_shadow_report, dict) else {}

    families = promotion_shadow_report.get("families")
    family_report = families.get(family) if isinstance(families, dict) and isinstance(families.get(family), dict) else {}
    evidence = family_report.get("promotion_evaluable") if isinstance(family_report.get("promotion_evaluable"), dict) else {}
    if evidence:
        return {
            "settled": int(evidence.get("settled") or 0),
            "pending": int(evidence.get("pending") or 0),
            "shadow_roi_per_settled_unit": _num(evidence.get("roi_per_settled_unit")),
            "sample_status": str(evidence.get("sample_status") or "DATA_BLOCKED"),
            "negative_directional_stages": list(evidence.get("negative_directional_stages") or []),
            "family_discrimination_ready": evidence.get("family_discrimination_ready"),
            "not_ready_classes": list(evidence.get("not_ready_classes") or []),
            "evidence_source": "POSTGRES_PHASE16_REPLAY:PROMOTION_EVALUABLE",
        }

    # Backward-compatible read for the earlier 1X2-only report shape.
    if (
        family == "1X2"
        and _norm(promotion_shadow_report.get("market_family")) == "1X2"
        and isinstance(promotion_shadow_report.get("promotion_evaluable"), dict)
    ):
        evidence = promotion_shadow_report["promotion_evaluable"]
        return {
            "settled": int(evidence.get("settled") or 0),
            "pending": int(evidence.get("pending") or 0),
            "shadow_roi_per_settled_unit": _num(evidence.get("roi_per_settled_unit")),
            "sample_status": str(evidence.get("sample_status") or "DATA_BLOCKED"),
            "negative_directional_stages": list(evidence.get("negative_directional_stages") or []),
            "family_discrimination_ready": evidence.get("family_discrimination_ready"),
            "not_ready_classes": list(evidence.get("not_ready_classes") or []),
            "evidence_source": "POSTGRES_PHASE16_REPLAY:PROMOTION_EVALUABLE",
        }

    if family != "1X2":
        return {
            "settled": 0,
            "pending": 0,
            "shadow_roi_per_settled_unit": None,
            "sample_status": "QUALITY_DIAGNOSTICS_NOT_IMPLEMENTED",
            "negative_directional_stages": [],
            "family_discrimination_ready": None,
            "not_ready_classes": [],
            "evidence_source": None,
        }

    if _norm(shadow_selection_diagnostics.get("market_family")) != "FT_1X2":
        return {
            "settled": 0,
            "pending": 0,
            "shadow_roi_per_settled_unit": None,
            "sample_status": "QUALITY_DIAGNOSTICS_MISSING",
            "negative_directional_stages": [],
            "family_discrimination_ready": None,
            "not_ready_classes": [],
            "evidence_source": None,
        }

    evidence = (
        shadow_selection_diagnostics.get("promotion_evaluable")
        if isinstance(shadow_selection_diagnostics.get("promotion_evaluable"), dict)
        else {}
    )
    return {
        "settled": int(evidence.get("settled") or 0),
        "pending": int(evidence.get("pending") or 0),
        "shadow_roi_per_settled_unit": _num(evidence.get("roi_per_settled_unit")),
        "sample_status": str(evidence.get("sample_status") or "DATA_BLOCKED"),
        "negative_directional_stages": list(evidence.get("negative_directional_stages") or []),
        "family_discrimination_ready": evidence.get("family_discrimination_ready"),
        "not_ready_classes": list(evidence.get("not_ready_classes") or []),
        "evidence_source": "SHADOW_SELECTION_DIAGNOSTICS:PROMOTION_EVALUABLE",
    }

def _stability_for_family(stability_report: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    families = stability_report.get("families")
    if not isinstance(families, dict):
        return {}

    matches = [families[key] for key in keys if isinstance(families.get(key), dict)]
    if not matches:
        return {}
    if len(matches) == 1:
        return matches[0]

    overalls = [value.get("overall") for value in matches if isinstance(value.get("overall"), dict)]
    rows = sum(int(value.get("rows") or 0) for value in overalls)
    # Conservative for combined home/away families: do not sum unique fixtures,
    # because the same fixture can contribute to both sides.
    unique_fixtures = max((int(value.get("unique_fixtures") or 0) for value in overalls), default=0)
    weighted_parts = []
    for value in overalls:
        n = int(value.get("rows") or 0)
        clv = _num(value.get("fixture_weighted_avg_probability_clv_pp"))
        if n > 0 and clv is not None:
            weighted_parts.append((n, clv))
    avg_clv = (
        sum(n * clv for n, clv in weighted_parts) / sum(n for n, _ in weighted_parts)
        if weighted_parts else None
    )

    negative_leagues = sorted({
        item
        for value in matches
        for item in (value.get("negative_directional_leagues") or [])
    })
    negative_stages = sorted({
        item
        for value in matches
        for item in (value.get("negative_directional_stages") or [])
    })

    if unique_fixtures < DIRECTIONAL_READ_MIN:
        status = "DATA_BLOCKED"
    elif unique_fixtures < TIER_B_REVIEW_MIN:
        status = "DIRECTIONAL_ONLY"
    elif negative_leagues or negative_stages:
        status = "SEGMENT_REVIEW"
    else:
        status = "STABILITY_REVIEW_READY"

    return {
        "status": status,
        "overall": {
            "rows": rows,
            "unique_fixtures": unique_fixtures,
            "fixture_weighted_avg_probability_clv_pp": round(avg_clv, 6) if avg_clv is not None else None,
        },
        "negative_directional_leagues": negative_leagues,
        "negative_directional_stages": negative_stages,
    }


def review_market(
    *,
    market_family: str,
    unique_fixtures: int,
    settled: int,
    roi_per_settled_unit: float | None,
    clv_rows: int,
    avg_clv_pp: float | None,
    stability_status: str,
    shadow_settled: int = 0,
    shadow_roi_per_settled_unit: float | None = None,
    shadow_sample_status: str = "MISSING",
    shadow_negative_directional_stages: list[str] | None = None,
    shadow_family_discrimination_ready: bool | None = None,
    shadow_not_ready_classes: list[str] | None = None,
    validation_blockers: list[str] | None = None,
    current_state: str = "RESEARCH",
    manual_approval: bool = False,
) -> dict[str, Any]:
    current = _norm(current_state)
    if current not in STATES:
        current = "RESEARCH"
    validation_blockers = list(validation_blockers or [])
    shadow_negative_directional_stages = sorted(set(shadow_negative_directional_stages or []))
    shadow_not_ready_classes = sorted(set(str(value) for value in (shadow_not_ready_classes or []) if value))

    blockers: list[str] = []
    warnings: list[str] = []

    if unique_fixtures < DIRECTIONAL_READ_MIN:
        blockers.append(f"UNIQUE_FIXTURES_{unique_fixtures}_LT_DIRECTIONAL_{DIRECTIONAL_READ_MIN}")
    if unique_fixtures < TIER_B_REVIEW_MIN:
        blockers.append(f"UNIQUE_FIXTURES_{unique_fixtures}_LT_TIER_B_REVIEW_{TIER_B_REVIEW_MIN}")
    if settled < DIRECTIONAL_READ_MIN:
        blockers.append(f"SETTLED_{settled}_LT_DIRECTIONAL_{DIRECTIONAL_READ_MIN}")
    if settled < TIER_B_REVIEW_MIN:
        blockers.append(f"SETTLED_{settled}_LT_TIER_B_REVIEW_{TIER_B_REVIEW_MIN}")
    if clv_rows <= 0 or avg_clv_pp is None:
        blockers.append("FAMILY_TRUE_CLV_MISSING")
    elif avg_clv_pp < 0:
        blockers.append("FAMILY_TRUE_CLV_NEGATIVE")
    if roi_per_settled_unit is None:
        blockers.append("ROI_MISSING")
    elif roi_per_settled_unit <= 0:
        blockers.append("ROI_NOT_POSITIVE")
    if stability_status != "STABILITY_REVIEW_READY":
        blockers.append(f"STABILITY_{stability_status or 'MISSING'}")
    if shadow_settled < DIRECTIONAL_READ_MIN:
        blockers.append(f"PROMOTION_SHADOW_SETTLED_{shadow_settled}_LT_DIRECTIONAL_{DIRECTIONAL_READ_MIN}")
    elif shadow_roi_per_settled_unit is None:
        blockers.append("PROMOTION_SHADOW_ROI_MISSING")
    elif shadow_roi_per_settled_unit <= 0:
        blockers.append("PROMOTION_SHADOW_ROI_NOT_POSITIVE")
    if shadow_negative_directional_stages:
        blockers.append("PROMOTION_SHADOW_NEGATIVE_DIRECTIONAL_STAGES:" + ",".join(shadow_negative_directional_stages))
    if market_family == "1X2" and shadow_family_discrimination_ready is not True:
        detail = ",".join(shadow_not_ready_classes or ["UNKNOWN"])
        blockers.append(f"PROMOTION_SHADOW_1X2_CLASS_DISCRIMINATION_NOT_READY:{detail}")
    blockers.extend(f"VALIDATION:{value}" for value in validation_blockers)

    tier_review_eligibility = {
        "directional_read": unique_fixtures >= DIRECTIONAL_READ_MIN and settled >= DIRECTIONAL_READ_MIN,
        "tier_b_review": unique_fixtures >= TIER_B_REVIEW_MIN and settled >= TIER_B_REVIEW_MIN,
        "tier_a_review": unique_fixtures >= TIER_A_REVIEW_MIN and settled >= TIER_A_REVIEW_MIN,
        "tier_s_review": unique_fixtures >= TIER_S_REVIEW_MIN and settled >= TIER_S_REVIEW_MIN,
        "model_weight_change_review": unique_fixtures >= MODEL_WEIGHT_CHANGE_MIN,
        "promotion_shadow_directional_read": shadow_settled >= DIRECTIONAL_READ_MIN,
        "promotion_shadow_review": shadow_settled >= TIER_B_REVIEW_MIN and shadow_roi_per_settled_unit is not None and shadow_roi_per_settled_unit > 0,
        "promotion_shadow_stage_stability": not shadow_negative_directional_stages,
        "promotion_shadow_family_discrimination": market_family != "1X2" or shadow_family_discrimination_ready is True,
    }

    collapse = (
        current in PRODUCTION_STATES
        and unique_fixtures >= TIER_B_REVIEW_MIN
        and settled >= TIER_B_REVIEW_MIN
        and roi_per_settled_unit is not None
        and roi_per_settled_unit < 0
        and avg_clv_pp is not None
        and avg_clv_pp < 0
    )

    if collapse:
        recommended_state = "DEMOTED"
        automatic_demotion_candidate = True
    elif unique_fixtures < DIRECTIONAL_READ_MIN:
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
        if unique_fixtures >= TIER_S_REVIEW_MIN and settled >= TIER_S_REVIEW_MIN:
            recommended_state = "TIER_S"
        elif unique_fixtures >= TIER_A_REVIEW_MIN and settled >= TIER_A_REVIEW_MIN:
            recommended_state = "TIER_A"
        elif unique_fixtures >= TIER_B_REVIEW_MIN and settled >= TIER_B_REVIEW_MIN:
            recommended_state = "TIER_B"
        else:
            recommended_state = "LEAN_ELIGIBLE"
        automatic_demotion_candidate = False

    return {
        "market_family": market_family,
        "current_state": current,
        "recommended_state": recommended_state,
        "unique_fixtures": unique_fixtures,
        "settled": settled,
        "roi_per_settled_unit": roi_per_settled_unit,
        "true_clv_rows": clv_rows,
        "avg_true_clv_probability_pp": avg_clv_pp,
        "stability_status": stability_status or "MISSING",
        "promotion_shadow_settled": shadow_settled,
        "promotion_shadow_roi_per_settled_unit": shadow_roi_per_settled_unit,
        "promotion_shadow_sample_status": shadow_sample_status,
        "promotion_shadow_negative_directional_stages": shadow_negative_directional_stages,
        "promotion_shadow_family_discrimination_ready": shadow_family_discrimination_ready,
        "promotion_shadow_not_ready_classes": shadow_not_ready_classes,
        "validation_blockers": validation_blockers,
        "tier_review_eligibility": tier_review_eligibility,
        "manual_approval_present": bool(manual_approval),
        "automatic_demotion_candidate": automatic_demotion_candidate,
        "blockers": blockers,
        "warnings": warnings,
    }


def build_report(
    market_performance: dict[str, Any],
    stability_report: dict[str, Any],
    validation_reports: dict[str, dict[str, Any]],
    shadow_performance: dict[str, Any] | None = None,
    shadow_selection_diagnostics: dict[str, Any] | None = None,
    promotion_shadow_report: dict[str, Any] | None = None,
    *,
    current_states: dict[str, str] | None = None,
    manual_approvals: dict[str, bool] | None = None,
) -> dict[str, Any]:
    current_states = current_states or {}
    manual_approvals = manual_approvals or {}
    shadow_performance = shadow_performance if isinstance(shadow_performance, dict) else {}
    shadow_selection_diagnostics = (
        shadow_selection_diagnostics
        if isinstance(shadow_selection_diagnostics, dict)
        else {}
    )
    promotion_shadow_report = (
        promotion_shadow_report
        if isinstance(promotion_shadow_report, dict)
        else {}
    )

    reviews: list[dict[str, Any]] = []
    for family, spec in FAMILY_SPECS.items():
        stability = _stability_for_family(stability_report, tuple(spec["stability_keys"]))
        overall = stability.get("overall") if isinstance(stability.get("overall"), dict) else {}
        unique_fixtures = int(overall.get("unique_fixtures") or 0)
        clv_rows = int(overall.get("rows") or 0)
        avg_clv = _num(overall.get("fixture_weighted_avg_probability_clv_pp"))

        perf = _performance_for_family(market_performance, tuple(spec["performance_aliases"]))
        watch_shadow = _shadow_for_family(shadow_performance, tuple(spec["performance_aliases"]))
        promotion_shadow = _promotion_shadow_for_family(
            family,
            shadow_selection_diagnostics,
            promotion_shadow_report,
        )
        settled = int(perf.get("settled") or 0)
        roi_per = _num(perf.get("roi_per_decision_units"))
        roi_units = _num(perf.get("roi_units"))
        if roi_per is None and roi_units is not None and settled > 0:
            roi_per = roi_units / settled

        validation = validation_reports.get(family) or {}
        blockers = _validation_blockers(validation)
        if not validation:
            blockers.append("VALIDATION_REPORT_MISSING")

        review = review_market(
            market_family=family,
            unique_fixtures=unique_fixtures,
            settled=settled,
            roi_per_settled_unit=roi_per,
            clv_rows=clv_rows,
            avg_clv_pp=avg_clv,
            stability_status=str(stability.get("status") or "MISSING"),
            shadow_settled=int(promotion_shadow.get("settled") or 0),
            shadow_roi_per_settled_unit=_num(promotion_shadow.get("shadow_roi_per_settled_unit")),
            shadow_sample_status=str(promotion_shadow.get("sample_status") or "MISSING"),
            shadow_negative_directional_stages=list(promotion_shadow.get("negative_directional_stages") or []),
            shadow_family_discrimination_ready=promotion_shadow.get("family_discrimination_ready"),
            shadow_not_ready_classes=list(promotion_shadow.get("not_ready_classes") or []),
            validation_blockers=blockers,
            current_state=current_states.get(family, "RESEARCH"),
            manual_approval=bool(manual_approvals.get(family, False)),
        )
        review["validation_status"] = validation.get("status")
        review["validation_model_version"] = validation.get("model_version")
        review["promotion_shadow_evidence_source"] = promotion_shadow.get("evidence_source")
        review["promotion_shadow_pending"] = int(promotion_shadow.get("pending") or 0)
        review["watch_shadow_settled"] = int(watch_shadow.get("settled") or 0)
        review["watch_shadow_roi_per_settled_unit"] = _num(watch_shadow.get("shadow_roi_per_settled_unit"))
        review["watch_shadow_sample_status"] = str(watch_shadow.get("sample_status") or "MISSING")
        review["watch_shadow_negative_directional_stages"] = list(watch_shadow.get("negative_directional_stages") or [])
        reviews.append(review)

    counts: dict[str, int] = {}
    for review in reviews:
        state = review["recommended_state"]
        counts[state] = counts.get(state, 0) + 1

    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "phase": "FASE_19_PROMOTION_FRAMEWORK",
        "validation_block": "G6_PROMOTION_REPORTS",
        "status": "PROMOTION_REVIEW_FRAMEWORK_ACTIVE",
        "states": list(STATES),
        "sample_policy": {
            "primary_sample_unit": "UNIQUE_FIXTURES",
            "directional_read": DIRECTIONAL_READ_MIN,
            "tier_b_review_preferred": TIER_B_REVIEW_MIN,
            "tier_a_review": TIER_A_REVIEW_MIN,
            "tier_s_review": TIER_S_REVIEW_MIN,
            "model_weight_change": MODEL_WEIGHT_CHANGE_MIN,
            "settlements_required_in_parallel": True,
            "promotion_shadow_directional_minimum": DIRECTIONAL_READ_MIN,
            "promotion_shadow_review_minimum": TIER_B_REVIEW_MIN,
            "promotion_shadow_roi_must_be_positive_for_lean_eligibility": True,
            "watch_alert_rows_do_not_count_toward_promotion": True,
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
        "market_family_reviews": reviews,
        "recommended_state_counts": counts,
        "notes": [
            "Promotion sample gates use unique fixtures from G5, not raw CLV row counts.",
            "Settled decisions and ROI are required in parallel with OOS/calibration/CLV/stability evidence.",
            "WATCH shadow observations remain research diagnostics and do not count toward promotion unless a market-specific quality diagnostic marks them promotion-evaluable.",
            "For 1X2, FT_TOTALS and BTTS, persisted Phase16 primary rankable candidates replayed from Postgres are the preferred promotion-shadow source; pending rows become settled automatically when final results arrive.",
            "1X2 promotion review additionally requires current family-level Home/Draw/Away discrimination readiness; partial class readiness remains SHADOW even if settled ROI and CLV are otherwise sufficient.",
            "Legacy WATCH diagnostics remain fallback-only for 1X2 and never count when clean Phase16 replay evidence exists.",
            "No market is automatically promoted. Tier B/A/S requires explicit manual approval after every evidence gate passes.",
            "Automatic demotion can only be flagged for an already-production market with sufficient unique fixtures and settlements plus negative ROI and negative family CLV.",
        ],
    }


def _load_validation_reports(analysis_dir: str) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for family, spec in FAMILY_SPECS.items():
        reports[family] = _load_json(os.path.join(analysis_dir, str(spec["validation_file"])))
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="G6 / Phase 19 family-aware promotion framework report.")
    parser.add_argument("--analysis-dir", required=True)
    parser.add_argument("--market-performance", required=True)
    parser.add_argument("--stability-report", required=True)
    parser.add_argument("--shadow-performance")
    parser.add_argument("--shadow-selection-diagnostics")
    parser.add_argument("--promotion-shadow-report")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = build_report(
        _load_json(args.market_performance),
        _load_json(args.stability_report),
        _load_validation_reports(args.analysis_dir),
        _load_json(args.shadow_performance) if args.shadow_performance else {},
        _load_json(args.shadow_selection_diagnostics) if args.shadow_selection_diagnostics else {},
        _load_json(args.promotion_shadow_report) if args.promotion_shadow_report else {},
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({
        "model_version": report["model_version"],
        "status": report["status"],
        "recommended_state_counts": report["recommended_state_counts"],
        "automatic_promotion_allowed": report["automatic_promotion_allowed"],
        "provider_requests_added": report["provider_requests_added"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
