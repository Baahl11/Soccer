from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0.0"

FAMILIES = {
    "FT_GOALS": ["ft_totals_validation.json", "ft_goals_shadow_validation.json"],
    "TEAM_TOTALS": ["team_totals_validation.json"],
    "CORRECT_SCORE": ["correct_score_validation.json"],
    "BTTS": ["btts_validation.json"],
    "1X2": ["one_x_two_calibration.json", "one_x_two_model_selection.json"],
    "DOUBLE_CHANCE": ["double_chance_validation.json"],
    "DNB": ["dnb_validation.json"],
    "ASIAN_HANDICAP": ["asian_handicap_validation.json"],
    "1H_GOALS": ["one_h_goals_model.json", "one_h_goals_calibration.json"],
    "2H_GOALS_PREGAME": ["two_h_goals_model.json"],
    "2H_HALFTIME": ["two_h_halftime_conditioned.json"],
    "CORNERS_FT": ["corners_baseline.json"],
    "TEAM_CORNERS": ["team_corners_validation.json"],
    "CARDS_TOTAL": ["cards_baseline.json"],
    "PLAYER_SHOTS": ["player_shots_model_sanity.json"],
    "PLAYER_SOT": ["player_sot_model_sanity.json"],
    "GOALSCORER": ["player_goalscorer_model_sanity.json"],
    "ASSISTS": ["player_assists_model_sanity.json"],
    "GK_SAVES": ["gk_saves_model_sanity.json"],
    "PLAYER_CARDS": ["player_cards_model_sanity.json"],
    "XG_XGA": ["xg_registry_sanity.json"],
}

SAMPLE_KEYS = {
    "n", "sample_n", "sample_size", "evaluated", "evaluated_fixtures",
    "evaluated_matches", "oos_n", "oos_matches", "observations",
}
GATE_KEYS = {
    "market_comparison_sample_gate_met",
    "actionable_review_sample_gate_met",
    "promotion_gate_met",
    "eligible_for_promotion_review",
    "enabled",
}
METRIC_TOKENS = ("brier", "log_loss", "logloss", "negative_log", "clv", "roi", "mae")


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def walk(obj: Any, prefix: str = "") -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            out.append((path, value))
            out.extend(walk(value, path))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj[:50]):
            out.extend(walk(value, f"{prefix}[{idx}]"))
    return out


def numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_report(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    flat = walk(payload)
    samples: list[dict[str, Any]] = []
    gates: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []

    for path, value in flat:
        leaf = path.split(".")[-1].split("[")[0].lower()
        num = numeric(value)
        if leaf in SAMPLE_KEYS and num is not None:
            samples.append({"path": path, "value": num})
        if leaf in GATE_KEYS and isinstance(value, bool):
            gates.append({"path": path, "value": value})
        if any(token in leaf for token in METRIC_TOKENS) and num is not None:
            metrics.append({"path": path, "value": num})

    max_sample = max((row["value"] for row in samples), default=None)
    true_gates = sum(1 for row in gates if row["value"] is True)
    false_gates = sum(1 for row in gates if row["value"] is False)

    return {
        "file": name,
        "status": payload.get("status"),
        "model": payload.get("model") or payload.get("model_version"),
        "max_detected_sample": max_sample,
        "sample_evidence": samples[:12],
        "gate_evidence": gates[:20],
        "metric_evidence": metrics[:20],
        "gate_true_count": true_gates,
        "gate_false_count": false_gates,
    }


def family_row(analysis_dir: Path, family: str, files: list[str]) -> dict[str, Any]:
    summaries = []
    missing = []
    for name in files:
        payload = load_json(analysis_dir / name)
        if payload is None:
            missing.append(name)
        else:
            summaries.append(summarize_report(name, payload))

    max_sample = max(
        (s["max_detected_sample"] for s in summaries if s["max_detected_sample"] is not None),
        default=None,
    )
    any_false_gate = any(s["gate_false_count"] > 0 for s in summaries)
    any_metrics = any(bool(s["metric_evidence"]) for s in summaries)

    blockers = []
    if missing:
        blockers.append("REQUIRED_REPORT_MISSING")
    if not summaries:
        blockers.append("NO_VALIDATION_REPORT")
    if not any_metrics:
        blockers.append("STANDARD_CALIBRATION_METRICS_NOT_DETECTED")
    if any_false_gate:
        blockers.append("ONE_OR_MORE_PROMOTION_GATES_NOT_MET")
    blockers.append("MANUAL_REVIEW_REQUIRED")
    blockers.append("NO_AUTOMATIC_WEIGHT_OR_PRODUCTION_CHANGE")

    if not summaries:
        review_status = "REPORT_MISSING"
    elif any_false_gate:
        review_status = "DATA_ACCUMULATING"
    elif missing:
        review_status = "INCOMPLETE_EVIDENCE"
    else:
        review_status = "READY_FOR_MANUAL_PROMOTION_REVIEW"

    return {
        "family": family,
        "review_status": review_status,
        "reports_expected": files,
        "reports_found": [s["file"] for s in summaries],
        "reports_missing": missing,
        "max_detected_sample": max_sample,
        "calibration_metrics_detected": any_metrics,
        "reports": summaries,
        "blockers": blockers,
        "automatic_promotion_allowed": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", default="soccer_edge_state/analysis")
    ap.add_argument("--output", default="soccer_edge_state/analysis/promotion_review.json")
    args = ap.parse_args()

    analysis_dir = Path(args.analysis_dir)
    rows = [family_row(analysis_dir, family, files) for family, files in FAMILIES.items()]
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "STANDARDIZED_MANUAL_PROMOTION_REVIEW",
        "policy": (
            "CONSOLIDATE_OOS_BRIER_LOGLOSS_CLV_ROI_AND_SAMPLE_GATES; "
            "NEVER_AUTO_PROMOTE; NEVER_AUTO_CHANGE_WEIGHTS; VERSIONED_MANUAL_REVIEW_REQUIRED"
        ),
        "families": rows,
        "counts": {
            "families": len(rows),
            "ready_for_manual_review": sum(r["review_status"] == "READY_FOR_MANUAL_PROMOTION_REVIEW" for r in rows),
            "data_accumulating": sum(r["review_status"] == "DATA_ACCUMULATING" for r in rows),
            "incomplete_evidence": sum(r["review_status"] == "INCOMPLETE_EVIDENCE" for r in rows),
            "report_missing": sum(r["review_status"] == "REPORT_MISSING" for r in rows),
        },
        "global_blockers": [
            "PROMOTION_REQUIRES_VERSIONED_MANUAL_DECISION",
            "TRUE_CLV_REQUIRED_WHERE_MARKET_HISTORY_EXISTS",
            "NO_SINGLE_GAME_OR_SMALL_SAMPLE_RECALIBRATION",
        ],
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
