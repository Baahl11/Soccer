from __future__ import annotations

import argparse
import json
import os
from typing import Any


def load_json(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def fnum(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def challenger_row(name: str, report: dict[str, Any]) -> dict[str, Any]:
    n = int(report.get("walk_forward_evaluated") or 0)
    baseline = report.get("baseline") if isinstance(report.get("baseline"), dict) else {}
    challenger = report.get("challenger") if isinstance(report.get("challenger"), dict) else {}
    improvement = report.get("improvement") if isinstance(report.get("improvement"), dict) else {}
    base_brier = fnum(baseline.get("brier"))
    chal_brier = fnum(challenger.get("brier"))
    base_ll = fnum(baseline.get("log_loss"))
    chal_ll = fnum(challenger.get("log_loss"))
    quality_better = (
        base_brier is not None and chal_brier is not None and chal_brier < base_brier
        and base_ll is not None and chal_ll is not None and chal_ll < base_ll
    )
    class_discrimination = (
        improvement.get("class_discrimination")
        if isinstance(improvement.get("class_discrimination"), dict)
        else {}
    )
    draw_discrimination = (
        class_discrimination.get("D")
        if isinstance(class_discrimination.get("D"), dict)
        else {}
    )
    class_discrimination_available = all(
        isinstance(class_discrimination.get(label), dict)
        for label in ("H", "D", "A")
    )
    draw_discrimination_ready = draw_discrimination.get("challenger_discrimination_ready") is True

    sample_gate = n >= 200
    if not sample_gate:
        status = "INSUFFICIENT_SAME_METHOD_OOS_SAMPLE"
    elif not quality_better:
        status = "MIXED_OR_WORSE_THAN_MATCHED_BASELINE"
    elif not class_discrimination_available:
        status = "CLASS_DISCRIMINATION_NOT_MEASURED"
    elif not draw_discrimination_ready:
        status = "DRAW_DISCRIMINATION_NOT_READY"
    else:
        status = "PROMOTION_REVIEW_ELIGIBLE_RESEARCH_ONLY"
    return {
        "candidate": name,
        "walk_forward_evaluated": n,
        "baseline_brier": base_brier,
        "challenger_brier": chal_brier,
        "baseline_log_loss": base_ll,
        "challenger_log_loss": chal_ll,
        "brier_delta": fnum(improvement.get("brier_delta")),
        "log_loss_delta": fnum(improvement.get("log_loss_delta")),
        "accuracy_delta_pp": fnum(improvement.get("accuracy_delta_pp")),
        "quality_better_on_matched_report": quality_better,
        "minimum_same_method_sample_gate_met": sample_gate,
        "class_discrimination_available": class_discrimination_available,
        "draw_discrimination_ready": draw_discrimination_ready,
        "draw_auc_lower_95": fnum(draw_discrimination.get("challenger_auc_lower_95")),
        "draw_auc_lower_95_delta_vs_baseline": fnum(draw_discrimination.get("auc_lower_95_delta")),
        "class_discrimination": class_discrimination,
        "status": status,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Consolidate Soccer Edge 1X2 research challengers without auto-selecting a production winner.")
    ap.add_argument("--analysis-dir", default="soccer_edge_state/analysis")
    ap.add_argument("--output", default="soccer_edge_state/analysis/one_x_two_model_selection.json")
    args = ap.parse_args()

    calibration = load_json(os.path.join(args.analysis_dir, "one_x_two_calibration.json"))
    relative = load_json(os.path.join(args.analysis_dir, "one_x_two_relative_strength.json"))
    dixon = load_json(os.path.join(args.analysis_dir, "one_x_two_dixon_coles.json"))
    prior = load_json(os.path.join(args.analysis_dir, "one_x_two_prior_calibration.json"))

    canonical = {
        "sample_fixtures": int(calibration.get("sample_fixtures") or 0),
        "multiclass_brier": fnum(calibration.get("multiclass_brier")),
        "multiclass_log_loss": fnum(calibration.get("multiclass_log_loss")),
        "top1_accuracy": fnum(calibration.get("top1_accuracy")),
        "status": calibration.get("status") or "NOT_AVAILABLE",
    }
    candidates = [
        challenger_row("RELATIVE_STRENGTH", relative),
        challenger_row("DIXON_COLES", dixon),
        challenger_row("CLASS_PRIOR_CALIBRATION", prior),
    ]
    review_eligible = [row["candidate"] for row in candidates if row["status"] == "PROMOTION_REVIEW_ELIGIBLE_RESEARCH_ONLY"]

    # Different challenger reports can contain different fixture cohorts. Never rank
    # candidates across non-identical samples from summary metrics alone.
    if not review_eligible:
        recommendation = "KEEP_CANONICAL_RESEARCH_BASELINE; NO_CHALLENGER_READY_FOR_FORMAL_REVIEW"
    elif len(review_eligible) == 1:
        recommendation = f"RUN_SAME_FIXTURE_HEAD_TO_HEAD_BEFORE_ANY_PROMOTION:{review_eligible[0]}"
    else:
        recommendation = "RUN_SAME_FIXTURE_HEAD_TO_HEAD_AMONG_REVIEW_ELIGIBLE_CHALLENGERS"

    output = {
        "schema_version": "1.0.0",
        "status": "RESEARCH_ONLY_MODEL_SELECTION",
        "canonical": canonical,
        "challengers": candidates,
        "review_eligible_challengers": review_eligible,
        "recommendation": recommendation,
        "production_winner_selected": False,
        "promotion_gate": {
            "enabled": False,
            "minimum_same_method_oos": 200,
            "minimum_same_fixture_head_to_head_oos": 300,
            "minimum_actionable_review_oos": 400,
            "requires": [
                "same-fixture head-to-head comparison for shortlisted challengers",
                "Brier and log-loss both improve versus canonical on the same cohort",
                "same-cohort H/D/A class discrimination is measured and Draw AUC lower-95 exceeds 0.50",
                "H/D/A calibration stable by probability bucket and competition",
                "verified historical 1X2 prices and true CLV evidence",
                "validated market shrinkage after final model selection",
                "manual/versioned promotion; never automatic weight changes",
            ],
        },
        "methodology_notes": [
            "Each challenger is first judged only against its own matched baseline cohort.",
            "Summary metrics from different fixture cohorts are not used to declare a winner.",
            "Accuracy alone is insufficient; probabilistic calibration metrics are primary.",
            "This report cannot promote 1X2 to BET/LEAN/Galaxy.",
        ],
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
