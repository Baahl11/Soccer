from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any


def poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def over_15(lam: float) -> float:
    return max(0.0, min(1.0, 1.0 - poisson_pmf(0, lam) - poisson_pmf(1, lam)))


def metrics(rows: list[tuple[float, int]]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {"n": 0}
    brier = sum((p - y) ** 2 for p, y in rows) / n
    logloss = sum(-(y * math.log(max(p, 1e-12)) + (1-y) * math.log(max(1-p, 1e-12))) for p, y in rows) / n
    acc = sum(int((p >= 0.5) == bool(y)) for p, y in rows) / n
    return {
        "n": n,
        "accuracy_at_0_5": round(acc, 4),
        "brier": round(brier, 4),
        "log_loss": round(logloss, 4),
        "mean_predicted_probability": round(sum(p for p, _ in rows) / n, 4),
        "observed_rate": round(sum(y for _, y in rows) / n, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Walk-forward calibration challenger for explicit Soccer Edge 1H goals model.")
    ap.add_argument("--model-report", default="soccer_edge_state/analysis/one_h_goals_model.json")
    ap.add_argument("--output", default="soccer_edge_state/analysis/one_h_goals_calibration.json")
    ap.add_argument("--minimum-calibration-history", type=int, default=30)
    ap.add_argument("--scale-prior-observations", type=float, default=50.0)
    args = ap.parse_args()

    report = json.load(open(args.model_report, encoding="utf-8"))
    predictions = list(report.get("predictions") or [])
    predictions.sort(key=lambda r: (str(r.get("kickoff_local") or ""), int(r.get("fixture_id") or 0)))

    baseline_rows: list[tuple[float, int]] = []
    challenger_rows: list[tuple[float, int]] = []
    diagnostics: list[dict[str, Any]] = []
    prior_pred_total = 0.0
    prior_actual_total = 0.0
    prior_n = 0

    for row in predictions:
        actual = row.get("actual_halftime") or {}
        try:
            actual_total = int(actual.get("total"))
            base_lambda = float(row.get("total_lambda_1h"))
        except (TypeError, ValueError):
            continue
        if base_lambda <= 0:
            continue

        if prior_n >= args.minimum_calibration_history:
            raw_ratio = prior_actual_total / max(prior_pred_total, 1e-9)
            # Shrink multiplicative calibration toward 1.0. This prevents the
            # early small sample from overcorrecting the explicit period model.
            weight = prior_n / (prior_n + args.scale_prior_observations)
            scale = (1.0 - weight) + weight * raw_ratio
            scale = max(0.65, min(1.35, scale))
            challenger_lambda = max(0.05, min(4.0, base_lambda * scale))
            p_base = float(row.get("p_over_1_5_1h"))
            p_challenger = over_15(challenger_lambda)
            y = int(actual_total >= 2)
            baseline_rows.append((p_base, y))
            challenger_rows.append((p_challenger, y))
            diagnostics.append({
                "fixture_id": row.get("fixture_id"),
                "kickoff_local": row.get("kickoff_local"),
                "league": row.get("league"),
                "base_lambda_1h": round(base_lambda, 6),
                "calibration_scale": round(scale, 6),
                "challenger_lambda_1h": round(challenger_lambda, 6),
                "baseline_p_over_1_5": round(p_base, 6),
                "challenger_p_over_1_5": round(p_challenger, 6),
                "actual_over_1_5": y,
                "prior_calibration_n": prior_n,
            })

        prior_pred_total += base_lambda
        prior_actual_total += actual_total
        prior_n += 1

    b = metrics(baseline_rows)
    c = metrics(challenger_rows)
    n = c.get("n", 0)
    final_raw_ratio = prior_actual_total / max(prior_pred_total, 1e-9) if prior_n else None
    result = {
        "schema_version": "1.0.0",
        "status": "RESEARCH_ONLY_NOT_ACTIONABLE",
        "timezone_basis": "America/Mexico_City",
        "method": "WALK_FORWARD_SHRUNK_MULTIPLICATIVE_1H_LAMBDA_CALIBRATION",
        "source_model": report.get("model"),
        "explicit_period_model": True,
        "reuses_ft_probability": False,
        "minimum_calibration_history": args.minimum_calibration_history,
        "scale_prior_observations": args.scale_prior_observations,
        "walk_forward_evaluated": n,
        "baseline_same_sample": b,
        "challenger": c,
        "improvement": {
            "accuracy_delta_pp": round((c.get("accuracy_at_0_5", 0)-b.get("accuracy_at_0_5", 0))*100, 2) if n else None,
            "brier_delta": round(c.get("brier", 0)-b.get("brier", 0), 4) if n else None,
            "log_loss_delta": round(c.get("log_loss", 0)-b.get("log_loss", 0), 4) if n else None,
            "mean_probability_delta_pp": round((c.get("mean_predicted_probability", 0)-b.get("mean_predicted_probability", 0))*100, 2) if n else None,
        },
        "full_history_actual_to_predicted_goal_ratio": round(final_raw_ratio, 6) if final_raw_ratio is not None else None,
        "promotion_gate": {
            "enabled": False,
            "decision": "KEEP_RESEARCH_ONLY",
            "minimum_calibrated_oos_n": 100,
            "sample_gate_met": n >= 100,
            "requires": [
                "challenger Brier <= baseline Brier",
                "challenger log-loss <= baseline log-loss",
                "dedicated 1H market comparison",
                "no automatic BET/LEAN promotion",
            ],
        },
        "diagnostics": diagnostics[-250:],
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "diagnostics"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
