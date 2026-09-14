from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from typing import Any


def fnum(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def final_outcome(result: dict[str, Any] | None) -> str | None:
    if not isinstance(result, dict) or not result:
        return None
    goals = result.get("goals") or {}
    score = result.get("score") or {}
    ft = score.get("fulltime") or {}
    h = goals.get("home", ft.get("home"))
    a = goals.get("away", ft.get("away"))
    try:
        h, a = int(h), int(a)
    except (TypeError, ValueError):
        return None
    return "H" if h > a else "A" if a > h else "D"


def bucket(p: float) -> str:
    lo = int(min(0.9, max(0.0, p)) * 10) * 10
    hi = lo + 10
    return f"{lo:02d}-{hi:02d}%"


def main() -> None:
    ap = argparse.ArgumentParser(description="Research-only 1X2 calibration diagnostics for Soccer Edge.")
    ap.add_argument("--ledger", default="soccer_edge_state/analysis/signal_ledger.jsonl")
    ap.add_argument("--output", default="soccer_edge_state/analysis/one_x_two_calibration.json")
    args = ap.parse_args()

    # Keep the latest pre-kickoff raw projection per fixture to avoid repeated-stage weighting.
    latest: dict[int, dict[str, Any]] = {}
    with open(args.ledger, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            fid = row.get("fixture_id")
            raw = row.get("raw_projection") or {}
            result = row.get("result")
            probs = [fnum(raw.get(k)) for k in ("raw_home_win_prob", "raw_draw_prob", "raw_away_win_prob")]
            if not fid or any(p is None for p in probs) or final_outcome(result) is None:
                continue
            if any(p < 0 or p > 1 for p in probs):
                continue
            s = sum(probs)
            if s <= 0:
                continue
            probs = [p / s for p in probs]
            ts = str(row.get("generated_at_local") or "")
            rec = {
                "fixture_id": int(fid), "league": row.get("league"), "data_tier": row.get("data_tier"),
                "stage": row.get("stage"), "timestamp": ts, "p": probs, "actual": final_outcome(result),
            }
            old = latest.get(int(fid))
            if old is None or ts > old["timestamp"]:
                latest[int(fid)] = rec

    labels = ["H", "D", "A"]
    n = len(latest)
    correct = 0
    brier_sum = 0.0
    logloss_sum = 0.0
    predicted_counts: Counter[str] = Counter()
    actual_counts: Counter[str] = Counter()
    favorite_groups: dict[str, list[int]] = defaultdict(list)
    calibration: dict[str, dict[str, list[int]]] = {lab: defaultdict(list) for lab in labels}
    by_stage: dict[str, list[int]] = defaultdict(list)

    for rec in latest.values():
        p = rec["p"]
        actual = rec["actual"]
        actual_idx = labels.index(actual)
        pred_idx = max(range(3), key=lambda i: p[i])
        pred = labels[pred_idx]
        hit = int(pred == actual)
        correct += hit
        predicted_counts[pred] += 1
        actual_counts[actual] += 1
        y = [1.0 if i == actual_idx else 0.0 for i in range(3)]
        brier_sum += sum((p[i] - y[i]) ** 2 for i in range(3))
        logloss_sum += -math.log(max(1e-12, p[actual_idx]))
        maxp = p[pred_idx]
        if maxp >= 0.60:
            favorite_groups[">=60%"].append(hit)
        elif maxp >= 0.50:
            favorite_groups["50-59%"].append(hit)
        elif maxp >= 0.40:
            favorite_groups["40-49%"].append(hit)
        else:
            favorite_groups["<40%"].append(hit)
        by_stage[str(rec.get("stage") or "UNKNOWN")].append(hit)
        for i, lab in enumerate(labels):
            calibration[lab][bucket(p[i])].append(1 if actual == lab else 0)

    cal_out: dict[str, dict[str, Any]] = {}
    for lab, bins in calibration.items():
        cal_out[lab] = {}
        for b, vals in sorted(bins.items()):
            lo = int(b[:2]) / 100.0
            hi = int(b[3:5]) / 100.0
            cal_out[lab][b] = {
                "n": len(vals),
                "mean_forecast_midpoint": round((lo + hi) / 2, 3),
                "observed_rate": round(sum(vals) / len(vals), 4) if vals else None,
            }

    fav_out = {k: {"n": len(v), "accuracy": round(sum(v)/len(v), 4) if v else None} for k, v in sorted(favorite_groups.items())}
    stage_out = {k: {"n": len(v), "accuracy": round(sum(v)/len(v), 4) if v else None} for k, v in sorted(by_stage.items())}
    result = {
        "schema_version": "1.0.0",
        "timezone_basis": "America/Mexico_City",
        "status": "RESEARCH_ONLY_NOT_ACTIONABLE",
        "sample_fixtures": n,
        "top1_accuracy": round(correct / n, 4) if n else None,
        "multiclass_brier": round(brier_sum / n, 4) if n else None,
        "multiclass_log_loss": round(logloss_sum / n, 4) if n else None,
        "predicted_outcome_counts": dict(predicted_counts),
        "actual_outcome_counts": dict(actual_counts),
        "accuracy_by_max_probability": fav_out,
        "accuracy_by_latest_stage": stage_out,
        "calibration_by_outcome": cal_out,
        "activation_gate": {
            "enabled": False,
            "reason": "1X2 remains research-only until sample size and calibration quality are sufficient; no backend classification may be upgraded from this report.",
        },
        "notes": [
            "Uses one latest available raw 1X2 projection per finalized fixture to avoid overweighting repeated refresh stages.",
            "Probabilities are normalized to sum to 1 before scoring.",
            "Calibration bins are descriptive and require larger samples before model changes are justified."
        ],
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
