from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from datetime import datetime
from typing import Any

LABELS = ["H", "D", "A"]
MIN_TRAIN = 30
PRIOR_STRENGTH = 30.0


def fnum(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def parse_dt(v: Any) -> datetime | None:
    if not v:
        return None
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except ValueError:
        return None


def final_outcome(result: dict[str, Any] | None) -> str | None:
    if not isinstance(result, dict):
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


def recalibrate(train: list[dict[str, Any]], p: list[float]) -> tuple[list[float], list[float]]:
    n = len(train)
    mean_pred = [sum(r["p"][i] for r in train) / n for i in range(3)]
    obs = [sum(1 for r in train if r["actual"] == lab) / n for lab in LABELS]
    # Shrink class-frequency correction toward no correction (factor=1).
    raw_factor = [obs[i] / max(1e-9, mean_pred[i]) for i in range(3)]
    w = n / (n + PRIOR_STRENGTH)
    factor = [1.0 + w * (raw_factor[i] - 1.0) for i in range(3)]
    q = [max(1e-9, p[i] * factor[i]) for i in range(3)]
    s = sum(q)
    return [x / s for x in q], factor


def metrics(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {"n": 0}
    correct = 0
    brier = 0.0
    ll = 0.0
    pred_counts: Counter[str] = Counter()
    mean_draw = 0.0
    draw_actual = 0
    for r in rows:
        p = r[key]
        ai = LABELS.index(r["actual"])
        pi = max(range(3), key=lambda i: p[i])
        pred_counts[LABELS[pi]] += 1
        correct += int(pi == ai)
        y = [1.0 if i == ai else 0.0 for i in range(3)]
        brier += sum((p[i] - y[i]) ** 2 for i in range(3))
        ll += -math.log(max(1e-12, p[ai]))
        mean_draw += p[1]
        draw_actual += int(ai == 1)
    return {
        "n": n,
        "accuracy": round(correct / n, 4),
        "brier": round(brier / n, 4),
        "log_loss": round(ll / n, 4),
        "predicted_counts": dict(pred_counts),
        "mean_draw_probability": round(mean_draw / n, 4),
        "observed_draw_rate": round(draw_actual / n, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Walk-forward class-prior calibration challenger for Soccer Edge 1X2.")
    ap.add_argument("--ledger", default="soccer_edge_state/analysis/signal_ledger.jsonl")
    ap.add_argument("--output", default="soccer_edge_state/analysis/one_x_two_prior_calibration.json")
    args = ap.parse_args()

    rows: list[dict[str, Any]] = []
    finals: dict[int, str] = {}
    with open(args.ledger, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append(r)
            fid = r.get("fixture_id")
            out = final_outcome(r.get("result"))
            if fid and out:
                finals[int(fid)] = out

    latest: dict[int, dict[str, Any]] = {}
    for r in rows:
        fid = r.get("fixture_id")
        if not fid or int(fid) not in finals:
            continue
        raw = r.get("raw_projection") or {}
        p = [fnum(raw.get(k)) for k in ("raw_home_win_prob", "raw_draw_prob", "raw_away_win_prob")]
        ts = parse_dt(r.get("generated_at_local"))
        ko = parse_dt(r.get("kickoff_local"))
        if any(x is None for x in p) or ts is None or ko is None or ts >= ko:
            continue
        s = sum(p)
        if s <= 0:
            continue
        rec = {"fixture_id": int(fid), "timestamp": r.get("generated_at_local"), "p": [x/s for x in p], "actual": finals[int(fid)]}
        old = latest.get(int(fid))
        if old is None or str(rec["timestamp"]) > str(old["timestamp"]):
            latest[int(fid)] = rec

    ordered = sorted(latest.values(), key=lambda x: str(x["timestamp"]))
    evaluated: list[dict[str, Any]] = []
    factors: list[list[float]] = []
    for i in range(MIN_TRAIN, len(ordered)):
        q, factor = recalibrate(ordered[:i], ordered[i]["p"])
        rec = dict(ordered[i])
        rec["calibrated"] = q
        evaluated.append(rec)
        factors.append(factor)

    base = metrics(evaluated, "p")
    cal = metrics(evaluated, "calibrated")
    result = {
        "schema_version": "1.0.0",
        "timezone_basis": "America/Mexico_City",
        "status": "RESEARCH_ONLY_NOT_ACTIONABLE",
        "method": "WALK_FORWARD_SHRUNK_CLASS_PRIOR_CALIBRATION",
        "minimum_training_fixtures": MIN_TRAIN,
        "prior_strength": PRIOR_STRENGTH,
        "eligible_fixtures": len(ordered),
        "walk_forward_evaluated": len(evaluated),
        "baseline": base,
        "challenger": cal,
        "improvement": {
            "accuracy_delta_pp": round((cal.get("accuracy",0)-base.get("accuracy",0))*100,2) if evaluated else None,
            "brier_delta": round(cal.get("brier",0)-base.get("brier",0),4) if evaluated else None,
            "log_loss_delta": round(cal.get("log_loss",0)-base.get("log_loss",0),4) if evaluated else None,
            "draw_probability_delta_pp": round((cal.get("mean_draw_probability",0)-base.get("mean_draw_probability",0))*100,2) if evaluated else None,
        },
        "latest_class_factors": [round(x,4) for x in factors[-1]] if factors else None,
        "diagnostic_interpretation": "If calibration improves materially but ranking/accuracy does not, output calibration is part of the issue. If it does not, upstream lambda/team-strength construction is the primary target.",
        "promotion_gate": {
            "enabled": False,
            "minimum_evaluation_n": 100,
            "sample_gate_met": len(evaluated) >= 100,
            "decision": "KEEP_RESEARCH_ONLY",
        },
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
