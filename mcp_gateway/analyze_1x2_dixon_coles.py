from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from datetime import datetime
from typing import Any

RHO_GRID = [-0.15, -0.12, -0.10, -0.08, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04]
MIN_TRAIN = 30
LABELS = ["H", "D", "A"]


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


def poisson(k: int, lam: float) -> float:
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def tau(h: int, a: int, lh: float, la: float, rho: float) -> float:
    if h == 0 and a == 0:
        return 1.0 - lh * la * rho
    if h == 0 and a == 1:
        return 1.0 + lh * rho
    if h == 1 and a == 0:
        return 1.0 + la * rho
    if h == 1 and a == 1:
        return 1.0 - rho
    return 1.0


def dc_probs(lh: float, la: float, rho: float, max_goals: int = 10) -> list[float] | None:
    hwin = draw = awin = 0.0
    total = 0.0
    for h in range(max_goals + 1):
        ph = poisson(h, lh)
        for a in range(max_goals + 1):
            p = ph * poisson(a, la) * tau(h, a, lh, la, rho)
            if p <= 0:
                return None
            total += p
            if h > a:
                hwin += p
            elif h == a:
                draw += p
            else:
                awin += p
    if total <= 0:
        return None
    return [hwin / total, draw / total, awin / total]


def auc_discrimination(observations: list[tuple[float, int]]) -> dict[str, Any]:
    positives = [row for row in observations if row[1] == 1]
    negatives = [row for row in observations if row[1] == 0]
    n_pos = len(positives)
    n_neg = len(negatives)
    if n_pos == 0 or n_neg == 0:
        return {
            "auc": None,
            "auc_standard_error": None,
            "auc_lower_95": None,
            "positive_count": n_pos,
            "negative_count": n_neg,
            "discrimination_ready": False,
        }

    ranked = sorted(observations, key=lambda item: item[0])
    rank = 1
    sum_positive_ranks = 0.0
    index = 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            end += 1
        count = end - index
        average_rank = (rank + (rank + count - 1)) / 2.0
        for _, outcome in ranked[index:end]:
            if outcome == 1:
                sum_positive_ranks += average_rank
        rank += count
        index = end

    auc = (
        sum_positive_ranks - (n_pos * (n_pos + 1) / 2.0)
    ) / (n_pos * n_neg)
    q1 = auc / (2.0 - auc) if auc < 2.0 else 0.0
    q2 = (2.0 * auc * auc) / (1.0 + auc) if auc > -1.0 else 0.0
    variance = (
        auc * (1.0 - auc)
        + (n_pos - 1) * (q1 - auc * auc)
        + (n_neg - 1) * (q2 - auc * auc)
    ) / (n_pos * n_neg)
    standard_error = math.sqrt(max(variance, 0.0))
    lower_95 = max(0.0, auc - 1.96 * standard_error)
    return {
        "auc": round(auc, 8),
        "auc_standard_error": round(standard_error, 8),
        "auc_lower_95": round(lower_95, 8),
        "positive_count": n_pos,
        "negative_count": n_neg,
        "discrimination_ready": lower_95 > 0.50,
    }


def class_discrimination(rows: list[dict[str, Any]], prob_key: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for index, label in enumerate(LABELS):
        observations = [
            (float(row[prob_key][index]), 1 if row["actual"] == label else 0)
            for row in rows
        ]
        out[label] = auc_discrimination(observations)
    return out


def metrics(rows: list[dict[str, Any]], prob_key: str) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {"n": 0, "accuracy": None, "brier": None, "log_loss": None, "predicted_counts": {}}
    correct = 0
    brier = 0.0
    ll = 0.0
    counts: Counter[str] = Counter()
    draw_forecast = 0.0
    draw_actual = 0
    for r in rows:
        p = r[prob_key]
        actual = r["actual"]
        ai = LABELS.index(actual)
        pi = max(range(3), key=lambda i: p[i])
        pred = LABELS[pi]
        counts[pred] += 1
        correct += int(pred == actual)
        y = [1.0 if i == ai else 0.0 for i in range(3)]
        brier += sum((p[i] - y[i]) ** 2 for i in range(3))
        ll += -math.log(max(1e-12, p[ai]))
        draw_forecast += p[1]
        draw_actual += int(actual == "D")
    return {
        "n": n,
        "accuracy": round(correct / n, 4),
        "brier": round(brier / n, 4),
        "log_loss": round(ll / n, 4),
        "predicted_counts": dict(counts),
        "mean_draw_probability": round(draw_forecast / n, 4),
        "observed_draw_rate": round(draw_actual / n, 4),
        "class_discrimination": class_discrimination(rows, prob_key),
    }


def fit_rho(train: list[dict[str, Any]]) -> float:
    best_rho = 0.0
    best_ll = float("inf")
    for rho in RHO_GRID:
        ll = 0.0
        valid = True
        for r in train:
            p = dc_probs(r["lh"], r["la"], rho)
            if p is None:
                valid = False
                break
            ll += -math.log(max(1e-12, p[LABELS.index(r["actual"])]))
        if valid and ll < best_ll:
            best_ll = ll
            best_rho = rho
    return best_rho


def walk_forward_evaluate(ordered: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[float]]:
    if len(ordered) <= MIN_TRAIN:
        return [], []

    probability_grid: dict[float, list[list[float] | None]] = {
        rho: [dc_probs(row["lh"], row["la"], rho) for row in ordered]
        for rho in RHO_GRID
    }
    cumulative_log_loss = {rho: 0.0 for rho in RHO_GRID}
    rho_valid = {rho: True for rho in RHO_GRID}

    def add_training_row(index: int) -> None:
        actual_index = LABELS.index(ordered[index]["actual"])
        for rho in RHO_GRID:
            if not rho_valid[rho]:
                continue
            probs = probability_grid[rho][index]
            if probs is None:
                rho_valid[rho] = False
                cumulative_log_loss[rho] = float("inf")
                continue
            cumulative_log_loss[rho] += -math.log(max(1e-12, probs[actual_index]))

    for index in range(MIN_TRAIN):
        add_training_row(index)

    evaluated: list[dict[str, Any]] = []
    rho_history: list[float] = []
    for index in range(MIN_TRAIN, len(ordered)):
        best_rho = RHO_GRID[0]
        best_loss = float("inf")
        for rho in RHO_GRID:
            loss = cumulative_log_loss[rho] if rho_valid[rho] else float("inf")
            if loss < best_loss:
                best_loss = loss
                best_rho = rho

        prediction = probability_grid[best_rho][index]
        if prediction is not None and math.isfinite(best_loss):
            rec = dict(ordered[index])
            rec["challenger"] = prediction
            rec["rho"] = best_rho
            evaluated.append(rec)
            rho_history.append(best_rho)

        add_training_row(index)

    return evaluated, rho_history


def main() -> None:
    ap = argparse.ArgumentParser(description="Walk-forward Dixon-Coles challenger for Soccer Edge 1X2 research.")
    ap.add_argument("--ledger", default="soccer_edge_state/analysis/signal_ledger.jsonl")
    ap.add_argument("--output", default="soccer_edge_state/analysis/one_x_two_dixon_coles.json")
    args = ap.parse_args()

    rows: list[dict[str, Any]] = []
    finals: dict[int, str] = {}
    with open(args.ledger, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
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
        lh = fnum(raw.get("raw_home_goal_rate"))
        la = fnum(raw.get("raw_away_goal_rate"))
        bp = [fnum(raw.get(k)) for k in ("raw_home_win_prob", "raw_draw_prob", "raw_away_win_prob")]
        ts = parse_dt(r.get("generated_at_local"))
        ko = parse_dt(r.get("kickoff_local"))
        if lh is None or la is None or any(x is None for x in bp) or ts is None or ko is None or ts >= ko:
            continue
        s = sum(bp)
        if s <= 0:
            continue
        rec = {
            "fixture_id": int(fid),
            "timestamp": r.get("generated_at_local"),
            "lh": lh,
            "la": la,
            "baseline": [x / s for x in bp],
            "actual": finals[int(fid)],
        }
        old = latest.get(int(fid))
        if old is None or str(rec["timestamp"]) > str(old["timestamp"]):
            latest[int(fid)] = rec

    ordered = sorted(latest.values(), key=lambda x: str(x["timestamp"]))
    evaluated, rho_history = walk_forward_evaluate(ordered)

    baseline = metrics(evaluated, "baseline")
    challenger = metrics(evaluated, "challenger")
    class_improvement: dict[str, Any] = {}
    for label in LABELS:
        baseline_class = (baseline.get("class_discrimination") or {}).get(label) or {}
        challenger_class = (challenger.get("class_discrimination") or {}).get(label) or {}
        base_auc = fnum(baseline_class.get("auc"))
        chal_auc = fnum(challenger_class.get("auc"))
        base_l95 = fnum(baseline_class.get("auc_lower_95"))
        chal_l95 = fnum(challenger_class.get("auc_lower_95"))
        class_improvement[label] = {
            "baseline_auc": base_auc,
            "challenger_auc": chal_auc,
            "auc_delta": round(chal_auc - base_auc, 8) if base_auc is not None and chal_auc is not None else None,
            "baseline_auc_lower_95": base_l95,
            "challenger_auc_lower_95": chal_l95,
            "auc_lower_95_delta": round(chal_l95 - base_l95, 8) if base_l95 is not None and chal_l95 is not None else None,
            "challenger_discrimination_ready": challenger_class.get("discrimination_ready") is True,
            "positive_count": int(challenger_class.get("positive_count") or 0),
            "negative_count": int(challenger_class.get("negative_count") or 0),
        }

    improvement = {
        "accuracy_delta_pp": round((challenger["accuracy"] - baseline["accuracy"]) * 100, 2) if evaluated else None,
        "brier_delta": round(challenger["brier"] - baseline["brier"], 4) if evaluated else None,
        "log_loss_delta": round(challenger["log_loss"] - baseline["log_loss"], 4) if evaluated else None,
        "draw_probability_delta_pp": round((challenger["mean_draw_probability"] - baseline["mean_draw_probability"]) * 100, 2) if evaluated else None,
        "class_discrimination": class_improvement,
    }
    wins_quality = bool(evaluated and challenger["brier"] < baseline["brier"] and challenger["log_loss"] < baseline["log_loss"])
    draw_discrimination = class_improvement.get("D") or {}
    draw_ready = draw_discrimination.get("challenger_discrimination_ready") is True
    result = {
        "schema_version": "1.0.0",
        "timezone_basis": "America/Mexico_City",
        "status": "RESEARCH_ONLY_NOT_ACTIONABLE",
        "method": "DIXON_COLES_LOW_SCORE_CORRECTION_WALK_FORWARD",
        "anti_leakage": "Each challenger prediction fits rho only on fixtures earlier than that prediction.",
        "minimum_training_fixtures": MIN_TRAIN,
        "candidate_rho_grid": RHO_GRID,
        "eligible_fixtures": len(ordered),
        "walk_forward_evaluated": len(evaluated),
        "baseline": baseline,
        "challenger": challenger,
        "improvement": improvement,
        "rho_history_summary": {
            "n": len(rho_history),
            "latest": rho_history[-1] if rho_history else None,
            "min": min(rho_history) if rho_history else None,
            "max": max(rho_history) if rho_history else None,
            "mean": round(sum(rho_history) / len(rho_history), 4) if rho_history else None,
        },
        "promotion_gate": {
            "enabled": False,
            "quality_metrics_better": wins_quality,
            "minimum_evaluation_n": 100,
            "sample_gate_met": len(evaluated) >= 100,
            "same_fixture_class_discrimination_available": bool(evaluated),
            "draw_discrimination_ready": draw_ready,
            "draw_auc_lower_95": draw_discrimination.get("challenger_auc_lower_95"),
            "draw_auc_lower_95_delta_vs_baseline": draw_discrimination.get("auc_lower_95_delta"),
            "decision": "KEEP_RESEARCH_ONLY",
            "reason": "Never promotes 1X2 automatically. Requires same-fixture Brier/log-loss improvement plus Draw AUC lower-95 above 0.50 and downstream CLV/settlement evidence.",
        },
        "notes": [
            "Baseline probabilities are the exact stored pre-kickoff Soccer Edge 1X2 probabilities.",
            "Dixon-Coles changes only low-score dependence using the stored home/away goal rates; it does not fabricate xG or team-strength inputs.",
            "Negative rho typically increases 0-0/1-1 mass and can improve draw calibration, but rho is selected walk-forward rather than hard-coded.",
            "H/D/A discrimination is measured on the exact same walk-forward cohort with the same AUC lower-95 > 0.50 research gate used by Phase16.",
            "This report cannot upgrade any backend BET/LEAN classification."
        ],
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
