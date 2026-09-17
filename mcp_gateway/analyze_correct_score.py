from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict
from typing import Any

GRID_MAX = 5


def fnum(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def pmf(goals: int, lam: float) -> float:
    return math.exp(-lam) * (lam ** goals) / math.factorial(goals)


def score_probability(home: int, away: int, home_lambda: float, away_lambda: float) -> float:
    return max(0.0, min(1.0, pmf(home, home_lambda) * pmf(away, away_lambda)))


def result_goals(event: dict[str, Any]) -> tuple[int, int] | None:
    result = event.get("result") if isinstance(event.get("result"), dict) else {}
    goals = result.get("goals") if isinstance(result.get("goals"), dict) else {}
    score = result.get("score") if isinstance(result.get("score"), dict) else {}
    fulltime = score.get("fulltime") if isinstance(score.get("fulltime"), dict) else {}
    fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
    fixture_goals = fixture.get("goals") if isinstance(fixture.get("goals"), dict) else {}
    h = goals.get("home", fulltime.get("home", fixture_goals.get("home")))
    a = goals.get("away", fulltime.get("away", fixture_goals.get("away")))
    try:
        return int(h), int(a)
    except (TypeError, ValueError):
        return None


def load(history_dir: str) -> tuple[dict[int, dict[str, Any]], dict[int, tuple[int, int]]]:
    predictions: dict[int, dict[str, Any]] = {}
    results: dict[int, tuple[int, int]] = {}
    for path in sorted(glob.glob(os.path.join(history_dir, "*.jsonl"))):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    tick = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    continue
                stamp = str(tick.get("generated_at_utc") or tick.get("generated_at_local") or "")
                for event in tick.get("events") or []:
                    if not isinstance(event, dict):
                        continue
                    fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
                    fid = fixture.get("fixture_id")
                    if fid is None:
                        continue
                    fid = int(fid)
                    if event.get("stage") == "POSTGAME":
                        final = result_goals(event)
                        if final is not None:
                            results[fid] = final
                        continue
                    raw = event.get("raw_projection") if isinstance(event.get("raw_projection"), dict) else {}
                    hl = fnum(raw.get("raw_home_goal_rate"))
                    al = fnum(raw.get("raw_away_goal_rate"))
                    if hl is None or al is None or hl <= 0 or al <= 0:
                        continue
                    old = predictions.get(fid)
                    if old is None or stamp > old["timestamp"]:
                        predictions[fid] = {
                            "fixture_id": fid,
                            "timestamp": stamp,
                            "stage": event.get("stage"),
                            "league": fixture.get("league"),
                            "home_team": fixture.get("home_team"),
                            "away_team": fixture.get("away_team"),
                            "home_lambda": hl,
                            "away_lambda": al,
                        }
    return predictions, results


def fixture_metrics(pred: dict[str, Any], actual: tuple[int, int]) -> dict[str, Any]:
    hl = float(pred["home_lambda"])
    al = float(pred["away_lambda"])
    ah, aa = actual
    p_actual = max(1e-12, score_probability(ah, aa, hl, al))

    ranked: list[tuple[float, int, int]] = []
    categories: list[tuple[str, float]] = []
    grid_mass = 0.0
    for h in range(GRID_MAX + 1):
        for a in range(GRID_MAX + 1):
            p = score_probability(h, a, hl, al)
            ranked.append((p, h, a))
            categories.append((f"{h}-{a}", p))
            grid_mass += p
    categories.append(("OTHER", max(0.0, 1.0 - grid_mass)))
    ranked.sort(reverse=True)

    actual_key = f"{ah}-{aa}" if ah <= GRID_MAX and aa <= GRID_MAX else "OTHER"
    multiclass_brier = 0.0
    for key, p in categories:
        y = 1.0 if key == actual_key else 0.0
        multiclass_brier += (p - y) ** 2

    rank_lookup = {(h, a): idx + 1 for idx, (_, h, a) in enumerate(ranked)}
    rank = rank_lookup.get((ah, aa))
    if rank is None:
        rank = len(ranked) + 1

    return {
        "fixture_id": pred["fixture_id"],
        "stage": pred.get("stage"),
        "league": pred.get("league"),
        "home_team": pred.get("home_team"),
        "away_team": pred.get("away_team"),
        "home_lambda": round(hl, 6),
        "away_lambda": round(al, 6),
        "actual_score": f"{ah}-{aa}",
        "actual_score_probability": round(p_actual, 8),
        "negative_log_likelihood": -math.log(p_actual),
        "multiclass_brier_grid_0_5_plus_other": multiclass_brier,
        "actual_score_rank_within_0_5_grid": rank,
        "top1_hit": int(rank == 1),
        "top3_hit": int(rank <= 3),
        "top5_hit": int(rank <= 5),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "mean_negative_log_likelihood": None,
            "mean_multiclass_brier": None,
            "top1_hit_rate": None,
            "top3_hit_rate": None,
            "top5_hit_rate": None,
        }
    n = len(rows)
    return {
        "n": n,
        "mean_negative_log_likelihood": round(sum(r["negative_log_likelihood"] for r in rows) / n, 6),
        "mean_multiclass_brier": round(sum(r["multiclass_brier_grid_0_5_plus_other"] for r in rows) / n, 6),
        "top1_hit_rate": round(sum(r["top1_hit"] for r in rows) / n, 6),
        "top3_hit_rate": round(sum(r["top3_hit"] for r in rows) / n, 6),
        "top5_hit_rate": round(sum(r["top5_hit"] for r in rows) / n, 6),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate research-only Correct Score probabilities from canonical home/away lambdas.")
    ap.add_argument("--history-dir", default="soccer_edge_state/history")
    ap.add_argument("--output", default="soccer_edge_state/analysis/correct_score_validation.json")
    args = ap.parse_args()

    predictions, results = load(args.history_dir)
    rows = [fixture_metrics(pred, results[fid]) for fid, pred in predictions.items() if fid in results]
    by_league: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_league[str(row.get("league") or "UNKNOWN")].append(row)

    fixture_n = len(rows)
    report = {
        "schema_version": "1.0.0",
        "status": "RESEARCH_ONLY_CORRECT_SCORE_VALIDATION",
        "model": "CANONICAL_HOME_AWAY_LAMBDA_INDEPENDENT_POISSON_EXACT_SCORE_v0.1",
        "evaluated_fixtures": fixture_n,
        "overall": summarize(rows),
        "by_league": {k: summarize(v) for k, v in sorted(by_league.items())},
        "promotion_gate": {
            "enabled": False,
            "minimum_oos_fixtures_for_market_comparison": 300,
            "minimum_oos_fixtures_for_actionable_review": 500,
            "market_comparison_sample_gate_met": fixture_n >= 300,
            "actionable_review_sample_gate_met": fixture_n >= 500,
            "requires": [
                "stable multiclass Brier and negative log likelihood",
                "stable top-1/top-3/top-5 hit rates across competitions",
                "verified historical correct-score prices and CLV evidence",
                "validated shrinkage for sparse exact-score outcomes",
                "no material degradation versus canonical FT-goals calibration",
            ],
        },
        "notes": [
            "Latest persisted pregame canonical home/away lambdas are evaluated against final exact score.",
            "Exact-score probability is sport-first independent Poisson and does not use bookmaker price.",
            "Multiclass Brier uses a 0-5 by 0-5 score grid plus OTHER tail bucket.",
            "This report cannot promote Correct Score to BET/LEAN/Galaxy by itself.",
        ],
        "rows": rows[-600:],
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps({k: report[k] for k in ("status", "evaluated_fixtures", "overall", "promotion_gate")}, indent=2))


if __name__ == "__main__":
    main()
