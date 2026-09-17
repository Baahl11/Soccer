from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from typing import Any

from mcp_gateway import analyze_cards_baseline as cards


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def shrunk(events: float, n: int, prior: float, pseudo: float) -> float:
    return (events + pseudo * prior) / (n + pseudo)


def model(target: dict[str, Any], prior: list[dict[str, Any]]) -> dict[str, Any] | None:
    if len(prior) < 100:
        return None
    gh = sum(1 for r in prior if float(r.get("home_red") or 0) > 0) / len(prior)
    ga = sum(1 for r in prior if float(r.get("away_red") or 0) > 0) / len(prior)
    gany = sum(1 for r in prior if float(r.get("home_red") or 0) + float(r.get("away_red") or 0) > 0) / len(prior)

    league = [r for r in prior if r.get("league_id") == target.get("league_id")]
    lh = shrunk(sum(1 for r in league if float(r.get("home_red") or 0) > 0), len(league), gh, 80.0)
    la = shrunk(sum(1 for r in league if float(r.get("away_red") or 0) > 0), len(league), ga, 80.0)
    lany = shrunk(sum(1 for r in league if float(r.get("home_red") or 0) + float(r.get("away_red") or 0) > 0), len(league), gany, 80.0)

    home_rows = [r for r in prior if r.get("home_team_id") == target.get("home_team_id")]
    away_rows = [r for r in prior if r.get("away_team_id") == target.get("away_team_id")]
    h_own = shrunk(sum(1 for r in home_rows if float(r.get("home_red") or 0) > 0), len(home_rows), lh, 24.0)
    h_draw_away = shrunk(sum(1 for r in home_rows if float(r.get("away_red") or 0) > 0), len(home_rows), la, 24.0)
    a_own = shrunk(sum(1 for r in away_rows if float(r.get("away_red") or 0) > 0), len(away_rows), la, 24.0)
    a_draw_home = shrunk(sum(1 for r in away_rows if float(r.get("home_red") or 0) > 0), len(away_rows), lh, 24.0)

    p_home = clamp(math.sqrt(max(1e-9, h_own * a_draw_home)), 0.002, 0.30)
    p_away = clamp(math.sqrt(max(1e-9, a_own * h_draw_away)), 0.002, 0.30)
    base_any = 1.0 - (1.0 - p_home) * (1.0 - p_away)

    referee = str(target.get("referee") or "").strip()
    ref_rows = [r for r in prior if referee and str(r.get("referee") or "").strip() == referee]
    ref_scale = 1.0
    if len(ref_rows) >= 20:
        ref_events = sum(1 for r in ref_rows if float(r.get("home_red") or 0) + float(r.get("away_red") or 0) > 0)
        ref_rate = shrunk(ref_events, len(ref_rows), lany, 30.0)
        ref_scale = clamp(ref_rate / max(lany, 1e-9), 0.60, 1.80)
    p_any = clamp(base_any * ref_scale, 0.003, 0.60)
    return {"p_any_red": p_any, "referee_prior_n": len(ref_rows), "league_n": len(league)}


def log_loss(p: float, y: int) -> float:
    q = min(max(p, 1e-9), 1.0 - 1e-9)
    return -(y * math.log(q) + (1-y) * math.log(1-q))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "brier": None, "log_loss": None, "observed_rate": None, "mean_probability": None}
    n = len(rows)
    return {
        "n": n,
        "brier": round(sum(r["brier"] for r in rows) / n, 6),
        "log_loss": round(sum(r["log_loss"] for r in rows) / n, 6),
        "observed_rate": round(sum(r["actual_any_red"] for r in rows) / n, 6),
        "mean_probability": round(sum(r["p_any_red"] for r in rows) / n, 6),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Walk-forward calibration for any-red-card match model; research only.")
    ap.add_argument("--history-dir", default="soccer_edge_state/history")
    ap.add_argument("--output", default="soccer_edge_state/analysis/red_cards_validation.json")
    args = ap.parse_args()

    source = cards.load_rows(args.history_dir)
    evals: list[dict[str, Any]] = []
    for idx, target in enumerate(source):
        pred = model(target, source[:idx])
        if pred is None:
            continue
        y = int(float(target.get("home_red") or 0) + float(target.get("away_red") or 0) > 0)
        p = float(pred["p_any_red"])
        evals.append({
            "fixture_id": target.get("fixture_id"),
            "league_id": target.get("league_id"),
            "league": target.get("league"),
            "p_any_red": round(p, 6),
            "actual_any_red": y,
            "brier": (p-y) ** 2,
            "log_loss": log_loss(p, y),
            "referee_prior_n": pred["referee_prior_n"],
            "referee_adjusted": pred["referee_prior_n"] >= 20,
            "bucket": f"{int(p*20)*5:02d}-{min(100, int(p*20)*5+4):02d}%",
        })

    by_league: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evals:
        by_league[str(row.get("league") or "UNKNOWN")].append(row)
        by_bucket[str(row.get("bucket") or "UNKNOWN")].append(row)
    referee_rows = [r for r in evals if r["referee_adjusted"]]
    n = len(evals)
    report = {
        "schema_version": "1.0.0",
        "status": "RESEARCH_ONLY_RED_CARD_VALIDATION",
        "model": "EMPIRICAL_BAYES_ANY_RED_CARD_MATCH_v0.1",
        "walk_forward_evaluated": n,
        "overall": summarize(evals),
        "referee_adjusted": summarize(referee_rows),
        "by_league": {k: summarize(v) for k, v in sorted(by_league.items())},
        "by_probability_bucket": {k: summarize(v) for k, v in sorted(by_bucket.items())},
        "promotion_gate": {
            "enabled": False,
            "minimum_oos_for_market_review": 500,
            "minimum_oos_for_actionable_review": 1000,
            "minimum_referee_adjusted_oos": 200,
            "market_review_sample_gate_met": n >= 500,
            "actionable_review_sample_gate_met": n >= 1000,
            "referee_sample_gate_met": len(referee_rows) >= 200,
            "requires": [
                "stable rare-event Brier/log-loss overall and by league/bucket",
                "verified match-red-card YES/NO price history and true CLV",
                "stable referee adjustment with verified assignments",
                "explicit sportsbook settlement compatibility",
            ],
        },
        "policy": "RED CARDS ONLY; WALK-FORWARD ANTI-LEAKAGE; YELLOW CARD COUNTS EXCLUDED; NO AUTOMATIC PROMOTION",
        "rows": evals[-1000:],
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps({k: report[k] for k in ("status", "walk_forward_evaluated", "overall", "promotion_gate")}, indent=2))


if __name__ == "__main__":
    main()
