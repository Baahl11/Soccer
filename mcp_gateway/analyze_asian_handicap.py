from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict
from typing import Any

MAX_GOALS = 12
VALIDATION_LINES = (-1.5, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5)


def fnum(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def poisson(k: int, lam: float) -> float:
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def split_line(line: float) -> list[float]:
    if abs(line * 2 - round(line * 2)) < 1e-8:
        return [line]
    return [math.floor(line * 2) / 2.0, math.ceil(line * 2) / 2.0]


def fractions_for_margin(margin: int, line: float) -> tuple[float, float, float]:
    components = split_line(line)
    w = p = l = 0.0
    for component in components:
        adjusted = margin + component
        if adjusted > 1e-9:
            w += 1.0 / len(components)
        elif adjusted < -1e-9:
            l += 1.0 / len(components)
        else:
            p += 1.0 / len(components)
    return w, p, l


def model_settlement(lh: float, la: float, line: float) -> tuple[float, float, float]:
    w = p = l = mass = 0.0
    for h in range(MAX_GOALS + 1):
        ph = poisson(h, lh)
        for a in range(MAX_GOALS + 1):
            prob = ph * poisson(a, la)
            mass += prob
            fw, fp, fl = fractions_for_margin(h - a, line)
            w += prob * fw
            p += prob * fp
            l += prob * fl
    return w / mass, p / mass, l / mass


def final_goals(event: dict[str, Any]) -> tuple[int, int] | None:
    result = event.get("result") if isinstance(event.get("result"), dict) else {}
    goals = result.get("goals") if isinstance(result.get("goals"), dict) else {}
    score = result.get("score") if isinstance(result.get("score"), dict) else {}
    ft = score.get("fulltime") if isinstance(score.get("fulltime"), dict) else {}
    fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
    fg = fixture.get("goals") if isinstance(fixture.get("goals"), dict) else {}
    h = goals.get("home", ft.get("home", fg.get("home")))
    a = goals.get("away", ft.get("away", fg.get("away")))
    try:
        return int(h), int(a)
    except (TypeError, ValueError):
        return None


def load(history_dir: str) -> tuple[dict[int, dict[str, Any]], dict[int, tuple[int, int]]]:
    predictions: dict[int, dict[str, Any]] = {}
    finals: dict[int, tuple[int, int]] = {}
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
                        score = final_goals(event)
                        if score is not None:
                            finals[fid] = score
                        continue
                    raw = event.get("raw_projection") if isinstance(event.get("raw_projection"), dict) else {}
                    lh = fnum(raw.get("raw_home_goal_rate"))
                    la = fnum(raw.get("raw_away_goal_rate"))
                    if lh is None or la is None or lh <= 0 or la <= 0:
                        continue
                    old = predictions.get(fid)
                    if old is None or stamp > old["timestamp"]:
                        predictions[fid] = {
                            "fixture_id": fid,
                            "timestamp": stamp,
                            "stage": event.get("stage"),
                            "league": fixture.get("league"),
                            "home_lambda": lh,
                            "away_lambda": la,
                        }
    return predictions, finals


def evaluate_fixture(pred: dict[str, Any], score: tuple[int, int]) -> list[dict[str, Any]]:
    margin = score[0] - score[1]
    rows: list[dict[str, Any]] = []
    for line in VALIDATION_LINES:
        mw, mp, ml = model_settlement(float(pred["home_lambda"]), float(pred["away_lambda"]), line)
        aw, ap, al = fractions_for_margin(margin, line)
        squared = (mw - aw) ** 2 + (mp - ap) ** 2 + (ml - al) ** 2
        rows.append({
            "fixture_id": pred["fixture_id"],
            "league": pred.get("league"),
            "stage": pred.get("stage"),
            "home_handicap": line,
            "model_win_fraction": round(mw, 6),
            "model_push_fraction": round(mp, 6),
            "model_loss_fraction": round(ml, 6),
            "actual_win_fraction": aw,
            "actual_push_fraction": ap,
            "actual_loss_fraction": al,
            "settlement_brier_3state": squared,
        })
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "mean_settlement_brier_3state": None}
    return {
        "n": len(rows),
        "mean_settlement_brier_3state": round(sum(r["settlement_brier_3state"] for r in rows) / len(rows), 6),
        "mean_model_win_fraction": round(sum(r["model_win_fraction"] for r in rows) / len(rows), 6),
        "mean_actual_win_fraction": round(sum(r["actual_win_fraction"] for r in rows) / len(rows), 6),
        "mean_model_push_fraction": round(sum(r["model_push_fraction"] for r in rows) / len(rows), 6),
        "mean_actual_push_fraction": round(sum(r["actual_push_fraction"] for r in rows) / len(rows), 6),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate research-only Asian Handicap settlement distribution from canonical goal lambdas.")
    ap.add_argument("--history-dir", default="soccer_edge_state/history")
    ap.add_argument("--output", default="soccer_edge_state/analysis/asian_handicap_validation.json")
    args = ap.parse_args()

    predictions, finals = load(args.history_dir)
    rows: list[dict[str, Any]] = []
    fixture_count = 0
    for fid, pred in predictions.items():
        if fid not in finals:
            continue
        fixture_count += 1
        rows.extend(evaluate_fixture(pred, finals[fid]))

    by_line: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_league: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_line[str(row["home_handicap"])].append(row)
        by_league[str(row.get("league") or "UNKNOWN")].append(row)

    report = {
        "schema_version": "1.0.0",
        "status": "RESEARCH_ONLY_ASIAN_HANDICAP_VALIDATION",
        "model": "CANONICAL_SCORE_MARGIN_ASIAN_HANDICAP_v0.1",
        "evaluated_fixtures": fixture_count,
        "settlement_rows": len(rows),
        "overall": summarize(rows),
        "by_home_handicap": {k: summarize(v) for k, v in sorted(by_line.items(), key=lambda x: float(x[0]))},
        "by_league": {k: summarize(v) for k, v in sorted(by_league.items())},
        "promotion_gate": {
            "enabled": False,
            "minimum_oos_bets_for_market_comparison": 250,
            "minimum_oos_bets_for_actionable_review": 500,
            "requires": [
                "verified historical observed Asian Handicap lines and prices",
                "true CLV by exact line and side",
                "stable settlement calibration by line bucket and competition",
                "validated push/quarter-line market shrinkage",
                "adequate parent score-margin calibration",
            ],
        },
        "notes": [
            "Validation lines are diagnostic settlement thresholds, not claims that a sportsbook offered those lines.",
            "Quarter-line outcomes are represented as fractional win/push/loss settlement states.",
            "Observed-market actionability still requires exact real line, price, source and OOS/CLV gates.",
            "This report cannot promote Asian Handicap to BET/LEAN/Galaxy.",
        ],
        "rows": rows[-1200:],
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps({k: report[k] for k in ("status", "evaluated_fixtures", "overall", "promotion_gate")}, indent=2))


if __name__ == "__main__":
    main()
