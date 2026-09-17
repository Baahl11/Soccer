from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict
from typing import Any


def fnum(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def final_outcome(event: dict[str, Any]) -> str | None:
    result = event.get("result") if isinstance(event.get("result"), dict) else {}
    goals = result.get("goals") if isinstance(result.get("goals"), dict) else {}
    score = result.get("score") if isinstance(result.get("score"), dict) else {}
    fulltime = score.get("fulltime") if isinstance(score.get("fulltime"), dict) else {}
    fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
    fixture_goals = fixture.get("goals") if isinstance(fixture.get("goals"), dict) else {}
    h = goals.get("home", fulltime.get("home", fixture_goals.get("home")))
    a = goals.get("away", fulltime.get("away", fixture_goals.get("away")))
    try:
        h, a = int(h), int(a)
    except (TypeError, ValueError):
        return None
    return "HOME" if h > a else "AWAY" if a > h else "DRAW"


def load(history_dir: str) -> tuple[dict[int, dict[str, Any]], dict[int, str]]:
    predictions: dict[int, dict[str, Any]] = {}
    results: dict[int, str] = {}
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
                        outcome = final_outcome(event)
                        if outcome:
                            results[fid] = outcome
                        continue
                    raw = event.get("raw_projection") if isinstance(event.get("raw_projection"), dict) else {}
                    vals = [fnum(raw.get(k)) for k in ("raw_home_win_prob", "raw_draw_prob", "raw_away_win_prob")]
                    if any(v is None or v < 0 for v in vals):
                        continue
                    total = sum(float(v) for v in vals if v is not None)
                    if total <= 0:
                        continue
                    h, d, a = [float(v) / total for v in vals if v is not None]
                    non_draw = h + a
                    if non_draw <= 0:
                        continue
                    old = predictions.get(fid)
                    if old is None or stamp > old["timestamp"]:
                        predictions[fid] = {
                            "fixture_id": fid,
                            "timestamp": stamp,
                            "stage": event.get("stage"),
                            "league": fixture.get("league"),
                            "p_draw": d,
                            "p_home_conditional": h / non_draw,
                            "p_away_conditional": a / non_draw,
                        }
    return predictions, results


def evaluate(pred: dict[str, Any], outcome: str) -> list[dict[str, Any]]:
    if outcome == "DRAW":
        return []
    rows: list[dict[str, Any]] = []
    for selection, p in (("HOME", pred["p_home_conditional"]), ("AWAY", pred["p_away_conditional"])):
        y = 1 if selection == outcome else 0
        q = min(max(float(p), 1e-9), 1.0 - 1e-9)
        rows.append({
            "fixture_id": pred["fixture_id"],
            "stage": pred.get("stage"),
            "league": pred.get("league"),
            "selection": selection,
            "probability_win_given_no_draw": round(float(p), 6),
            "actual_win_given_no_draw": y,
            "brier": (float(p) - y) ** 2,
            "log_loss": -(y * math.log(q) + (1 - y) * math.log(1 - q)),
        })
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "brier": None, "log_loss": None, "observed_win_rate": None, "mean_probability": None}
    n = len(rows)
    return {
        "n": n,
        "brier": round(sum(r["brier"] for r in rows) / n, 6),
        "log_loss": round(sum(r["log_loss"] for r in rows) / n, 6),
        "observed_win_rate": round(sum(r["actual_win_given_no_draw"] for r in rows) / n, 6),
        "mean_probability": round(sum(r["probability_win_given_no_draw"] for r in rows) / n, 6),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate research-only Draw No Bet probabilities conditioned on no draw.")
    ap.add_argument("--history-dir", default="soccer_edge_state/history")
    ap.add_argument("--output", default="soccer_edge_state/analysis/dnb_validation.json")
    args = ap.parse_args()

    predictions, results = load(args.history_dir)
    rows: list[dict[str, Any]] = []
    total_with_result = 0
    draws = 0
    non_draw_fixtures = 0
    for fid, pred in predictions.items():
        outcome = results.get(fid)
        if outcome is None:
            continue
        total_with_result += 1
        if outcome == "DRAW":
            draws += 1
            continue
        non_draw_fixtures += 1
        rows.extend(evaluate(pred, outcome))

    by_selection: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_league: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_selection[row["selection"]].append(row)
        by_league[str(row.get("league") or "UNKNOWN")].append(row)

    report = {
        "schema_version": "1.0.0",
        "status": "RESEARCH_ONLY_DNB_VALIDATION",
        "model": "CANONICAL_1X2_CONDITIONAL_DNB_v0.1",
        "fixtures_with_result": total_with_result,
        "draw_push_fixtures": draws,
        "observed_draw_push_rate": round(draws / total_with_result, 6) if total_with_result else None,
        "non_draw_oos_fixtures": non_draw_fixtures,
        "evaluated_binary_rows": len(rows),
        "overall_non_draw": summarize(rows),
        "by_selection": {k: summarize(v) for k, v in sorted(by_selection.items())},
        "by_league": {k: summarize(v) for k, v in sorted(by_league.items())},
        "promotion_gate": {
            "enabled": False,
            "minimum_non_draw_oos_for_market_comparison": 200,
            "minimum_non_draw_oos_for_actionable_review": 400,
            "market_comparison_sample_gate_met": non_draw_fixtures >= 200,
            "actionable_review_sample_gate_met": non_draw_fixtures >= 400,
            "requires": [
                "parent 1X2 production model approved",
                "stable HOME/AWAY conditional calibration on non-draw fixtures",
                "verified historical DNB prices and true CLV",
                "push-aware price settlement verified",
                "validated market shrinkage after parent-model selection",
            ],
        },
        "notes": [
            "Draw fixtures are pushes and are excluded from conditional HOME/AWAY calibration scoring.",
            "DNB fair probability is P(win | no draw), while push probability is tracked separately.",
            "This report cannot promote DNB to BET/LEAN/Galaxy while parent 1X2 remains research-only.",
        ],
        "rows": rows[-900:],
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps({k: report[k] for k in ("status", "fixtures_with_result", "non_draw_oos_fixtures", "overall_non_draw", "promotion_gate")}, indent=2))


if __name__ == "__main__":
    main()
