from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict
from typing import Any

LINES = (0.5, 1.5, 2.5)


def fnum(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def poisson_over(lam: float, line: float) -> float:
    threshold = int(math.floor(line)) + 1
    cdf = sum(math.exp(-lam) * (lam ** k) / math.factorial(k) for k in range(threshold))
    return max(0.0, min(1.0, 1.0 - cdf))


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate research-only team-total goal probabilities from canonical team lambdas.")
    ap.add_argument("--history-dir", default="soccer_edge_state/history")
    ap.add_argument("--output", default="soccer_edge_state/analysis/team_totals_validation.json")
    args = ap.parse_args()

    predictions, results = load(args.history_dir)
    rows: list[dict[str, Any]] = []
    for fid, pred in predictions.items():
        final = results.get(fid)
        if final is None:
            continue
        home_goals, away_goals = final
        for role, lam, actual in (("HOME", pred["home_lambda"], home_goals), ("AWAY", pred["away_lambda"], away_goals)):
            for line in LINES:
                p_over = poisson_over(lam, line)
                for side, p, y in (
                    ("OVER", p_over, int(actual > line)),
                    ("UNDER", 1.0 - p_over, int(actual < line)),
                ):
                    p = max(1e-9, min(1.0 - 1e-9, p))
                    rows.append({
                        "fixture_id": fid,
                        "stage": pred.get("stage"),
                        "league": pred.get("league"),
                        "home_team": pred.get("home_team"),
                        "away_team": pred.get("away_team"),
                        "team_role": role,
                        "line": line,
                        "selection": side,
                        "lambda": round(lam, 6),
                        "probability": round(p, 6),
                        "actual_team_goals": actual,
                        "outcome": y,
                        "brier": (p - y) ** 2,
                        "log_loss": -(y * math.log(p) + (1 - y) * math.log(1 - p)),
                    })

    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        if not group:
            return {"n": 0, "mean_brier": None, "mean_log_loss": None, "mean_probability": None, "observed_rate": None}
        return {
            "n": len(group),
            "mean_brier": round(sum(r["brier"] for r in group) / len(group), 6),
            "mean_log_loss": round(sum(r["log_loss"] for r in group) / len(group), 6),
            "mean_probability": round(sum(r["probability"] for r in group) / len(group), 6),
            "observed_rate": round(sum(r["outcome"] for r in group) / len(group), 6),
        }

    by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_line: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_selection: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_role_line: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_role[row["team_role"]].append(row)
        by_line[str(row["line"])].append(row)
        by_selection[row["selection"]].append(row)
        by_role_line[f'{row["team_role"]}:{row["line"]}:{row["selection"]}'].append(row)

    fixture_n = len({r["fixture_id"] for r in rows})
    report = {
        "schema_version": "1.0.0",
        "status": "RESEARCH_ONLY_TEAM_TOTALS_VALIDATION",
        "model": "CANONICAL_TEAM_LAMBDA_POISSON_DERIVATION_v0.1",
        "evaluated_fixtures": fixture_n,
        "evaluated_probability_rows": len(rows),
        "overall": summarize(rows),
        "by_team_role": {k: summarize(v) for k, v in sorted(by_role.items())},
        "by_line": {k: summarize(v) for k, v in sorted(by_line.items(), key=lambda kv: float(kv[0]))},
        "by_selection": {k: summarize(v) for k, v in sorted(by_selection.items())},
        "by_role_line_selection": {k: summarize(v) for k, v in sorted(by_role_line.items())},
        "promotion_gate": {
            "enabled": False,
            "minimum_oos_fixtures_for_market_comparison": 100,
            "minimum_oos_fixtures_for_actionable_review": 200,
            "market_comparison_sample_gate_met": fixture_n >= 100,
            "actionable_review_sample_gate_met": fixture_n >= 200,
            "requires": [
                "stable Brier/log-loss by home/away and 0.5/1.5/2.5",
                "verified historical team-total prices and CLV evidence",
                "stable calibration across competitions",
                "no material degradation versus FT-goals calibration",
            ],
        },
        "notes": [
            "Latest persisted pregame raw home/away lambda per fixture is used; postgame results provide the target.",
            "Only half-goal lines 0.5, 1.5 and 2.5 are validated in v1.",
            "No market odds are used to create sport probabilities.",
            "This report cannot promote Team Totals to BET/LEAN by itself.",
        ],
        "rows": rows[-600:],
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps({k: report[k] for k in ("status", "evaluated_fixtures", "evaluated_probability_rows", "overall", "promotion_gate")}, indent=2))


if __name__ == "__main__":
    main()
