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
                    p = fnum(raw.get("raw_btts_yes_prob"))
                    if p is None or not 0 < p < 1:
                        continue
                    old = predictions.get(fid)
                    if old is None or stamp > old["timestamp"]:
                        coverage = event.get("coverage") if isinstance(event.get("coverage"), dict) else {}
                        predictions[fid] = {
                            "fixture_id": fid,
                            "timestamp": stamp,
                            "stage": event.get("stage"),
                            "league": fixture.get("league"),
                            "data_tier": coverage.get("data_tier"),
                            "p_yes": p,
                        }
    return predictions, results


def evaluate(pred: dict[str, Any], final: tuple[int, int]) -> dict[str, Any]:
    p = float(pred["p_yes"])
    y = 1.0 if final[0] > 0 and final[1] > 0 else 0.0
    clipped = min(max(p, 1e-9), 1.0 - 1e-9)
    return {
        "fixture_id": pred["fixture_id"],
        "stage": pred.get("stage"),
        "league": pred.get("league"),
        "data_tier": pred.get("data_tier"),
        "p_yes": round(p, 6),
        "actual_btts_yes": int(y),
        "brier": (p - y) ** 2,
        "log_loss": -(y * math.log(clipped) + (1.0 - y) * math.log(1.0 - clipped)),
        "bucket": f"{int(p * 10) * 10:02d}-{min(100, int(p * 10) * 10 + 9):02d}%",
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "brier": None, "log_loss": None, "observed_rate": None, "mean_probability": None}
    n = len(rows)
    return {
        "n": n,
        "brier": round(sum(r["brier"] for r in rows) / n, 6),
        "log_loss": round(sum(r["log_loss"] for r in rows) / n, 6),
        "observed_rate": round(sum(r["actual_btts_yes"] for r in rows) / n, 6),
        "mean_probability": round(sum(r["p_yes"] for r in rows) / n, 6),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate research-only BTTS probability from canonical score matrix.")
    ap.add_argument("--history-dir", default="soccer_edge_state/history")
    ap.add_argument("--output", default="soccer_edge_state/analysis/btts_validation.json")
    args = ap.parse_args()

    predictions, results = load(args.history_dir)
    rows = [evaluate(pred, results[fid]) for fid, pred in predictions.items() if fid in results]
    by_league: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_tier: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_league[str(row.get("league") or "UNKNOWN")].append(row)
        by_tier[str(row.get("data_tier") or "UNKNOWN")].append(row)
        by_bucket[str(row.get("bucket") or "UNKNOWN")].append(row)

    n = len(rows)
    report = {
        "schema_version": "1.0.0",
        "status": "RESEARCH_ONLY_BTTS_VALIDATION",
        "model": "CANONICAL_SCORE_MATRIX_BTTS_v0.1",
        "evaluated_fixtures": n,
        "overall": summarize(rows),
        "by_league": {k: summarize(v) for k, v in sorted(by_league.items())},
        "by_data_tier": {k: summarize(v) for k, v in sorted(by_tier.items())},
        "by_probability_bucket": {k: summarize(v) for k, v in sorted(by_bucket.items())},
        "promotion_gate": {
            "enabled": False,
            "minimum_oos_fixtures_for_market_comparison": 150,
            "minimum_oos_fixtures_for_actionable_review": 300,
            "market_comparison_sample_gate_met": n >= 150,
            "actionable_review_sample_gate_met": n >= 300,
            "requires": [
                "stable Brier/log-loss overall and across competitions",
                "stable calibration by probability bucket and data tier",
                "verified historical BTTS prices and true CLV evidence",
                "validated market shrinkage policy",
                "no material degradation versus canonical FT-goals calibration",
            ],
        },
        "rows": rows[-800:],
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps({k: report[k] for k in ("status", "evaluated_fixtures", "overall", "promotion_gate")}, indent=2))


if __name__ == "__main__":
    main()
