from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from typing import Any


def fnum(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def final_total(result: dict[str, Any] | None) -> int | None:
    if not isinstance(result, dict) or not result:
        return None
    goals = result.get("goals") or {}
    score = result.get("score") or {}
    ft = score.get("fulltime") or {}
    h = goals.get("home", ft.get("home"))
    a = goals.get("away", ft.get("away"))
    try:
        return int(h) + int(a)
    except (TypeError, ValueError):
        return None


def norm(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "").strip()).lower()


def canonical_total_market(best: dict[str, Any] | None) -> bool:
    if not isinstance(best, dict) or not best:
        return False
    market = norm(best.get("market"))
    if market not in {"goals over/under", "over/under"}:
        return False
    banned = ("first half", "second half", "1h", "2h", "team total", "corners", "cards", "player", "asian", "alternate")
    return not any(x in market for x in banned)


def parse_line(best: dict[str, Any]) -> float | None:
    line = fnum(best.get("line"))
    if line is not None:
        return line
    m = re.search(r"(?:over|under)\s*([0-9]+(?:\.[0-9]+)?)", str(best.get("selection") or ""), re.I)
    return float(m.group(1)) if m else None


def selection_side(best: dict[str, Any]) -> str | None:
    s = norm(best.get("selection"))
    if s.startswith("over") or " over " in f" {s} ":
        return "OVER"
    if s.startswith("under") or " under " in f" {s} ":
        return "UNDER"
    return None


def grade_total(total: int, side: str, line: float) -> str:
    if math.isclose(total, line):
        return "PUSH"
    if side == "OVER":
        return "WIN" if total > line else "LOSS"
    return "WIN" if total < line else "LOSS"


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate canonical full-match Soccer Edge totals decisions and probabilities.")
    ap.add_argument("--ledger", default="soccer_edge_state/analysis/signal_ledger.jsonl")
    ap.add_argument("--output", default="soccer_edge_state/analysis/ft_totals_validation.json")
    args = ap.parse_args()

    rows = []
    results: dict[int, int] = {}
    observations: list[dict[str, Any]] = []
    with open(args.ledger, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            fid = row.get("fixture_id")
            if not fid:
                continue
            total = final_total(row.get("result"))
            if total is not None:
                results[int(fid)] = total
            best = row.get("best_market") or {}
            if canonical_total_market(best):
                line_v = parse_line(best)
                side = selection_side(best)
                price = fnum(best.get("decimal_price"))
                p = fnum(best.get("p_shrunk"))
                if line_v is None or side is None:
                    continue
                observations.append({
                    "fixture_id": int(fid),
                    "timestamp": str(row.get("generated_at_local") or ""),
                    "stage": row.get("stage"),
                    "classification": str(row.get("classification") or "").upper(),
                    "tier": row.get("tier") or best.get("tier"),
                    "line": line_v,
                    "side": side,
                    "price": price,
                    "p_shrunk": p,
                    "edge_pp": fnum(best.get("prob_edge_pp")),
                    "availability_confidence": fnum(row.get("availability_confidence")),
                    "data_tier": row.get("data_tier"),
                    "league": row.get("league"),
                })

    # De-duplicate repeated snapshots: use latest pre-kickoff observation for same exact decision.
    latest: dict[tuple, dict[str, Any]] = {}
    for r in observations:
        key = (r["fixture_id"], r["line"], r["side"], r["classification"])
        old = latest.get(key)
        if old is None or r["timestamp"] > old["timestamp"]:
            latest[key] = r

    eval_rows = []
    for r in latest.values():
        total = results.get(r["fixture_id"])
        if total is None:
            continue
        outcome = grade_total(total, r["side"], r["line"])
        rr = dict(r)
        rr["final_total"] = total
        rr["outcome"] = outcome
        if outcome in {"WIN", "LOSS"} and r["p_shrunk"] is not None:
            y = 1.0 if outcome == "WIN" else 0.0
            rr["brier"] = (r["p_shrunk"] - y) ** 2
            rr["log_loss"] = -(math.log(max(1e-12, r["p_shrunk"])) if y else math.log(max(1e-12, 1-r["p_shrunk"])))
        else:
            rr["brier"] = None
            rr["log_loss"] = None
        if outcome == "WIN" and r["price"] is not None:
            rr["roi_units"] = r["price"] - 1.0
        elif outcome == "LOSS":
            rr["roi_units"] = -1.0
        else:
            rr["roi_units"] = 0.0
        eval_rows.append(rr)

    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        c = Counter(r["outcome"] for r in group)
        decided = c["WIN"] + c["LOSS"]
        briers = [r["brier"] for r in group if r.get("brier") is not None]
        lls = [r["log_loss"] for r in group if r.get("log_loss") is not None]
        roi = sum(float(r.get("roi_units") or 0.0) for r in group)
        return {
            "n": len(group), "win": c["WIN"], "loss": c["LOSS"], "push": c["PUSH"],
            "hit_rate_ex_push": round(c["WIN"]/decided, 4) if decided else None,
            "roi_units_flat_1u": round(roi, 4),
            "roi_per_decision": round(roi/len(group), 4) if group else None,
            "mean_brier": round(sum(briers)/len(briers), 4) if briers else None,
            "mean_log_loss": round(sum(lls)/len(lls), 4) if lls else None,
        }

    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_line: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_stage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_side: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in eval_rows:
        by_class[r["classification"]].append(r)
        by_line[str(r["line"])].append(r)
        by_stage[str(r.get("stage") or "UNKNOWN")].append(r)
        by_side[r["side"]].append(r)

    actionable = [r for r in eval_rows if r["classification"] in {"BET", "LEAN"}]
    watch = [r for r in eval_rows if r["classification"] == "WATCH"]
    result = {
        "schema_version": "1.0.0",
        "timezone_basis": "America/Mexico_City",
        "status": "VALIDATION_ONLY_DO_NOT_UPGRADE_BACKEND",
        "evaluated_decisions": len(eval_rows),
        "actionable": summarize(actionable),
        "watch_research": summarize(watch),
        "by_classification": {k: summarize(v) for k, v in sorted(by_class.items())},
        "by_line": {k: summarize(v) for k, v in sorted(by_line.items(), key=lambda kv: float(kv[0]))},
        "by_stage": {k: summarize(v) for k, v in sorted(by_stage.items())},
        "by_side": {k: summarize(v) for k, v in sorted(by_side.items())},
        "notes": [
            "Only canonical full-match Goals Over/Under observations are included.",
            "Period, team-total, alternate and other derivative markets are excluded.",
            "Flat 1-unit ROI is descriptive only and is not used to upgrade classifications.",
            "Latest snapshot for each fixture/line/side/classification is used to avoid repeated-stage weighting."
        ]
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
