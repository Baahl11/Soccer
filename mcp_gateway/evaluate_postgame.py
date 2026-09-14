from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from typing import Any

CANONICAL_MARKETS = {"match winner", "winner", "goals over/under", "over/under", "both teams to score", "btts"}


def fnum(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def norm(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "").strip()).lower()


def period_type(market: str) -> str:
    n = norm(market)
    if any(s in n for s in ("first half", "1st half", "1h", "half time", "halftime")):
        return "1H"
    if any(s in n for s in ("second half", "2nd half", "2h")):
        return "2H"
    return "FT"


def canonical_full_match(best: dict[str, Any] | None) -> bool:
    if not best:
        return False
    market = norm(best.get("market"))
    if period_type(market) != "FT":
        return False
    banned = (
        "team total", "corners", "cards", "player", "double chance", "draw no bet",
        "asian", "handicap", "correct score", "odd/even", "both halves", "result/total",
        "winner &", "win and",
    )
    if any(x in market for x in banned):
        return False
    return market in CANONICAL_MARKETS


def parse_line(best: dict[str, Any]) -> float | None:
    line = fnum(best.get("line"))
    if line is not None:
        return line
    for field in (best.get("selection"), best.get("market")):
        m = re.search(r"(?:over|under)\s*([0-9]+(?:\.[0-9]+)?)", str(field or ""), re.I)
        if m:
            return float(m.group(1))
    return None


def final_score(result: dict[str, Any] | None) -> tuple[int | None, int | None, int | None, int | None]:
    if not result:
        return None, None, None, None
    goals = result.get("goals") or {}
    score = result.get("score") or {}
    ft = score.get("fulltime") or {}
    ht = score.get("halftime") or {}
    vals = [goals.get("home", ft.get("home")), goals.get("away", ft.get("away")), ht.get("home"), ht.get("away")]
    out = []
    for v in vals:
        try:
            out.append(int(v) if v is not None else None)
        except Exception:
            out.append(None)
    return out[0], out[1], out[2], out[3]


def grade_market(best: dict[str, Any] | None, result: dict[str, Any] | None, home: str, away: str) -> str:
    if not best or not result:
        return "NO_MARKET_OR_FINAL"
    h, a, hh, ha = final_score(result)
    if h is None or a is None:
        return "NO_FINAL"
    market = norm(best.get("market"))
    sel = norm(best.get("selection"))
    p = period_type(market)
    if p == "FT":
        ph, pa = h, a
    elif p == "1H" and hh is not None and ha is not None:
        ph, pa = hh, ha
    elif p == "2H" and hh is not None and ha is not None:
        ph, pa = h - hh, a - ha
    else:
        return "UNGRADABLE"

    if "both teams to score" in market or market == "btts":
        yes = ph > 0 and pa > 0
        if sel in {"yes", "y", "btts yes"}:
            return "WIN" if yes else "LOSS"
        if sel in {"no", "n", "btts no"}:
            return "WIN" if not yes else "LOSS"
        return "UNGRADABLE"

    if "over/under" in market or market in {"goals over/under", "over/under"} or " over " in f" {sel} " or " under " in f" {sel} ":
        line = parse_line(best)
        if line is None:
            return "UNGRADABLE"
        total = ph + pa
        is_over = sel.startswith("over") or " over " in f" {sel} "
        is_under = sel.startswith("under") or " under " in f" {sel} "
        if not (is_over or is_under):
            return "UNGRADABLE"
        if math.isclose(total, line):
            return "PUSH"
        if is_over:
            return "WIN" if total > line else "LOSS"
        return "WIN" if total < line else "LOSS"

    if market in {"match winner", "winner"}:
        actual = "home" if h > a else "away" if a > h else "draw"
        if sel in {"home", "1", norm(home)}:
            pick = "home"
        elif sel in {"away", "2", norm(away)}:
            pick = "away"
        elif sel in {"draw", "x"}:
            pick = "draw"
        else:
            return "UNGRADABLE"
        return "WIN" if pick == actual else "LOSS"

    return "UNGRADABLE"


def research_signal_grades(row: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    h, a, _, _ = final_score(result)
    if h is None or a is None:
        return {}
    total = h + a
    shortlist = row.get("sporting_shortlist") or {}
    tracks = set(shortlist.get("tracks") or [])
    raw = row.get("raw_projection") or {}
    out: dict[str, Any] = {}

    if "GOALS_OVER" in tracks:
        out["GOALS_OVER"] = {"hit_3plus": total >= 3, "actual_total": total}
    if "GOALS_UNDER" in tracks:
        out["GOALS_UNDER"] = {"hit_0_2": total <= 2, "actual_total": total}
    if "TWO_WAY" in tracks:
        out["TWO_WAY"] = {"btts_yes": h > 0 and a > 0, "actual_total": total}
    if "SIDE" in tracks:
        probs = {
            "home": fnum(raw.get("raw_home_win_prob")),
            "draw": fnum(raw.get("raw_draw_prob")),
            "away": fnum(raw.get("raw_away_win_prob")),
        }
        valid = {k: v for k, v in probs.items() if v is not None}
        if len(valid) == 3:
            predicted = max(valid, key=valid.get)
            actual = "home" if h > a else "away" if a > h else "draw"
            out["SIDE"] = {
                "predicted": predicted,
                "actual": actual,
                "hit": predicted == actual,
                "raw_probs": probs,
            }
    return out


def roi_units(outcome: str, price: float | None, stake: float | None) -> float | None:
    if price is None:
        return None
    s = stake if stake and stake > 0 else 1.0
    if outcome == "WIN":
        return round(s * (price - 1.0), 6)
    if outcome == "LOSS":
        return round(-s, 6)
    if outcome == "PUSH":
        return 0.0
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate Soccer Edge ledger rows once final results exist.")
    ap.add_argument("--ledger", default="soccer_edge_state/analysis/signal_ledger.jsonl")
    ap.add_argument("--output", default="soccer_edge_state/analysis/postgame_evaluation.jsonl")
    ap.add_argument("--summary-output", default="soccer_edge_state/analysis/postgame_summary.json")
    args = ap.parse_args()

    rows: list[dict[str, Any]] = []
    with open(args.ledger, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    finals: dict[int, dict[str, Any]] = {}
    for row in rows:
        if row.get("result"):
            finals[int(row["fixture_id"])] = row["result"]

    evaluations: list[dict[str, Any]] = []
    for row in rows:
        fid = int(row["fixture_id"])
        result = finals.get(fid)
        if not result:
            continue
        best = row.get("best_market")
        outcome = grade_market(best, result, row.get("home_team") or "", row.get("away_team") or "") if best else None
        price = fnum((best or {}).get("decimal_price"))
        classification = str(row.get("classification") or "").upper()
        stake = fnum(row.get("stake_units"))
        eval_row = {
            "event_key": row.get("event_key"),
            "fixture_id": fid,
            "kickoff_local": row.get("kickoff_local"),
            "generated_at_local": row.get("generated_at_local"),
            "stage": row.get("stage"),
            "league": row.get("league"),
            "home_team": row.get("home_team"),
            "away_team": row.get("away_team"),
            "classification": classification,
            "tier": row.get("tier"),
            "data_tier": row.get("data_tier"),
            "availability_confidence": row.get("availability_confidence"),
            "bet_eligible": row.get("bet_eligible"),
            "canonical_ft_market": canonical_full_match(best),
            "best_market": best,
            "market_outcome": outcome,
            "roi_units": roi_units(outcome, price, stake) if classification in {"BET", "LEAN"} else None,
            "research_signals": research_signal_grades(row, result),
            "result": result,
        }
        evaluations.append(eval_row)

    # One decision per fixture/market/classification for ROI to avoid counting repeated refresh snapshots twice.
    decision_best: dict[tuple[Any, ...], dict[str, Any]] = {}
    for ev in evaluations:
        best = ev.get("best_market") or {}
        key = (
            ev["fixture_id"],
            ev["classification"],
            norm(best.get("market")),
            norm(best.get("selection")),
            fnum(best.get("line")),
        )
        prev = decision_best.get(key)
        if prev is None or str(ev.get("generated_at_local") or "") > str(prev.get("generated_at_local") or ""):
            decision_best[key] = ev

    actionable = [
        ev for ev in decision_best.values()
        if ev["classification"] in {"BET", "LEAN"} and ev["canonical_ft_market"] and ev.get("market_outcome") in {"WIN", "LOSS", "PUSH"}
    ]
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ev in actionable:
        by_class[ev["classification"]].append(ev)

    signal_stats: dict[str, Counter[str]] = defaultdict(Counter)
    signal_unique: set[tuple[int, str]] = set()
    for ev in evaluations:
        for signal, payload in (ev.get("research_signals") or {}).items():
            key = (ev["fixture_id"], signal)
            if key in signal_unique:
                continue
            signal_unique.add(key)
            if signal == "GOALS_OVER":
                signal_stats[signal]["n"] += 1
                signal_stats[signal]["hit"] += int(bool(payload.get("hit_3plus")))
            elif signal == "GOALS_UNDER":
                signal_stats[signal]["n"] += 1
                signal_stats[signal]["hit"] += int(bool(payload.get("hit_0_2")))
            elif signal == "TWO_WAY":
                signal_stats[signal]["n"] += 1
                signal_stats[signal]["hit"] += int(bool(payload.get("btts_yes")))
            elif signal == "SIDE" and payload.get("hit") is not None:
                signal_stats[signal]["n"] += 1
                signal_stats[signal]["hit"] += int(bool(payload.get("hit")))

    def decision_summary(group: list[dict[str, Any]]) -> dict[str, Any]:
        c = Counter(ev["market_outcome"] for ev in group)
        decided = c["WIN"] + c["LOSS"]
        roi = sum(float(ev.get("roi_units") or 0.0) for ev in group)
        risk = sum((fnum(ev.get("best_market", {}).get("decimal_price")) is not None) * (fnum(ev.get("roi_units")) is not None) * (fnum(ev.get("best_market", {}).get("decimal_price")) is not None) for ev in group)
        # risk above is count-based because LEAN may not carry stake; ROI is still unit-normalized when stake absent.
        return {
            "n": len(group),
            "win": c["WIN"],
            "loss": c["LOSS"],
            "push": c["PUSH"],
            "hit_rate_ex_push": round(c["WIN"] / decided, 4) if decided else None,
            "roi_units": round(roi, 4),
            "roi_per_decision_units": round(roi / len(group), 4) if group else None,
        }

    summary = {
        "schema_version": "1.0.0",
        "timezone_basis": "America/Mexico_City",
        "ledger_rows_read": len(rows),
        "final_fixture_count": len(finals),
        "evaluation_rows": len(evaluations),
        "canonical_ft_actionable_decisions": len(actionable),
        "actionable_by_classification": {k: decision_summary(v) for k, v in sorted(by_class.items())},
        "research_signal_performance": {
            signal: {
                "n": counts["n"],
                "hit": counts["hit"],
                "hit_rate": round(counts["hit"] / counts["n"], 4) if counts["n"] else None,
                "definition": {
                    "GOALS_OVER": "3+ final-match goals (research proxy; not a wager line)",
                    "GOALS_UNDER": "0-2 final-match goals (research proxy; not a wager line)",
                    "TWO_WAY": "BTTS Yes final result (research proxy)",
                    "SIDE": "argmax raw 1X2 probability matched final 1/X/2 (research only)",
                }.get(signal),
            }
            for signal, counts in sorted(signal_stats.items())
        },
        "safety_note": "Only canonical full-match Winner/1X2, Goals O/U and BTTS are included in actionable ROI summaries. Period/team/compound/player/corners/cards derivatives remain research-only and are excluded.",
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        for ev in evaluations:
            fh.write(json.dumps(ev, ensure_ascii=False, separators=(",", ":")) + "\n")
    with open(args.summary_output, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
