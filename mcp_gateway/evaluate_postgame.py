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


def market_family(best: dict[str, Any] | None) -> str:
    """Return a stable family bucket for settlement/performance reporting.

    The family is intentionally conservative. Derivative markets are bucketed,
    and only markets with explicit settlement logic in grade_market() produce
    WIN/LOSS/PUSH.
    """
    if not best:
        return "NO_MARKET"
    market = norm(best.get("market"))
    selection = norm(best.get("selection"))
    period = period_type(market)

    if "team total" in market or "team goals" in market:
        return f"{period}_TEAM_TOTAL"
    if "corner" in market:
        return f"{period}_CORNERS"
    if "card" in market or "booking" in market:
        return f"{period}_CARDS"
    if "player" in market or any(token in market for token in ("shots", "saves", "assists", "goalscorer")):
        return "PLAYER_PROP"
    if "double chance" in market:
        return f"{period}_DOUBLE_CHANCE"
    if "draw no bet" in market or market == "dnb":
        return f"{period}_DNB"
    if "asian" in market or "handicap" in market:
        return f"{period}_HANDICAP"
    if "correct score" in market:
        return f"{period}_CORRECT_SCORE"
    if "result/total" in market or "winner &" in market or "win and" in market:
        return f"{period}_COMPOUND"
    if "both teams to score" in market or market == "btts":
        return f"{period}_BTTS"
    if market in {"match winner", "winner"}:
        return f"{period}_1X2"
    if (
        "over/under" in market
        or market in {"goals over/under", "over/under"}
        or selection.startswith("over")
        or selection.startswith("under")
        or " over " in f" {selection} "
        or " under " in f" {selection} "
    ):
        return f"{period}_TOTALS"
    return f"{period}_OTHER"


def canonical_full_match(best: dict[str, Any] | None) -> bool:
    if not best:
        return False
    market = norm(best.get("market"))
    if period_type(market) != "FT":
        return False
    banned = (
        "team total", "team goals", "corners", "cards", "booking", "player", "double chance", "draw no bet",
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


def period_scores(result: dict[str, Any] | None, market: str) -> tuple[int | None, int | None]:
    h, a, hh, ha = final_score(result)
    if h is None or a is None:
        return None, None
    p = period_type(market)
    if p == "FT":
        return h, a
    if p == "1H" and hh is not None and ha is not None:
        return hh, ha
    if p == "2H" and hh is not None and ha is not None:
        return h - hh, a - ha
    return None, None


def _team_aliases(name: str) -> set[str]:
    n = norm(name)
    aliases = {n}
    if n:
        aliases.add(n.replace(" fc", ""))
        aliases.add(n.replace(" cf", ""))
    return {a for a in aliases if a}


def team_total_side(best: dict[str, Any], home: str, away: str) -> str | None:
    """Infer whether a team-total selection applies to home or away.

    Supported examples:
    - "Home Over 1.5", "Away Under 0.5"
    - "Team 1 Over 1.5", "Team 2 Under 1.5"
    - "<home team name> Over 1.5", "<away team name> Under 0.5"
    - best["team"] / best["team_name"] / best["participant"] when available.
    """
    fields = [
        best.get("selection"),
        best.get("team"),
        best.get("team_name"),
        best.get("participant"),
        best.get("label"),
        best.get("name"),
    ]
    text = " ".join(norm(x) for x in fields if x)
    if not text:
        return None

    if re.search(r"\b(home|team\s*1|1)\b", text) and not re.search(r"\b(away|team\s*2|2)\b", text):
        return "home"
    if re.search(r"\b(away|team\s*2|2)\b", text) and not re.search(r"\b(home|team\s*1|1)\b", text):
        return "away"

    for alias in _team_aliases(home):
        if alias and alias in text:
            return "home"
    for alias in _team_aliases(away):
        if alias and alias in text:
            return "away"
    return None


def is_over_under_selection(best: dict[str, Any]) -> tuple[bool, bool]:
    sel = norm(best.get("selection"))
    market = norm(best.get("market"))
    text = f" {sel} {market} "
    return (
        sel.startswith("over") or " over " in text,
        sel.startswith("under") or " under " in text,
    )


def grade_over_under(total: int | float, line: float | None, is_over: bool, is_under: bool) -> str:
    if line is None or not (is_over or is_under):
        return "UNGRADABLE"
    if math.isclose(float(total), float(line)):
        return "PUSH"
    if is_over:
        return "WIN" if total > line else "LOSS"
    return "WIN" if total < line else "LOSS"


def grade_market(best: dict[str, Any] | None, result: dict[str, Any] | None, home: str, away: str) -> str:
    if not best or not result:
        return "NO_MARKET_OR_FINAL"

    family = market_family(best)
    market = norm(best.get("market"))
    sel = norm(best.get("selection"))
    ph, pa = period_scores(result, market)
    if ph is None or pa is None:
        return "NO_FINAL"

    if family.endswith("_BTTS"):
        yes = ph > 0 and pa > 0
        if sel in {"yes", "y", "btts yes"}:
            return "WIN" if yes else "LOSS"
        if sel in {"no", "n", "btts no"}:
            return "WIN" if not yes else "LOSS"
        return "UNGRADABLE"

    if family.endswith("_TOTALS"):
        is_over, is_under = is_over_under_selection(best)
        return grade_over_under(ph + pa, parse_line(best), is_over, is_under)

    if family.endswith("_TEAM_TOTAL"):
        side = team_total_side(best, home, away)
        if side is None:
            return "UNGRADABLE_TEAM_TOTAL_SIDE"
        is_over, is_under = is_over_under_selection(best)
        team_goals = ph if side == "home" else pa
        return grade_over_under(team_goals, parse_line(best), is_over, is_under)

    if family.endswith("_1X2"):
        actual = "home" if ph > pa else "away" if pa > ph else "draw"
        if sel in {"home", "1", norm(home)}:
            pick = "home"
        elif sel in {"away", "2", norm(away)}:
            pick = "away"
        elif sel in {"draw", "x"}:
            pick = "draw"
        else:
            return "UNGRADABLE"
        return "WIN" if pick == actual else "LOSS"

    if family in {"FT_CORNERS", "1H_CORNERS", "2H_CORNERS", "FT_CARDS", "1H_CARDS", "2H_CARDS", "PLAYER_PROP"}:
        return "UNSUPPORTED_DERIVATIVE"

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


def decision_key(ev: dict[str, Any]) -> tuple[Any, ...]:
    best = ev.get("best_market") or {}
    return (
        ev["fixture_id"],
        ev["classification"],
        market_family(best),
        norm(best.get("market")),
        norm(best.get("selection")),
        fnum(best.get("line")),
    )


def settlement_row(ev: dict[str, Any]) -> dict[str, Any]:
    best = ev.get("best_market") or {}
    price = fnum(best.get("decimal_price"))
    stake = fnum(ev.get("stake_units")) or 1.0
    outcome = ev.get("market_outcome")
    return {
        "schema_version": "1.1.0",
        "event_key": ev.get("event_key"),
        "fixture_id": ev.get("fixture_id"),
        "kickoff_local": ev.get("kickoff_local"),
        "generated_at_local": ev.get("generated_at_local"),
        "stage": ev.get("stage"),
        "league": ev.get("league"),
        "home_team": ev.get("home_team"),
        "away_team": ev.get("away_team"),
        "classification": ev.get("classification"),
        "tier": ev.get("tier"),
        "data_tier": ev.get("data_tier"),
        "availability_confidence": ev.get("availability_confidence"),
        "bet_eligible": ev.get("bet_eligible"),
        "period": period_type(norm(best.get("market"))),
        "market_family": market_family(best),
        "canonical_ft_market": ev.get("canonical_ft_market"),
        "market": best.get("market"),
        "selection": best.get("selection"),
        "line": fnum(best.get("line")),
        "decimal_price": price,
        "bookmaker": best.get("bookmaker") or best.get("book"),
        "stake_units": stake,
        "settlement_status": outcome,
        "settled": outcome in {"WIN", "LOSS", "PUSH"},
        "roi_units": ev.get("roi_units"),
        "result": ev.get("result"),
    }


def performance_summary(group: list[dict[str, Any]]) -> dict[str, Any]:
    c = Counter(row.get("settlement_status") for row in group)
    decided = c["WIN"] + c["LOSS"]
    settled = decided + c["PUSH"]
    roi = sum(float(row.get("roi_units") or 0.0) for row in group)
    return {
        "n": len(group),
        "settled": settled,
        "win": c["WIN"],
        "loss": c["LOSS"],
        "push": c["PUSH"],
        "ungraded": len(group) - settled,
        "unsupported_derivative": c["UNSUPPORTED_DERIVATIVE"],
        "ungradable_team_total_side": c["UNGRADABLE_TEAM_TOTAL_SIDE"],
        "hit_rate_ex_push": round(c["WIN"] / decided, 4) if decided else None,
        "roi_units": round(roi, 4),
        "roi_per_decision_units": round(roi / len(group), 4) if group else None,
    }


def tier_gate(summary: dict[str, Any]) -> dict[str, Any]:
    """Conservative promotion gate for market buckets.

    This is not a betting recommendation. It prevents premature promotion by
    requiring enough settled decisions and positive ROI before a bucket can
    leave research mode. CLV still must be checked separately before Tier A/S.
    """
    n = int(summary.get("n") or 0)
    settled = int(summary.get("settled") or 0)
    unsupported = int(summary.get("unsupported_derivative") or 0)
    ungraded = int(summary.get("ungraded") or 0)
    roi = float(summary.get("roi_units") or 0.0)
    hit_rate = summary.get("hit_rate_ex_push")

    if unsupported and settled == 0:
        status = "RESEARCH_ONLY_NEEDS_EXPLICIT_SETTLEMENT"
        reason = "bucket has derivative picks but no explicit win/loss grading yet"
    elif ungraded and settled == 0:
        status = "RESEARCH_ONLY_UNGRADABLE"
        reason = "bucket has rows but cannot infer settlement from stored market metadata"
    elif settled < 20:
        status = "RESEARCH_ONLY_SAMPLE_TOO_SMALL"
        reason = "fewer than 20 settled decisions"
    elif hit_rate is None:
        status = "HOLD_NO_DECIDED_OUTCOMES"
        reason = "no win/loss decisions after pushes/ungraded rows"
    elif roi <= 0:
        status = "HOLD_NEGATIVE_OR_FLAT_ROI"
        reason = "settled sample is non-positive ROI"
    elif settled >= 100:
        status = "TIER_S_CANDIDATE_REQUIRES_CLV"
        reason = "100+ settled decisions and positive ROI; require CLV/segmentation confirmation"
    elif settled >= 50:
        status = "TIER_A_CANDIDATE_REQUIRES_CLV"
        reason = "50+ settled decisions and positive ROI; require CLV/segmentation confirmation"
    else:
        status = "TIER_B_CANDIDATE"
        reason = "20+ settled decisions and positive ROI; keep low stake until CLV/sample improves"

    return {
        "status": status,
        "reason": reason,
        "n": n,
        "settled": settled,
        "hit_rate_ex_push": hit_rate,
        "roi_units": round(roi, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate Soccer Edge ledger rows once final results exist.")
    ap.add_argument("--ledger", default="soccer_edge_state/analysis/signal_ledger.jsonl")
    ap.add_argument("--output", default="soccer_edge_state/analysis/postgame_evaluation.jsonl")
    ap.add_argument("--summary-output", default="soccer_edge_state/analysis/postgame_summary.json")
    ap.add_argument("--settlement-output", default="soccer_edge_state/analysis/bet_settlement_ledger.jsonl")
    ap.add_argument("--market-summary-output", default="soccer_edge_state/analysis/market_performance_summary.json")
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
            "market_family": market_family(best),
            "best_market": best,
            "market_outcome": outcome,
            "roi_units": roi_units(outcome, price, stake) if classification in {"BET", "LEAN"} else None,
            "research_signals": research_signal_grades(row, result),
            "result": result,
        }
        evaluations.append(eval_row)

    decision_best: dict[tuple[Any, ...], dict[str, Any]] = {}
    for ev in evaluations:
        key = decision_key(ev)
        prev = decision_best.get(key)
        if prev is None or str(ev.get("generated_at_local") or "") > str(prev.get("generated_at_local") or ""):
            decision_best[key] = ev

    settlement_decisions = [
        settlement_row(ev)
        for ev in decision_best.values()
        if ev["classification"] in {"BET", "LEAN"} and ev.get("best_market") and ev.get("result")
    ]

    actionable = [
        ev for ev in decision_best.values()
        if ev["classification"] in {"BET", "LEAN"} and ev["canonical_ft_market"] and ev.get("market_outcome") in {"WIN", "LOSS", "PUSH"}
    ]
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ev in actionable:
        by_class[ev["classification"]].append(ev)

    settlement_by_market: dict[str, list[dict[str, Any]]] = defaultdict(list)
    settlement_by_market_class: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in settlement_decisions:
        family = str(row.get("market_family") or "UNKNOWN")
        classification = str(row.get("classification") or "UNKNOWN")
        settlement_by_market[family].append(row)
        settlement_by_market_class[family][classification].append(row)

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
        return {
            "n": len(group),
            "win": c["WIN"],
            "loss": c["LOSS"],
            "push": c["PUSH"],
            "hit_rate_ex_push": round(c["WIN"] / decided, 4) if decided else None,
            "roi_units": round(roi, 4),
            "roi_per_decision_units": round(roi / len(group), 4) if group else None,
        }

    by_market_family = {k: performance_summary(v) for k, v in sorted(settlement_by_market.items())}
    by_market_family_and_classification = {
        family: {classification: performance_summary(rows) for classification, rows in sorted(groups.items())}
        for family, groups in sorted(settlement_by_market_class.items())
    }
    market_summary = {
        "schema_version": "1.2.0",
        "timezone_basis": "America/Mexico_City",
        "settlement_decisions": len(settlement_decisions),
        "by_market_family": by_market_family,
        "by_market_family_and_classification": by_market_family_and_classification,
        "promotion_gate_review": {family: tier_gate(summary) for family, summary in by_market_family.items()},
        "promotion_gate_by_market_family_and_classification": {
            family: {classification: tier_gate(summary) for classification, summary in groups.items()}
            for family, groups in by_market_family_and_classification.items()
        },
        "explicit_settlement_families": [
            "FT_1X2",
            "1H_1X2",
            "2H_1X2",
            "FT_TOTALS",
            "1H_TOTALS",
            "2H_TOTALS",
            "FT_BTTS",
            "1H_BTTS",
            "2H_BTTS",
            "FT_TEAM_TOTAL",
            "1H_TEAM_TOTAL",
            "2H_TEAM_TOTAL",
        ],
        "minimum_sample_policy": {
            "directional_read": 20,
            "tier_b_candidate": 20,
            "tier_a_candidate": 50,
            "tier_s_candidate": 100,
        },
        "safety_note": "Team totals and 1H/2H goal/BTTS/1X2 derivatives now settle when the stored market metadata identifies period, side, line and selection. Corners/cards/player props still require explicit stat feeds before promotion.",
    }

    summary = {
        "schema_version": "1.2.0",
        "timezone_basis": "America/Mexico_City",
        "ledger_rows_read": len(rows),
        "final_fixture_count": len(finals),
        "evaluation_rows": len(evaluations),
        "canonical_ft_actionable_decisions": len(actionable),
        "settlement_decisions": len(settlement_decisions),
        "market_family_actionable_decisions": {k: len(v) for k, v in sorted(settlement_by_market.items())},
        "actionable_by_classification": {k: decision_summary(v) for k, v in sorted(by_class.items())},
        "actionable_by_market_family": market_summary["by_market_family"],
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
        "safety_note": "Canonical full-match Winner/1X2, Goals O/U and BTTS remain the only promoted ROI group. The settlement ledger now also settles explicit 1H/2H goal/BTTS/1X2 and team-total markets when metadata is sufficient.",
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        for ev in evaluations:
            fh.write(json.dumps(ev, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.makedirs(os.path.dirname(args.settlement_output), exist_ok=True)
    with open(args.settlement_output, "w", encoding="utf-8") as fh:
        for row in settlement_decisions:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    with open(args.summary_output, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    with open(args.market_summary_output, "w", encoding="utf-8") as fh:
        json.dump(market_summary, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
