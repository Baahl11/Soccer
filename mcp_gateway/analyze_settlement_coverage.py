from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from typing import Any, Iterable

ACTIONABLE_CLASSES = {"BET", "LEAN"}
SETTLED_OUTCOMES = {"WIN", "LOSS", "PUSH"}


def norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def fnum(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_jsonl(path: str) -> list[dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")


def write_jsonl(path: str, rows: Iterable[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def period_type(market: Any) -> str:
    m = norm(market)
    if any(token in m for token in ("first half", "1st half", "1h", "half time", "halftime")):
        return "1H"
    if any(token in m for token in ("second half", "2nd half", "2h")):
        return "2H"
    return "FT"


def market_family(best: dict[str, Any] | None) -> str:
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


def best_market(row: dict[str, Any]) -> dict[str, Any] | None:
    best = row.get("best_market")
    return best if isinstance(best, dict) and best else None


def decision_key_from_best(row: dict[str, Any], best: dict[str, Any] | None = None) -> tuple[Any, ...] | None:
    best = best or best_market(row)
    if not best:
        return None
    return (
        row.get("fixture_id"),
        str(row.get("classification") or "").upper(),
        market_family(best),
        norm(best.get("market")),
        norm(best.get("selection")),
        fnum(best.get("line")),
    )


def decision_key_from_settlement(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("fixture_id"),
        str(row.get("classification") or "").upper(),
        row.get("market_family") or "NO_MARKET",
        norm(row.get("market")),
        norm(row.get("selection")),
        fnum(row.get("line")),
    )


def final_fixture_ids(rows: Iterable[dict[str, Any]]) -> set[Any]:
    return {row.get("fixture_id") for row in rows if row.get("fixture_id") is not None and row.get("result")}


def eval_by_event_key(rows: Iterable[dict[str, Any]]) -> dict[Any, dict[str, Any]]:
    out: dict[Any, dict[str, Any]] = {}
    for row in rows:
        key = row.get("event_key")
        if key:
            out[key] = row
    return out


def latest_source_by_decision(rows: Iterable[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    latest: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        if str(row.get("classification") or "").upper() not in ACTIONABLE_CLASSES:
            continue
        key = decision_key_from_best(row)
        if key is None:
            continue
        prev = latest.get(key)
        if prev is None or str(row.get("generated_at_local") or "") > str(prev.get("generated_at_local") or ""):
            latest[key] = row
    return latest


def reason_for_row(
    row: dict[str, Any],
    *,
    finals: set[Any],
    settled_keys: set[tuple[Any, ...]],
    latest_by_key: dict[tuple[Any, ...], dict[str, Any]],
    evals: dict[Any, dict[str, Any]],
) -> tuple[str, str]:
    best = best_market(row)
    fid = row.get("fixture_id")
    has_final = fid in finals or bool(row.get("result"))
    if not best:
        if has_final:
            return "MISSING_BEST_MARKET_WITH_FINAL", "store exact market/selection/line/price before classifying as BET or LEAN"
        return "PENDING_FINAL_AND_MISSING_MARKET", "wait for result and store exact market metadata"
    key = decision_key_from_best(row, best)
    if not has_final:
        return "PENDING_FINAL_RESULT", "wait for fixture final/postgame result"
    if key in settled_keys:
        latest = latest_by_key.get(key)
        if latest and latest.get("event_key") != row.get("event_key"):
            return "DUPLICATE_OLDER_SNAPSHOT", "no action; latest snapshot for this same decision is already settled"
        return "IN_SETTLEMENT_LEDGER", "already settled"
    ev = evals.get(row.get("event_key"))
    outcome = ev.get("market_outcome") if ev else None
    if outcome in SETTLED_OUTCOMES:
        return "FINAL_WITH_MARKET_NOT_IN_SETTLEMENT_LEDGER", "investigate settlement key mismatch or classification capture"
    if outcome:
        return f"UNSETTLED_{outcome}", "add/repair grading metadata or settlement logic if this market should be supported"
    return "FINAL_WITH_MARKET_NOT_EVALUATED", "investigate missing postgame evaluation row"


def build_report(
    source_rows: list[dict[str, Any]],
    evaluation_rows: list[dict[str, Any]],
    settlement_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    actionable = [row for row in source_rows if str(row.get("classification") or "").upper() in ACTIONABLE_CLASSES]
    finals = final_fixture_ids(source_rows)
    settled_keys = {decision_key_from_settlement(row) for row in settlement_rows}
    evals = eval_by_event_key(evaluation_rows)
    latest_by_key = latest_source_by_decision(actionable)

    by_class: dict[str, Counter[str]] = defaultdict(Counter)
    by_reason: Counter[str] = Counter()
    by_market_reason: dict[str, Counter[str]] = defaultdict(Counter)
    backlog: list[dict[str, Any]] = []

    for row in actionable:
        cls = str(row.get("classification") or "").upper()
        best = best_market(row)
        family = market_family(best)
        reason, action = reason_for_row(row, finals=finals, settled_keys=settled_keys, latest_by_key=latest_by_key, evals=evals)
        by_class[cls]["source_rows"] += 1
        by_class[cls][reason] += 1
        by_reason[reason] += 1
        by_market_reason[family][reason] += 1
        if row.get("fixture_id") in finals or row.get("result"):
            by_class[cls]["rows_with_final"] += 1
        if best:
            by_class[cls]["rows_with_market"] += 1
        if best and (row.get("fixture_id") in finals or row.get("result")):
            by_class[cls]["rows_with_final_and_market"] += 1
        if reason != "IN_SETTLEMENT_LEDGER":
            backlog.append({
                "schema_version": "1.0.0",
                "event_key": row.get("event_key"),
                "fixture_id": row.get("fixture_id"),
                "classification": cls,
                "stage": row.get("stage"),
                "generated_at_local": row.get("generated_at_local"),
                "kickoff_local": row.get("kickoff_local"),
                "league": row.get("league"),
                "home_team": row.get("home_team"),
                "away_team": row.get("away_team"),
                "market_family": family,
                "market": (best or {}).get("market"),
                "selection": (best or {}).get("selection"),
                "line": fnum((best or {}).get("line")),
                "decimal_price": fnum((best or {}).get("decimal_price")),
                "coverage_reason": reason,
                "next_action": action,
            })

    class_summary: dict[str, dict[str, Any]] = {}
    for cls, counts in sorted(by_class.items()):
        source_count = counts.get("source_rows", 0)
        in_settlement = counts.get("IN_SETTLEMENT_LEDGER", 0)
        class_summary[cls] = {
            "source_rows": source_count,
            "settlement_rows": in_settlement,
            "settlement_coverage_rate": round(in_settlement / source_count, 4) if source_count else None,
            "rows_with_final": counts.get("rows_with_final", 0),
            "rows_with_market": counts.get("rows_with_market", 0),
            "rows_with_final_and_market": counts.get("rows_with_final_and_market", 0),
            "reason_counts": {k: v for k, v in sorted(counts.items()) if k not in {"source_rows", "rows_with_final", "rows_with_market", "rows_with_final_and_market"}},
        }

    report = {
        "schema_version": "1.0.0",
        "status": "SETTLEMENT_COVERAGE_AUDIT",
        "policy": {
            "purpose": "Increase settled sample by showing why BET/LEAN source rows are not yet in the settlement ledger.",
            "no_runtime_changes": True,
            "no_tier_or_stake_changes": True,
            "promotion_still_requires": ["20+ settled for Tier B review", "positive ROI", "CLV review", "stage/league/odds-band stability"],
        },
        "source_actionable_rows": len(actionable),
        "settlement_rows": len(settlement_rows),
        "settlement_coverage_rate": round(len(settlement_rows) / len(actionable), 4) if actionable else None,
        "backlog_rows": len(backlog),
        "by_classification": class_summary,
        "by_reason": dict(sorted(by_reason.items())),
        "by_market_family_reason": {family: dict(sorted(counter.items())) for family, counter in sorted(by_market_reason.items())},
        "recommended_next_actions": recommended_actions(by_reason),
    }
    return report, backlog


def recommended_actions(by_reason: Counter[str]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    if by_reason.get("PENDING_FINAL_RESULT"):
        actions.append({"reason": "PENDING_FINAL_RESULT", "count": by_reason["PENDING_FINAL_RESULT"], "action": "No code change; these should enter settlement when final results arrive."})
    missing_market = by_reason.get("MISSING_BEST_MARKET_WITH_FINAL", 0) + by_reason.get("PENDING_FINAL_AND_MISSING_MARKET", 0)
    if missing_market:
        actions.append({"reason": "MISSING_BEST_MARKET", "count": missing_market, "action": "Require exact market/selection/line/price for any future BET/LEAN classification, or downgrade to WATCH/RESEARCH."})
    duplicate = by_reason.get("DUPLICATE_OLDER_SNAPSHOT", 0)
    if duplicate:
        actions.append({"reason": "DUPLICATE_OLDER_SNAPSHOT", "count": duplicate, "action": "No action; dedupe keeps the latest decision snapshot."})
    ungraded = sum(count for reason, count in by_reason.items() if reason.startswith("UNSETTLED_") or reason == "FINAL_WITH_MARKET_NOT_EVALUATED" or reason == "FINAL_WITH_MARKET_NOT_IN_SETTLEMENT_LEDGER")
    if ungraded:
        actions.append({"reason": "UNSETTLED_OR_NOT_EVALUATED", "count": ungraded, "action": "Add specific settlement metadata/logic only for markets that are real BET/LEAN candidates."})
    return actions


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit BET/LEAN rows that are not yet represented in the settlement ledger.")
    parser.add_argument("--ledger", default="soccer_edge_state/analysis/signal_ledger.jsonl")
    parser.add_argument("--postgame-evaluation", default="soccer_edge_state/analysis/postgame_evaluation.jsonl")
    parser.add_argument("--settlement-ledger", default="soccer_edge_state/analysis/bet_settlement_ledger.jsonl")
    parser.add_argument("--output", default="soccer_edge_state/analysis/settlement_coverage_report.json")
    parser.add_argument("--backlog-output", default="soccer_edge_state/analysis/settlement_backlog.jsonl")
    args = parser.parse_args()

    source_rows = load_jsonl(args.ledger)
    evaluation_rows = load_jsonl(args.postgame_evaluation)
    settlement_rows = load_jsonl(args.settlement_ledger)
    report, backlog = build_report(source_rows, evaluation_rows, settlement_rows)
    write_json(args.output, report)
    write_jsonl(args.backlog_output, backlog)
    print(json.dumps({"status": report["status"], "source_actionable_rows": report["source_actionable_rows"], "settlement_rows": report["settlement_rows"], "backlog_rows": report["backlog_rows"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
