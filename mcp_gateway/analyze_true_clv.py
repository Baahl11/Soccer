from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

CANONICAL = {
    "match winner", "winner", "goals over/under", "over/under",
    "goals over under", "both teams to score", "both teams score", "btts",
}


def fnum(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def line_from_selection(selection: Any) -> float | None:
    match = re.search(r"(?:over|under)\s*([0-9]+(?:\.[0-9]+)?)", str(selection or ""), re.I)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def canonical_ft_market(market: Any) -> bool:
    n = norm(market)
    if n not in CANONICAL:
        return False
    banned = (
        "first half", "second half", "1st half", "2nd half", "1h", "2h",
        "team total", "corner", "card", "player", "double chance", "handicap",
        "correct score", "both halves", "result/total", "winner &", "win and",
    )
    return not any(token in n for token in banned)


def same_line(a: float | None, b: float | None) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(a - b) < 1e-6


def snapshot_from_event(event: dict[str, Any]) -> dict[str, Any] | None:
    snap = event.get("closing_market_snapshot")
    if isinstance(snap, dict):
        return snap
    # Compatibility escape hatch if a compact persistence layer nests it under best_market.
    best = event.get("best_market")
    if isinstance(best, dict) and isinstance(best.get("closing_market_snapshot"), dict):
        return best["closing_market_snapshot"]
    return None


def group_fair_probability(
    group: dict[str, Any], selection: str, line: float | None
) -> tuple[float | None, float | None]:
    implied: list[tuple[dict[str, Any], float]] = []
    for value in group.get("values") or []:
        if not isinstance(value, dict):
            continue
        price = fnum(value.get("decimal_price") if value.get("decimal_price") is not None else value.get("price"))
        if price is None or price <= 1.0:
            continue
        implied.append((value, 1.0 / price))
    total = sum(prob for _, prob in implied)
    if total <= 0:
        return None, None
    for value, prob in implied:
        value_line = fnum(value.get("line"))
        if value_line is None:
            value_line = line_from_selection(value.get("selection"))
        if norm(value.get("selection")) == norm(selection) and same_line(value_line, line):
            price = fnum(value.get("decimal_price") if value.get("decimal_price") is not None else value.get("price"))
            return prob / total, price
    return None, None


def main() -> None:
    ap = argparse.ArgumentParser(description="Dedicated pre-kickoff true CLV analysis for canonical FT Soccer Edge markets.")
    ap.add_argument("--history-dir", default="soccer_edge_state/history")
    ap.add_argument("--output", default="soccer_edge_state/analysis/true_clv_tracking.jsonl")
    ap.add_argument("--summary-output", default="soccer_edge_state/analysis/true_clv_summary.json")
    args = ap.parse_args()

    ticks: list[dict[str, Any]] = []
    for path in sorted(glob.glob(os.path.join(args.history_dir, "*.jsonl"))):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    tick = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(tick, dict):
                    ticks.append(tick)
    ticks.sort(key=lambda t: str(t.get("generated_at_local") or ""))

    signals: list[dict[str, Any]] = []
    closes: dict[int, list[dict[str, Any]]] = defaultdict(list)
    dedicated_close_events = 0
    pre_kickoff_close_events = 0
    late_close_events = 0

    for tick in ticks:
        tick_ts = parse_dt(tick.get("generated_at_local"))
        for event in tick.get("events") or []:
            if not isinstance(event, dict):
                continue
            fixture = event.get("fixture") or {}
            fid = fixture.get("fixture_id")
            ko = parse_dt(fixture.get("kickoff"))
            if not fid or tick_ts is None or ko is None:
                continue

            if event.get("stage") == "CLOSE":
                snap = snapshot_from_event(event)
                if snap is not None:
                    dedicated_close_events += 1
                    record = {
                        "fixture_id": int(fid),
                        "timestamp": tick_ts,
                        "kickoff": ko,
                        "snapshot": snap,
                        "pre_kickoff": tick_ts < ko,
                    }
                    closes[int(fid)].append(record)
                    if tick_ts < ko:
                        pre_kickoff_close_events += 1
                    else:
                        late_close_events += 1

            best = event.get("best_market") or {}
            cls = str(event.get("classification") or "").upper()
            if cls not in {"BET", "LEAN", "WATCH"} or event.get("stage") not in {"T-40", "T-20", "T-10"}:
                continue
            if not isinstance(best, dict) or not canonical_ft_market(best.get("market")):
                continue
            price = fnum(best.get("decimal_price"))
            if price is None or price <= 1.0 or tick_ts >= ko:
                continue
            line = fnum(best.get("line"))
            if line is None:
                line = line_from_selection(best.get("selection"))
            signals.append({
                "fixture_id": int(fid),
                "league": fixture.get("league"),
                "home_team": fixture.get("home_team"),
                "away_team": fixture.get("away_team"),
                "kickoff_local": fixture.get("kickoff"),
                "signal_timestamp_local": tick.get("generated_at_local"),
                "signal_ts": tick_ts,
                "stage": event.get("stage"),
                "classification": cls,
                "tier": event.get("tier"),
                "market": best.get("market"),
                "selection": best.get("selection"),
                "line": line,
                "bookmaker": best.get("bookmaker"),
                "signal_price": price,
                "signal_fair_prob": fnum(best.get("p_market_fair")) or (1.0 / price),
            })

    tracked: list[dict[str, Any]] = []
    counts = Counter()
    same_book_count = 0
    consensus_count = 0

    for signal in signals:
        candidates = [c for c in closes.get(signal["fixture_id"], []) if c["pre_kickoff"] and c["timestamp"] >= signal["signal_ts"]]
        if not candidates:
            continue
        close = max(candidates, key=lambda c: c["timestamp"])
        groups = close["snapshot"].get("groups") or []

        exact_groups = [
            g for g in groups
            if isinstance(g, dict) and norm(g.get("market")) == norm(signal["market"])
        ]
        if not exact_groups:
            continue

        fair = price = None
        method = None
        same_book = [g for g in exact_groups if norm(g.get("bookmaker")) == norm(signal["bookmaker"])]
        for group in same_book:
            fair, price = group_fair_probability(group, signal["selection"], signal["line"])
            if fair is not None:
                method = "TRUE_PREKICKOFF_CLOSE_SAME_BOOK_DE_VIG"
                same_book_count += 1
                break

        if fair is None:
            fair_values: list[float] = []
            price_values: list[float] = []
            for group in exact_groups:
                gf, gp = group_fair_probability(group, signal["selection"], signal["line"])
                if gf is not None:
                    fair_values.append(gf)
                    if gp is not None:
                        price_values.append(gp)
            if fair_values:
                fair = statistics.median(fair_values)
                price = statistics.median(price_values) if price_values else None
                method = "TRUE_PREKICKOFF_CLOSE_CROSS_BOOK_MEDIAN_DE_VIG"
                consensus_count += 1

        if fair is None:
            continue

        signal_fair = float(signal["signal_fair_prob"])
        clv_pp = (fair - signal_fair) * 100.0
        direction = "POSITIVE" if clv_pp > 0.05 else "NEGATIVE" if clv_pp < -0.05 else "FLAT"
        counts[direction] += 1
        tracked.append({
            "schema_version": "1.0.0",
            "timezone_basis": "America/Mexico_City",
            "fixture_id": signal["fixture_id"],
            "league": signal["league"],
            "home_team": signal["home_team"],
            "away_team": signal["away_team"],
            "kickoff_local": signal["kickoff_local"],
            "stage": signal["stage"],
            "classification": signal["classification"],
            "tier": signal["tier"],
            "market": signal["market"],
            "selection": signal["selection"],
            "line": signal["line"],
            "bookmaker_at_signal": signal["bookmaker"],
            "signal_timestamp_local": signal["signal_timestamp_local"],
            "signal_price": round(signal["signal_price"], 6),
            "signal_fair_probability": round(signal_fair, 8),
            "close_timestamp_local": close["timestamp"].isoformat(),
            "close_price": round(price, 6) if price is not None else None,
            "close_fair_probability": round(fair, 8),
            "clv_probability_pp": round(clv_pp, 4),
            "clv_direction": direction,
            "is_true_closing_line": True,
            "closing_line_status": method,
        })

    clv_values = [row["clv_probability_pp"] for row in tracked]
    summary = {
        "schema_version": "1.0.0",
        "timezone_basis": "America/Mexico_City",
        "definition": "True CLV requires a dedicated canonical FT CLOSE snapshot captured strictly before kickoff. Closing fair probability is de-vigged; same-book matching is preferred, otherwise cross-book median fair probability is used.",
        "signal_rows_considered": len(signals),
        "dedicated_close_events": dedicated_close_events,
        "pre_kickoff_dedicated_close_events": pre_kickoff_close_events,
        "late_close_events_excluded_from_true_clv": late_close_events,
        "true_clv_rows": len(tracked),
        "same_book_true_clv_rows": same_book_count,
        "cross_book_consensus_true_clv_rows": consensus_count,
        "positive_clv": counts["POSITIVE"],
        "negative_clv": counts["NEGATIVE"],
        "flat_clv": counts["FLAT"],
        "positive_clv_rate": round(counts["POSITIVE"] / len(tracked), 4) if tracked else None,
        "avg_true_clv_probability_pp": round(sum(clv_values) / len(clv_values), 4) if clv_values else None,
        "status": "COLLECTING_DEDICATED_CLOSE_SNAPSHOTS" if len(tracked) < 50 else "ACTIVE_TRUE_CLV_SAMPLE",
        "minimum_sample_policy": {"directional_read": 50, "model_change": 200},
        "safety_note": "Research metric only. It cannot upgrade BET/LEAN classifications. Period, team-total and other derivatives are excluded.",
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        for row in tracked:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    with open(args.summary_output, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
