from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

CANONICAL = {"match winner", "winner", "goals over/under", "over/under", "both teams to score", "btts"}


def fnum(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def norm(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "").strip()).lower()


def parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def period_type(market: str) -> str:
    n = norm(market)
    if any(s in n for s in ("first half", "1st half", "1h", "half time", "halftime")):
        return "1H"
    if any(s in n for s in ("second half", "2nd half", "2h")):
        return "2H"
    return "FT"


def canonical_ft(m: dict[str, Any]) -> bool:
    market = norm(m.get("market"))
    if period_type(market) != "FT":
        return False
    banned = ("team total", "corners", "cards", "player", "double chance", "draw no bet", "asian", "handicap", "correct score", "odd/even", "both halves", "result/total", "winner &", "win and")
    return market in CANONICAL and not any(x in market for x in banned)


def market_key(row: dict[str, Any]) -> tuple:
    m = row.get("best_market") or {}
    line = fnum(m.get("line"))
    return (
        int(row["fixture_id"]),
        norm(m.get("family")),
        norm(m.get("market")),
        norm(m.get("selection")),
        round(line, 4) if line is not None else None,
    )


def implied(price: float | None) -> float | None:
    if price is None or price <= 1.0:
        return None
    return 1.0 / price


def main() -> None:
    ap = argparse.ArgumentParser(description="Track observed Soccer Edge price movement and CLV proxies from the historical ledger.")
    ap.add_argument("--ledger", default="soccer_edge_state/analysis/signal_ledger.jsonl")
    ap.add_argument("--output", default="soccer_edge_state/analysis/clv_tracking.jsonl")
    ap.add_argument("--summary-output", default="soccer_edge_state/analysis/clv_summary.json")
    args = ap.parse_args()

    rows: list[dict[str, Any]] = []
    with open(args.ledger, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            m = row.get("best_market") or {}
            price = fnum(m.get("decimal_price"))
            if not m or price is None or price <= 1.0 or not canonical_ft(m):
                continue
            ts = parse_dt(row.get("generated_at_local"))
            ko = parse_dt(row.get("kickoff_local"))
            if ts is None or ko is None or ts >= ko:
                continue
            row["_ts"] = ts
            row["_ko"] = ko
            rows.append(row)

    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[market_key(row)].append(row)

    out: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    positive = negative = flat = 0
    for key, obs in groups.items():
        obs.sort(key=lambda r: r["_ts"])
        close = obs[-1]
        close_m = close["best_market"]
        close_price = fnum(close_m.get("decimal_price"))
        close_imp = implied(close_price)
        for entry in obs:
            cls = str(entry.get("classification") or "").upper()
            if cls not in {"BET", "LEAN", "WATCH"}:
                continue
            em = entry["best_market"]
            entry_price = fnum(em.get("decimal_price"))
            entry_imp = implied(entry_price)
            if entry_price is None or close_price is None or entry_imp is None or close_imp is None:
                continue
            # Positive when the same selection became shorter by close.
            clv_prob_pp = (close_imp - entry_imp) * 100.0
            clv_price_pct = ((entry_price / close_price) - 1.0) * 100.0
            if clv_prob_pp > 0.05:
                direction = "POSITIVE"
                positive += 1
            elif clv_prob_pp < -0.05:
                direction = "NEGATIVE"
                negative += 1
            else:
                direction = "FLAT"
                flat += 1
            class_counts[cls] += 1
            out.append({
                "schema_version": "1.0.0",
                "timezone_basis": "America/Mexico_City",
                "fixture_id": entry.get("fixture_id"),
                "league": entry.get("league"),
                "home_team": entry.get("home_team"),
                "away_team": entry.get("away_team"),
                "kickoff_local": entry.get("kickoff_local"),
                "stage": entry.get("stage"),
                "classification": cls,
                "tier": entry.get("tier"),
                "market": em.get("market"),
                "selection": em.get("selection"),
                "line": fnum(em.get("line")),
                "bookmaker_at_signal": em.get("bookmaker"),
                "signal_timestamp_local": entry.get("generated_at_local"),
                "signal_price": round(entry_price, 6),
                "signal_implied_prob": round(entry_imp, 8),
                "last_observed_pre_kickoff_timestamp_local": close.get("generated_at_local"),
                "last_observed_pre_kickoff_stage": close.get("stage"),
                "last_observed_pre_kickoff_price": round(close_price, 6),
                "last_observed_pre_kickoff_implied_prob": round(close_imp, 8),
                "clv_probability_pp": round(clv_prob_pp, 4),
                "clv_price_pct": round(clv_price_pct, 4),
                "clv_direction": direction,
                "is_true_closing_line": False,
                "closing_line_status": "LAST_OBSERVED_PRE_KICKOFF_PROXY",
                "result": entry.get("result"),
            })

    out.sort(key=lambda r: (r.get("signal_timestamp_local") or "", r["fixture_id"], r.get("market") or ""))
    clvs = [r["clv_probability_pp"] for r in out]
    summary = {
        "schema_version": "1.0.0",
        "timezone_basis": "America/Mexico_City",
        "definition": "CLV proxy compares signal price with the last observed pre-kickoff price for the exact same canonical FT market/selection/line. It is not labeled a true bookmaker closing line unless a dedicated close snapshot exists.",
        "ledger_market_observations": len(rows),
        "market_series": len(groups),
        "tracked_signal_rows": len(out),
        "classification_counts": dict(sorted(class_counts.items())),
        "positive_clv": positive,
        "negative_clv": negative,
        "flat_clv": flat,
        "positive_clv_rate": round(positive / len(out), 4) if out else None,
        "avg_clv_probability_pp": round(sum(clvs) / len(clvs), 4) if clvs else None,
        "true_closing_lines": 0,
        "proxy_closing_lines": len(out),
        "safety_note": "Only canonical FT Winner/1X2, Goals O/U and BTTS observations are tracked. Derivatives are excluded. CLV is a research metric and does not upgrade backend classifications.",
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        for row in out:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    with open(args.summary_output, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
