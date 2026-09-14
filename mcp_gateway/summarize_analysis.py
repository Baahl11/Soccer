from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {"n": len(rows), "graded": 0, "win": 0, "loss": 0, "push": 0, "no_final": 0, "ungradable": 0,
           "hit_rate_ex_push": None, "flat_profit_units": 0.0, "flat_roi": None, "priced_graded": 0}
    risked = 0.0
    profit = 0.0
    for r in rows:
        o = r.get("outcome")
        if o in {"WIN", "LOSS", "PUSH"}:
            out["graded"] += 1
            out[o.lower()] += 1
            p = r.get("price")
            if isinstance(p, (int, float)) and p > 1:
                out["priced_graded"] += 1
                risked += 1.0
                if o == "WIN": profit += p - 1.0
                elif o == "LOSS": profit -= 1.0
        elif o == "NO_FINAL": out["no_final"] += 1
        else: out["ungradable"] += 1
    decided = out["win"] + out["loss"]
    out["hit_rate_ex_push"] = round(out["win"] / decided, 4) if decided else None
    out["flat_profit_units"] = round(profit, 3)
    out["flat_roi"] = round(profit / risked, 4) if risked else None
    return out


def compact(r: dict[str, Any]) -> dict[str, Any]:
    fx = r.get("fixture") or {}
    return {
        "fixture_id": r.get("fixture_id"), "league": fx.get("league"),
        "matchup": f"{fx.get('home_team')} vs {fx.get('away_team')}", "kickoff": fx.get("kickoff"),
        "classification": r.get("classification"), "tier": r.get("tier"), "stage": r.get("stage"),
        "market": r.get("market"), "selection": r.get("selection"), "line": r.get("line"),
        "price": r.get("price"), "bookmaker": r.get("bookmaker"), "edge_pp": r.get("edge_pp"),
        "availability_confidence": r.get("availability_confidence"), "version": r.get("version"),
        "outcome": r.get("outcome"), "canonical_ft": r.get("canonical_ft"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    data = json.load(open(args.input, encoding="utf-8"))
    rows = data.get("market_call_rows") or []
    canonical = [r for r in rows if r.get("canonical_ft")]
    legacy = [r for r in rows if not r.get("canonical_ft")]

    class_market: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    version_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in canonical:
        cls = str(r.get("classification") or "NONE")
        market = str(r.get("market") or r.get("family") or "UNKNOWN")
        class_market[cls][market].append(r)
        version_groups[str(r.get("version") or "unknown")].append(r)

    out = {
        "generated_at": data.get("generated_at"), "period": data.get("period"),
        "overall_canonical": summarize(canonical), "legacy_invalid": summarize(legacy),
        "by_classification_market": {
            cls: {m: summarize(rs) for m, rs in sorted(markets.items())}
            for cls, markets in sorted(class_market.items())
        },
        "by_version": {v: summarize(rs) for v, rs in sorted(version_groups.items())},
        "bets": [compact(r) for r in canonical if r.get("classification") == "BET"],
        "leans": [compact(r) for r in canonical if r.get("classification") == "LEAN"],
        "watches": [compact(r) for r in canonical if r.get("classification") == "WATCH"],
        "legacy_graded_examples": [compact(r) for r in legacy if r.get("outcome") in {"WIN","LOSS","PUSH"}],
        "sporting_signal_diagnostics": data.get("sporting_signal_diagnostics"),
        "score_bin_diagnostics": data.get("score_bin_diagnostics"),
        "generic_watch_count": data.get("generic_watch_count"),
        "generic_watch_reasons": data.get("generic_watch_reasons"),
        "counts": {
            "unique_market_calls": len(rows), "canonical_ft": len(canonical), "legacy_invalid_model": len(legacy),
            "unique_fixtures_with_shortlist": data.get("unique_fixtures_with_shortlist"),
            "final_results_found": data.get("final_results_found"),
        },
    }
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
