from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter
from typing import Any


def load_json(path: str) -> dict[str, Any] | None:
    if not os.path.exists(path):
        return None
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def market_snapshot_counts(history_dir: str) -> Counter:
    counts = Counter()
    quote_counts = Counter()
    for path in sorted(glob.glob(os.path.join(history_dir, "*.jsonl"))):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    tick = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for event in tick.get("events") or []:
                    if not isinstance(event, dict):
                        continue
                    snap = event.get("derivative_research_market_snapshot")
                    if not isinstance(snap, dict):
                        continue
                    for group in snap.get("groups") or []:
                        if not isinstance(group, dict):
                            continue
                        fam = str(group.get("family") or "UNKNOWN")
                        counts[fam] += 1
                        quote_counts[fam] += len(group.get("values") or [])
    counts["1H_GOALS_QUOTES"] = quote_counts["1H_GOALS"]
    counts["2H_GOALS_QUOTES"] = quote_counts["2H_GOALS"]
    return counts


def model_summary(report: dict[str, Any] | None, calibration: dict[str, Any] | None = None) -> dict[str, Any] | None:
    if not report:
        return None
    summary = {
        "model": report.get("model"),
        "explicit_period_model": report.get("explicit_period_model"),
        "reuses_ft_probability": report.get("reuses_ft_probability"),
        "walk_forward_evaluated": report.get("walk_forward_evaluated"),
        "metrics": report.get("metrics"),
        "promotion_gate": report.get("promotion_gate"),
    }
    if calibration:
        summary["calibration"] = {
            "method": calibration.get("method"),
            "walk_forward_evaluated": calibration.get("walk_forward_evaluated"),
            "baseline_same_sample": calibration.get("baseline_same_sample"),
            "challenger": calibration.get("challenger"),
            "improvement": calibration.get("improvement"),
            "promotion_gate": calibration.get("promotion_gate"),
        }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Refresh derivative readiness registry from explicit research models.")
    ap.add_argument("--history-dir", default="soccer_edge_state/history")
    ap.add_argument("--analysis-dir", default="soccer_edge_state/analysis")
    ap.add_argument("--output", default="soccer_edge_state/analysis/derivative_model_registry.json")
    args = ap.parse_args()

    market_counts = market_snapshot_counts(args.history_dir)
    one_h = load_json(os.path.join(args.analysis_dir, "one_h_goals_model.json"))
    one_h_cal = load_json(os.path.join(args.analysis_dir, "one_h_goals_calibration.json"))
    two_h = load_json(os.path.join(args.analysis_dir, "two_h_goals_model.json"))

    models: dict[str, Any] = {
        "1H_GOALS": {
            "status": "RESEARCH_MODEL_ACTIVE_NOT_ACTIONABLE" if one_h else "BLOCKED_PENDING_EXPLICIT_MODEL",
            "actionable": False,
            "market_groups_persisted": int(market_counts["1H_GOALS"]),
            "market_quotes_persisted": int(market_counts["1H_GOALS_QUOTES"]),
            "research_model": model_summary(one_h, one_h_cal),
            "requirements_remaining": [
                "dedicated 1H market-vs-model calibration",
                "stable OOS Brier/log-loss improvement",
                "minimum 200 OOS observations before actionable review",
                "availability/XI gates remain mandatory if ever activated",
            ] if one_h else ["period-specific target", "1H feature set", "out-of-sample calibration"],
        },
        "2H_GOALS": {
            "status": "RESEARCH_MODEL_ACTIVE_NOT_ACTIONABLE" if two_h else "BLOCKED_PENDING_EXPLICIT_MODEL",
            "actionable": False,
            "market_groups_persisted": int(market_counts["2H_GOALS"]),
            "market_quotes_persisted": int(market_counts["2H_GOALS_QUOTES"]),
            "research_model": model_summary(two_h),
            "requirements_remaining": [
                "dedicated pregame 2H market-vs-model calibration",
                "stable OOS validation",
                "separate live halftime-conditioned model if live 2H betting is ever desired",
                "minimum 200 OOS observations before actionable review",
            ] if two_h else ["period-specific target", "2H state/context features", "out-of-sample calibration"],
        },
        "TEAM_TOTALS": {
            "status": "BLOCKED_PENDING_EXPLICIT_MODEL", "actionable": False,
            "requirements_remaining": ["team-specific scoring distribution", "opponent defensive model", "market-line calibration"],
        },
        "CORNERS": {
            "status": "BLOCKED_PENDING_EXPLICIT_MODEL", "actionable": False,
            "requirements_remaining": ["corner event history", "cross/shot pressure features", "team/league calibration"],
        },
        "CARDS": {
            "status": "BLOCKED_PENDING_EXPLICIT_MODEL", "actionable": False,
            "requirements_remaining": ["referee/card history", "foul/tackle context", "league calibration"],
        },
        "PLAYER_SHOTS": {
            "status": "BLOCKED_PENDING_EXPLICIT_MODEL", "actionable": False,
            "requirements_remaining": ["verified starters/minutes", "player shot volume", "opponent matchup", "lineup gate"],
        },
    }

    output = {
        "schema_version": "2.0.0",
        "policy": "NEVER_REUSE_FT_PROBABILITIES_FOR_DERIVATIVES",
        "models": models,
        "note": "Explicit period research models may exist while remaining non-actionable. A research model does not become BET/LEAN eligible until its dedicated market calibration and OOS gates pass.",
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
