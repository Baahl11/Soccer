from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from datetime import datetime
from typing import Any, Iterable

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_OOS_BACKTEST_V4_1.0.0"
ROLLING_WINDOWS = (20, 50, 100)


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _parse_dt(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def chronological_splits(
    rows: Iterable[dict[str, Any]],
    *,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    timestamp_key: str = "generated_at_local",
) -> dict[str, list[dict[str, Any]]]:
    source = [row for row in rows if isinstance(row, dict) and _parse_dt(row.get(timestamp_key)) is not None]
    source.sort(key=lambda row: _parse_dt(row.get(timestamp_key)))
    n = len(source)
    train_end = int(n * max(min(train_fraction, 1.0), 0.0))
    validation_end = train_end + int(n * max(min(validation_fraction, 1.0 - train_fraction), 0.0))
    return {
        "train": source[:train_end],
        "validation": source[train_end:validation_end],
        "test": source[validation_end:],
    }


def settlement_metrics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    settled = [
        row for row in rows
        if isinstance(row, dict)
        and bool(row.get("settled"))
        and str(row.get("settlement_status") or "").upper() in {"WIN", "LOSS", "PUSH", "HALF_WIN", "HALF_LOSS", "HALF_WIN_HALF_LOSS"}
    ]
    settled.sort(key=lambda row: _parse_dt(row.get("generated_at_local")) or datetime.min)

    returns: list[float] = []
    stake_sum = 0.0
    wins = losses = pushes = 0
    longest_losing_streak = current_losing_streak = 0
    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0

    for row in settled:
        value = _num(row.get("roi_units"))
        if value is None:
            continue
        stake = _num(row.get("stake_units"))
        if stake is not None and stake > 0:
            stake_sum += stake
        returns.append(value)
        status = str(row.get("settlement_status") or "").upper()
        if status in {"WIN", "HALF_WIN"}:
            wins += 1
            current_losing_streak = 0
        elif status in {"LOSS", "HALF_LOSS"}:
            losses += 1
            current_losing_streak += 1
            longest_losing_streak = max(longest_losing_streak, current_losing_streak)
        else:
            pushes += 1
            current_losing_streak = 0

        cumulative += value
        peak = max(peak, cumulative)
        max_drawdown = max(max_drawdown, peak - cumulative)

    decided = wins + losses
    roi_units = sum(returns)
    return {
        "settled_rows": len(returns),
        "wins": wins,
        "losses": losses,
        "push_like": pushes,
        "hit_rate_ex_push": round(wins / decided, 6) if decided else None,
        "roi_units": round(roi_units, 6),
        "stake_units": round(stake_sum, 6),
        "roi_per_staked_unit": round(roi_units / stake_sum, 6) if stake_sum > 0 else None,
        "max_drawdown_units": round(max_drawdown, 6),
        "max_losing_streak": longest_losing_streak,
        "return_volatility_stddev": round(statistics.pstdev(returns), 6) if len(returns) >= 2 else 0.0 if returns else None,
    }


def rolling_settlement_metrics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    settled = [
        row for row in rows
        if isinstance(row, dict)
        and bool(row.get("settled"))
        and _num(row.get("roi_units")) is not None
    ]
    settled.sort(key=lambda row: _parse_dt(row.get("generated_at_local")) or datetime.min)
    out: dict[str, Any] = {}
    for window in ROLLING_WINDOWS:
        if len(settled) < window:
            out[str(window)] = {
                "available": False,
                "reason": f"SETTLED_{len(settled)}_LT_WINDOW_{window}",
            }
            continue
        out[str(window)] = {
            "available": True,
            "metrics": settlement_metrics(settled[-window:]),
        }
    return out


def timestamp_discipline(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    source = [row for row in rows if isinstance(row, dict)]
    decision_timestamp_rows = 0
    pre_kickoff_rows = 0
    bookmaker_timestamp_rows = 0
    lineup_timestamp_rows = 0
    feature_timestamp_rows = 0

    for row in source:
        generated = _parse_dt(row.get("generated_at_local") or row.get("prediction_timestamp"))
        kickoff = _parse_dt(row.get("kickoff_local"))
        if generated is not None:
            decision_timestamp_rows += 1
            if kickoff is not None and generated <= kickoff:
                pre_kickoff_rows += 1
        if _parse_dt(row.get("bookmaker_timestamp") or row.get("quote_timestamp")) is not None:
            bookmaker_timestamp_rows += 1
        if _parse_dt(row.get("lineup_timestamp") or row.get("lineup_captured_at")) is not None:
            lineup_timestamp_rows += 1
        if _parse_dt(row.get("feature_timestamp") or row.get("feature_captured_at")) is not None:
            feature_timestamp_rows += 1

    n = len(source)
    return {
        "rows": n,
        "decision_timestamp_coverage": round(decision_timestamp_rows / n, 6) if n else 0.0,
        "pre_kickoff_decision_rate": round(pre_kickoff_rows / decision_timestamp_rows, 6) if decision_timestamp_rows else None,
        "bookmaker_timestamp_coverage": round(bookmaker_timestamp_rows / n, 6) if n else 0.0,
        "lineup_timestamp_coverage": round(lineup_timestamp_rows / n, 6) if n else 0.0,
        "feature_timestamp_coverage": round(feature_timestamp_rows / n, 6) if n else 0.0,
    }


def build_report(
    settlement_rows: Iterable[dict[str, Any]],
    clv_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = [row for row in settlement_rows if isinstance(row, dict)]
    splits = chronological_splits(rows)
    discipline = timestamp_discipline(rows)
    realized = settlement_metrics(rows)
    rolling = rolling_settlement_metrics(rows)
    clv_report = clv_report if isinstance(clv_report, dict) else {}

    split_summary: dict[str, Any] = {}
    for name, partition in splits.items():
        timestamps = [_parse_dt(row.get("generated_at_local")) for row in partition]
        timestamps = [value for value in timestamps if value is not None]
        split_summary[name] = {
            "rows": len(partition),
            "start": min(timestamps).isoformat() if timestamps else None,
            "end": max(timestamps).isoformat() if timestamps else None,
        }

    blockers: list[str] = []
    warnings: list[str] = []
    if discipline["bookmaker_timestamp_coverage"] < 1.0:
        blockers.append("BOOKMAKER_TIMESTAMP_DISCIPLINE_INCOMPLETE")
    if discipline["lineup_timestamp_coverage"] < 1.0:
        blockers.append("LINEUP_TIMESTAMP_DISCIPLINE_INCOMPLETE")
    if discipline["feature_timestamp_coverage"] < 1.0:
        blockers.append("FEATURE_TIMESTAMP_DISCIPLINE_INCOMPLETE")
    if discipline["pre_kickoff_decision_rate"] is not None and discipline["pre_kickoff_decision_rate"] < 1.0:
        blockers.append("POST_KICKOFF_DECISION_ROWS_DETECTED")
    if realized["settled_rows"] < 50:
        blockers.append(f"SETTLED_{realized['settled_rows']}_LT_50")
    if clv_report.get("status") != "CLV_ANALYSIS_AVAILABLE":
        blockers.append("CLV_ENGINE_NOT_COMPLETE")

    if len(splits["test"]) < 10:
        warnings.append("CHRONOLOGICAL_TEST_PARTITION_SMALL")

    metric_availability = {
        "brier": False,
        "log_loss": False,
        "calibration": False,
        "mae": False,
        "rmse": False,
        "hit_rate": realized["hit_rate_ex_push"] is not None,
        "roi": realized["roi_per_staked_unit"] is not None,
        "clv": (clv_report.get("overall") or {}).get("avg_probability_clv_pp") is not None,
        "drawdown": realized["settled_rows"] > 0,
        "max_losing_streak": realized["settled_rows"] > 0,
        "volatility": realized["return_volatility_stddev"] is not None,
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "phase": "FASE_18_OOS_BACKTEST",
        "status": "OOS_FRAMEWORK_READY_FOR_MODEL_PREDICTIONS" if not blockers else "OOS_FRAMEWORK_DATA_GAPS",
        "production_promotion_allowed": False,
        "manual_review_required": True,
        "provider_requests_added": 0,
        "model_weights_changed": False,
        "canonical_bet_logic_changed": False,
        "framework": {
            "chronological_splits": True,
            "rolling_windows": list(ROLLING_WINDOWS),
            "walk_forward_validation_supported": True,
            "no_training_performance_promotion": True,
            "no_leakage_required": True,
        },
        "chronological_split_preview": split_summary,
        "timestamp_discipline": discipline,
        "realized_settlement_metrics": realized,
        "rolling_settlement_metrics": rolling,
        "metric_availability_on_current_settlement_ledger": metric_availability,
        "clv_context": {
            "status": clv_report.get("status"),
            "rows": clv_report.get("rows"),
            "true_closing_line_rows": clv_report.get("true_closing_line_rows"),
            "overall": clv_report.get("overall"),
        },
        "blockers": blockers,
        "warnings": warnings,
        "notes": [
            "The settlement ledger can measure realized hit rate, ROI, drawdown, losing streak and volatility.",
            "Brier/log-loss/calibration/MAE/RMSE require point-in-time model predictions joined to final outcomes; settlement ROI alone cannot substitute for those metrics.",
            "Chronological split preview is diagnostic only. Production model evaluation must use frozen prediction snapshots and walk-forward/rolling retraining without leakage.",
            "Bookmaker, lineup and feature timestamps must be explicit per prediction row before production promotion.",
        ],
    }


def _load_json(path: str) -> dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path or not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 18 OOS/backtest framework report.")
    parser.add_argument("--settlement-ledger", required=True)
    parser.add_argument("--clv-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build_report(_load_jsonl(args.settlement_ledger), _load_json(args.clv_report))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
