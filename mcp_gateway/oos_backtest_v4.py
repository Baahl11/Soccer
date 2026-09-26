from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from collections import Counter
from datetime import datetime
from typing import Any, Iterable

CLV_COMPLETE_STATUSES = {
    "CLV_ANALYSIS_AVAILABLE",
    "CLV_CAPTURE_COMPLETE_ANALYSIS_AVAILABLE",
    "ACTIVE_TRUE_CLV_SAMPLE",
}

SCHEMA_VERSION = "1.2.0"
MODEL_VERSION = "SOCCER_OOS_BACKTEST_V4_1.2.0"
ROLLING_WINDOWS = (20, 50, 100)
MODERN_SETTLEMENT_SOURCES = {"POSTGRES_REFRESH_EVENT"}


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


def _timestamp_sort_key(value: Any) -> tuple[int, float]:
    parsed = _parse_dt(value)
    if parsed is None:
        return (1, 0.0)
    try:
        return (0, parsed.timestamp())
    except (OSError, OverflowError, ValueError):
        return (1, 0.0)


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
    settled.sort(key=lambda row: _timestamp_sort_key(row.get("generated_at_local")))

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
    settled.sort(key=lambda row: _timestamp_sort_key(row.get("generated_at_local")))
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

    modern = [
        row for row in source
        if str(row.get("source") or "").upper() in MODERN_SETTLEMENT_SOURCES
    ]
    legacy = [row for row in source if row not in modern]

    def _bookmaker_ts(row: dict[str, Any]) -> datetime | None:
        return _parse_dt(row.get("bookmaker_timestamp") or row.get("quote_timestamp"))

    def _lineup_ts(row: dict[str, Any]) -> datetime | None:
        return _parse_dt(
            row.get("lineup_timestamp")
            or row.get("lineup_captured_at")
            or row.get("lineup_observation_timestamp")
        )

    def _feature_ts(row: dict[str, Any]) -> datetime | None:
        return _parse_dt(row.get("feature_timestamp") or row.get("feature_captured_at"))

    for row in source:
        generated = _parse_dt(row.get("generated_at_local") or row.get("prediction_timestamp"))
        kickoff = _parse_dt(row.get("kickoff_local"))
        if generated is not None:
            decision_timestamp_rows += 1
            if kickoff is not None and generated <= kickoff:
                pre_kickoff_rows += 1
        if _bookmaker_ts(row) is not None:
            bookmaker_timestamp_rows += 1
        if _lineup_ts(row) is not None:
            lineup_timestamp_rows += 1
        if _feature_ts(row) is not None:
            feature_timestamp_rows += 1

    modern_bookmaker = sum(1 for row in modern if _bookmaker_ts(row) is not None)
    modern_lineup = sum(1 for row in modern if _lineup_ts(row) is not None)
    modern_feature = sum(1 for row in modern if _feature_ts(row) is not None)

    n = len(source)
    modern_n = len(modern)
    legacy_n = len(legacy)
    return {
        "rows": n,
        "decision_timestamp_coverage": round(decision_timestamp_rows / n, 6) if n else 0.0,
        "pre_kickoff_decision_rate": round(pre_kickoff_rows / decision_timestamp_rows, 6) if decision_timestamp_rows else None,
        "bookmaker_timestamp_coverage": round(bookmaker_timestamp_rows / n, 6) if n else 0.0,
        "lineup_timestamp_coverage": round(lineup_timestamp_rows / n, 6) if n else 0.0,
        "feature_timestamp_coverage": round(feature_timestamp_rows / n, 6) if n else 0.0,
        "modern_rows": modern_n,
        "legacy_rows": legacy_n,
        "modern_bookmaker_timestamp_coverage": round(modern_bookmaker / modern_n, 6) if modern_n else None,
        "modern_lineup_timestamp_coverage": round(modern_lineup / modern_n, 6) if modern_n else None,
        "modern_feature_timestamp_coverage": round(modern_feature / modern_n, 6) if modern_n else None,
        "modern_bookmaker_timestamp_present": modern_bookmaker,
        "modern_lineup_timestamp_present": modern_lineup,
        "modern_feature_timestamp_present": modern_feature,
        "legacy_bookmaker_timestamp_unknown": sum(1 for row in legacy if _bookmaker_ts(row) is None),
        "legacy_lineup_timestamp_unknown": sum(1 for row in legacy if _lineup_ts(row) is None),
        "legacy_feature_timestamp_unknown": sum(1 for row in legacy if _feature_ts(row) is None),
        "policy": (
            "Modern Postgres settlement rows require explicit point-in-time quote, lineup-state observation "
            "and feature timestamps. Legacy Git-history rows keep unknown provenance visible as warnings and "
            "are never assigned fabricated timestamps."
        ),
    }

def canonical_clv_context(
    clv_report: dict[str, Any] | None,
    tracking_rows: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Normalize legacy and strict Postgres CLV reports into one Phase18 context."""
    report = clv_report if isinstance(clv_report, dict) else {}
    tracking = [row for row in (tracking_rows or []) if isinstance(row, dict)]

    strict_rows = report.get("comparable_true_clv_rows")
    if strict_rows is None:
        strict_rows = report.get("rows")
    try:
        row_count = int(strict_rows or 0)
    except (TypeError, ValueError):
        row_count = len(tracking)
    if tracking:
        row_count = len(tracking)

    true_close_count = report.get("true_closing_line_rows")
    if true_close_count is None and tracking:
        true_close_count = sum(1 for row in tracking if bool(row.get("is_true_closing_line")))
    if true_close_count is None:
        true_close_count = row_count
    try:
        true_close_count = int(true_close_count or 0)
    except (TypeError, ValueError):
        true_close_count = 0

    overall = report.get("overall") if isinstance(report.get("overall"), dict) else {}
    if tracking:
        probability_values = [
            value
            for value in (_num(row.get("probability_clv") if row.get("probability_clv") is not None else row.get("clv_probability_pp")) for row in tracking)
            if value is not None
        ]
        price_values = [
            value
            for value in (_num(row.get("price_clv") if row.get("price_clv") is not None else row.get("clv_price_pct")) for row in tracking)
            if value is not None
        ]
        overall = {
            "avg_price_clv_pct": round(statistics.fmean(price_values), 6) if price_values else None,
            "avg_probability_clv_pp": round(statistics.fmean(probability_values), 6) if probability_values else None,
            "positive_probability_clv_rate": (
                round(sum(1 for value in probability_values if value > 0) / len(probability_values), 6)
                if probability_values else None
            ),
        }

    family_counts = (
        report.get("family_counts")
        if isinstance(report.get("family_counts"), dict)
        else {}
    )
    if tracking:
        family_counts = dict(
            Counter(str(row.get("market_family") or "UNKNOWN") for row in tracking)
        )

    return {
        "status": report.get("status"),
        "model_version": report.get("model_version"),
        "rows": row_count,
        "true_closing_line_rows": true_close_count,
        "overall": overall,
        "family_counts": family_counts,
        "ft_totals_maturation_funnel": (
            report.get("ft_totals_maturation_funnel")
            if isinstance(report.get("ft_totals_maturation_funnel"), dict)
            else {}
        ),
        "source": (
            "STRICT_POSTGRES_CLV_V4_TRACKING"
            if report.get("comparable_true_clv_rows") is not None
            else "LEGACY_PHASE17_CLV_REPORT"
        ),
    }


def build_report(
    settlement_rows: Iterable[dict[str, Any]],
    clv_report: dict[str, Any] | None = None,
    oos_calibration_report: dict[str, Any] | None = None,
    clv_tracking_rows: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rows = [row for row in settlement_rows if isinstance(row, dict)]
    splits = chronological_splits(rows)
    discipline = timestamp_discipline(rows)
    realized = settlement_metrics(rows)
    rolling = rolling_settlement_metrics(rows)
    clv_report = clv_report if isinstance(clv_report, dict) else {}
    clv_context = canonical_clv_context(clv_report, clv_tracking_rows)
    oos_calibration_report = (
        oos_calibration_report if isinstance(oos_calibration_report, dict) else {}
    )
    oos_fixture_rows = int(oos_calibration_report.get("fixture_rows") or 0)
    oos_ready_targets = int(oos_calibration_report.get("ready_target_count") or 0)
    oos_improving_targets = int(
        oos_calibration_report.get("targets_improving_brier_and_log_loss") or 0
    )
    anti_leakage = (
        oos_calibration_report.get("anti_leakage")
        if isinstance(oos_calibration_report.get("anti_leakage"), dict)
        else {}
    )
    anti_leakage_required = {
        "prediction_timestamp_before_kickoff_required": True,
        "one_latest_pre_kickoff_prediction_per_fixture": True,
        "historical_predictions_recomputed": False,
        "market_fields_used": False,
        "final_result_from_postgame_or_final_status_only": True,
    }
    anti_leakage_ok = all(
        anti_leakage.get(key) is expected
        for key, expected in anti_leakage_required.items()
    )

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

    modern_rows = int(discipline.get("modern_rows") or 0)
    if modern_rows:
        if discipline["modern_bookmaker_timestamp_coverage"] < 1.0:
            blockers.append("BOOKMAKER_TIMESTAMP_DISCIPLINE_INCOMPLETE")
        if discipline["modern_lineup_timestamp_coverage"] < 1.0:
            blockers.append("LINEUP_TIMESTAMP_DISCIPLINE_INCOMPLETE")
        if discipline["modern_feature_timestamp_coverage"] < 1.0:
            blockers.append("FEATURE_TIMESTAMP_DISCIPLINE_INCOMPLETE")
    else:
        warnings.append("NO_MODERN_SETTLEMENT_ROWS_FOR_TIMESTAMP_VALIDATION")

    legacy_rows = int(discipline.get("legacy_rows") or 0)
    if legacy_rows:
        if discipline.get("legacy_bookmaker_timestamp_unknown"):
            warnings.append(
                f"LEGACY_BOOKMAKER_TIMESTAMP_UNKNOWN_{discipline['legacy_bookmaker_timestamp_unknown']}"
            )
        if discipline.get("legacy_lineup_timestamp_unknown"):
            warnings.append(
                f"LEGACY_LINEUP_TIMESTAMP_UNKNOWN_{discipline['legacy_lineup_timestamp_unknown']}"
            )
        if discipline.get("legacy_feature_timestamp_unknown"):
            warnings.append(
                f"LEGACY_FEATURE_TIMESTAMP_UNKNOWN_{discipline['legacy_feature_timestamp_unknown']}"
            )
    if discipline["pre_kickoff_decision_rate"] is not None and discipline["pre_kickoff_decision_rate"] < 1.0:
        blockers.append("POST_KICKOFF_DECISION_ROWS_DETECTED")
    if oos_fixture_rows < 50:
        blockers.append(f"OOS_MODEL_FIXTURES_{oos_fixture_rows}_LT_50")
    if oos_ready_targets <= 0:
        blockers.append("NO_OOS_CALIBRATION_TARGET_READY")
    if not anti_leakage_ok:
        blockers.append("OOS_ANTI_LEAKAGE_CONTRACT_INCOMPLETE")
    if str(clv_context.get("status") or "") not in CLV_COMPLETE_STATUSES:
        blockers.append("CLV_ENGINE_NOT_COMPLETE")

    if realized["settled_rows"] < 50:
        warnings.append(
            f"REAL_BET_SETTLEMENT_{realized['settled_rows']}_LT_50_COMMERCIAL_PERFORMANCE_ONLY"
        )
    if len(splits["test"]) < 10:
        warnings.append("CHRONOLOGICAL_REAL_BET_TEST_PARTITION_SMALL")

    metric_availability = {
        "brier": oos_ready_targets > 0,
        "log_loss": oos_ready_targets > 0,
        "calibration": oos_ready_targets > 0,
        "mae": False,
        "rmse": False,
        "hit_rate": realized["hit_rate_ex_push"] is not None,
        "roi": realized["roi_per_staked_unit"] is not None,
        "clv": (clv_context.get("overall") or {}).get("avg_probability_clv_pp") is not None,
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
        "oos_model_evidence": {
            "status": oos_calibration_report.get("status"),
            "model_version": oos_calibration_report.get("model_version"),
            "fixture_rows": oos_fixture_rows,
            "minimum_fixture_rows": 50,
            "ready_target_count": oos_ready_targets,
            "targets_improving_brier_and_log_loss": oos_improving_targets,
            "anti_leakage_ok": anti_leakage_ok,
            "anti_leakage": anti_leakage,
            "source_counts": oos_calibration_report.get("source_counts") or {},
            "run_type_counts": oos_calibration_report.get("run_type_counts") or {},
            "role": "MODEL_OOS_VALIDATION_NOT_REAL_BET_PERFORMANCE",
        },
        "realized_settlement_metrics": realized,
        "rolling_settlement_metrics": rolling,
        "metric_availability_on_current_settlement_ledger": metric_availability,
        "clv_context": clv_context,
        "blockers": blockers,
        "warnings": warnings,
        "notes": [
            "The settlement ledger can measure realized hit rate, ROI, drawdown, losing streak and volatility.",
            "Brier/log-loss/calibration come from frozen OOS model predictions joined to final outcomes; real BET/LEAN settlement ROI is tracked separately and cannot substitute for model OOS evidence.",
            "Phase18 research readiness must not depend on producing BET/LEAN classifications before promotion; doing so would create a circular promotion gate.",
            "Chronological split preview is diagnostic only. Production model evaluation must use frozen prediction snapshots and walk-forward/rolling retraining without leakage.",
            "Bookmaker, lineup-state observation and feature timestamps must be explicit for modern Postgres prediction rows before production promotion.",
            "Legacy rows with unavailable provenance remain visible as warnings and are never assigned fabricated timestamps.",
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
    parser.add_argument("--clv-tracking", required=False)
    parser.add_argument("--oos-calibration-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build_report(
        _load_jsonl(args.settlement_ledger),
        _load_json(args.clv_report),
        _load_json(args.oos_calibration_report),
        _load_jsonl(args.clv_tracking) if args.clv_tracking else None,
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
