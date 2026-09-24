from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from datetime import datetime
from typing import Any, Iterable

from mcp_gateway import calibration_v4

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_OOS_HISTORY_MERGE_V4_1.0.0"
PREGAME_STAGES = {
    "EARLY_RESEARCH",
    "T-90",
    "T-60",
    "T-40",
    "T-30",
    "T-20",
    "T-10",
    "CLOSE",
}
FINAL_STATUSES = {"FT", "AET", "PEN"}
TARGET_KEYS = ("home_win", "draw", "away_win", "btts", "over_2_5")


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        out = datetime.fromisoformat(text)
    except ValueError:
        return None
    return out if out.tzinfo is not None else None


def _num_probability(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) and 0.0 <= out <= 1.0 else None


def _result_goals(row: dict[str, Any]) -> tuple[int, int] | None:
    result = row.get("result") if isinstance(row.get("result"), dict) else {}
    goals = result.get("goals") if isinstance(result.get("goals"), dict) else {}
    score = result.get("score") if isinstance(result.get("score"), dict) else {}
    fulltime = score.get("fulltime") if isinstance(score.get("fulltime"), dict) else {}
    home = goals.get("home", fulltime.get("home"))
    away = goals.get("away", fulltime.get("away"))
    try:
        return int(home), int(away)
    except (TypeError, ValueError):
        return None


def _outcomes(home_goals: int, away_goals: int) -> dict[str, int]:
    total = home_goals + away_goals
    return {
        "home_win": int(home_goals > away_goals),
        "draw": int(home_goals == away_goals),
        "away_win": int(home_goals < away_goals),
        "btts": int(home_goals > 0 and away_goals > 0),
        "over_2_5": int(total > 2),
    }


def historical_rows(signal_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    latest_prediction: dict[int, tuple[datetime, dict[str, Any]]] = {}
    finals: dict[int, tuple[int, int]] = {}

    for row in signal_rows:
        if not isinstance(row, dict):
            continue
        fixture_id = row.get("fixture_id")
        try:
            fixture_id = int(fixture_id)
        except (TypeError, ValueError):
            continue

        stage = str(row.get("stage") or "").upper()
        fixture_status = str(row.get("fixture_status") or "").upper()
        result = _result_goals(row)
        if result is not None and (stage == "POSTGAME" or fixture_status in FINAL_STATUSES):
            finals[fixture_id] = result

        raw = row.get("raw_projection")
        if not isinstance(raw, dict) or stage not in PREGAME_STAGES:
            continue

        generated_at = _parse_dt(row.get("generated_at_utc"))
        kickoff = _parse_dt(row.get("kickoff_local"))
        if generated_at is None or kickoff is None or generated_at >= kickoff:
            continue

        prior = latest_prediction.get(fixture_id)
        if prior is None or generated_at > prior[0]:
            latest_prediction[fixture_id] = (generated_at, row)

    out: list[dict[str, Any]] = []
    for fixture_id, (generated_at, row) in latest_prediction.items():
        final = finals.get(fixture_id)
        if final is None:
            continue
        kickoff = _parse_dt(row.get("kickoff_local"))
        if kickoff is None:
            continue
        home_goals, away_goals = final
        raw = row.get("raw_projection") or {}
        predictions = {
            "home_win": _num_probability(raw.get("raw_home_win_prob")),
            "draw": _num_probability(raw.get("raw_draw_prob")),
            "away_win": _num_probability(raw.get("raw_away_win_prob")),
            "btts": _num_probability(raw.get("raw_btts_yes_prob")),
            "over_2_5": _num_probability(raw.get("raw_over_2_5_prob")),
        }
        if not any(value is not None for value in predictions.values()):
            continue
        out.append({
            "fixture_id": fixture_id,
            "run_timestamp": generated_at.isoformat(),
            "kickoff": kickoff.isoformat(),
            "run_type": row.get("stage"),
            "model_version": row.get("model_version"),
            "home_goals": home_goals,
            "away_goals": away_goals,
            "predictions": predictions,
            "outcomes": _outcomes(home_goals, away_goals),
            "anti_leakage": True,
            "source": "HISTORICAL_SIGNAL_LEDGER",
        })

    out.sort(key=lambda row: (str(row.get("run_timestamp") or ""), int(row.get("fixture_id") or 0)))
    return out


def merge_rows(postgres_rows: Iterable[dict[str, Any]], history_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    chosen: dict[int, dict[str, Any]] = {}
    for source_name, rows in (("HISTORICAL_SIGNAL_LEDGER", history_rows), ("POSTGRES_NATIVE", postgres_rows)):
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                fixture_id = int(row.get("fixture_id"))
            except (TypeError, ValueError):
                continue
            candidate = dict(row)
            candidate.setdefault("source", source_name)
            timestamp = _parse_dt(candidate.get("run_timestamp"))
            kickoff = _parse_dt(candidate.get("kickoff"))
            if timestamp is None or kickoff is None or timestamp >= kickoff:
                continue
            current = chosen.get(fixture_id)
            if current is None:
                chosen[fixture_id] = candidate
                continue
            current_ts = _parse_dt(current.get("run_timestamp"))
            if current_ts is None or timestamp > current_ts or (
                timestamp == current_ts and candidate.get("source") == "POSTGRES_NATIVE"
            ):
                chosen[fixture_id] = candidate

    return sorted(chosen.values(), key=lambda row: int(row.get("fixture_id") or 0))


def summarize(rows: list[dict[str, Any]], *, postgres_n: int, historical_n: int) -> dict[str, Any]:
    version_counts = Counter(str(row.get("model_version") or "UNKNOWN") for row in rows)
    run_type_counts = Counter(str(row.get("run_type") or "UNKNOWN") for row in rows)
    source_counts = Counter(str(row.get("source") or "UNKNOWN") for row in rows)

    targets: dict[str, Any] = {}
    ready_targets = 0
    improving_targets = 0
    for target in TARGET_KEYS:
        observations = []
        for row in rows:
            predictions = row.get("predictions") if isinstance(row.get("predictions"), dict) else {}
            outcomes = row.get("outcomes") if isinstance(row.get("outcomes"), dict) else {}
            probability = _num_probability(predictions.get(target))
            outcome = outcomes.get(target)
            if probability is None or outcome not in (0, 1):
                continue
            observations.append({"probability": probability, "outcome": int(outcome)})

        report = calibration_v4.calibration_report(observations)
        raw_metrics = report.get("raw_metrics") if isinstance(report.get("raw_metrics"), dict) else {}
        calibrated_metrics = report.get("calibrated_metrics") if isinstance(report.get("calibrated_metrics"), dict) else {}
        calibrator = report.get("calibrator") if isinstance(report.get("calibrator"), dict) else {}
        fitted = calibrator.get("status") == "RESEARCH_CALIBRATOR_FITTED"
        if fitted:
            ready_targets += 1

        brier_delta = report.get("brier_delta")
        log_loss_delta = report.get("log_loss_delta")
        improves_both = (
            isinstance(brier_delta, (int, float))
            and isinstance(log_loss_delta, (int, float))
            and brier_delta < 0
            and log_loss_delta < 0
        )
        if improves_both:
            improving_targets += 1

        targets[target] = {
            "rows": len(observations),
            "status": report.get("status"),
            "raw_metrics": raw_metrics,
            "calibrator": calibrator,
            "calibrated_metrics": calibrated_metrics or None,
            "brier_delta": brier_delta,
            "log_loss_delta": log_loss_delta,
            "calibration_improves_brier_and_log_loss": improves_both,
            "production_promotion_allowed": False,
        }

    minimum_rows = calibration_v4.MIN_OOS_ROWS
    status = (
        "OOS_CALIBRATION_MATERIALIZED"
        if rows and ready_targets == len(TARGET_KEYS)
        else "COLLECTING_OOS_PREDICTIONS"
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": status,
        "fixture_rows": len(rows),
        "postgres_fixture_rows": postgres_n,
        "historical_fixture_rows": historical_n,
        "minimum_rows_per_target": minimum_rows,
        "latest_pre_kickoff_prediction_per_fixture": True,
        "anti_leakage": {
            "prediction_timestamp_before_kickoff_required": True,
            "final_result_from_postgame_or_final_status_only": True,
            "one_latest_pre_kickoff_prediction_per_fixture": True,
            "market_fields_used": False,
            "historical_predictions_recomputed": False,
        },
        "source_counts": dict(sorted(source_counts.items())),
        "model_version_counts": dict(sorted(version_counts.items())),
        "run_type_counts": dict(sorted(run_type_counts.items())),
        "targets": targets,
        "ready_target_count": ready_targets,
        "targets_improving_brier_and_log_loss": improving_targets,
        "rows": rows,
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
        "runtime_prediction_weight": 0.0,
        "canonical_bet_logic_changed": False,
        "notes": [
            "Historical rows come only from predictions already persisted in signal_ledger.jsonl; no model is rerun or backcast.",
            "For each fixture the latest prediction strictly before kickoff is selected, then joined to a postgame/final result.",
            "Postgres-native rows and historical rows are merged by fixture; the latest valid pre-kickoff timestamp wins.",
            "Calibration remains research-only and does not alter runtime prediction weights.",
        ],
    }


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


def _load_json(path: str) -> dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge strict historical pre-kickoff OOS rows with Postgres-native OOS rows.")
    parser.add_argument("--signal-ledger", required=True)
    parser.add_argument("--postgres-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    signal_rows = _load_jsonl(args.signal_ledger)
    postgres_report = _load_json(args.postgres_report)
    postgres_rows = postgres_report.get("rows") if isinstance(postgres_report.get("rows"), list) else []
    history = historical_rows(signal_rows)
    merged = merge_rows(postgres_rows, history)
    report = summarize(merged, postgres_n=len(postgres_rows), historical_n=len(history))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps({
        "model_version": report["model_version"],
        "status": report["status"],
        "fixture_rows": report["fixture_rows"],
        "postgres_fixture_rows": report["postgres_fixture_rows"],
        "historical_fixture_rows": report["historical_fixture_rows"],
        "ready_target_count": report["ready_target_count"],
        "targets_improving_brier_and_log_loss": report["targets_improving_brier_and_log_loss"],
        "source_counts": report["source_counts"],
        "provider_requests_added": report["provider_requests_added"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
