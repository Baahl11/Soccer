from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any

from mcp_gateway import calibration_v4, persistence

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_OOS_PREDICTION_LEDGER_V4_1.0.0"
TARGET_KEYS = (
    "home_win",
    "draw",
    "away_win",
    "btts",
    "over_2_5",
)
RAW_PROBABILITY_KEYS = {
    "home_win": "raw_home_win_prob",
    "draw": "raw_draw_prob",
    "away_win": "raw_away_win_prob",
    "btts": "raw_btts_yes_prob",
    "over_2_5": "raw_over_2_5_prob",
}


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if 0.0 <= out <= 1.0 else None
    except (TypeError, ValueError):
        return None


def outcomes(home_goals: int, away_goals: int) -> dict[str, int]:
    total = int(home_goals) + int(away_goals)
    return {
        "home_win": int(home_goals > away_goals),
        "draw": int(home_goals == away_goals),
        "away_win": int(home_goals < away_goals),
        "btts": int(home_goals > 0 and away_goals > 0),
        "over_2_5": int(total > 2.5),
    }


def normalize_row(row: dict[str, Any]) -> dict[str, Any] | None:
    raw = row.get("raw_projection")
    if not isinstance(raw, dict):
        return None
    try:
        home_goals = int(row.get("home_goals"))
        away_goals = int(row.get("away_goals"))
    except (TypeError, ValueError):
        return None

    target_outcomes = outcomes(home_goals, away_goals)
    predictions: dict[str, float | None] = {}
    for target, key in RAW_PROBABILITY_KEYS.items():
        predictions[target] = _num(raw.get(key))

    return {
        "fixture_id": row.get("fixture_id"),
        "run_timestamp": row.get("run_timestamp").isoformat()
        if hasattr(row.get("run_timestamp"), "isoformat")
        else row.get("run_timestamp"),
        "kickoff": row.get("kickoff").isoformat()
        if hasattr(row.get("kickoff"), "isoformat")
        else row.get("kickoff"),
        "run_type": row.get("run_type"),
        "model_version": row.get("model_version"),
        "home_goals": home_goals,
        "away_goals": away_goals,
        "predictions": predictions,
        "outcomes": target_outcomes,
        "anti_leakage": True,
    }


def _load_latest_pre_kickoff_rows(conn, *, lookback_days: int, max_fixtures: int) -> list[dict[str, Any]]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, int(lookback_days)))
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT ON (m.fixture_id)
                m.fixture_id,
                m.run_timestamp,
                m.run_type,
                m.model_version,
                m.raw_projection,
                r.home_goals,
                r.away_goals,
                f.kickoff
            FROM soccer_model_runs m
            JOIN soccer_results r ON r.fixture_id = m.fixture_id
            JOIN soccer_fixtures f ON f.fixture_id = m.fixture_id
            WHERE m.run_timestamp >= %s
              AND m.run_timestamp < f.kickoff
              AND r.home_goals IS NOT NULL
              AND r.away_goals IS NOT NULL
            ORDER BY m.fixture_id, m.run_timestamp DESC
            LIMIT %s
            """,
            (cutoff, max(1, int(max_fixtures))),
        )
        columns = [desc.name for desc in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def build_report(*, lookback_days: int = 180, max_fixtures: int = 10000) -> dict[str, Any]:
    persistence.ensure_schema()
    with persistence._connect() as conn:
        raw_rows = _load_latest_pre_kickoff_rows(
            conn,
            lookback_days=lookback_days,
            max_fixtures=max_fixtures,
        )

    rows = [normalized for row in raw_rows if (normalized := normalize_row(row)) is not None]
    version_counts = Counter(str(row.get("model_version") or "UNKNOWN") for row in rows)
    run_type_counts = Counter(str(row.get("run_type") or "UNKNOWN") for row in rows)

    targets: dict[str, Any] = {}
    ready_targets = 0
    improving_targets = 0

    for target in TARGET_KEYS:
        observations = [
            {
                "probability": row["predictions"].get(target),
                "outcome": row["outcomes"].get(target),
            }
            for row in rows
            if row["predictions"].get(target) is not None
        ]
        report = calibration_v4.calibration_report(observations)
        raw_metrics = report.get("raw_metrics") if isinstance(report.get("raw_metrics"), dict) else {}
        calibrated_metrics = (
            report.get("calibrated_metrics")
            if isinstance(report.get("calibrated_metrics"), dict)
            else {}
        )
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
        "lookback_days": int(lookback_days),
        "fixture_rows": len(rows),
        "minimum_rows_per_target": minimum_rows,
        "latest_pre_kickoff_prediction_per_fixture": True,
        "anti_leakage": {
            "prediction_timestamp_before_kickoff_required": True,
            "final_result_joined_after_prediction": True,
            "one_latest_pre_kickoff_prediction_per_fixture": True,
            "market_fields_used": False,
        },
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
            "The ledger uses the latest stored pre-kickoff model run per settled fixture.",
            "Outcomes are joined only after the prediction timestamp and never feed the prediction features.",
            "Calibration remains research-only; a fitted calibrator is not automatically promoted.",
            "Current rows can contain multiple historical runtime model versions, which are reported explicitly.",
        ],
    }
