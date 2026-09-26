from datetime import datetime, timedelta, timezone

from mcp_gateway import oos_backtest_v4 as v


def _row(i: int, status: str, roi: float, *, quote_ts=True, lineup_ts=True, feature_ts=True):
    base = datetime(2026, 9, 10, 10, 0, tzinfo=timezone(timedelta(hours=-6))) + timedelta(days=i)
    kickoff = base + timedelta(hours=1)
    row = {
        "fixture_id": i,
        "generated_at_local": base.isoformat(),
        "kickoff_local": kickoff.isoformat(),
        "settled": True,
        "settlement_status": status,
        "roi_units": roi,
        "stake_units": 1.0,
        "source": "POSTGRES_REFRESH_EVENT",
    }
    if quote_ts:
        row["quote_timestamp"] = (base - timedelta(minutes=5)).isoformat()
    if lineup_ts:
        row["lineup_captured_at"] = (base - timedelta(minutes=10)).isoformat()
        row["lineup_observation_timestamp"] = row["lineup_captured_at"]
    if feature_ts:
        row["feature_captured_at"] = (base - timedelta(minutes=15)).isoformat()
    return row


def test_phase18_chronological_split_has_no_shuffle():
    rows = [_row(i, "WIN", 1.0) for i in range(10)]
    rows.reverse()
    split = v.chronological_splits(rows)
    assert len(split["train"]) == 6
    assert len(split["validation"]) == 2
    assert len(split["test"]) == 2
    assert split["train"][0]["fixture_id"] == 0
    assert split["test"][-1]["fixture_id"] == 9


def test_phase18_settlement_metrics_drawdown_streak_and_volatility():
    rows = [
        _row(0, "WIN", 1.0),
        _row(1, "LOSS", -1.0),
        _row(2, "LOSS", -1.0),
        _row(3, "WIN", 2.0),
    ]
    metrics = v.settlement_metrics(rows)
    assert metrics["settled_rows"] == 4
    assert metrics["wins"] == 2
    assert metrics["losses"] == 2
    assert metrics["hit_rate_ex_push"] == 0.5
    assert metrics["roi_units"] == 1.0
    assert metrics["max_drawdown_units"] == 2.0
    assert metrics["max_losing_streak"] == 2
    assert metrics["return_volatility_stddev"] > 0


def test_phase18_blocks_missing_timestamp_discipline():
    rows = [
        _row(i, "WIN" if i % 2 == 0 else "LOSS", 1.0 if i % 2 == 0 else -1.0,
             quote_ts=False, lineup_ts=False, feature_ts=False)
        for i in range(30)
    ]
    report = v.build_report(rows, {"status": "CLV_TRACKING_INCOMPLETE", "overall": {}})
    assert report["status"] == "OOS_FRAMEWORK_DATA_GAPS"
    assert "BOOKMAKER_TIMESTAMP_DISCIPLINE_INCOMPLETE" in report["blockers"]
    assert "LINEUP_TIMESTAMP_DISCIPLINE_INCOMPLETE" in report["blockers"]
    assert "FEATURE_TIMESTAMP_DISCIPLINE_INCOMPLETE" in report["blockers"]
    assert "REAL_BET_SETTLEMENT_30_LT_50_COMMERCIAL_PERFORMANCE_ONLY" in report["warnings"]
    assert report["production_promotion_allowed"] is False


def test_phase18_current_ledger_metrics_are_separate_from_probability_metrics():
    rows = [_row(i, "WIN", 1.0) for i in range(60)]
    report = v.build_report(
        rows,
        {
            "status": "CLV_ANALYSIS_AVAILABLE",
            "rows": 60,
            "true_closing_line_rows": 60,
            "overall": {"avg_probability_clv_pp": 1.2},
        },
    )
    availability = report["metric_availability_on_current_settlement_ledger"]
    assert availability["hit_rate"] is True
    assert availability["roi"] is True
    assert availability["clv"] is True
    assert availability["brier"] is False
    assert availability["log_loss"] is False
    assert availability["calibration"] is False



def test_phase18_accepts_v159_capture_complete_clv_status():
    rows = [_row(i, "WIN", 1.0) for i in range(60)]
    report = v.build_report(
        rows,
        {
            "status": "CLV_CAPTURE_COMPLETE_ANALYSIS_AVAILABLE",
            "rows": 179,
            "true_closing_line_rows": 179,
            "overall": {"avg_probability_clv_pp": 0.15},
        },
    )
    assert "CLV_ENGINE_NOT_COMPLETE" not in report["blockers"]
    assert report["clv_context"]["status"] == "CLV_CAPTURE_COMPLETE_ANALYSIS_AVAILABLE"



def test_phase18_legacy_missing_timestamps_are_warnings_not_modern_blockers():
    rows = []
    for i in range(60):
        row = _row(
            i,
            "WIN" if i % 2 == 0 else "LOSS",
            1.0 if i % 2 == 0 else -1.0,
            quote_ts=False,
            lineup_ts=False,
            feature_ts=False,
        )
        row.pop("source", None)
        rows.append(row)

    report = v.build_report(
        rows,
        {
            "status": "CLV_CAPTURE_COMPLETE_ANALYSIS_AVAILABLE",
            "rows": 179,
            "true_closing_line_rows": 179,
            "overall": {"avg_probability_clv_pp": 0.1},
        },
    )

    assert "BOOKMAKER_TIMESTAMP_DISCIPLINE_INCOMPLETE" not in report["blockers"]
    assert "LINEUP_TIMESTAMP_DISCIPLINE_INCOMPLETE" not in report["blockers"]
    assert "FEATURE_TIMESTAMP_DISCIPLINE_INCOMPLETE" not in report["blockers"]
    assert "NO_MODERN_SETTLEMENT_ROWS_FOR_TIMESTAMP_VALIDATION" in report["warnings"]
    assert "LEGACY_BOOKMAKER_TIMESTAMP_UNKNOWN_60" in report["warnings"]
    assert report["timestamp_discipline"]["modern_rows"] == 0
    assert report["timestamp_discipline"]["legacy_rows"] == 60


def test_phase18_accepts_lineup_observation_timestamp_without_claiming_confirmed_lineup():
    row = _row(1, "WIN", 1.0, lineup_ts=False)
    row["lineup_observation_timestamp"] = row["feature_captured_at"]
    discipline = v.timestamp_discipline([row])
    assert discipline["modern_lineup_timestamp_coverage"] == 1.0



def _oos_report(rows=703):
    return {
        "status": "OOS_CALIBRATION_MATERIALIZED",
        "model_version": "SOCCER_OOS_HISTORY_MERGE_V4_1.0.0",
        "fixture_rows": rows,
        "ready_target_count": 5,
        "targets_improving_brier_and_log_loss": 5,
        "source_counts": {"POSTGRES_NATIVE": 209, "HISTORICAL_SIGNAL_LEDGER": 494},
        "run_type_counts": {"T-10": 358, "EARLY_RESEARCH": 89},
        "anti_leakage": {
            "prediction_timestamp_before_kickoff_required": True,
            "one_latest_pre_kickoff_prediction_per_fixture": True,
            "historical_predictions_recomputed": False,
            "market_fields_used": False,
            "final_result_from_postgame_or_final_status_only": True,
        },
    }


def test_phase18_uses_frozen_oos_predictions_not_bet_count_for_model_readiness():
    rows = [_row(i, "WIN", 1.0) for i in range(28)]
    report = v.build_report(
        rows,
        {
            "status": "CLV_CAPTURE_COMPLETE_ANALYSIS_AVAILABLE",
            "rows": 179,
            "true_closing_line_rows": 179,
            "overall": {"avg_probability_clv_pp": 0.15},
        },
        _oos_report(),
    )

    assert "OOS_MODEL_FIXTURES_703_LT_50" not in report["blockers"]
    assert not any("SETTLED_" in blocker for blocker in report["blockers"])
    assert "REAL_BET_SETTLEMENT_28_LT_50_COMMERCIAL_PERFORMANCE_ONLY" in report["warnings"]
    assert report["oos_model_evidence"]["fixture_rows"] == 703
    assert report["oos_model_evidence"]["anti_leakage_ok"] is True
    assert report["metric_availability_on_current_settlement_ledger"]["brier"] is True
    assert report["metric_availability_on_current_settlement_ledger"]["log_loss"] is True
    assert report["metric_availability_on_current_settlement_ledger"]["calibration"] is True


def test_phase18_blocks_oos_if_anti_leakage_contract_breaks():
    oos = _oos_report()
    oos["anti_leakage"]["market_fields_used"] = True
    rows = [_row(i, "WIN", 1.0) for i in range(60)]
    report = v.build_report(
        rows,
        {
            "status": "CLV_CAPTURE_COMPLETE_ANALYSIS_AVAILABLE",
            "rows": 179,
            "true_closing_line_rows": 179,
            "overall": {"avg_probability_clv_pp": 0.15},
        },
        oos,
    )
    assert "OOS_ANTI_LEAKAGE_CONTRACT_INCOMPLETE" in report["blockers"]



def test_phase18_normalizes_strict_postgres_clv_tracking():
    rows = [_row(i, "WIN", 1.0) for i in range(60)]
    tracking = [
        {
            "fixture_id": 100,
            "market_family": "1X2",
            "is_true_closing_line": True,
            "probability_clv": 1.0,
            "price_clv": 2.0,
        },
        {
            "fixture_id": 101,
            "market_family": "BTTS",
            "is_true_closing_line": True,
            "probability_clv": -0.5,
            "price_clv": -1.0,
        },
    ]
    report = v.build_report(
        rows,
        {
            "status": "ACTIVE_TRUE_CLV_SAMPLE",
            "model_version": "SOCCER_TRUE_CLV_POSTGRES_V4_1.1.10",
            "comparable_true_clv_rows": 2,
            "family_counts": {"1X2": 1, "BTTS": 1},
            "ft_totals_maturation_funnel": {
                "priced_entry_rows": 108,
                "true_clv_rows": 0,
            },
        },
        _oos_report(),
        tracking,
    )

    assert "CLV_ENGINE_NOT_COMPLETE" not in report["blockers"]
    assert report["clv_context"]["source"] == "STRICT_POSTGRES_CLV_V4_TRACKING"
    assert report["clv_context"]["rows"] == 2
    assert report["clv_context"]["true_closing_line_rows"] == 2
    assert report["clv_context"]["family_counts"] == {"1X2": 1, "BTTS": 1}
    assert report["clv_context"]["overall"]["avg_probability_clv_pp"] == 0.25
    assert report["clv_context"]["overall"]["avg_price_clv_pct"] == 0.5
    assert report["clv_context"]["ft_totals_maturation_funnel"]["true_clv_rows"] == 0
