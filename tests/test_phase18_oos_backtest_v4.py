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
    assert "SETTLED_30_LT_50" in report["blockers"]
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
