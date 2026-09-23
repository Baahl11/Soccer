from mcp_gateway import calibration_v4


def _rows(n: int = 240):
    rows = []
    for index in range(n):
        outcome = 1 if index % 3 == 0 else 0
        probability = 0.72 if outcome else 0.38
        rows.append({"probability": probability, "outcome": outcome})
    return rows


def test_v4_013_blocks_insufficient_oos():
    report = calibration_v4.calibration_report(
        [{"probability": 0.6, "outcome": 1}] * 20,
    )
    assert report["status"] == "DATA_BLOCKED"
    assert report["calibrator"]["status"] == "INSUFFICIENT_OOS_FOR_CALIBRATION"
    assert report["production_promotion_allowed"] is False


def test_v4_013_reliability_metrics_are_bounded():
    metrics = calibration_v4.reliability_metrics(_rows())
    assert metrics["status"] == "OK"
    assert 0 <= metrics["brier"] <= 1
    assert metrics["log_loss"] > 0
    assert 0 <= metrics["ece"] <= 1
    assert 0 <= metrics["mce"] <= 1
    assert metrics["row_count"] == 240


def test_v4_013_platt_fit_is_research_only_and_deterministic():
    rows = _rows()
    first = calibration_v4.fit_platt(rows)
    second = calibration_v4.fit_platt(rows)
    assert first["status"] == "RESEARCH_CALIBRATOR_FITTED"
    assert first["parameters"] == second["parameters"]
    assert first["market_fields_used"] is False
    assert first["production_promotion_allowed"] is False
    assert first["runtime_prediction_weight"] == 0.0


def test_v4_013_calibrated_probability_is_monotonic():
    fitted = calibration_v4.fit_platt(_rows())
    low = calibration_v4.calibrate_probability(0.2, fitted)
    high = calibration_v4.calibrate_probability(0.8, fitted)
    assert low is not None and high is not None
    assert 0 < low < high < 1


def test_v4_013_report_exposes_before_after_metrics_without_promotion():
    report = calibration_v4.calibration_report(_rows())
    assert report["status"] == "RESEARCH_CALIBRATION_AVAILABLE"
    assert report["raw_metrics"]["status"] == "OK"
    assert report["calibrated_metrics"]["status"] == "OK"
    assert report["production_promotion_allowed"] is False
    assert report["market_fields_used"] is False
