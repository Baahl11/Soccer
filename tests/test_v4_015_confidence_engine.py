from mcp_gateway import confidence_engine_v4


def test_v4_015_blocks_without_enough_oos():
    result = confidence_engine_v4.score(
        oos_rows=50,
        calibration_ece=0.04,
        disagreement_max_range=0.05,
        feature_missing_rate=0.10,
    )
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert result["confidence_band"] == "INSUFFICIENT_DATA"
    assert result["production_promotion_allowed"] is False


def test_v4_015_high_confidence_requires_good_evidence():
    result = confidence_engine_v4.score(
        oos_rows=1200,
        calibration_ece=0.02,
        disagreement_max_range=0.03,
        feature_missing_rate=0.05,
    )
    assert result["status"] == "RESEARCH_CONFIDENCE_AVAILABLE"
    assert result["confidence_band"] == "HIGH"
    assert result["confidence_score"] >= confidence_engine_v4.HIGH_CONFIDENCE_SCORE
    assert result["market_fields_used"] is False


def test_v4_015_low_confidence_for_poor_calibration_and_disagreement():
    result = confidence_engine_v4.score(
        oos_rows=250,
        calibration_ece=0.14,
        disagreement_max_range=0.19,
        feature_missing_rate=0.40,
    )
    assert result["status"] == "RESEARCH_CONFIDENCE_AVAILABLE"
    assert result["confidence_band"] == "LOW"
    assert result["runtime_prediction_weight"] == 0.0
    assert result["canonical_bet_logic_changed"] is False


def test_v4_015_score_is_bounded():
    result = confidence_engine_v4.score(
        oos_rows=5000,
        calibration_ece=0.0,
        disagreement_max_range=0.0,
        feature_missing_rate=0.0,
    )
    assert 0 <= result["confidence_score"] <= 100
    assert result["production_promotion_allowed"] is False
