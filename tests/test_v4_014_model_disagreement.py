from mcp_gateway import model_disagreement_v4


BASE = {
    "home_win": 0.50,
    "draw": 0.28,
    "away_win": 0.22,
    "btts": 0.54,
    "over_1_5": 0.75,
    "over_2_5": 0.52,
    "over_3_5": 0.29,
}


def _pred(delta: float):
    return {
        "home_win": BASE["home_win"] + delta,
        "draw": BASE["draw"],
        "away_win": BASE["away_win"] - delta,
        "btts": BASE["btts"] + delta,
        "over_1_5": BASE["over_1_5"],
        "over_2_5": BASE["over_2_5"] + delta,
        "over_3_5": BASE["over_3_5"],
    }


def test_v4_014_blocks_missing_component():
    result = model_disagreement_v4.analyze({
        "DIXON_COLES": _pred(0.0),
        "BIVARIATE_POISSON": _pred(0.01),
    })
    assert result["status"] == "MISSING_COMPONENT_PREDICTIONS"
    assert "LIGHTGBM_GOALS" in result["missing_components"]
    assert result["production_promotion_allowed"] is False


def test_v4_014_low_disagreement():
    result = model_disagreement_v4.analyze({
        "DIXON_COLES": _pred(-0.01),
        "BIVARIATE_POISSON": _pred(0.0),
        "LIGHTGBM_GOALS": _pred(0.01),
    })
    assert result["status"] == "RESEARCH_DISAGREEMENT_AVAILABLE"
    assert result["disagreement"]["level"] == "LOW"
    assert result["market_fields_used"] is False
    assert result["runtime_prediction_weight"] == 0.0


def test_v4_014_high_disagreement_detected():
    result = model_disagreement_v4.analyze({
        "DIXON_COLES": _pred(-0.06),
        "BIVARIATE_POISSON": _pred(0.0),
        "LIGHTGBM_GOALS": _pred(0.06),
    })
    assert result["status"] == "RESEARCH_DISAGREEMENT_AVAILABLE"
    assert result["disagreement"]["level"] == "HIGH"
    assert result["disagreement"]["max_probability_range"] > 0.10
    assert result["production_promotion_allowed"] is False
