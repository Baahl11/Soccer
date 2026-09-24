from mcp_gateway import oos_stage_diagnostics_v4 as v


def _row(fid, stage, version, probs, outcome):
    home, draw, away = probs
    return {
        "fixture_id": fid,
        "run_type": stage,
        "model_version": version,
        "predictions": {
            "home_win": home,
            "draw": draw,
            "away_win": away,
            "btts": 0.55,
            "over_2_5": 0.60,
        },
        "outcomes": {
            "home_win": 1 if outcome == "home" else 0,
            "draw": 1 if outcome == "draw" else 0,
            "away_win": 1 if outcome == "away" else 0,
            "btts": 1,
            "over_2_5": 1,
        },
    }


def test_stage_diagnostics_separates_current_model_version():
    rows = [
        _row(1, "T-10", "SOCCER EDGE ENGINE v1.0", (0.6, 0.2, 0.2), "home"),
        _row(2, "T-10", "SOCCER EDGE ENGINE v1.7", (0.5, 0.25, 0.25), "home"),
        _row(3, "T-20", "SOCCER EDGE ENGINE v1.7", (0.3, 0.3, 0.4), "away"),
    ]
    report = v.build_report(rows)
    assert report["current_source_model_version"] == "SOCCER EDGE ENGINE v1.7"
    assert report["all_model_rows"] == 3
    assert report["current_model_rows"] == 2
    assert report["stage_counts_current_model"] == {"T-10": 1, "T-20": 1}


def test_stage_diagnostics_reports_binary_and_multiclass_metrics():
    rows = []
    for fid in range(1, 21):
        rows.append(_row(fid, "T-10", "SOCCER EDGE ENGINE v1.7", (0.6, 0.2, 0.2), "home"))
    report = v.build_report(rows)
    stage = report["current_model_by_stage"]["T-10"]
    assert stage["rows"] == 20
    assert stage["sample_status"] == "DIRECTIONAL_SAMPLE"
    assert stage["binary_targets"]["home_win"]["rows"] == 20
    assert stage["binary_targets"]["home_win"]["brier"] is not None
    assert stage["multiclass_1x2"]["n"] == 20
    assert stage["multiclass_1x2"]["multiclass_log_loss"] is not None


def test_stage_diagnostics_never_changes_runtime_or_calls_provider():
    report = v.build_report([
        _row(1, "T-40", "SOCCER EDGE ENGINE v1.7", (0.4, 0.3, 0.3), "draw"),
    ])
    assert report["provider_requests_added"] == 0
    assert report["production_promotion_allowed"] is False
    assert report["anti_leakage"]["stage_calibrators_fitted"] is False
    assert report["anti_leakage"]["runtime_weights_changed"] is False


def test_current_model_deployment_calibrators_are_research_only():
    rows = []
    for fid in range(1, 261):
        outcome = "home" if fid % 3 == 0 else "draw" if fid % 3 == 1 else "away"
        rows.append(_row(fid, "T-10", "SOCCER EDGE ENGINE v1.7", (0.6, 0.2, 0.2), outcome))
    report = v.build_report(rows)
    calibrators = report["current_model_deployment_calibrators"]
    assert report["current_source_model_version"] == "SOCCER EDGE ENGINE v1.7"
    assert calibrators["home_win"]["production_promotion_allowed"] is False
    assert calibrators["home_win"]["runtime_prediction_weight"] == 0.0
    assert calibrators["home_win"]["rows"] == 260
    assert calibrators["btts"]["rows"] == 260
