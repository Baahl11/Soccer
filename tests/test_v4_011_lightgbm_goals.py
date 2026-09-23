from mcp_gateway import lightgbm_goals_v4


def _row(fid: int, captured: str, hg: int, ag: int, *, market=False):
    row = {
        "fixture_id": fid,
        "feature_captured_at": captured,
        "market_fields_included": False,
        "features": {
            "team_performance.home_goal_rate_blend": 1.2 + fid * 0.01,
            "team_performance.away_goal_rate_blend": 0.9 + fid * 0.005,
            "context.league_id": 128,
            "context.venue": "NOT_NUMERIC",
        },
        "feature_missing": {
            "team_performance.home_goal_rate_blend": False,
            "team_performance.away_goal_rate_blend": False,
            "context.league_id": False,
            "context.venue": False,
        },
        "targets": {"home_goals": hg, "away_goals": ag},
    }
    if market:
        row["market"] = {"price": 1.9}
    return row


def test_v4_011_feature_schema_is_deterministic_and_numeric_only():
    rows = [_row(i, f"2026-06-{i:02d}T10:00:00+00:00", i % 3, (i + 1) % 2) for i in range(1, 6)]
    cols = lightgbm_goals_v4.feature_columns(rows)
    assert cols == sorted(cols)
    assert "context.venue" not in cols
    assert "context.league_id" in cols
    assert lightgbm_goals_v4.expanded_feature_names(cols)[1].endswith("__missing")


def test_v4_011_preserves_missingness_without_silent_imputation():
    row = _row(1, "2026-06-01T10:00:00+00:00", 1, 0)
    row["features"]["team_performance.home_goal_rate_blend"] = None
    row["feature_missing"]["team_performance.home_goal_rate_blend"] = True
    cols = ["team_performance.home_goal_rate_blend"]
    vec = lightgbm_goals_v4.vectorize(row, cols)
    assert vec[0] != vec[0]  # NaN, intentionally preserved for LightGBM native missing handling.
    assert vec[1] == 1.0


def test_v4_011_small_dataset_blocks_before_importing_lightgbm():
    rows = [
        _row(i, f"2026-06-{i:02d}T10:00:00+00:00", i % 4, (i + 1) % 3, market=True)
        for i in range(1, 12)
    ]
    report = lightgbm_goals_v4.walk_forward(rows)
    assert report["status"] == "INSUFFICIENT_TRAINING_SAMPLE"
    assert report["eligible_rows"] == 11
    assert report["walk_forward_evaluated"] == 0
    assert report["lightgbm_imported"] is False
    assert report["market_fields_used"] is False
    assert report["post_kickoff_features_used"] is False
    assert report["silent_imputation_used"] is False
    assert report["promotion_allowed"] is False


def test_v4_011_probability_projection_is_normalized():
    probs = lightgbm_goals_v4.probabilities(1.6, 1.1)
    assert abs(probs["home_win"] + probs["draw"] + probs["away_win"] - 1.0) < 1e-9
    assert 0.0 <= probs["btts"] <= 1.0
    assert 0.0 <= probs["over_2_5"] <= 1.0
