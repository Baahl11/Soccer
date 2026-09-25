from datetime import datetime, timedelta, timezone

from mcp_gateway import btts_poisson_ranking_v4 as v


def _row(fid, home_rate, away_rate, btts):
    captured = datetime(2026, 9, 1, tzinfo=timezone.utc) + timedelta(hours=fid)
    return {
        "fixture_id": fid,
        "feature_captured_at": captured.isoformat(),
        "kickoff": (captured + timedelta(hours=2)).isoformat(),
        "market_fields_included": False,
        "features": {
            "team_performance.home_goal_rate_blend": home_rate,
            "team_performance.away_goal_rate_blend": away_rate,
            "team_performance.total_goal_rate_blend": home_rate + away_rate,
        },
        "targets": {"btts": int(btts)},
    }


def test_btts_poisson_tracker_is_research_only_and_oos():
    rows = []
    for i in range(1, 181):
        home = 0.6 + (i % 11) * 0.14
        away = 0.5 + (i % 9) * 0.13
        p = v.poisson_probability(home, away, "btts")
        btts = p > 0.48 if i % 5 else p > 0.62
        rows.append(_row(i, home, away, btts))

    report = v.walk_forward(rows)
    assert report["status"] == "RESEARCH_ONLY"
    assert report["walk_forward_evaluated"] >= 50
    assert report["production_promotion_allowed"] is False
    assert report["runtime_prediction_weight"] == 0
    assert report["market_fields_used"] is False
    assert report["post_kickoff_features_used"] is False
    assert report["provider_requests_added"] == 0
    assert report["raw_metrics"]["rows"] == report["calibrated_metrics"]["rows"]


def test_positive_platt_slope_preserves_auc_on_synthetic_signal():
    rows = []
    for i in range(1, 181):
        high = i % 2 == 0
        home = 1.7 if high else 0.55
        away = 1.6 if high else 0.50
        rows.append(_row(i, home, away, high))

    report = v.walk_forward(rows)
    assert report["comparison"]["positive_slope_all_folds"] is True
    assert report["comparison"]["ranking_preserved"] is True
    assert abs(report["comparison"]["auc_delta"]) <= 1e-6
