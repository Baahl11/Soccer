from datetime import datetime, timedelta, timezone

from mcp_gateway import draw_challenger_v4 as v


def _row(fid, home_rate, away_rate, draw, *, market_fields=False):
    captured = datetime(2026, 9, 1, tzinfo=timezone.utc) + timedelta(hours=fid)
    kickoff = captured + timedelta(hours=2)
    return {
        "fixture_id": fid,
        "feature_captured_at": captured.isoformat(),
        "kickoff": kickoff.isoformat(),
        "market_fields_included": market_fields,
        "features": {
            "team_performance.home_goal_rate_blend": home_rate,
            "team_performance.away_goal_rate_blend": away_rate,
            "team_performance.total_goal_rate_blend": home_rate + away_rate,
            "availability.both_xi_confirmed": fid % 3 != 0,
            "availability.both_goalkeepers_confirmed": fid % 4 != 0,
            "availability.injury_report_count": fid % 5,
        },
        "targets": {"draw": int(draw)},
    }


def test_poisson_draw_is_higher_for_balanced_low_scoring_rates():
    balanced = v.poisson_draw_probability(0.9, 0.9)
    open_game = v.poisson_draw_probability(2.2, 1.8)
    imbalanced = v.poisson_draw_probability(2.0, 0.5)
    assert balanced > open_game
    assert balanced > imbalanced


def test_eligible_rows_enforces_pre_kickoff_and_no_market_fields():
    good = _row(1, 1.1, 1.0, 1)
    bad_market = _row(2, 1.1, 1.0, 0, market_fields=True)
    bad_time = _row(3, 1.1, 1.0, 0)
    bad_time["feature_captured_at"] = bad_time["kickoff"]
    rows = v.eligible_rows([bad_market, bad_time, good])
    assert [row["fixture_id"] for row in rows] == [1]


def test_insufficient_sample_stays_research_only_without_dependency():
    rows = [_row(i, 1.0 + (i % 5) * 0.1, 1.0, i % 4 == 0) for i in range(1, 40)]
    report = v.walk_forward(rows, min_train_rows=30, min_oos_rows=20)
    assert report["status"] == "INSUFFICIENT_TRAINING_SAMPLE"
    assert report["production_promotion_allowed"] is False
    assert report["market_fields_used"] is False
    assert report["post_kickoff_features_used"] is False


def test_lightgbm_walk_forward_is_chronological_and_research_only():
    rows = []
    for i in range(1, 181):
        # Draws are deliberately associated with balanced, lower-rate games.
        if i % 3 == 0:
            home_rate = 0.85 + (i % 5) * 0.02
            away_rate = 0.86 + (i % 4) * 0.02
            draw = 1
        else:
            home_rate = 1.7 + (i % 7) * 0.07
            away_rate = 0.65 + (i % 5) * 0.04
            draw = 0
        rows.append(_row(i, home_rate, away_rate, draw))

    report = v.walk_forward(rows, min_train_rows=100, min_oos_rows=50)

    assert report["status"] == "RESEARCH_ONLY"
    assert report["walk_forward_evaluated"] >= 50
    assert report["fitted_folds"] >= 1
    assert report["baseline_metrics"]["rows"] == report["challenger_metrics"]["rows"]
    assert report["challenger_metrics"]["discrimination"]["auc"] is not None
    assert report["production_promotion_allowed"] is False
    assert report["runtime_prediction_weight"] == 0
    assert report["provider_requests_added"] == 0
    assert report["market_fields_used"] is False
    assert report["post_kickoff_features_used"] is False
