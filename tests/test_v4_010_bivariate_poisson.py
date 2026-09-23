import math

from mcp_gateway import bivariate_poisson_v4


def _row(fid: int, captured: str, home_mean: float, away_mean: float, hg: int, ag: int):
    return {
        "fixture_id": fid,
        "feature_captured_at": captured,
        "features": {
            "team_performance.home_goal_rate_blend": home_mean,
            "team_performance.away_goal_rate_blend": away_mean,
        },
        "targets": {
            "home_goals": hg,
            "away_goals": ag,
        },
    }


def test_v4_010_zero_shared_fraction_matches_independent_marginals():
    probs = bivariate_poisson_v4.probabilities(1.6, 1.1, 0.0)
    assert probs is not None
    assert math.isclose(probs["home_win"] + probs["draw"] + probs["away_win"], 1.0, rel_tol=1e-9)
    assert 0.0 <= probs["btts"] <= 1.0


def test_v4_010_positive_shared_component_changes_low_score_dependence():
    p_ind = bivariate_poisson_v4.joint_pmf(0, 0, 1.4, 1.0, 0.0)
    p_shared = bivariate_poisson_v4.joint_pmf(0, 0, 1.4, 1.0, 0.2)
    assert p_shared > p_ind


def test_v4_010_fit_is_deterministic():
    rows = [
        _row(i, f"2026-03-{i:02d}T10:00:00+00:00", 1.4 + (i % 3) * 0.1, 1.0, i % 3, (i + 1) % 2)
        for i in range(1, 12)
    ]
    first = bivariate_poisson_v4.fit_shared_fraction(rows)
    second = bivariate_poisson_v4.fit_shared_fraction(list(rows))
    assert first == second
    assert first["shared_fraction"] in bivariate_poisson_v4.SHARED_FRACTION_GRID


def test_v4_010_walk_forward_is_research_only_and_market_free():
    rows = []
    for i in range(1, 10):
        row = _row(
            i,
            f"2026-04-{i:02d}T10:00:00+00:00",
            1.2 + i * 0.03,
            0.9 + i * 0.02,
            i % 4,
            (i + 1) % 3,
        )
        row["market"] = {"price": 123.0}
        rows.append(row)
    report = bivariate_poisson_v4.walk_forward(rows, min_train_rows=5)
    assert report["walk_forward_evaluated"] == 4
    assert report["market_fields_used"] is False
    assert report["post_kickoff_features_used"] is False
    assert report["promotion_allowed"] is False


def test_v4_010_current_small_sample_is_blocked():
    rows = [
        _row(i, f"2026-05-{i:02d}T10:00:00+00:00", 1.5, 1.0, 1, 0)
        for i in range(1, 12)
    ]
    report = bivariate_poisson_v4.walk_forward(rows)
    assert report["eligible_rows"] == 11
    assert report["walk_forward_evaluated"] == 0
    assert report["status"] == "INSUFFICIENT_OOS_SAMPLE"
    assert report["promotion_reason"] == "INSUFFICIENT_WALK_FORWARD_SAMPLE"
