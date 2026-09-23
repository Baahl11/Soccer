import math

from mcp_gateway import dixon_coles_v4


def _row(fid: int, captured: str, home_lambda: float, away_lambda: float, hg: int, ag: int):
    return {
        "fixture_id": fid,
        "feature_captured_at": captured,
        "features": {
            "team_performance.home_goal_rate_blend": home_lambda,
            "team_performance.away_goal_rate_blend": away_lambda,
        },
        "targets": {
            "home_goals": hg,
            "away_goals": ag,
        },
    }


def test_v4_009_tau_only_changes_low_scores():
    assert dixon_coles_v4.tau(2, 1, 1.5, 1.0, -0.1) == 1.0
    assert dixon_coles_v4.tau(0, 0, 1.5, 1.0, -0.1) != 1.0
    assert dixon_coles_v4.tau(1, 1, 1.5, 1.0, -0.1) != 1.0


def test_v4_009_probabilities_are_normalized_and_market_free():
    probs = dixon_coles_v4.probabilities(1.7, 1.1, -0.08)
    assert probs is not None
    assert math.isclose(probs["home_win"] + probs["draw"] + probs["away_win"], 1.0, rel_tol=1e-9)
    assert 0.0 <= probs["btts"] <= 1.0
    assert 0.0 <= probs["over_2_5"] <= 1.0


def test_v4_009_fit_rho_is_deterministic():
    rows = [
        _row(i, f"2026-01-{i:02d}T10:00:00+00:00", 1.4 + (i % 3) * 0.1, 1.0, i % 3, (i + 1) % 2)
        for i in range(1, 12)
    ]
    first = dixon_coles_v4.fit_rho(rows)
    second = dixon_coles_v4.fit_rho(list(rows))
    assert first == second
    assert first["rho"] in dixon_coles_v4.RHO_GRID


def test_v4_009_walk_forward_never_uses_market_fields():
    rows = []
    for i in range(1, 10):
        row = _row(
            i,
            f"2026-01-{i:02d}T10:00:00+00:00",
            1.2 + i * 0.03,
            0.9 + i * 0.02,
            i % 4,
            (i + 1) % 3,
        )
        row["market"] = {"price": 999.0}
        rows.append(row)
    report = dixon_coles_v4.walk_forward(rows, min_train_rows=5)
    assert report["walk_forward_evaluated"] == 4
    assert report["market_fields_used"] is False
    assert report["post_kickoff_features_used"] is False
    assert report["promotion_allowed"] is False


def test_v4_009_current_small_sample_is_explicitly_blocked():
    rows = [
        _row(i, f"2026-02-{i:02d}T10:00:00+00:00", 1.5, 1.0, 1, 0)
        for i in range(1, 12)
    ]
    report = dixon_coles_v4.walk_forward(rows)
    assert report["eligible_rows"] == 11
    assert report["walk_forward_evaluated"] == 0
    assert report["status"] == "INSUFFICIENT_OOS_SAMPLE"
    assert report["promotion_reason"] == "INSUFFICIENT_WALK_FORWARD_SAMPLE"
