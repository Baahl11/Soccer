from datetime import datetime, timedelta, timezone

from mcp_gateway import goals_binary_challenger_v4 as v


def _row(fid, home_rate, away_rate, btts, over25):
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
            "availability.both_xi_confirmed": fid % 3 != 0,
            "availability.both_goalkeepers_confirmed": fid % 4 != 0,
            "availability.injury_report_count": fid % 5,
        },
        "targets": {"btts": int(btts), "over_2_5": int(over25)},
    }


def test_poisson_baselines_behave_monotonically():
    assert v.poisson_probability(1.8, 1.7, "btts") > v.poisson_probability(1.8, 0.3, "btts")
    assert v.poisson_probability(2.0, 1.5, "over_2_5") > v.poisson_probability(0.7, 0.6, "over_2_5")


def test_binary_challenger_rejects_market_fields_and_postkickoff_rows():
    good = _row(1, 1.2, 1.1, True, False)
    market = _row(2, 1.2, 1.1, True, False)
    market["market_fields_included"] = True
    late = _row(3, 1.2, 1.1, True, False)
    late["feature_captured_at"] = late["kickoff"]
    assert [r["fixture_id"] for r in v.eligible_rows([market, late, good], "btts")] == [1]


def test_walk_forward_remains_research_only_and_uses_oos_rows():
    rows = []
    for i in range(1, 181):
        home = 0.7 + (i % 9) * 0.16
        away = 0.6 + (i % 7) * 0.15
        btts = home > 1.1 and away > 1.0
        over25 = home + away > 2.4
        rows.append(_row(i, home, away, btts, over25))

    report = v.build_report(rows)
    for target in ("btts", "over_2_5"):
        row = report["targets"][target]
        assert row["status"] == "RESEARCH_ONLY"
        assert row["walk_forward_evaluated"] >= 50
        assert row["baseline_metrics"]["rows"] == row["challenger_metrics"]["rows"]
        assert row["market_fields_used"] is False
        assert row["post_kickoff_features_used"] is False
        assert row["production_promotion_allowed"] is False
        assert row["provider_requests_added"] == 0
