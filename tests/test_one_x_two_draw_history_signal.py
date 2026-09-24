from datetime import datetime, timedelta, timezone

from mcp_gateway import analyze_1x2_draw_history_signal as v


def _fixture(fid, kickoff, league_id, home_id, away_id, actual, draw_prob=0.25):
    score = {"H": (2, 0), "D": (1, 1), "A": (0, 2)}[actual]
    generated = kickoff - timedelta(hours=2)
    return {
        "fixture_id": fid,
        "generated_at_local": generated.isoformat(),
        "kickoff_local": kickoff.isoformat(),
        "league_id": league_id,
        "home_team_id": home_id,
        "away_team_id": away_id,
        "raw_projection": {"raw_draw_prob": draw_prob},
        "result": {"goals": {"home": score[0], "away": score[1]}},
    }


def test_hierarchical_league_signal_is_strictly_prior_and_can_discriminate():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(120):
        league = 1 if i % 2 == 0 else 2
        if league == 1:
            actual = "D" if i >= 20 else ("H" if i % 4 == 0 else "A")
        else:
            actual = "H" if i % 4 == 1 else "A"
        rows.append(_fixture(
            i + 1,
            start + timedelta(days=i),
            league,
            1000 + (i % 8),
            2000 + (i % 8),
            actual,
        ))

    report = v.build_report(rows)
    assert report["evaluated_fixtures"] == 90
    league_signal = report["signals"]["league_shrunk_draw_rate"]
    assert league_signal["auc"] > 0.8
    assert league_signal["discrimination_ready"] is True
    assert report["history_coverage"]["league"]["max"] > 20


def test_history_signal_never_changes_runtime_or_promotion():
    report = v.build_report([])
    assert report["provider_requests_added"] == 0
    assert report["production_promotion_allowed"] is False
    assert report["model_weights_changed"] is False
    assert report["canonical_bet_logic_changed"] is False
