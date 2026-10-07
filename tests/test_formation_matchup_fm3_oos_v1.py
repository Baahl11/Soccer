from mcp_gateway import formation_matchup_fm3_oos_v1 as fm3


def _row(i, *, shots=True, matchup="4-3-3 vs 4-2-3-1"):
    row = {
        "fixture_id": i,
        "kickoff_local": f"2026-09-{i:02d}T12:00:00+00:00",
        "league_id": 39,
        "league": "Test League",
        "home_team_id": 10,
        "away_team_id": 20,
        "home_formation": "4-3-3",
        "away_formation": "4-2-3-1",
        "matchup_key": matchup,
        "home_goals": 2,
        "away_goals": 1,
        "total_goals": 3,
        "home_sot": 5,
        "away_sot": 3,
        "total_sot": 8,
    }
    if shots:
        row.update(
            {
                "home_shots": 14,
                "away_shots": 9,
                "total_shots": 23,
            }
        )
    else:
        row.update(
            {
                "home_shots": None,
                "away_shots": None,
                "total_shots": None,
            }
        )
    return row


def _source(n=10):
    return {
        "model_version": "FORMATION_MATCHUP_ENGINE_V1.0.0",
        "status": "RESEARCH_ONLY_FORMATION_MATCHUP_ENGINE",
        "fixtures_with_verified_formation_pair_and_final": n,
        "rows": [_row(i) for i in range(1, n + 1)],
    }


def test_fm3_requires_eight_prior_matchup_residuals_and_is_prior_only():
    report10 = fm3.build_report(_source(10))
    shots10 = report10["targets"]["SHOTS"]

    assert shots10["observed_rows"] == 10
    assert shots10["formation_adjusted_evaluations"] == 1
    eval10 = [
        row
        for row in report10["evaluation_rows"]
        if row["target"] == "SHOTS" and row["fixture_id"] == 10
    ][0]
    assert eval10["prior_matchup_n"] == 8

    source11 = _source(11)
    source11["rows"][-1]["home_shots"] = 100
    source11["rows"][-1]["away_shots"] = 80
    report11 = fm3.build_report(source11)
    eval10_after_future = [
        row
        for row in report11["evaluation_rows"]
        if row["target"] == "SHOTS" and row["fixture_id"] == 10
    ][0]

    assert eval10_after_future["baseline_home"] == eval10["baseline_home"]
    assert eval10_after_future["baseline_away"] == eval10["baseline_away"]
    assert eval10_after_future["challenger_home"] == eval10["challenger_home"]
    assert eval10_after_future["challenger_away"] == eval10["challenger_away"]


def test_fm3_missing_shots_are_not_zero():
    source = _source(10)
    source["rows"][4] = _row(5, shots=False)
    report = fm3.build_report(source)

    shots = report["targets"]["SHOTS"]
    goals = report["targets"]["GOALS"]
    assert shots["observed_rows"] == 9
    assert goals["observed_rows"] == 10
    assert shots["baseline_all_rows"]["home"]["n"] < goals["baseline_all_rows"]["home"]["n"]


def test_fm3_multiplier_shrinks_and_clips():
    high = fm3._formation_multiplier([10.0] * fm3.MIN_PRIOR_MATCHUP_N)
    low = fm3._formation_multiplier([0.0] * fm3.MIN_PRIOR_MATCHUP_N)
    assert high == fm3.FORMATION_RATIO_CLIP[1]
    assert low == fm3.FORMATION_RATIO_CLIP[0]
    assert fm3._formation_multiplier([1.0] * (fm3.MIN_PRIOR_MATCHUP_N - 1)) is None


def test_fm3_is_research_only_and_market_independent():
    report = fm3.build_report(_source(12))

    assert report["status"] == "RESEARCH_ONLY_FM3_SHOTS_SOT_GOALS"
    assert report["health"]["odds_consumed"] is False
    assert report["health"]["market_prices_consumed"] is False
    assert report["health"]["production_enabled"] is False
    assert report["health"]["decision_weight"] == 0.0
    assert report["health"]["provider_requests_added"] == 0
    assert report["health"]["model_weights_changed"] is False
    assert report["health"]["canonical_bet_logic_changed"] is False
    for target in ("SHOTS", "SOT", "GOALS"):
        assert report["targets"][target]["production_enabled"] is False
        assert report["targets"][target]["decision_weight"] == 0.0


def test_fm3_baseline_does_not_require_formation_history():
    source = _source(6)
    for idx, row in enumerate(source["rows"]):
        row["matchup_key"] = f"UNIQUE-{idx}"
    report = fm3.build_report(source)

    shots = report["targets"]["SHOTS"]
    assert shots["baseline_evaluations"] >= 5
    assert shots["formation_adjusted_evaluations"] == 0
