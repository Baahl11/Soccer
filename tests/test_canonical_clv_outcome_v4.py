from mcp_gateway import canonical_clv_outcome_v4 as v


def _final_row(fixture_id, home_goals, away_goals, generated="2026-10-06T20:00:00+00:00"):
    return {
        "fixture_id": fixture_id,
        "generated_at_utc": generated,
        "result": {"goals": {"home": home_goals, "away": away_goals}},
    }


def test_build_grades_one_x_two_and_team_totals_without_promoting():
    clv = [
        {
            "fixture_id": 1,
            "market_family": "1X2",
            "market": "Match Winner",
            "selection": "home",
            "home_team": "Home FC",
            "away_team": "Away FC",
            "signal_price": 2.0,
            "signal_fair_probability": 0.55,
            "clv_probability_pp": 1.2,
            "entry_timestamp": "2026-10-06T10:00:00+00:00",
        },
        {
            "fixture_id": 2,
            "market_family": "HOME_TT",
            "market": "Total - Home",
            "selection": "OVER",
            "line": 1.5,
            "home_team": "Alpha",
            "away_team": "Beta",
            "signal_price": 1.8,
            "signal_fair_probability": 0.60,
            "clv_probability_pp": 0.8,
            "entry_timestamp": "2026-10-06T10:00:00+00:00",
        },
        {
            "fixture_id": 2,
            "market_family": "AWAY_TT",
            "market": "Total - Away",
            "selection": "UNDER",
            "line": 0.5,
            "home_team": "Alpha",
            "away_team": "Beta",
            "signal_price": 2.2,
            "signal_fair_probability": 0.45,
            "clv_probability_pp": -0.2,
            "entry_timestamp": "2026-10-06T10:01:00+00:00",
        },
    ]
    signals = [_final_row(1, 2, 1), _final_row(2, 2, 0)]

    ledger, summary = v.build(clv, signals)

    assert len(ledger) == 3
    assert summary["production_promotion_allowed"] is False
    assert summary["model_weights_changed"] is False
    assert summary["families"]["1X2"]["settled"] == 1
    assert summary["families"]["1X2"]["win"] == 1
    assert summary["families"]["TEAM_TOTALS"]["settled"] == 2
    assert summary["families"]["TEAM_TOTALS"]["win"] == 2
    assert summary["families"]["TEAM_TOTALS"]["by_team_role"]["HOME"]["win"] == 1
    assert summary["families"]["TEAM_TOTALS"]["by_team_role"]["AWAY"]["win"] == 1


def test_team_totals_dedupes_same_exact_signal_by_earliest_entry():
    clv = [
        {
            "fixture_id": 3,
            "market_family": "HOME_TT",
            "market": "Total - Home",
            "selection": "OVER",
            "line": 1.5,
            "home_team": "Alpha",
            "away_team": "Beta",
            "signal_price": 2.1,
            "signal_fair_probability": 0.48,
            "entry_timestamp": "2026-10-06T11:00:00+00:00",
            "bookmaker": "Later",
        },
        {
            "fixture_id": 3,
            "market_family": "HOME_TT",
            "market": "Total - Home",
            "selection": "OVER",
            "line": 1.5,
            "home_team": "Alpha",
            "away_team": "Beta",
            "signal_price": 1.9,
            "signal_fair_probability": 0.52,
            "entry_timestamp": "2026-10-06T10:00:00+00:00",
            "bookmaker": "Earlier",
        },
    ]

    ledger, summary = v.build(clv, [_final_row(3, 2, 0)])

    assert len(ledger) == 1
    assert summary["duplicate_exact_rows_collapsed"] == 1
    assert ledger[0]["signal_price"] == 1.9
    assert ledger[0]["entry_timestamp"] == "2026-10-06T10:00:00+00:00"


def test_team_total_multiple_lines_are_kept_but_fixture_weighting_is_explicit():
    clv = [
        {
            "fixture_id": 4,
            "market_family": "HOME_TT",
            "selection": "OVER",
            "line": 0.5,
            "home_team": "Alpha",
            "away_team": "Beta",
            "signal_price": 1.5,
            "signal_fair_probability": 0.70,
            "entry_timestamp": "2026-10-06T10:00:00+00:00",
        },
        {
            "fixture_id": 4,
            "market_family": "HOME_TT",
            "selection": "OVER",
            "line": 1.5,
            "home_team": "Alpha",
            "away_team": "Beta",
            "signal_price": 2.5,
            "signal_fair_probability": 0.40,
            "entry_timestamp": "2026-10-06T10:00:00+00:00",
        },
    ]

    _, summary = v.build(clv, [_final_row(4, 1, 0)])
    team = summary["families"]["TEAM_TOTALS"]

    assert team["rows"] == 2
    assert team["unique_fixtures"] == 1
    assert team["settled"] == 2
    assert team["fixture_equal_weight_roi"]["unique_fixtures"] == 1
    assert "MULTIPLE_TEAM_TOTAL_LINES_WITHIN_ONE_FIXTURE_ARE_CORRELATED" in summary["policy"]


def test_missing_final_remains_ungraded_and_does_not_fake_probability_score():
    clv = [
        {
            "fixture_id": 5,
            "market_family": "1X2",
            "selection": "away",
            "home_team": "Home FC",
            "away_team": "Away FC",
            "signal_price": 3.0,
            "signal_fair_probability": 0.35,
            "entry_timestamp": "2026-10-06T10:00:00+00:00",
        }
    ]

    ledger, summary = v.build(clv, [])

    assert ledger[0]["settlement_status"] == "NO_FINAL"
    assert ledger[0]["settled"] is False
    assert ledger[0]["brier"] is None
    assert summary["families"]["1X2"]["settled"] == 0
    assert summary["families"]["1X2"]["probability_scored_rows"] == 0


def test_non_target_clv_families_are_not_pulled_into_p1a_report():
    clv = [
        {
            "fixture_id": 6,
            "market_family": "FT_TOTALS",
            "selection": "OVER",
            "line": 2.5,
        }
    ]
    ledger, summary = v.build(clv, [_final_row(6, 2, 2)])
    assert ledger == []
    assert summary["raw_supported_clv_rows"] == 0
    assert summary["families"]["1X2"]["rows"] == 0
    assert summary["families"]["TEAM_TOTALS"]["rows"] == 0
