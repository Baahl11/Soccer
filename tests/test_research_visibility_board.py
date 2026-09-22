from mcp_gateway import automation_v91 as v91


def test_research_visibility_turns_pass_event_into_visible_watch_row():
    payload = {
        "actionable_refresh_count": 0,
        "bet_candidate_count": 0,
        "events": [
            {
                "event_type": "SOCCER_REFRESH",
                "stage": "T-90",
                "classification": "PASS",
                "fixture": {
                    "fixture_id": 123,
                    "kickoff": "2026-09-22T18:00:00-06:00",
                    "league": "Test League",
                    "country": "Testland",
                    "home_team": "Home FC",
                    "away_team": "Away FC",
                    "status": "NS",
                },
                "coverage": {"data_tier": "A"},
                "sporting_shortlist": {
                    "shortlisted": False,
                    "rank": 0.0,
                    "tracks": [],
                    "side_edge_score": 45.0,
                    "goal_environment_score": 48.0,
                    "two_way_scoring_score": 43.0,
                    "reason": "SPORTING_SCREEN_BELOW_SHORTLIST",
                },
                "notes": ["SPORT FIRST screen below shortlist threshold."],
            }
        ],
    }

    v91._annotate_research_visibility(payload)

    assert payload["research_visible_count"] == 1
    row = payload["match_table_rows"][0]
    assert row["classification"] == "WATCH"
    assert row["event_classification"] == "PASS"
    assert row["home"] == "Home FC"
    assert row["away"] == "Away FC"
    assert row["market_family"] == "SPORTING_SCREEN"
    assert row["reason"] == "SPORTING_SCREEN_BELOW_SHORTLIST"
    assert payload["research_visibility"]["silent_tick_prevention"] is True
    assert payload["research_visibility"]["raw_event_classification_counts"] == {"PASS": 1}


def test_research_visibility_preserves_best_market_and_scores_for_lean():
    payload = {
        "actionable_refresh_count": 1,
        "bet_candidate_count": 0,
        "events": [
            {
                "event_type": "SOCCER_REFRESH",
                "stage": "T-20",
                "classification": "LEAN",
                "fixture": {
                    "fixture_id": 456,
                    "kickoff": "2026-09-22T20:00:00-06:00",
                    "league": "Test League",
                    "country": "Testland",
                    "home_team": "Over FC",
                    "away_team": "Tempo FC",
                    "status": "NS",
                },
                "coverage": {"data_tier": "B"},
                "sporting_shortlist": {
                    "shortlisted": True,
                    "rank": 74.0,
                    "tracks": ["GOALS_OVER", "TWO_WAY"],
                    "side_edge_score": 51.0,
                    "goal_environment_score": 74.0,
                    "two_way_scoring_score": 68.0,
                    "reason": "SPORTING_SCREEN_PASS",
                },
                "best_market": {
                    "family": "FT_TOTALS",
                    "market": "Goals Over/Under",
                    "selection": "Over",
                    "line": 2.5,
                    "decimal_price": 2.05,
                    "bookmaker": "TestBook",
                    "prob_edge_pp": 3.4,
                    "estimated_ev": 0.061,
                    "tier": "C",
                    "classification": "LEAN",
                    "stake_units": 0.0,
                },
            }
        ],
    }

    v91._annotate_research_visibility(payload)

    row = payload["match_table_rows"][0]
    assert row["classification"] == "LEAN"
    assert row["market_family"] == "FT_TOTALS"
    assert row["market"] == "Goals Over/Under"
    assert row["selection"] == "Over"
    assert row["line"] == 2.5
    assert row["price"] == 2.05
    assert row["goals_score"] == 74.0
    assert row["two_way_score"] == 68.0
    assert payload["research_visibility"]["track_counts"] == {"GOALS_OVER": 1, "TWO_WAY": 1}
