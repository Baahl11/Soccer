from collections import Counter
from datetime import datetime, timezone

from mcp_gateway import formation_tactical_history_backfill_v1 as v


def test_compact_tactical_stats_maps_provider_fields():
    raw = {
        "response": [
            {
                "team": {"id": 1, "name": "Home"},
                "statistics": [
                    {"type": "Total Shots", "value": 14},
                    {"type": "Shots on Goal", "value": 6},
                    {"type": "Blocked Shots", "value": 3},
                    {"type": "Shots insidebox", "value": 9},
                    {"type": "Ball Possession", "value": "61%"},
                    {"type": "Fouls", "value": 10},
                    {"type": "Yellow Cards", "value": 2},
                ],
            },
            {
                "team": {"id": 2, "name": "Away"},
                "statistics": [
                    {"type": "Total Shots", "value": 8},
                    {"type": "Shots on Goal", "value": 2},
                    {"type": "Blocked Shots", "value": 2},
                    {"type": "Shots insidebox", "value": 4},
                    {"type": "Ball Possession", "value": "39%"},
                    {"type": "Fouls", "value": 13},
                    {"type": "Yellow Cards", "value": 3},
                ],
            },
        ]
    }

    compact = v.compact_tactical_stats(raw)

    assert v._sufficient_stats(compact) is True
    assert compact["teams"][0]["possession"] == 61.0
    assert compact["teams"][1]["shots_inside_box"] == 4.0
    assert compact["totals"]["total_shots"] == 22.0
    assert compact["totals"]["yellow_cards"] == 5.0


def test_candidate_selection_prioritizes_undercovered_target_teams():
    now = datetime(2026, 9, 10, tzinfo=timezone.utc)
    rows = [
        {
            "fixture_id": 1,
            "kickoff": now,
            "home_team_id": 10,
            "away_team_id": 20,
        },
        {
            "fixture_id": 2,
            "kickoff": now,
            "home_team_id": 30,
            "away_team_id": 40,
        },
        {
            "fixture_id": 3,
            "kickoff": now,
            "home_team_id": 10,
            "away_team_id": 99,
        },
    ]
    counts = Counter({10: 2, 20: 2, 30: 0, 40: 0})

    selected = v.select_candidates(
        rows,
        team_ids={10, 20, 30, 40},
        prior_counts=counts,
        max_fixtures=2,
    )

    assert selected[0]["fixture_id"] == 2
    assert len(selected) == 2


def test_make_event_is_research_only_and_preserves_provenance():
    kickoff = datetime(2026, 9, 1, 12, tzinfo=timezone.utc)
    candidate = {
        "fixture_id": 7,
        "kickoff": kickoff,
        "league_id": 39,
        "league": "League",
        "country": "Country",
        "season": 2026,
        "round": "Round 1",
        "home_team_id": 1,
        "home_team": "Home",
        "away_team_id": 2,
        "away_team": "Away",
    }
    tactical = {
        "teams": [
            {
                "team_id": 1,
                "total_shots": 10.0,
                "shots_on_goal": 4.0,
                "blocked_shots": 2.0,
                "possession": 55.0,
                "fouls": 9.0,
                "yellow_cards": 1.0,
            },
            {
                "team_id": 2,
                "total_shots": 8.0,
                "shots_on_goal": 3.0,
                "blocked_shots": 1.0,
                "possession": 45.0,
                "fouls": 11.0,
                "yellow_cards": 2.0,
            },
        ],
        "totals": {},
        "provider": "API_FOOTBALL",
        "endpoint": "fixtures/statistics",
    }
    retrieved = datetime(2026, 10, 7, 13, tzinfo=timezone.utc)

    event = v.make_event(
        candidate,
        tactical,
        retrieved_at=retrieved,
        provider_daily_remaining=500,
    )

    assert event["stage"] == "FM4_TACTICAL_BACKFILL"
    assert event["classification"] == "PASS"
    assert event["bet_eligible"] is False
    assert event["decision_weight"] == 0.0
    assert event["backfill"]["retroactive_prediction_rewrite"] is False
    assert event["backfill"]["retroactive_market_created"] is False
    assert event["backfill"]["retroactive_bet_created"] is False
    assert event["postgame_tactical_stats"]["retrieved_at"] == retrieved.isoformat()
    assert event["fixture"]["kickoff"] == kickoff.isoformat()


def test_incomplete_provider_stats_do_not_qualify():
    compact = {
        "teams": [
            {"team_id": 1, "total_shots": 10, "shots_on_goal": 4},
            {"team_id": 2, "total_shots": 8, "shots_on_goal": 2},
        ]
    }
    assert v._sufficient_stats(compact) is False
