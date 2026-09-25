# cards/props evidence audit v141
from mcp_gateway import research_derivative_postgres_audit as v


def test_classify_research_derivative_markets():
    assert v.classify_market("Total Yellow Cards") == "CARDS"
    assert v.classify_market("Booking Points") == "CARDS"
    assert v.classify_market("Player Cards") == "PLAYER_CARDS"
    assert v.classify_market("Player Shots") == "SHOTS"
    assert v.classify_market("Player Shots On Target") == "SOT"
    assert v.classify_market("Anytime Goalscorer") == "GOALSCORER_ANYTIME"
    assert v.classify_market("Player Assists") == "ASSISTS"
    assert v.classify_market("Goalkeeper Saves") == "GK_SAVES"
    assert v.classify_market("Home Anytime Goal Scorer") == "GOALSCORER_ANYTIME"
    assert v.classify_market("Away First Goal Scorer") == "GOALSCORER_FIRST"
    assert v.classify_market("Home Last Goal Scorer") == "GOALSCORER_LAST"
    assert v.classify_market("Cards Asian Handicap") == "CARDS"
    assert v.classify_market("Cards European Handicap") == "CARDS"
    assert v.classify_market("First Card Received (3 way)") == "CARDS"
    assert v.classify_market("RCARD") == "CARDS"
    assert v.classify_market("ShotOnTarget Handicap") == "TEAM_SOT"
    assert v.classify_market("ShotOnTarget 1x2") == "TEAM_SOT"
    assert v.classify_market("Total ShotOnGoal") == "TEAM_SOT"
    assert v.classify_market("Total Shots") == "TEAM_SHOTS"
    assert v.classify_market("Match Winner") is None
    assert v.classify_market("Total Corners") is None


def test_value_line_supports_sidecar_and_normalized_shapes():
    assert v.value_line({"parsed_line": 2.5}) == 2.5
    assert v.value_line({"line": 3.5}) == 3.5
    assert v.value_line({"handicap": "4.5"}) == 4.5
    assert v.value_line({"selection": "Player A Over 1.5"}) == 1.5
    assert v.value_line({"raw_selection": "Under 5.5"}) == 5.5
    assert v.value_line({"selection": "Player B To Be Booked"}) is None


def test_summarize_rows_tracks_unique_fixture_line_and_confirmed_xi_coverage():
    rows = [
        {
            "fixture_id": 100,
            "captured_at": "2026-09-25T10:00:00+00:00",
            "stage": "T-20",
            "bookmaker": "Book A",
            "market": "Player Shots",
            "values": [
                {"selection": "Player A Over 2.5", "price": "1.90", "parsed_line": 2.5},
                {"selection": "Player A Under 2.5", "price": "1.90", "parsed_line": 2.5},
            ],
            "provider_update": "2026-09-25T09:55:00+00:00",
            "pre_kickoff": True,
            "confirmed_xi_before_market": True,
        },
        {
            "fixture_id": 100,
            "captured_at": "2026-09-25T10:05:00+00:00",
            "stage": "T-10",
            "bookmaker": "Book B",
            "market": "Player Shots On Target",
            "values": [
                {"raw_selection": "Player A Over 0.5", "decimal_price": 1.7, "line": 0.5},
            ],
            "provider_update": "2026-09-25T10:00:00+00:00",
            "pre_kickoff": True,
            "confirmed_xi_before_market": True,
        },
        {
            "fixture_id": 101,
            "captured_at": "2026-09-25T10:00:00+00:00",
            "stage": "T-40",
            "bookmaker": "Book A",
            "market": "Total Yellow Cards",
            "values": [
                {"selection": "Over 4.5", "price": "1.95", "parsed_line": 4.5},
            ],
            "provider_update": None,
            "pre_kickoff": True,
            "confirmed_xi_before_market": False,
        },
        {
            "fixture_id": 102,
            "captured_at": "2026-09-25T10:00:00+00:00",
            "stage": "POSTGAME",
            "bookmaker": "Book A",
            "market": "Goalkeeper Saves",
            "values": [{"selection": "Keeper A Over 3.5", "price": "1.88", "parsed_line": 3.5}],
            "provider_update": "2026-09-25T09:58:00+00:00",
            "pre_kickoff": False,
            "confirmed_xi_before_market": True,
        },
    ]

    report = v.summarize_rows(rows, lookback_days=180)

    assert report["market_snapshot_rows"] == 4
    assert report["unique_fixtures"] == 3
    assert report["confirmed_xi_pre_kickoff_unique_fixtures"] == 1
    assert report["exact_line_value_rows"] == 5

    shots = report["families"]["SHOTS"]
    assert shots["market_snapshot_rows"] == 1
    assert shots["unique_fixtures"] == 1
    assert shots["confirmed_xi_pre_kickoff_unique_fixtures"] == 1
    assert shots["exact_line_value_rows"] == 2
    assert shots["exact_observed_market_history_materialized"] is True

    sot = report["families"]["SOT"]
    assert sot["unique_fixtures"] == 1
    assert sot["confirmed_xi_pre_kickoff_unique_fixtures"] == 1

    cards = report["families"]["CARDS"]
    assert cards["unique_fixtures"] == 1
    assert cards["provider_update_unique_fixtures"] == 0
    assert cards["confirmed_xi_overlap_materialized"] is False

    gk = report["families"]["GK_SAVES"]
    assert gk["unique_fixtures"] == 1
    assert gk["pre_kickoff_unique_fixtures"] == 0
    assert gk["confirmed_xi_pre_kickoff_unique_fixtures"] == 0



def test_summarize_rows_exposes_unclassified_taxonomy_gaps():
    rows = [
        {
            "fixture_id": 201,
            "market": "Total Yellow Cards",
            "values": [{"selection": "Over 4.5", "price": "1.90", "parsed_line": 4.5}],
            "pre_kickoff": True,
            "confirmed_xi_before_market": False,
        },
        {
            "fixture_id": 202,
            "market": "Player Total Passes",
            "values": [{"selection": "Player A Over 45.5", "price": "1.90"}],
            "pre_kickoff": True,
            "confirmed_xi_before_market": True,
        },
        {
            "fixture_id": 203,
            "market": "Shots Inside Box",
            "values": [{"selection": "Over 7.5", "price": "1.85"}],
            "pre_kickoff": True,
            "confirmed_xi_before_market": False,
        },
    ]

    report = v.summarize_rows(rows, lookback_days=180)

    assert report["candidate_rows"] == 3
    assert report["classified_rows"] == 1
    assert report["unclassified_rows"] == 2
    assert report["market_name_counts"]["Total Yellow Cards"] == 1
    assert report["unclassified_market_name_counts"]["Player Total Passes"] == 1
    assert report["unclassified_market_name_counts"]["Shots Inside Box"] == 1
    assert len(report["unclassified_sample_rows"]) == 2


def test_xi_alignment_requires_exact_confirmed_starter_and_goalkeeper_role():
    lineup = {
        "both_xi_confirmed": True,
        "teams": [
            {
                "team_id": 10,
                "team": "Home",
                "starters": [
                    {"id": 501, "name": "Ángel Di María", "pos": "F"},
                    {"id": 502, "name": "Keeper One", "pos": "G"},
                ],
            },
            {
                "team_id": 20,
                "team": "Away",
                "starters": [
                    {"id": 601, "name": "Player B", "pos": "M"},
                    {"id": 602, "name": "Keeper Two", "pos": "G"},
                ],
            },
        ],
    }

    shot = v.align_value_to_confirmed_xi(
        {"selection": "Angel Di Maria Over 2.5", "price": "1.90", "parsed_line": 2.5},
        lineup_payload=lineup,
        family="SHOTS",
    )
    assert shot["xi_alignment_status"] == "MATCHED_CONFIRMED_XI"
    assert shot["player_id"] == 501
    assert shot["team_id"] == 10
    assert shot["confirmed_starter"] is True

    missing = v.align_value_to_confirmed_xi(
        {"selection": "Bench Player Over 1.5", "price": "1.80", "parsed_line": 1.5},
        lineup_payload=lineup,
        family="SHOTS",
    )
    assert missing["xi_alignment_status"] == "PLAYER_NOT_MATCHED_TO_CONFIRMED_XI"

    wrong_gk = v.align_value_to_confirmed_xi(
        {"selection": "Angel Di Maria Over 2.5", "price": "2.10", "parsed_line": 2.5},
        lineup_payload=lineup,
        family="GK_SAVES",
    )
    assert wrong_gk["xi_alignment_status"] == "MATCHED_NON_GOALKEEPER"

    right_gk = v.align_value_to_confirmed_xi(
        {"selection": "Keeper One Over 3.5", "price": "1.88", "parsed_line": 3.5},
        lineup_payload=lineup,
        family="GK_SAVES",
    )
    assert right_gk["xi_alignment_status"] == "MATCHED_CONFIRMED_XI"
    assert right_gk["position"] == "G"


def test_summarize_rows_counts_only_priced_xi_aligned_player_values():
    lineup = {
        "both_xi_confirmed": True,
        "teams": [
            {
                "team_id": 10,
                "team": "Home",
                "starters": [{"id": 501, "name": "Player A", "pos": "F"}],
            },
            {
                "team_id": 20,
                "team": "Away",
                "starters": [{"id": 601, "name": "Player B", "pos": "M"}],
            },
        ],
    }
    rows = [
        {
            "fixture_id": 300,
            "market": "Player Shots",
            "values": [
                {"selection": "Player A Over 2.5", "price": "1.90", "parsed_line": 2.5},
                {"selection": "Bench Player Over 1.5", "price": "1.80", "parsed_line": 1.5},
            ],
            "pre_kickoff": True,
            "confirmed_xi_before_market": True,
            "confirmed_lineup_payload": lineup,
        }
    ]

    report = v.summarize_rows(rows, lookback_days=180)
    shots = report["families"]["SHOTS"]
    assert shots["priced_value_rows"] == 2
    assert shots["xi_aligned_priced_value_rows"] == 1
    assert shots["xi_aligned_exact_line_value_rows"] == 1
    assert shots["confirmed_xi_player_aligned_unique_fixtures"] == 1
    assert shots["player_xi_alignment_materialized"] is True
