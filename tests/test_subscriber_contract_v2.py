from __future__ import annotations

from mcp_gateway import subscriber_app_v4
from mcp_gateway import subscriber_contract_v2
from mcp_gateway import subscription_entitlements_v4


def _payload(rows):
    return {
        "status": "ok",
        "version": "test-pipeline",
        "model_version": "test-runtime-model",
        "generated_at_utc": "2026-10-07T17:00:00Z",
        "match_table_rows": rows,
        "market_mismatch_rows": [],
    }


def _base_row(**extra):
    row = {
        "fixture_id": 101,
        "status": "NS",
        "stage": "T-10",
        "kickoff": "2026-10-07T20:00:00Z",
        "league": "Test League",
        "country": "Test",
        "home_team_id": 10,
        "home_team": "Alpha",
        "away_team_id": 20,
        "away_team": "Beta",
        "market_family": "1X2",
        "market": "Match Winner",
        "selection": "Alpha",
        "line": None,
        "model_signal": "STRONG",
        "model_signal_score": 0.91,
        "model_version": "soccer-test-v1",
    }
    row.update(extra)
    return row


def _pro_entitlement():
    return {
        "ok": True,
        "authenticated": True,
        "effective_plan": subscription_entitlements_v4.PRO_PLAN,
        "feature_access": subscription_entitlements_v4.feature_access(
            subscription_entitlements_v4.PRO_PLAN
        ),
        "user": {"id": "test-user", "role": "USER"},
    }


def test_v2_keeps_raw_shrunk_calibrated_and_market_probabilities_distinct():
    row = _base_row(
        classification="BET",
        tier="B",
        decimal_price=1.91,
        bookmaker="Book",
        p_raw=0.64,
        p_shrunk=0.60,
        p_model_calibrated=0.61,
        p_market_fair=0.54,
        p_breakeven=0.52356,
        prob_edge_pp=6.0,
        estimated_ev=0.074,
        availability_confidence=0.92,
        data_tier="A",
    )
    candidate = subscriber_contract_v2.adapt_candidate(row)
    projections = candidate["projections"]

    assert projections["raw_sport_probability"] == 0.64
    assert projections["market_shrunk_probability"] == 0.60
    assert projections["calibrated_model_probability"] == 0.61
    assert projections["fair_market_probability"] == 0.54
    assert projections["breakeven_probability"] == 0.52356
    assert projections["probability_edge_pp"] == 6.0
    assert candidate["decision"]["classification"] == "BET"
    assert candidate["decision"]["tier"] == "B"


def test_v2_does_not_invent_raw_or_shrunk_probability_from_calibrated_probability():
    candidate = subscriber_contract_v2.adapt_candidate(
        _base_row(
            classification="WATCH",
            p_model_calibrated=0.62,
            p_market_fair=0.55,
        )
    )
    projections = candidate["projections"]

    assert projections["raw_sport_probability"] is None
    assert projections["market_shrunk_probability"] is None
    assert projections["calibrated_model_probability"] == 0.62
    assert projections["fair_market_probability"] == 0.55


def test_v2_ready_or_strong_signal_is_never_promoted_to_bet_without_explicit_classification():
    row = _base_row(
        execution_status="READY",
        p_raw=0.67,
        p_shrunk=0.62,
        p_market_fair=0.55,
        prob_edge_pp=7.0,
        decimal_price=1.90,
    )
    contract = subscriber_contract_v2.build_contract(_payload([row]), _pro_entitlement())

    assert contract["counts"]["picks"] == 0
    assert contract["picks"]["rows"] == []


def test_v2_only_explicit_bet_and_lean_rows_enter_their_respective_resources():
    bet = _base_row(
        fixture_id=1,
        classification="BET",
        tier="B",
        decimal_price=1.88,
        p_raw=0.63,
        p_shrunk=0.59,
        p_market_fair=0.53,
        prob_edge_pp=6.0,
    )
    lean = _base_row(
        fixture_id=2,
        classification="LEAN",
        tier="B",
        decimal_price=1.92,
        p_raw=0.60,
        p_shrunk=0.57,
        p_market_fair=0.53,
        prob_edge_pp=4.0,
    )
    ready_unclassified = _base_row(fixture_id=3, execution_status="READY")

    contract = subscriber_contract_v2.build_contract(
        _payload([bet, lean, ready_unclassified]),
        _pro_entitlement(),
    )

    assert contract["counts"]["picks"] == 1
    assert contract["counts"]["leans"] == 1
    assert contract["picks"]["rows"][0]["fixture"]["fixture_id"] == 1
    assert contract["leans"]["rows"][0]["fixture"]["fixture_id"] == 2


def test_v2_wait_states_are_watch_not_bet():
    row = _base_row(
        fixture_id=4,
        execution_status="WAIT_XI",
        reason="Waiting for confirmed XI",
        p_raw=0.66,
    )
    contract = subscriber_contract_v2.build_contract(_payload([row]), _pro_entitlement())

    assert contract["counts"]["picks"] == 0
    assert contract["counts"]["watches"] == 1
    assert contract["watches"]["rows"][0]["decision"]["execution_status"] == "WAIT_XI"


def test_v2_missing_availability_fields_stay_not_verified():
    candidate = subscriber_contract_v2.adapt_candidate(_base_row(classification="WATCH"))
    availability = candidate["availability"]

    assert availability["confidence"] is None
    assert availability["data_tier"] == "NOT VERIFIED"
    assert availability["lineup_status"] == "NOT VERIFIED"
    assert availability["starting_xi_status"] == "NOT VERIFIED"
    assert availability["goalkeeper_status"] == "NOT VERIFIED"
    assert availability["injury_status"] == "NOT VERIFIED"
    assert availability["weather_status"] == "NOT VERIFIED"


def test_v2_does_not_derive_breakeven_probability_from_price():
    candidate = subscriber_contract_v2.adapt_candidate(
        _base_row(classification="BET", decimal_price=2.00)
    )
    assert candidate["projections"]["breakeven_probability"] is None


def test_v2_price_format_is_only_claimed_when_verified_by_field_or_explicit_format():
    decimal = subscriber_contract_v2.adapt_candidate(
        _base_row(classification="BET", decimal_price=1.95)
    )
    generic = subscriber_contract_v2.adapt_candidate(
        _base_row(classification="BET", price=-110)
    )

    assert decimal["market"]["price"]["format"] == "DECIMAL"
    assert generic["market"]["price"]["format"] == "NOT VERIFIED"


def test_v2_free_access_redacts_picks_and_leans_but_keeps_public_watch_state():
    bet = _base_row(
        fixture_id=11,
        classification="BET",
        decimal_price=1.90,
        p_raw=0.65,
        p_shrunk=0.60,
        p_market_fair=0.54,
        prob_edge_pp=6.0,
    )
    watch = _base_row(
        fixture_id=12,
        execution_status="WAIT_PRICE",
        market_family="BTTS",
        market="Both Teams To Score",
        p_raw=0.61,
        p_market_fair=0.55,
    )
    contract = subscriber_contract_v2.build_contract(
        _payload([bet, watch]),
        subscriber_app_v4.anonymous_entitlement(),
    )

    assert contract["picks"]["locked"] is True
    assert contract["picks"]["total"] == 1
    assert contract["picks"]["rows"] == []
    assert contract["watches"]["total"] == 1
    assert "projections" not in contract["watches"]["rows"][0]
    assert contract["slate"]["premium_values_redacted"] is True


def test_v2_contract_firewall_is_explicit():
    contract = subscriber_contract_v2.contract()

    assert contract["frontend_creates_bet_or_lean"] is False
    assert contract["raw_sport_probability_is_distinct"] is True
    assert contract["market_shrunk_probability_is_distinct"] is True
    assert contract["fair_market_probability_is_distinct"] is True
    assert contract["provider_requests_added"] == 0
    assert contract["canonical_bet_logic_changed"] is False
    assert contract["model_weights_changed"] is False
    assert contract["production_promotion_allowed"] is False


def test_v2_percentage_point_fields_are_not_heuristically_rescaled():
    candidate = subscriber_contract_v2.adapt_candidate(
        _base_row(
            classification="BET",
            prob_edge_pp=0.8,
            estimated_ev=0.012,
            ev_pct=1.2,
        )
    )
    projections = candidate["projections"]

    assert projections["probability_edge_pp"] == 0.8
    assert projections["estimated_ev"] == 0.012
    assert projections["estimated_ev_pct"] == 1.2

def test_v2_full_registry_slate_keeps_fixture_without_analysis_visible():
    payload = _payload([
        _base_row(
            fixture_id=101,
            classification="WATCH",
            market_family="1X2",
            market="Match Winner",
            reason="WAIT_MARKET",
        )
    ])
    contract = subscriber_contract_v2.build_contract(payload, _pro_entitlement())
    registry = {
        "status": "FULL_SLATE_READY",
        "slate_date": "2026-10-07",
        "timezone": "America/Mexico_City",
        "rows": [
            {
                "fixture_id": 101,
                "kickoff": "2026-10-07T20:00:00+00:00",
                "status": "NS",
                "league": "Test League",
                "country": "Test",
                "home_team_id": 10,
                "home_team": "Alpha",
                "away_team_id": 20,
                "away_team": "Beta",
            },
            {
                "fixture_id": 202,
                "kickoff": "2026-10-07T22:00:00+00:00",
                "status": "NS",
                "league": "Other League",
                "country": "Test",
                "home_team_id": 30,
                "home_team": "Gamma",
                "away_team_id": 40,
                "away_team": "Delta",
            },
        ],
    }

    result = subscriber_contract_v2._attach_full_registry_slate(contract, payload, registry)

    assert result["counts"]["fixtures"] == 2
    assert result["slate"]["full_slate"] is True
    assert result["slate"]["source"] == "POSTGRES_SOCCER_FIXTURES+PERSISTED_ANALYSIS_COVERAGE"
    assert result["slate"]["rows"][0]["coverage"]["analysis_rows"] == 1
    assert result["slate"]["rows"][0]["coverage"]["markets"] == ["Match Winner"]
    assert result["slate"]["rows"][1]["state"]["display_status"] == "INSUFFICIENT DATA"
    assert result["slate"]["rows"][1]["coverage"]["analysis_rows"] == 0


def test_v2_slate_context_prefers_persisted_local_day_and_timezone():
    day, timezone_name = subscriber_contract_v2._slate_context(
        {
            "generated_at_utc": "2026-10-08T00:15:00Z",
            "generated_at_local": "2026-10-07T18:15:00-06:00",
            "timezone": "America/Mexico_City",
        }
    )

    assert day == "2026-10-07"
    assert timezone_name == "America/Mexico_City"

def test_v2_match_contract_opens_registry_only_fixture_without_fabricating_data():
    payload = _payload([])
    registry_fixture = {
        "fixture_id": 909,
        "kickoff": "2026-10-07T23:00:00+00:00",
        "status": "NS",
        "league": "Registry League",
        "country": "Test",
        "home_team_id": 91,
        "home_team": "Registry Home",
        "away_team_id": 92,
        "away_team": "Registry Away",
    }

    result = subscriber_contract_v2.build_match_contract(
        payload,
        909,
        registry_fixture=registry_fixture,
    )

    assert result is not None
    assert result["status"] == "MATCH_INTELLIGENCE_INSUFFICIENT_DATA"
    assert result["fixture"]["fixture_id"] == 909
    assert result["selected_candidate"] is None
    assert result["projection_ladder"]["raw_sport_probability"] is None
    assert result["market_context"]["candidate_count"] == 0
    assert result["analyst_review"]["human_review_allowed"] is True
    assert result["analyst_review"]["available_sections"] == ["FIXTURE_IDENTITY"]
    assert result["data_disclosure"]["unknown_policy"] == "NOT VERIFIED"


def test_v2_match_contract_exposes_available_markets_for_human_review():
    payload = _payload([
        _base_row(
            fixture_id=333,
            classification="WATCH",
            market_family="BTTS",
            market="Both Teams To Score",
            p_raw=0.61,
        ),
        _base_row(
            fixture_id=333,
            classification="WATCH",
            market_family="FT_TOTALS",
            market="Over/Under",
            p_raw=0.58,
        ),
    ])

    result = subscriber_contract_v2.build_match_contract(payload, 333)

    assert result is not None
    assert "BTTS" in result["analyst_review"]["available_markets"]
    assert "Goals" in result["analyst_review"]["available_markets"]
    assert result["analyst_review"]["analysis_rows"] == 2
    assert result["analyst_review"]["human_review_allowed"] is True

def test_v2_registry_slate_marks_relational_evidence_as_data_available():
    payload = _payload([])
    contract = subscriber_contract_v2.build_contract(payload, _pro_entitlement())
    registry = {
        "status": "FULL_SLATE_READY",
        "slate_date": "2026-10-07",
        "timezone": "America/Mexico_City",
        "rows": [
            {
                "fixture_id": 808,
                "kickoff": "2026-10-07T23:00:00+00:00",
                "status": "NS",
                "league": "Evidence League",
                "country": "Test",
                "home_team_id": 81,
                "home_team": "Evidence Home",
                "away_team_id": 82,
                "away_team": "Evidence Away",
                "evidence_inventory": {
                    "refresh_events": 2,
                    "market_snapshots": 3,
                    "model_runs": 1,
                    "feature_snapshots": 1,
                    "lineup_snapshots": 0,
                    "availability_snapshots": 1,
                    "market_names": ["Match Winner", "Goals Over/Under"],
                },
            }
        ],
    }

    result = subscriber_contract_v2._attach_full_registry_slate(contract, payload, registry)
    row = result["slate"]["rows"][0]

    assert row["state"]["status_code"] == "SPORT_DATA_AVAILABLE"
    assert row["coverage"]["persisted_evidence_count"] == 8
    assert "market" in row["coverage"]["data_sources"]
    assert row["coverage"]["persisted_market_names"] == ["Match Winner", "Goals Over/Under"]


def test_v2_match_contract_hydrates_sport_context_from_persisted_model_run():
    payload = _payload([
        _base_row(
            fixture_id=444,
            classification="WATCH",
            market_family="1X2",
            market="Match Winner",
            p_raw=0.61,
        )
    ])
    relational = {
        "counts": {
            "refresh_events": 1,
            "market_snapshots": 0,
            "model_runs": 1,
            "feature_snapshots": 1,
            "lineup_snapshots": 0,
            "availability_snapshots": 0,
            "market_names": [],
        },
        "model_runs": [
            {
                "run_timestamp": "2026-10-08T18:00:00Z",
                "model_version": "SOCCER EDGE ENGINE v1.7",
                "raw_projection": {
                    "status": "MODELED_LIMITED",
                    "model_version": "SOCCER EDGE ENGINE v1.7",
                    "projection_model": "POISSON_GOAL_RATE_BASELINE",
                    "raw_home_goal_rate": 1.72,
                    "raw_away_goal_rate": 0.94,
                    "raw_home_win_prob": 0.58,
                    "raw_draw_prob": 0.24,
                    "raw_away_win_prob": 0.18,
                    "top_scorelines": [
                        {"home": 1, "away": 0, "prob": 0.14},
                        {"home": 2, "away": 0, "prob": 0.12},
                    ],
                    "screen_scores": {
                        "side_edge_score": 74.0,
                        "goal_environment_score": 63.0,
                        "two_way_scoring_score": 58.0,
                    },
                },
            }
        ],
        "refresh_events": [],
        "feature_snapshots": [],
        "market_snapshots": [],
        "lineup": None,
        "availability": None,
    }

    result = subscriber_contract_v2.build_match_contract(
        payload,
        444,
        relational_evidence=relational,
    )

    assert result is not None
    sport = result["sport_context"]
    assert sport["source"] == "POSTGRES_SOCCER_MODEL_RUNS_RAW_PROJECTION"
    assert sport["goal_rate_semantics"] == "POISSON_LAMBDA_NOT_XG"
    assert sport["expected_goals"]["home"] == 1.72
    assert sport["expected_goals"]["away"] == 0.94
    assert sport["expected_goals"]["xg_verified"] is False
    assert round(sport["outcome_probabilities"]["home"], 6) == 0.58
    assert sport["score_matrix"][0]["score"] == "1-0"
    assert sport["sport_profile"][0]["label"] == "Side edge"
    assert "OUTCOME_PROBABILITIES" in result["analyst_review"]["available_sections"]
    assert "EXPECTED_GOALS" in result["analyst_review"]["available_sections"]


def test_v2_match_contract_uses_registry_identity_when_analysis_row_omits_logos():
    # Regression: React Match Center rendered initials even with real team IDs in the registry.
    row = _base_row(home_team_id=None, away_team_id=None, home_team_logo=None, away_team_logo=None)
    registry = {
        "fixture_id": 101,
        "kickoff": "2026-10-07T20:00:00Z",
        "league": "Test League",
        "country": "Test",
        "home_team_id": 111,
        "home_team": "Alpha",
        "away_team_id": 222,
        "away_team": "Beta",
    }
    result = subscriber_contract_v2.build_match_contract(_payload([row]), 101, registry)
    assert result is not None
    assert result["fixture"]["home_team_id"] == 111
    assert result["fixture"]["away_team_id"] == 222
    assert result["fixture"]["home_team_logo"] == "https://media.api-sports.io/football/teams/111.png"
    assert result["fixture"]["away_team_logo"] == "https://media.api-sports.io/football/teams/222.png"
    assert result["model_weights_changed"] is False
    assert result["canonical_bet_logic_changed"] is False


def test_v2_registry_identity_does_not_overwrite_explicit_analysis_identity():
    row = _base_row()
    selected = subscriber_contract_v2._match_fixture_with_registry_identity(row, {
        "home_team_id": 111, "away_team_id": 222,
        "home_team": "Wrong", "away_team": "Wrong",
    })
    assert selected["home_team_id"] == 10
    assert selected["away_team_id"] == 20
    assert selected["home_team"] == "Alpha"
    assert selected["away_team"] == "Beta"



def test_match_evidence_sections_show_only_actual_snapshot_values():
    sections = subscriber_contract_v2._match_evidence_sections({
        "feature_snapshots": [{
            "captured_at": "2026-10-08T18:00:00Z",
            "data_tier": "B",
            "model_version": "v1",
            "payload": {"features": {
                "team_performance.home_goals_for_avg": {
                    "value": 1.8, "source": "API_FOOTBALL_TEAM_STATS", "sample_n": 16
                },
                "corners.home_avg": {
                    "value": 5.0, "source": "API_FOOTBALL_FIXTURES", "sample_n": 10
                },
                "cards.away_avg": {"value": None, "source": "API_FOOTBALL_FIXTURES"},
                "players.no_matches": {
                    "value": 12, "source": "API_FOOTBALL_TEAM_STATS", "sample_n": 0
                },
                "context.missing": {"value": ""},
                "context.confirmed": {"value": False, "source": "PERSISTED_SNAPSHOT"},
            }},
        }]
    })
    assert [section["category"] for section in sections] == ["TEAMS", "CORNERS", "CONTEXT"]
    assert sections[0]["items"][0]["value"] == 1.8
    assert sections[0]["items"][0]["sample_n"] == 16
    assert sections[0]["items"][0]["source"] == "API_FOOTBALL_TEAM_STATS"
    assert sections[1]["items"][0]["value"] == 5.0
    assert sections[2]["items"][0]["value"] is False
    assert sections[0]["items"][0]["captured_at"] == "2026-10-08T18:00:00Z"


def test_match_evidence_sections_accept_sparse_or_absent_snapshots():
    assert subscriber_contract_v2._match_evidence_sections({}) == []
    assert subscriber_contract_v2._match_evidence_sections({"feature_snapshots": [{"payload": {"features": {}}}]}) == []


def test_fixture_without_analysis_still_exposes_persisted_sport_features():
    registry = {
        "fixture_id": 90210, "kickoff": "2026-10-08T20:00:00Z",
        "league": "Test", "home_team_id": 11, "home_team": "Home",
        "away_team_id": 12, "away_team": "Away",
    }
    relational = {
        "counts": {"feature_snapshots": 1},
        "feature_snapshots": [{"captured_at": "2026-10-08T18:00:00Z",
                               "payload": {"features": {"team_performance.home_wins_total": {
                                   "value": 8, "source": "API_FOOTBALL_TEAM_STATS", "sample_n": 10
                               }}}}],
    }
    result = subscriber_contract_v2.build_match_contract(_payload([]), 90210, registry, relational)
    assert result is not None
    assert result["selected_candidate"] is None
    assert result["evidence_sections"][0]["items"][0]["value"] == 8
    assert result["model_weights_changed"] is False
    assert result["canonical_bet_logic_changed"] is False


def test_registry_only_fixture_retains_existing_persisted_raw_sport_projection():
    """Regression: a later empty pipeline tick must not erase earlier visible sport data."""
    registry = {
        "fixture_id": 1612077,
        "kickoff": "2026-10-08T22:00:00Z",
        "league": "Reserve League",
        "home_team_id": 18681, "home_team": "Boca Juniors Res.",
        "away_team_id": 18683, "away_team": "Colón Res.",
    }
    relational = {
        "counts": {
            "refresh_events": 0, "market_snapshots": 1, "model_runs": 1,
            "feature_snapshots": 1, "lineup_snapshots": 0, "availability_snapshots": 0,
        },
        "model_runs": [{
            "run_timestamp": "2026-10-08T20:00:00Z",
            "model_version": "v1-test",
            "raw_projection": {
                "projection_model": "POISSON_GOAL_RATE_BASELINE",
                "raw_home_win_prob": 0.595,
                "raw_draw_prob": 0.258,
                "raw_away_win_prob": 0.147,
                "raw_home_goal_rate": 1.51,
                "raw_away_goal_rate": 0.60,
                "top_scorelines": [{"home": 1, "away": 0, "prob": 0.183}],
                "screen_scores": {"side_edge_score": 70.0},
            },
        }],
        "feature_snapshots": [{
            "captured_at": "2026-10-08T19:55:00Z",
            "data_tier": "D",
            "payload": {"features": {
                "team_performance.home_goals_for_avg": {
                    "value": 1.8, "source": "API_FOOTBALL_TEAM_STATS", "sample_n": 16,
                }
            }},
        }],
        "market_snapshots": [{
            "captured_at": "2026-10-08T19:00:00Z",
            "bookmaker": "Past-only", "values": [{"odds": 1.8}],
        }],
    }
    result = subscriber_contract_v2.build_match_contract(
        _payload([]), 1612077, registry_fixture=registry, relational_evidence=relational,
    )
    assert result is not None
    assert result["sport_context"]["outcome_probabilities"]["home"] == 0.595
    assert result["sport_context"]["expected_goals"]["home"] == 1.51
    assert result["sport_context"]["expected_goals"]["away"] == 0.60
    assert result["sport_context"]["score_matrix"][0]["score"] == "1-0"
    assert result["sport_context"]["sport_profile"][0]["label"] == "Side edge"
    assert result["sport_context"]["captured_at"] == "2026-10-08T20:00:00Z"
    assert result["sport_context"]["snapshot_scope"] == "HISTORICAL_PERSISTED_SPORT_ONLY"
    assert result["model_context"]["model_version"] == "v1-test"
    assert result["availability"]["data_tier"] == "D"
    assert result["availability"]["confidence"] is None
    assert "OUTCOME_PROBABILITIES" in result["analyst_review"]["available_sections"]
    assert "EXPECTED_GOALS" not in result["analyst_review"]["missing_sections"]
    assert result["evidence_sections"][0]["items"][0]["value"] == 1.8
    assert result["selected_candidate"] is None
    assert result["projection_ladder"]["fair_market_probability"] is None
    assert result["projection_ladder"]["probability_edge_pp"] is None
    assert result["decision_summary"]["classification"] is None
    assert result["canonical_bet_logic_changed"] is False
    assert result["model_weights_changed"] is False


def test_registry_only_fixture_with_no_model_does_not_invent_probabilities():
    registry = {
        "fixture_id": 88, "kickoff": "2026-10-08T22:00:00Z",
        "league": "Test League", "home_team_id": 1, "home_team": "A",
        "away_team_id": 2, "away_team": "B",
    }
    result = subscriber_contract_v2.build_match_contract(
        _payload([]), 88, registry_fixture=registry,
        relational_evidence={"counts": {"model_runs": 0}, "model_runs": [], "feature_snapshots": []},
    )
    assert result["sport_context"]["outcome_probabilities"] is None
    assert result["sport_context"]["expected_goals"] is None
    assert "OUTCOME_PROBABILITIES" in result["analyst_review"]["missing_sections"]
    assert "OUTCOME_PROBABILITIES" not in result["analyst_review"]["available_sections"]
    assert result["selected_candidate"] is None



def test_edge_radar_reconciles_all_registry_fixtures_without_guessing_model_coverage():
    payload = _payload([
        _base_row(fixture_id=101, classification="WATCH", p_raw=0.61,
                  market_family="1X2", market="Match Winner"),
        _base_row(fixture_id=202, classification="WATCH", p_raw=None,
                  market_family="BTTS", market="Both Teams To Score"),
    ])
    contract = subscriber_contract_v2.build_contract(payload, _pro_entitlement())
    rows = [
        {"fixture_id": fid, "kickoff": "2026-11-10T23:00:00Z",
         "status": "NS", "league": "Test", "country": "Test",
         "home_team_id": fid*2, "away_team_id": fid*2+1,
         "home_team": "Home", "away_team": "Away",
         "evidence_inventory": inventory}
        for fid, inventory in (
            (101, {"model_runs": 1, "feature_snapshots": 1,
                   "lineup_snapshots": 0, "availability_snapshots": 0}),
            (202, {"refresh_events": 1}),
            (303, {"model_runs": 1, "lineup_snapshots": 2,
                   "availability_snapshots": 1}),
            (404, {"market_snapshots": 2}),
            (505, {}),
        )
    ]
    result = subscriber_contract_v2._attach_full_registry_slate(
        contract, payload, {"status": "FULL_SLATE_READY",
                            "slate_date": "2026-11-10",
                            "timezone": "America/Mexico_City",
                            "rows": rows}
    )
    radar = result["radar"]
    assert radar["status"] == "PERSISTED_EVIDENCE_AUDIT_READY"
    assert radar["eligible_registry_count"] == 5
    assert radar["reconciliation_complete"] is True
    assert radar["raw_sport_recorded_count"] == 1
    assert radar["partial_analysis_count"] == 1
    assert radar["sport_only_count"] == 1
    assert radar["market_only_count"] == 1
    assert radar["no_persisted_analysis_count"] == 1
    states = {row["fixture"]["fixture_id"]: row["screening"] for row in result["slate"]["rows"]}
    assert states[101]["status_code"] == "RAW_SPORT_RECORDED"
    assert states[101]["classification"] == "RESEARCH_ONLY_NOT_BET"
    assert states[101]["raw_projection_recorded"] is True
    assert states[202]["status_code"] == "PARTIAL_ANALYSIS"
    assert "RAW_SPORT_NOT_RECORDED" in states[202]["missing_evidence_codes"]
    assert states[303]["status_code"] == "SPORT_EVIDENCE_ONLY"
    assert "LINEUP_SNAPSHOT_NOT_RECORDED" not in states[303]["missing_evidence_codes"]
    # A lineup snapshot exists; Radar does NOT infer both XI/GK confirmed.
    assert "both_xi_confirmed" not in states[303]
    assert states[404]["status_code"] == "MARKET_ONLY"
    assert "sin modelo deportivo" in states[404]["reason"]
    assert states[505]["status_code"] == "NO_PERSISTED_ANALYSIS"
    assert "puede faltar" in states[505]["reason"]
    assert radar["not_a_full_sport_screening_claim"] is True
    assert radar["not_real_time_provider_coverage"] is True
    assert radar["provider_requests_added"] == 0
    assert radar["production_promotion_allowed"] is False


def test_edge_radar_does_not_misreport_complete_coverage_when_registry_missing():
    contract = subscriber_contract_v2.build_contract(_payload([]), _pro_entitlement())
    result = subscriber_contract_v2._attach_full_registry_slate(contract, _payload([]), None)
    assert result["radar"]["status"] == "REGISTRY_UNAVAILABLE"
    assert result["radar"]["eligible_registry_count"] is None
    assert result["radar"]["raw_sport_recorded_count"] is None
    assert result["slate"]["full_slate"] is False
    assert result["radar"]["production_promotion_allowed"] is False


def test_highlightly_research_visible_to_owner_only_and_never_promotes_bet():
    payload = _payload([])
    admin = _pro_entitlement()
    admin["user"] = {"id": "admin", "role": "OWNER"}
    contract=subscriber_contract_v2.build_contract(payload,admin)
    registry={"status":"FULL_SLATE_READY","slate_date":"2026-11-11",
        "timezone":"America/Mexico_City","rows":[{
            "fixture_id":1581906,"kickoff":"2026-11-11T09:30:00Z","status":"NS",
            "league":"Persha Liga","country":"Ukraine",
            "home_team_id":100,"away_team_id":200,
            "home_team":"Chernihiv","away_team":"Inhulets",
        }]}
    contract=subscriber_contract_v2._attach_full_registry_slate(contract,payload,registry)
    supplement={"1581906":{
        "identity_status":"IDENTITY_EXACT_VERIFIED",
        "evidence_status":"PREMATCH_RESEARCH_LIMITED",
        "families_research":["FORM"],
        "captured_at":"2026-11-11T05:02:00+00:00",
    }}
    contract=subscriber_contract_v2._attach_highlightly_owner_research(contract,supplement)
    assert contract["radar"]["highlightly_research"]["prematch_research_limited_count"]==1
    assert contract["counts"]["picks"]==0
    assert contract["radar"]["raw_sport_recorded_count"]==0
    assert contract["slate"]["rows"][0]["screening"]["supplementary_sport_research"]["families_research"]==["FORM"]
    assert contract["slate"]["rows"][0]["screening"]["supplementary_sport_research"]["model_input_allowed"] is False
    assert contract["slate"]["rows"][0]["screening"]["supplementary_sport_research"]["bet_eligible"] is False

    free=subscriber_contract_v2.build_contract(payload,subscriber_app_v4.anonymous_entitlement())
    free=subscriber_contract_v2._attach_full_registry_slate(free,payload,registry)
    before=free["slate"]["rows"][0]["screening"].copy()
    free=subscriber_contract_v2._attach_highlightly_owner_research(free,supplement)
    assert "highlightly_research" not in free["radar"]
    assert free["slate"]["rows"][0]["screening"]==before


def test_highlightly_research_unavailable_is_unknown_not_zero_for_owner():
    payload=_payload([])
    entitlement=_pro_entitlement()
    entitlement["user"]={"id":"owner","role":"OWNER"}
    contract=subscriber_contract_v2.build_contract(payload,entitlement)
    contract=subscriber_contract_v2._attach_highlightly_owner_research(contract,None)
    assert contract["radar"]["highlightly_research"]["status"]=="SOURCE_NOT_VERIFIED"


def test_p0_sport_screening_health_does_not_confuse_fixture_discovery_and_models():
    payload=_payload([])
    payload.update({
        "fixture_scan_count": 538,
        "p0_sport_preview_attempted_count": 2,
        "p0_sport_preview_modeled_limited_count": 1,
        "p0_sport_preview_missing_inputs_count": 1,
        "p0_sport_preview_cached_count": 5,
        "p0_sport_preview_deferred_quota_count": 1,
        "p0_sport_preview_api_calls": 4,
    })
    anonymous=subscriber_contract_v2.build_contract(
        payload, subscriber_app_v4.anonymous_entitlement()
    )
    health=anonymous["sport_screening"]
    assert health["status"]=="SPORT_FIRST_SCREEN_RECORDED"
    assert health["scope"]=="LAST_PIPELINE_TICK_ONLY"
    assert health["attempted"]==2
    assert health["modeled_research"]==1
    assert health["inputs_insufficient"]==1
    assert health["quota_deferred"]==1
    assert health["provider_calls"]==4
    assert health["not_full_slate_coverage"] is True
    assert health["source_redisplay_rights_verified"] is False
    assert health["production_promotion_allowed"] is False
    assert "raw_probabilities" not in health
    assert "fixtures" not in health

    # An old runtime without P0 keys is NOT counted as zero attempts.
    missing=subscriber_contract_v2.build_contract(
        _payload([]), subscriber_app_v4.anonymous_entitlement()
    )["sport_screening"]
    assert missing["status"]=="NO_RUNTIME_SPORT_SCREEN_TELEMETRY"
    assert missing["attempted"] is None
    assert missing["modeled_research"] is None


def test_universal_sport_status_exists_for_every_registry_fixture_even_when_pass():
    rows=[
        {"fixture_id":70001,"kickoff":"2026-11-12T20:00:00+00:00","status":"NS",
         "home_team":"A","away_team":"B",
         "screen_queue":{"status":"QUEUED","attempt_count":0},
         "evidence_inventory":{"refresh_events":0,"market_snapshots":0,"model_runs":0,
                               "feature_snapshots":0,"lineup_snapshots":0,"availability_snapshots":0}},
        {"fixture_id":70002,"kickoff":"2026-11-12T21:00:00+00:00","status":"NS",
         "home_team":"C","away_team":"D",
         "screen_queue":{"status":"MODELED_LIMITED","attempt_count":1},
         "evidence_inventory":{"refresh_events":1,"market_snapshots":0,"model_runs":1,
                               "feature_snapshots":1,"lineup_snapshots":0,"availability_snapshots":0}},
    ]
    contract=subscriber_contract_v2.build_contract(
        _payload([]), subscriber_app_v4.anonymous_entitlement()
    )
    result=subscriber_contract_v2._attach_full_registry_slate(
        contract, _payload([]),
        {"status":"FULL_SLATE_READY","slate_date":"2026-11-12",
         "timezone":"America/Mexico_City","rows":rows}
    )
    slate=result["slate"]["rows"]
    assert len(slate)==2
    assert all("sport_analysis" in row and "opportunity" in row for row in slate)
    assert slate[0]["sport_analysis"]["status"]=="NOT_ANALYZED"
    assert slate[0]["opportunity"]["status"]=="MERCADO_NO_EVALUADO"
    # The independent persisted P0 queue confirms a genuine limited RAW model.
    assert slate[1]["sport_analysis"]["status"]=="MODELED_LIMITED"
    assert slate[1]["sport_analysis"]["probabilities"] is None
    assert slate[1]["screen_queue"]["attempt_count"]==1
    assert result["sport_coverage_queue"]["total_eligible_fixtures"]==2
    assert result["sport_coverage_queue"]["all_fixtures_accounted_for"] is True
    assert result["sport_intelligence_policy"]["market_not_evaluated_distinct_from_pass"] is True
