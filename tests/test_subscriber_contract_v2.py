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

    assert row["state"]["status_code"] == "DATA_AVAILABLE"
    assert row["coverage"]["persisted_evidence_count"] == 8
    assert "market" in row["coverage"]["data_sources"]
    assert row["coverage"]["persisted_market_names"] == ["Match Winner", "Goals Over/Under"]
