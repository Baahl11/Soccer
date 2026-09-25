import asyncio

from mcp_gateway import price_resolver_v4 as v


def _calibration_state(model_version="SOCCER EDGE ENGINE v1.7"):
    identity = {
        "status": "RESEARCH_CALIBRATOR_FITTED",
        "parameters": {"intercept": 0.0, "slope": 1.0},
    }
    return {
        "binary": {
            "current_source_model_version": model_version,
            "current_model_deployment_calibrators": {
                "btts": {
                    "eligible_for_phase16_research": True,
                    "calibrator": identity,
                },
                "over_2_5": {
                    "eligible_for_phase16_research": True,
                    "calibrator": identity,
                },
                "home_win": {
                    "eligible_for_phase16_research": True,
                    "rows": 400,
                    "brier_delta": -0.01,
                    "log_loss_delta": -0.02,
                    "discrimination": {
                        "auc": 0.64,
                        "auc_lower_95": 0.56,
                        "positive_count": 160,
                        "negative_count": 240,
                    },
                    "calibrator": identity,
                },
                "draw": {
                    "eligible_for_phase16_research": False,
                    "rows": 400,
                    "brier_delta": -0.004,
                    "log_loss_delta": -0.006,
                    "discrimination": {
                        "auc": 0.54,
                        "auc_lower_95": 0.48,
                        "positive_count": 100,
                        "negative_count": 300,
                    },
                    "calibrator": identity,
                },
                "away_win": {
                    "eligible_for_phase16_research": True,
                    "rows": 400,
                    "brier_delta": -0.008,
                    "log_loss_delta": -0.012,
                    "discrimination": {
                        "auc": 0.62,
                        "auc_lower_95": 0.55,
                        "positive_count": 140,
                        "negative_count": 260,
                    },
                    "calibrator": identity,
                },
            },
        },
        "multiclass_1x2": {
            "source_model_version": model_version,
            "research_deployment_calibrator": {
                "status": "RESEARCH_DEPLOYMENT_CALIBRATOR_FITTED",
                "temperature": 1.5,
            },
        },
    }


def test_normalizes_and_devigs_match_winner():
    payload = {
        "response": [{
            "fixture": {"id": 123},
            "update": "2026-09-23T20:00:00+00:00",
            "bookmakers": [{
                "id": 1,
                "name": "Book A",
                "bets": [{
                    "id": 1,
                    "name": "Match Winner",
                    "values": [
                        {"value": "Home", "odd": "2.00"},
                        {"value": "Draw", "odd": "3.50"},
                        {"value": "Away", "odd": "4.00"},
                    ],
                }],
            }],
        }],
    }
    rows = v.normalize_api_response(payload)
    assert len(rows) == 1
    values = rows[0]["values"]
    assert abs(sum(x["fair_probability"] for x in values) - 1.0) < 1e-9
    assert rows[0]["source"] == "API_FOOTBALL_ODDS_V3"


def test_normalizes_totals_by_exact_line():
    payload = {
        "response": [{
            "fixture": {"id": 123},
            "bookmakers": [{
                "id": 1,
                "name": "Book A",
                "bets": [{
                    "id": 5,
                    "name": "Goals Over/Under",
                    "values": [
                        {"value": "Over 2.5", "odd": "1.95"},
                        {"value": "Under 2.5", "odd": "1.90"},
                        {"value": "Over 3.5", "odd": "2.80"},
                        {"value": "Under 3.5", "odd": "1.42"},
                    ],
                }],
            }],
        }],
    }
    rows = v.normalize_api_response(payload)
    offer = v.choose_reference_offer(rows, family="FT_TOTALS", selection="Over", line=2.5)
    assert offer is not None
    assert offer["line"] == 2.5
    assert offer["decimal_price"] == 1.95
    assert 0 < offer["fair_probability"] < 1


def test_research_totals_maps_to_real_2_5_offer_without_fake_calibration():
    event = {"raw_projection": {"raw_over_2_5_prob": 0.62}}
    row = {
        "market_family": "FT_TOTALS_RESEARCH",
        "selection": "Over research",
    }
    markets = [{
        "fixture_id": 1,
        "bookmaker_id": 10,
        "bookmaker": "Book",
        "market_id": 5,
        "market": "Goals Over/Under",
        "values": [
            {"selection": "Over", "line": 2.5, "decimal_price": 2.0, "fair_probability": 0.48},
            {"selection": "Under", "line": 2.5, "decimal_price": 1.85, "fair_probability": 0.52},
        ],
        "source": "API_FOOTBALL_ODDS_V3",
    }]
    status = v._enrich_row(row, event, markets, "PRICE_API_RESOLVED")
    assert status == "PRICE_API_RESOLVED"
    assert row["market_family"] == "FT_TOTALS"
    assert row["selection"] == "Over"
    assert row["line"] == 2.5
    assert row["price"] == 2.0
    assert row["p_market_fair"] == 0.48
    assert row["p_raw"] == 0.62
    assert row["price_resolution_calibrated_probability_added"] is False
    assert "p_model_calibrated" not in row


def test_btts_maps_to_yes_offer():
    event = {"raw_projection": {"raw_btts_yes_prob": 0.58}}
    row = {"market_family": "FT_BTTS_RESEARCH", "selection": "BTTS research"}
    markets = [{
        "market": "Both Teams To Score",
        "bookmaker": "Book",
        "values": [
            {"selection": "Yes", "line": None, "decimal_price": 1.9, "fair_probability": 0.51},
            {"selection": "No", "line": None, "decimal_price": 1.95, "fair_probability": 0.49},
        ],
    }]
    v._enrich_row(row, event, markets, "PRICE_API_RESOLVED")
    assert row["market_family"] == "BTTS"
    assert row["selection"] == "Yes"
    assert row["price"] == 1.9


def test_no_exact_market_remains_explicitly_unresolved():
    event = {"raw_projection": {"raw_over_2_5_prob": 0.62}}
    row = {"market_family": "FT_TOTALS_RESEARCH", "selection": "Over research"}
    status = v._enrich_row(row, event, [], "PRICE_API_RESOLVED")
    assert status == "PRICE_API_NO_MARKET"
    assert row["price_resolution_status"] == "PRICE_API_NO_MARKET"


def test_quota_accounting_adds_price_resolver_calls():
    payload = {
        "api_calls_this_tick": 70,
        "last_daily_remaining": 6957,
        "quota": {"daily_remaining": 6957},
    }
    v._apply_quota_accounting(payload, 3, 6954)
    assert payload["api_calls_this_tick"] == 73
    assert payload["last_daily_remaining"] == 6954
    assert payload["quota"]["daily_remaining"] == 6954


def test_missing_exact_total_line_is_not_misclassified_as_missing_market():
    event = {"raw_projection": {"raw_over_2_5_prob": 0.62}}
    row = {
        "market_family": "FT_TOTALS_RESEARCH",
        "selection": "Over research",
    }
    markets = [{
        "market": "Goals Over/Under",
        "bookmaker": "Book",
        "values": [
            {"selection": "Over", "line": 3.5, "decimal_price": 2.8, "fair_probability": 0.33},
            {"selection": "Under", "line": 3.5, "decimal_price": 1.42, "fair_probability": 0.67},
        ],
    }]
    status = v._enrich_row(row, event, markets, "PRICE_API_RESOLVED")
    assert status == "PRICE_API_NO_EXACT_LINE"
    assert row["price_resolution_available_lines"] == [3.5]


def test_current_model_calibration_is_applied_to_totals():
    event = {"raw_projection": {"raw_over_2_5_prob": 0.62}}
    row = {"market_family": "FT_TOTALS_RESEARCH", "selection": "Over research"}
    markets = [{
        "market": "Goals Over/Under",
        "bookmaker": "Book",
        "values": [
            {"selection": "Over", "line": 2.5, "decimal_price": 2.0, "fair_probability": 0.48},
            {"selection": "Under", "line": 2.5, "decimal_price": 1.85, "fair_probability": 0.52},
        ],
    }]
    v._enrich_row(
        row,
        event,
        markets,
        "PRICE_API_RESOLVED",
        calibration_state=_calibration_state(),
        model_version="SOCCER EDGE ENGINE v1.7",
    )
    assert row["price_resolution_calibrated_probability_added"] is True
    assert abs(row["p_model_calibrated"] - 0.62) < 1e-8
    assert row["phase16_calibration_source"] == "CURRENT_MODEL_OOS_PLATT:OVER_2_5"


def test_current_model_calibration_is_not_applied_on_version_mismatch():
    event = {"raw_projection": {"raw_btts_yes_prob": 0.58}}
    row = {"market_family": "FT_BTTS_RESEARCH", "selection": "BTTS research"}
    markets = [{
        "market": "Both Teams To Score",
        "bookmaker": "Book",
        "values": [
            {"selection": "Yes", "line": None, "decimal_price": 1.9, "fair_probability": 0.51},
            {"selection": "No", "line": None, "decimal_price": 1.95, "fair_probability": 0.49},
        ],
    }]
    v._enrich_row(
        row,
        event,
        markets,
        "PRICE_API_RESOLVED",
        calibration_state=_calibration_state("SOCCER EDGE ENGINE v1.6"),
        model_version="SOCCER EDGE ENGINE v1.7",
    )
    assert row["price_resolution_calibrated_probability_added"] is False
    assert "p_model_calibrated" not in row


def test_1x2_draw_is_blocked_when_class_discrimination_is_not_ready():
    event = {
        "raw_projection": {
            "raw_home_win_prob": 0.30,
            "raw_draw_prob": 0.45,
            "raw_away_win_prob": 0.25,
        }
    }
    row = {"market_family": "FT_1X2_RESEARCH", "selection": "Side research"}
    family, selection, line, p_raw = v._desired_offer(row, event)
    assert family == "1X2"
    assert selection == "Draw"
    assert line is None
    assert p_raw == 0.45

    markets = [{
        "market": "Match Winner",
        "bookmaker": "Book",
        "values": [
            {"selection": "Home", "line": None, "decimal_price": 2.8, "fair_probability": 0.34},
            {"selection": "Draw", "line": None, "decimal_price": 3.1, "fair_probability": 0.31},
            {"selection": "Away", "line": None, "decimal_price": 3.3, "fair_probability": 0.35},
        ],
    }]
    v._enrich_row(
        row,
        event,
        markets,
        "PRICE_API_RESOLVED",
        calibration_state=_calibration_state(),
        model_version="SOCCER EDGE ENGINE v1.7",
    )
    assert row["selection"] == "Draw"
    assert row["price_resolution_calibrated_probability_added"] is False
    assert "p_model_calibrated" not in row
    assert row["phase16_calibration_status"] == "SELECTION_DISCRIMINATION_NOT_READY"
    assert row["phase16_1x2_family_discrimination_ready"] is False
    assert row["phase16_1x2_not_ready_classes"] == ["DRAW"]
    assert row["phase16_1x2_class_discrimination_ready"]["home_win"] is True
    assert row["phase16_1x2_class_discrimination_ready"]["draw"] is False
    assert row["phase16_1x2_class_discrimination_ready"]["away_win"] is True
    diagnostics = row["phase16_1x2_class_discrimination_diagnostics"]
    assert diagnostics["draw"]["rows"] == 400
    assert diagnostics["draw"]["positive_count"] == 100
    assert diagnostics["draw"]["negative_count"] == 300
    assert diagnostics["draw"]["auc_lower_95"] == 0.48
    assert diagnostics["draw"]["auc_lower_95_gap_to_gate"] == -0.02
    assert diagnostics["draw"]["ready"] is False


def test_legacy_cached_value_odd_rows_are_normalized():
    values = [
        {"value": "Over 2.5", "odd": "1.95"},
        {"value": "Under 2.5", "odd": "1.90"},
    ]
    parsed = v._normalize_market_values("Goals Over/Under", values)
    assert parsed[0]["selection"] == "Over"
    assert parsed[0]["line"] == 2.5
    assert parsed[0]["decimal_price"] == 1.95
    assert parsed[1]["selection"] == "Under"
    assert parsed[1]["line"] == 2.5
    assert abs(sum(x["fair_probability"] for x in parsed) - 1.0) < 1e-9


def test_period_totals_never_match_full_time_total_family():
    assert v._market_kind("Goals Over/Under - Second Half") is None
    assert v._market_kind("Goals Over/Under - First Half") is None
    assert v._market_kind("Goals Over/Under") == "FT_TOTALS"


def test_handicap_field_can_supply_total_line():
    values = [
        {"value": "Over", "handicap": "2.5", "odd": "2.05"},
        {"value": "Under", "handicap": "2.5", "odd": "1.80"},
    ]
    parsed = v._normalize_market_values("Goals Over/Under", values)
    assert {x["line"] for x in parsed} == {2.5}
    assert {x["selection"] for x in parsed} == {"Over", "Under"}


def test_1x2_home_is_allowed_when_class_discrimination_is_ready():
    event = {
        "raw_projection": {
            "raw_home_win_prob": 0.55,
            "raw_draw_prob": 0.20,
            "raw_away_win_prob": 0.25,
        }
    }
    row = {"market_family": "FT_1X2_RESEARCH", "selection": "Side research"}
    markets = [{
        "market": "Match Winner",
        "bookmaker": "Book",
        "values": [
            {"selection": "Home", "line": None, "decimal_price": 2.1, "fair_probability": 0.45},
            {"selection": "Draw", "line": None, "decimal_price": 3.2, "fair_probability": 0.30},
            {"selection": "Away", "line": None, "decimal_price": 3.8, "fair_probability": 0.25},
        ],
    }]
    v._enrich_row(
        row,
        event,
        markets,
        "PRICE_API_RESOLVED",
        calibration_state=_calibration_state(),
        model_version="SOCCER EDGE ENGINE v1.7",
    )
    assert row["selection"] == "Home"
    assert row["price_resolution_calibrated_probability_added"] is True
    assert 0 < row["p_model_calibrated"] < 1
    assert row["phase16_calibration_source"] == "CURRENT_MODEL_OOS_TEMPERATURE:1X2"
    assert row["phase16_calibration_policy"] == "MULTICLASS_TEMPERATURE+SELECTION_AUC_L95_GT_0_50"
    assert row["phase16_1x2_family_discrimination_ready"] is False
    assert row["phase16_1x2_not_ready_classes"] == ["DRAW"]


def test_research_cache_hydration_adds_zero_provider_calls(monkeypatch):
    payload = {
        "events": [{
            "event_type": "SOCCER_REFRESH",
            "stage": "T-40",
            "fixture": {
                "fixture_id": 222,
                "home_team": "Home FC",
                "away_team": "Away FC",
            },
            "raw_projection": {
                "raw_home_goal_rate": 1.6,
                "raw_away_goal_rate": 1.2,
            },
        }],
        "match_table_rows": [],
        "api_calls_this_tick": 12,
    }

    cached = [{
        "fixture_id": 222,
        "bookmaker_id": 1,
        "bookmaker": "Book",
        "market_id": 10,
        "market": "Home Team Total Goals",
        "values": [
            {"selection": "Over", "line": 1.5, "decimal_price": 1.95},
            {"selection": "Under", "line": 1.5, "decimal_price": 1.85},
        ],
        "source": "POSTGRES_MARKET_SNAPSHOT_CACHE",
    }]

    monkeypatch.setattr(v, "_load_cached_markets", lambda fixture_id, stage: cached if fixture_id == 222 else [])

    result = asyncio.run(v.resolve_payload(payload, max_api_calls=0, calibration_state={}))

    assert result["candidate_rows"] == 0
    assert result["api_calls_added"] == 0
    assert result["cache_hydrated_research_fixtures"] == 1
    assert result["cache_hydrated_research_market_rows"] == 1
    assert result["cache_hydration_provider_requests_added"] == 0
    assert payload["api_calls_this_tick"] == 12
    assert payload["events"][0]["market"]["markets"][0]["market"] == "Home Team Total Goals"



def test_team_totals_research_spillover_uses_leftover_budget(monkeypatch):
    payload = {
        "events": [{
            "event_type": "SOCCER_REFRESH",
            "stage": "T-40",
            "fixture": {
                "fixture_id": 333,
                "home_team": "Home FC",
                "away_team": "Away FC",
            },
            "raw_projection": {
                "raw_home_goal_rate": 1.7,
                "raw_away_goal_rate": 1.1,
            },
        }],
        "match_table_rows": [],
        "api_calls_this_tick": 4,
    }

    fetched = [{
        "fixture_id": 333,
        "bookmaker_id": 1,
        "bookmaker": "Book",
        "market_id": 10,
        "market": "Home Team Total Goals",
        "values": [
            {"selection": "Over", "line": 1.5, "decimal_price": 1.95},
            {"selection": "Under", "line": 1.5, "decimal_price": 1.85},
        ],
        "source": "API_FOOTBALL_ODDS_V3",
    }]

    calls = []

    async def fake_fetch(client, fixture_id, *, api_key, remaining_calls):
        calls.append((fixture_id, remaining_calls))
        return fetched, 1, "PRICE_API_RESOLVED", 7000

    monkeypatch.setenv("API_FOOTBALL_KEY", "test-key")
    monkeypatch.setattr(v, "_load_cached_markets", lambda fixture_id, stage: [])
    monkeypatch.setattr(v, "_fetch_fixture_odds", fake_fetch)

    result = asyncio.run(v.resolve_payload(payload, max_api_calls=2, calibration_state={}))

    assert calls == [(333, 2)]
    assert result["candidate_rows"] == 0
    assert result["api_calls_added"] == 1
    assert result["research_spillover_api_calls_added"] == 1
    assert result["research_spillover_fixtures_fetched"] == 1
    assert result["research_spillover_market_rows_fetched"] == 1
    assert result["research_spillover_budget_exhausted_fixtures"] == 0
    assert payload["api_calls_this_tick"] == 5
    assert payload["events"][0]["market"]["markets"][0]["market"] == "Home Team Total Goals"
    assert payload["events"][0]["research_price_spillover"]["research_only"] is True


def test_team_totals_research_spillover_never_preempts_primary_budget(monkeypatch):
    payload = {
        "model_version": "SOCCER EDGE ENGINE v1.7",
        "events": [
            {
                "event_type": "SOCCER_REFRESH",
                "stage": "T-20",
                "fixture": {"fixture_id": 401},
                "raw_projection": {"raw_over_2_5_prob": 0.61},
            },
            {
                "event_type": "SOCCER_REFRESH",
                "stage": "T-40",
                "fixture": {"fixture_id": 402},
                "raw_projection": {
                    "raw_home_goal_rate": 1.5,
                    "raw_away_goal_rate": 1.2,
                },
            },
        ],
        "match_table_rows": [{
            "row_index": 0,
            "fixture_id": 401,
            "stage": "T-20",
            "execution_status": "WAIT_PRICE",
            "market_family": "FT_TOTALS_RESEARCH",
            "selection": "Over research",
        }],
        "api_calls_this_tick": 8,
    }

    primary_markets = [{
        "fixture_id": 401,
        "bookmaker_id": 1,
        "bookmaker": "Book",
        "market_id": 5,
        "market": "Goals Over/Under",
        "values": [
            {"selection": "Over", "line": 2.5, "decimal_price": 2.0, "fair_probability": 0.48},
            {"selection": "Under", "line": 2.5, "decimal_price": 1.85, "fair_probability": 0.52},
        ],
        "source": "API_FOOTBALL_ODDS_V3",
    }]

    calls = []

    async def fake_fetch(client, fixture_id, *, api_key, remaining_calls):
        calls.append((fixture_id, remaining_calls))
        return primary_markets, 1, "PRICE_API_RESOLVED", 6999

    monkeypatch.setenv("API_FOOTBALL_KEY", "test-key")
    monkeypatch.setattr(v, "_load_cached_markets", lambda fixture_id, stage: [])
    monkeypatch.setattr(v, "_fetch_fixture_odds", fake_fetch)

    result = asyncio.run(v.resolve_payload(payload, max_api_calls=1, calibration_state={}))

    assert calls == [(401, 1)]
    assert result["candidate_rows"] == 1
    assert result["api_calls_added"] == 1
    assert result["research_spillover_api_calls_added"] == 0
    assert result["research_spillover_fixtures_fetched"] == 0
    assert result["research_spillover_budget_exhausted_fixtures"] == 1
    assert payload["api_calls_this_tick"] == 9
    assert "market" not in payload["events"][1]


def test_team_totals_research_spillover_uses_only_budget_remaining_after_primary(monkeypatch):
    payload = {
        "model_version": "SOCCER EDGE ENGINE v1.7",
        "events": [
            {
                "event_type": "SOCCER_REFRESH",
                "stage": "T-20",
                "fixture": {"fixture_id": 501},
                "raw_projection": {"raw_over_2_5_prob": 0.60},
            },
            {
                "event_type": "SOCCER_REFRESH",
                "stage": "T-40",
                "fixture": {
                    "fixture_id": 502,
                    "home_team": "Home FC",
                    "away_team": "Away FC",
                },
                "raw_projection": {
                    "raw_home_goal_rate": 1.8,
                    "raw_away_goal_rate": 1.0,
                },
            },
        ],
        "match_table_rows": [{
            "row_index": 0,
            "fixture_id": 501,
            "stage": "T-20",
            "execution_status": "WAIT_PRICE",
            "market_family": "FT_TOTALS_RESEARCH",
            "selection": "Over research",
        }],
        "api_calls_this_tick": 10,
    }

    primary = [{
        "fixture_id": 501,
        "bookmaker": "Book",
        "market": "Goals Over/Under",
        "values": [
            {"selection": "Over", "line": 2.5, "decimal_price": 1.95, "fair_probability": 0.49},
            {"selection": "Under", "line": 2.5, "decimal_price": 1.90, "fair_probability": 0.51},
        ],
        "source": "API_FOOTBALL_ODDS_V3",
    }]
    research = [{
        "fixture_id": 502,
        "bookmaker": "Book",
        "market": "Away Team Total Goals",
        "values": [
            {"selection": "Over", "line": 0.5, "decimal_price": 1.70},
            {"selection": "Under", "line": 0.5, "decimal_price": 2.10},
        ],
        "source": "API_FOOTBALL_ODDS_V3",
    }]
    calls = []

    async def fake_fetch(client, fixture_id, *, api_key, remaining_calls):
        calls.append((fixture_id, remaining_calls))
        if fixture_id == 501:
            return primary, 1, "PRICE_API_RESOLVED", 6998
        return research, 1, "PRICE_API_RESOLVED", 6997

    monkeypatch.setenv("API_FOOTBALL_KEY", "test-key")
    monkeypatch.setattr(v, "_load_cached_markets", lambda fixture_id, stage: [])
    monkeypatch.setattr(v, "_fetch_fixture_odds", fake_fetch)

    result = asyncio.run(v.resolve_payload(payload, max_api_calls=2, calibration_state={}))

    assert calls == [(501, 2), (502, 1)]
    assert result["api_calls_added"] == 2
    assert result["research_spillover_api_calls_added"] == 1
    assert result["research_spillover_fixtures_fetched"] == 1
    assert payload["api_calls_this_tick"] == 12
    assert payload["events"][1]["market"]["markets"][0]["market"] == "Away Team Total Goals"

def test_binary_discrimination_blocker_is_explicit_for_btts_and_totals():
    state = _calibration_state()
    for target in ("btts", "over_2_5"):
        state["binary"]["current_model_deployment_calibrators"][target]["eligible_for_phase16_research"] = False
        state["binary"]["current_model_deployment_calibrators"][target]["rows"] = 407
        state["binary"]["current_model_deployment_calibrators"][target]["brier_delta"] = -0.04
        state["binary"]["current_model_deployment_calibrators"][target]["log_loss_delta"] = -0.11
        state["binary"]["current_model_deployment_calibrators"][target]["discrimination"] = {
            "auc": 0.52,
            "auc_lower_95": 0.46,
            "positive_count": 240,
            "negative_count": 167,
        }

    btts_row = {"market_family": "FT_BTTS_RESEARCH", "selection": "BTTS research"}
    btts_event = {"raw_projection": {"raw_btts_yes_prob": 0.58}}
    btts_markets = [{
        "market": "Both Teams To Score",
        "bookmaker": "Book",
        "values": [
            {"selection": "Yes", "line": None, "decimal_price": 1.9, "fair_probability": 0.51},
            {"selection": "No", "line": None, "decimal_price": 1.95, "fair_probability": 0.49},
        ],
    }]
    v._enrich_row(
        btts_row,
        btts_event,
        btts_markets,
        "PRICE_API_RESOLVED",
        calibration_state=state,
        model_version="SOCCER EDGE ENGINE v1.7",
    )
    assert btts_row["phase16_calibration_status"] == "BINARY_DISCRIMINATION_NOT_READY"
    assert btts_row["phase16_binary_calibration_diagnostics"]["target"] == "btts"
    assert btts_row["phase16_binary_calibration_diagnostics"]["auc_lower_95"] == 0.46
    assert btts_row["phase16_binary_calibration_diagnostics"]["auc_lower_95_gap_to_gate"] == -0.04
    assert btts_row["phase16_calibration_promotion_shadow_eligible"] is False
    assert "p_model_calibrated" not in btts_row

    totals_row = {"market_family": "FT_TOTALS_RESEARCH", "selection": "Over research"}
    totals_event = {"raw_projection": {"raw_over_2_5_prob": 0.62}}
    totals_markets = [{
        "market": "Goals Over/Under",
        "bookmaker": "Book",
        "values": [
            {"selection": "Over", "line": 2.5, "decimal_price": 2.0, "fair_probability": 0.48},
            {"selection": "Under", "line": 2.5, "decimal_price": 1.85, "fair_probability": 0.52},
        ],
    }]
    v._enrich_row(
        totals_row,
        totals_event,
        totals_markets,
        "PRICE_API_RESOLVED",
        calibration_state=state,
        model_version="SOCCER EDGE ENGINE v1.7",
    )
    assert totals_row["phase16_calibration_status"] == "BINARY_DISCRIMINATION_NOT_READY"
    assert totals_row["phase16_binary_calibration_diagnostics"]["target"] == "over_2_5"
    assert totals_row["phase16_calibration_promotion_shadow_eligible"] is False
    assert "p_model_calibrated" not in totals_row


def test_existing_priced_1x2_row_gets_zero_call_calibration(monkeypatch):
    payload = {
        "model_version": "SOCCER EDGE ENGINE v1.7",
        "events": [{
            "event_type": "SOCCER_REFRESH",
            "stage": "T-20",
            "fixture": {"fixture_id": 321},
            "raw_projection": {
                "raw_home_win_prob": 0.55,
                "raw_draw_prob": 0.20,
                "raw_away_win_prob": 0.25,
            },
        }],
        "match_table_rows": [{
            "row_index": 0,
            "fixture_id": 321,
            "stage": "T-20",
            "execution_status": "RESEARCH_ONLY",
            "market_family": "1X2",
            "market": "Match Winner",
            "selection": "home",
            "price": 2.10,
            "p_market_fair": 0.45,
        }],
        "api_calls_this_tick": 5,
    }

    monkeypatch.setattr(v, "_load_cached_markets", lambda fixture_id, stage: [])

    result = asyncio.run(
        v.resolve_payload(
            payload,
            max_api_calls=0,
            calibration_state=_calibration_state(),
        )
    )

    row = payload["match_table_rows"][0]
    assert result["candidate_rows"] == 0
    assert result["api_calls_added"] == 0
    assert result["existing_price_calibration_provider_requests_added"] == 0
    assert result["existing_price_calibration_rows_considered"] == 1
    assert result["existing_price_calibrated_rows_added"] == 1
    assert row["phase16_calibration_status"] == "RESEARCH_CALIBRATION_APPLIED"
    assert row["phase16_calibration_promotion_shadow_eligible"] is True
    assert row["price_resolution_existing_price_calibration_added"] is True
    assert row["phase16_1x2_class_discrimination_ready"]["home_win"] is True
    assert row["phase16_1x2_class_discrimination_ready"]["draw"] is False
    assert row["phase16_1x2_class_discrimination_ready"]["away_win"] is True
    assert 0 < row["p_model_calibrated"] < 1
    assert payload["api_calls_this_tick"] == 5



def test_team_totals_diversity_backlog_uses_fresh_client_and_counts_exact_fixture(monkeypatch):
    payload = {
        "model_version": "SOCCER EDGE ENGINE v1.7",
        "events": [],
        "match_table_rows": [],
        "api_calls_this_tick": 2,
    }
    backlog_event = {
        "event_type": "TEAM_TOTALS_RESEARCH_SPILLOVER",
        "stage": "T-90",
        "fixture": {
            "fixture_id": 7001,
            "kickoff": "2026-09-25T03:00:00+00:00",
            "home_team": "Home FC",
            "away_team": "Away FC",
            "home_team_id": 71,
            "away_team_id": 72,
        },
        "raw_projection": {
            "raw_home_goal_rate": 1.8,
            "raw_away_goal_rate": 1.1,
        },
        "classification": "RESEARCH_ONLY",
        "research_only": True,
        "decision_weight": 0.0,
    }
    fetched = [{
        "fixture_id": 7001,
        "bookmaker_id": 1,
        "bookmaker": "Book",
        "market_id": 16,
        "market": "Total - Home",
        "values": [
            {"selection": "Over", "line": 1.5, "decimal_price": 1.90},
            {"selection": "Under", "line": 1.5, "decimal_price": 1.90},
        ],
        "source": "API_FOOTBALL_ODDS_V3",
    }]
    calls = []

    async def fake_fetch(client, fixture_id, *, api_key, remaining_calls):
        assert client.is_closed is False
        calls.append((fixture_id, remaining_calls))
        return fetched, 1, "PRICE_API_RESOLVED", 7200

    monkeypatch.setenv("API_FOOTBALL_KEY", "test-key")
    monkeypatch.setattr(v, "_load_cached_markets", lambda fixture_id, stage: [])
    monkeypatch.setattr(v, "_load_team_totals_diversity_backlog", lambda: {
        "existing_fixture_ids": {6000},
        "existing_unique_fixtures": 1,
        "target": 20,
        "gap": 19,
        "candidate_events": [backlog_event],
        "candidate_count": 1,
        "source": "TEST_BACKLOG",
    })
    monkeypatch.setattr(v, "_fetch_fixture_odds", fake_fetch)

    result = asyncio.run(v.resolve_payload(payload, max_api_calls=3, calibration_state={}))

    assert calls == [(7001, 3)]
    assert result["research_spillover_existing_unique_fixtures"] == 1
    assert result["research_spillover_new_unique_fixtures_this_tick"] == 1
    assert result["research_spillover_projected_unique_fixtures"] == 2
    assert result["research_spillover_diversity_gap_remaining"] == 18
    assert result["research_spillover_exact_team_total_fixtures_attached"] == 1
    assert result["research_spillover_synthetic_events_added"] == 1
    assert result["research_spillover_ft_team_total_market_rows_attached"] == 1
    assert result["research_spillover_primary_markets_preempted"] is False
    assert payload["api_calls_this_tick"] == 3
    assert payload["events"][0]["event_type"] == "TEAM_TOTALS_RESEARCH_SPILLOVER"
    assert payload["events"][0]["market"]["markets"][0]["market"] == "Total - Home"
    assert payload["events"][0]["research_price_spillover"]["ft_team_totals_present"] is True
    capture = payload["events"][0]["team_totals_diversity_capture"]
    assert capture["qualifies"] is True
    assert capture["phase19_true_clv_qualified"] is False
    assert capture["phase19_true_clv_requires_later_pre_kickoff_close"] is True
    assert result["research_spillover_diversity_counter_semantics"].startswith("EXPLICIT_STRICT_FT_TEAM_TOTAL_CAPTURE_MARKER")
    assert result["research_spillover_phase19_true_clv_gate_separate"] is True


def test_team_totals_diversity_does_not_count_generic_odds_payload(monkeypatch):
    payload = {
        "events": [],
        "match_table_rows": [],
        "api_calls_this_tick": 0,
    }
    backlog_event = {
        "event_type": "TEAM_TOTALS_RESEARCH_SPILLOVER",
        "stage": "T-90",
        "fixture": {"fixture_id": 7002, "kickoff": "2026-09-25T04:00:00+00:00"},
        "raw_projection": {
            "raw_home_goal_rate": 1.5,
            "raw_away_goal_rate": 1.3,
        },
        "classification": "RESEARCH_ONLY",
    }
    fetched = [{
        "fixture_id": 7002,
        "bookmaker": "Book",
        "market_id": 5,
        "market": "Goals Over/Under",
        "values": [
            {"selection": "Over", "line": 2.5, "decimal_price": 1.95},
            {"selection": "Under", "line": 2.5, "decimal_price": 1.90},
        ],
        "source": "API_FOOTBALL_ODDS_V3",
    }]

    async def fake_fetch(client, fixture_id, *, api_key, remaining_calls):
        return fetched, 1, "PRICE_API_RESOLVED", 7199

    monkeypatch.setenv("API_FOOTBALL_KEY", "test-key")
    monkeypatch.setattr(v, "_load_cached_markets", lambda fixture_id, stage: [])
    monkeypatch.setattr(v, "_load_team_totals_diversity_backlog", lambda: {
        "existing_fixture_ids": {6000},
        "existing_unique_fixtures": 1,
        "target": 20,
        "gap": 19,
        "candidate_events": [backlog_event],
        "candidate_count": 1,
        "source": "TEST_BACKLOG",
    })
    monkeypatch.setattr(v, "_fetch_fixture_odds", fake_fetch)

    result = asyncio.run(v.resolve_payload(payload, max_api_calls=2, calibration_state={}))

    assert result["research_spillover_new_unique_fixtures_this_tick"] == 0
    assert result["research_spillover_projected_unique_fixtures"] == 1
    assert result["research_spillover_diversity_gap_remaining"] == 19
    assert result["research_spillover_exact_team_total_fixtures_attached"] == 0
    assert result["research_spillover_synthetic_events_added"] == 0



def test_team_totals_scanned_upcoming_fixture_can_be_captured_without_model(monkeypatch):
    payload = {
        "events": [],
        "match_table_rows": [],
        "api_calls_this_tick": 6,
        "upcoming_market_capture_fixtures": [{
            "fixture_id": 8001,
            "kickoff": "2026-09-25T12:00:00+00:00",
            "league_id": 39,
            "league": "Premier League",
            "season": 2026,
            "home_team_id": 1,
            "home_team": "Home FC",
            "away_team_id": 2,
            "away_team": "Away FC",
            "status": "NS",
        }],
    }
    fetched = [
        {
            "fixture_id": 8001,
            "bookmaker_id": 1,
            "bookmaker": "Book",
            "market_id": 16,
            "market": "Total - Home",
            "values": [
                {"selection": "Over", "line": 1.5, "decimal_price": 1.90},
                {"selection": "Under", "line": 1.5, "decimal_price": 1.90},
            ],
            "source": "API_FOOTBALL_ODDS_V3",
        },
        {
            "fixture_id": 8001,
            "bookmaker_id": 1,
            "bookmaker": "Book",
            "market_id": 5,
            "market": "Goals Over/Under",
            "values": [
                {"selection": "Over", "line": 2.5, "decimal_price": 1.95},
                {"selection": "Under", "line": 2.5, "decimal_price": 1.90},
            ],
            "source": "API_FOOTBALL_ODDS_V3",
        },
    ]

    async def fake_fetch(client, fixture_id, *, api_key, remaining_calls):
        assert fixture_id == 8001
        return fetched, 1, "PRICE_API_RESOLVED", 7100

    monkeypatch.setenv("API_FOOTBALL_KEY", "test-key")
    monkeypatch.setattr(v, "_load_cached_markets", lambda fixture_id, stage: [])
    monkeypatch.setattr(v, "_load_team_totals_diversity_backlog", lambda: {
        "existing_fixture_ids": {7000},
        "existing_unique_fixtures": 1,
        "legacy_observed_unique_fixtures": 108,
        "target": 20,
        "gap": 19,
        "candidate_events": [],
        "candidate_count": 0,
        "source": "TEST_BACKLOG",
    })
    monkeypatch.setattr(v, "_fetch_fixture_odds", fake_fetch)

    result = asyncio.run(v.resolve_payload(payload, max_api_calls=5, calibration_state={}))

    assert result["research_spillover_scanned_upcoming_candidates"] == 1
    assert result["research_spillover_market_capture_only_candidates"] == 1
    assert result["research_spillover_new_unique_fixtures_this_tick"] == 1
    assert result["research_spillover_projected_unique_fixtures"] == 2
    assert result["research_spillover_ft_team_total_market_rows_attached"] == 1
    assert result["research_spillover_synthetic_events_added"] == 1
    assert payload["api_calls_this_tick"] == 7

    event = payload["events"][0]
    assert event["event_type"] == "TEAM_TOTALS_RESEARCH_SPILLOVER"
    assert "raw_projection" not in event
    assert event["team_totals_diversity_provenance"]["market_capture_only"] is True
    assert event["team_totals_diversity_capture"]["qualifies"] is True
    assert event["team_totals_diversity_capture"]["phase19_true_clv_qualified"] is False
    assert [row["market"] for row in event["market"]["markets"]] == ["Total - Home"]


def test_scanned_upcoming_capture_stops_when_strict_diversity_target_is_met(monkeypatch):
    payload = {
        "events": [],
        "match_table_rows": [],
        "api_calls_this_tick": 0,
        "upcoming_market_capture_fixtures": [
            {"fixture_id": 8101, "kickoff": "2026-09-25T12:00:00+00:00"},
            {"fixture_id": 8102, "kickoff": "2026-09-25T13:00:00+00:00"},
        ],
    }
    calls = []

    async def fake_fetch(client, fixture_id, *, api_key, remaining_calls):
        calls.append(fixture_id)
        return [{
            "fixture_id": fixture_id,
            "market_id": 16,
            "market": "Total - Home",
            "bookmaker": "Book",
            "values": [
                {"selection": "Over", "line": 1.5, "decimal_price": 1.90},
                {"selection": "Under", "line": 1.5, "decimal_price": 1.90},
            ],
        }], 1, "PRICE_API_RESOLVED", 7099

    monkeypatch.setenv("API_FOOTBALL_KEY", "test-key")
    monkeypatch.setattr(v, "_load_cached_markets", lambda fixture_id, stage: [])
    monkeypatch.setattr(v, "_load_team_totals_diversity_backlog", lambda: {
        "existing_fixture_ids": set(range(1, 20)),
        "existing_unique_fixtures": 19,
        "legacy_observed_unique_fixtures": 108,
        "target": 20,
        "gap": 1,
        "candidate_events": [],
        "candidate_count": 0,
        "source": "TEST_BACKLOG",
    })
    monkeypatch.setattr(v, "_fetch_fixture_odds", fake_fetch)

    result = asyncio.run(v.resolve_payload(payload, max_api_calls=5, calibration_state={}))

    assert calls == [8101]
    assert result["research_spillover_projected_unique_fixtures"] == 20
    assert result["research_spillover_diversity_gap_remaining"] == 0
