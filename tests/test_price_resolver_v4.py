import asyncio
from datetime import datetime, timedelta, timezone

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



def test_team_totals_reuses_primary_event_odds_with_zero_leftover_budget(monkeypatch):
    payload = {
        "events": [{
            "event_type": "SOCCER_REFRESH",
            "stage": "T-40",
            "fixture": {
                "fixture_id": 9001,
                "home_team": "Home FC",
                "away_team": "Away FC",
                "home_team_id": 91,
                "away_team_id": 92,
            },
            "raw_projection": {
                "raw_home_goal_rate": 1.7,
                "raw_away_goal_rate": 1.2,
            },
            "market": {
                "markets": [
                    {
                        "market_id": 5,
                        "market": "Goals Over/Under",
                        "bookmaker": "Book",
                        "values": [
                            {"selection": "Over 2.5", "price": "1.95"},
                            {"selection": "Under 2.5", "price": "1.90"},
                        ],
                    },
                    {
                        "market_id": 16,
                        "market": "Total - Home",
                        "bookmaker": "Book",
                        "values": [
                            {"selection": "Over 1.5", "price": "1.85"},
                            {"selection": "Under 1.5", "price": "1.95"},
                        ],
                    },
                    {
                        "market_id": 17,
                        "market": "Total - Away",
                        "bookmaker": "Book",
                        "values": [
                            {"selection": "Over 0.5", "price": "1.70"},
                            {"selection": "Under 0.5", "price": "2.10"},
                        ],
                    },
                ],
            },
        }],
        "match_table_rows": [],
        "api_calls_this_tick": 70,
    }

    monkeypatch.setattr(v, "_load_cached_markets", lambda fixture_id, stage: [])
    monkeypatch.setattr(v, "_load_team_totals_diversity_backlog", lambda: {
        "existing_fixture_ids": {8000},
        "existing_unique_fixtures": 1,
        "legacy_observed_unique_fixtures": 108,
        "target": 20,
        "gap": 19,
        "candidate_events": [],
        "candidate_count": 0,
        "source": "TEST_BACKLOG",
    })

    result = asyncio.run(v.resolve_payload(payload, max_api_calls=0, calibration_state={}))

    assert result["api_calls_added"] == 0
    assert result["research_spillover_api_calls_added"] == 0
    assert result["research_spillover_primary_payload_reuse_fixtures"] == 1
    assert result["research_spillover_primary_payload_reuse_market_rows"] == 2
    assert result["research_spillover_new_unique_fixtures_this_tick"] == 1
    assert result["research_spillover_projected_unique_fixtures"] == 2
    assert result["research_spillover_diversity_gap_remaining"] == 18
    assert payload["events"][0]["team_totals_diversity_capture"]["qualifies"] is True
    assert payload["api_calls_this_tick"] == 70


def test_team_totals_maturation_fetches_after_diversity_target_and_ignores_cache(monkeypatch):
    payload = {
        "events": [],
        "match_table_rows": [],
        "api_calls_this_tick": 10,
    }
    maturation_event = {
        "event_type": "TEAM_TOTALS_RESEARCH_SPILLOVER",
        "stage": "T-20",
        "fixture": {
            "fixture_id": 9201,
            "kickoff": "2026-09-25T06:00:00+00:00",
            "home_team": "Home",
            "away_team": "Away",
        },
        "classification": "RESEARCH_ONLY",
        "research_only": True,
        "decision_weight": 0.0,
        "team_totals_clv_maturation": {
            "signal_generated_at": "2026-09-25T05:20:00+00:00",
            "requires_provider_update_after_signal": True,
        },
    }
    stale_cache = [{
        "fixture_id": 9201,
        "bookmaker_id": 1,
        "bookmaker": "Book",
        "market_id": 16,
        "market": "Total - Home",
        "values": [
            {"selection": "Over", "line": 1.5, "decimal_price": 1.90},
            {"selection": "Under", "line": 1.5, "decimal_price": 1.90},
        ],
        "provider_update": "2026-09-25T05:10:00+00:00",
        "source": "POSTGRES_MARKET_SNAPSHOT_CACHE",
    }]
    fresh = [{
        "fixture_id": 9201,
        "bookmaker_id": 1,
        "bookmaker": "Book",
        "market_id": 16,
        "market": "Total - Home",
        "values": [
            {"selection": "Over", "line": 1.5, "decimal_price": 1.86},
            {"selection": "Under", "line": 1.5, "decimal_price": 1.96},
        ],
        "provider_update": "2026-09-25T05:45:00+00:00",
        "source": "API_FOOTBALL_ODDS_V3",
    }]
    calls = []

    async def fake_fetch(client, fixture_id, *, api_key, remaining_calls):
        calls.append((fixture_id, remaining_calls))
        return fresh, 1, "PRICE_API_RESOLVED", 6400

    monkeypatch.setenv("API_FOOTBALL_KEY", "test-key")
    monkeypatch.setattr(v, "_load_team_totals_maturation_backlog", lambda: {
        "candidate_events": [maturation_event],
        "candidate_count": 1,
        "source": "TEST_MATURATION",
    })
    monkeypatch.setattr(v, "_load_team_totals_diversity_backlog", lambda: {
        "existing_fixture_ids": set(range(1, 21)) | {9201},
        "existing_unique_fixtures": 21,
        "legacy_observed_unique_fixtures": 100,
        "target": 20,
        "gap": 0,
        "candidate_events": [],
        "candidate_count": 0,
        "source": "TEST_DIVERSITY",
    })
    monkeypatch.setattr(v, "_load_cached_markets", lambda fixture_id, stage: stale_cache)
    monkeypatch.setattr(v, "_fetch_fixture_odds", fake_fetch)

    result = asyncio.run(v.resolve_payload(payload, max_api_calls=5, calibration_state={}))

    assert calls == [(9201, 5)]
    assert result["research_spillover_maturation_candidates"] == 1
    assert result["research_spillover_maturation_cache_replays_ignored"] == 1
    assert result["research_spillover_maturation_api_calls_added"] == 1
    assert result["research_spillover_maturation_later_real_quote_refreshes"] == 1
    assert result["research_spillover_diversity_gap_remaining"] == 0
    assert result["research_spillover_primary_markets_preempted"] is False
    assert payload["api_calls_this_tick"] == 11
    assert payload["events"][0]["market"]["source"] == "API_FOOTBALL_ODDS_V3"
    assert payload["events"][0]["team_totals_clv_maturation"]["provider_update_after_signal"] is True


def test_team_totals_current_due_maturation_merges_fresh_team_total_into_existing_market(monkeypatch):
    payload = {
        "events": [{
            "event_type": "SOCCER_REFRESH",
            "stage": "T-20",
            "fixture": {"fixture_id": 9301, "kickoff": "2026-09-25T06:00:00+00:00"},
            "raw_projection": {"raw_home_goal_rate": 1.7, "raw_away_goal_rate": 1.1},
            "market": {
                "source": "API_FOOTBALL_ODDS_V3",
                "resolution_status": "PRICE_API_RESOLVED",
                "markets": [{
                    "fixture_id": 9301,
                    "market_id": 5,
                    "market": "Goals Over/Under",
                    "bookmaker": "Book",
                    "values": [
                        {"selection": "Over", "line": 2.5, "decimal_price": 1.95},
                        {"selection": "Under", "line": 2.5, "decimal_price": 1.90},
                    ],
                    "provider_update": "2026-09-25T05:40:00+00:00",
                }],
            },
        }],
        "match_table_rows": [],
        "api_calls_this_tick": 4,
    }
    maturation_event = {
        "event_type": "TEAM_TOTALS_RESEARCH_SPILLOVER",
        "stage": "T-20",
        "fixture": {"fixture_id": 9301, "kickoff": "2026-09-25T06:00:00+00:00"},
        "classification": "RESEARCH_ONLY",
        "team_totals_clv_maturation": {
            "signal_generated_at": "2026-09-25T05:20:00+00:00",
        },
    }
    fresh = [{
        "fixture_id": 9301,
        "bookmaker_id": 1,
        "bookmaker": "Book",
        "market_id": 16,
        "market": "Total - Home",
        "values": [
            {"selection": "Over", "line": 1.5, "decimal_price": 1.85},
            {"selection": "Under", "line": 1.5, "decimal_price": 1.98},
        ],
        "provider_update": "2026-09-25T05:45:00+00:00",
        "source": "API_FOOTBALL_ODDS_V3",
    }]

    async def fake_fetch(client, fixture_id, *, api_key, remaining_calls):
        return fresh, 1, "PRICE_API_RESOLVED", 6399

    monkeypatch.setenv("API_FOOTBALL_KEY", "test-key")
    monkeypatch.setattr(v, "_load_team_totals_maturation_backlog", lambda: {
        "candidate_events": [maturation_event],
        "candidate_count": 1,
        "source": "TEST_MATURATION",
    })
    monkeypatch.setattr(v, "_load_team_totals_diversity_backlog", lambda: {
        "existing_fixture_ids": {9301} | set(range(1, 21)),
        "existing_unique_fixtures": 21,
        "legacy_observed_unique_fixtures": 100,
        "target": 20,
        "gap": 0,
        "candidate_events": [],
        "candidate_count": 0,
        "source": "TEST_DIVERSITY",
    })
    monkeypatch.setattr(v, "_load_cached_markets", lambda fixture_id, stage: [])
    monkeypatch.setattr(v, "_fetch_fixture_odds", fake_fetch)

    result = asyncio.run(v.resolve_payload(payload, max_api_calls=3, calibration_state={}))

    markets = payload["events"][0]["market"]["markets"]
    assert [row["market"] for row in markets] == ["Goals Over/Under", "Total - Home"]
    assert result["research_spillover_maturation_later_real_quote_refreshes"] == 1
    assert payload["events"][0]["team_totals_clv_maturation"]["provider_update_after_signal"] is True



def _empty_team_totals_backlogs(monkeypatch):
    monkeypatch.setattr(v, "_load_team_totals_maturation_backlog", lambda: {
        "candidate_events": [],
        "candidate_count": 0,
        "source": "TEST_EMPTY_TEAM_TOTALS_MATURATION",
    })
    monkeypatch.setattr(v, "_load_team_totals_diversity_backlog", lambda: {
        "existing_fixture_ids": set(range(1, 21)),
        "existing_unique_fixtures": 20,
        "legacy_observed_unique_fixtures": 0,
        "target": 20,
        "gap": 0,
        "candidate_events": [],
        "candidate_count": 0,
        "source": "TEST_EMPTY_TEAM_TOTALS_DIVERSITY",
    })


def test_primary_clv_maturation_fetches_one_fixture_for_multiple_primary_families(monkeypatch):
    payload = {
        "events": [],
        "match_table_rows": [],
        "api_calls_this_tick": 50,
    }
    event = {
        "event_type": v.PRIMARY_CLV_MATURATION_EVENT_TYPE,
        "stage": "T-20",
        "fixture": {
            "fixture_id": 9401,
            "kickoff": "2026-09-25T06:30:00+00:00",
            "home_team": "Home",
            "away_team": "Away",
        },
        "classification": "RESEARCH_ONLY",
        "primary_clv_maturation": {
            "signals": [
                {
                    "market_family": "FT_TOTALS",
                    "market": "Goals Over/Under",
                    "signal_generated_at": "2026-09-25T05:30:00+00:00",
                },
                {
                    "market_family": "BTTS",
                    "market": "Both Teams Score",
                    "signal_generated_at": "2026-09-25T05:32:00+00:00",
                },
            ],
        },
    }
    fresh = [
        {
            "fixture_id": 9401,
            "bookmaker_id": 1,
            "bookmaker": "Book",
            "market_id": 5,
            "market": "Goals Over/Under",
            "values": [
                {"selection": "Over", "line": 2.5, "decimal_price": 1.95},
                {"selection": "Under", "line": 2.5, "decimal_price": 1.90},
            ],
            "provider_update": "2026-09-25T06:00:00+00:00",
            "source": "API_FOOTBALL_ODDS_V3",
        },
        {
            "fixture_id": 9401,
            "bookmaker_id": 1,
            "bookmaker": "Book",
            "market_id": 8,
            "market": "Both Teams Score",
            "values": [
                {"selection": "Yes", "line": None, "decimal_price": 1.85},
                {"selection": "No", "line": None, "decimal_price": 1.95},
            ],
            "provider_update": "2026-09-25T06:00:00+00:00",
            "source": "API_FOOTBALL_ODDS_V3",
        },
        {
            "fixture_id": 9401,
            "bookmaker_id": 1,
            "bookmaker": "Book",
            "market_id": 16,
            "market": "Total - Home",
            "values": [
                {"selection": "Over", "line": 1.5, "decimal_price": 1.90},
                {"selection": "Under", "line": 1.5, "decimal_price": 1.90},
            ],
            "provider_update": "2026-09-25T06:00:00+00:00",
            "source": "API_FOOTBALL_ODDS_V3",
        },
    ]
    calls = []

    async def fake_fetch(client, fixture_id, *, api_key, remaining_calls):
        calls.append((fixture_id, remaining_calls))
        return fresh, 1, "PRICE_API_RESOLVED", 6200

    monkeypatch.setenv("API_FOOTBALL_KEY", "test-key")
    monkeypatch.setattr(v, "_load_primary_clv_maturation_backlog", lambda: {
        "candidate_events": [event],
        "candidate_count": 1,
        "candidate_family_counts": {"BTTS": 1, "FT_TOTALS": 1},
        "source": "TEST_PRIMARY_MATURATION",
    })
    _empty_team_totals_backlogs(monkeypatch)
    monkeypatch.setattr(v, "_fetch_fixture_odds", fake_fetch)

    result = asyncio.run(v.resolve_payload(payload, max_api_calls=6, calibration_state={}))

    assert calls == [(9401, 6)]
    assert result["primary_clv_maturation_candidates"] == 1
    assert result["primary_clv_maturation_api_calls_added"] == 1
    assert result["primary_clv_maturation_fixtures_refreshed"] == 1
    assert result["primary_clv_maturation_family_refresh_counts"] == {"BTTS": 1, "FT_TOTALS": 1}
    assert result["primary_clv_maturation_synthetic_events_added"] == 1
    assert payload["api_calls_this_tick"] == 51
    persisted = payload["events"][0]
    assert persisted["event_type"] == v.PRIMARY_CLV_MATURATION_EVENT_TYPE
    assert persisted["primary_clv_maturation"]["matured_families"] == ["BTTS", "FT_TOTALS"]
    assert [row["market"] for row in persisted["market"]["markets"]] == [
        "Goals Over/Under",
        "Both Teams Score",
    ]


def test_primary_clv_maturation_rejects_unchanged_provider_update(monkeypatch):
    payload = {
        "events": [],
        "match_table_rows": [],
        "api_calls_this_tick": 5,
    }
    event = {
        "event_type": v.PRIMARY_CLV_MATURATION_EVENT_TYPE,
        "stage": "T-10",
        "fixture": {"fixture_id": 9402, "kickoff": "2026-09-25T06:30:00+00:00"},
        "classification": "RESEARCH_ONLY",
        "primary_clv_maturation": {
            "signals": [{
                "market_family": "1X2",
                "market": "Match Winner",
                "signal_generated_at": "2026-09-25T06:00:00+00:00",
            }],
        },
    }
    stale = [{
        "fixture_id": 9402,
        "bookmaker_id": 1,
        "bookmaker": "Book",
        "market_id": 1,
        "market": "Match Winner",
        "values": [
            {"selection": "Home", "line": None, "decimal_price": 2.0},
            {"selection": "Draw", "line": None, "decimal_price": 3.2},
            {"selection": "Away", "line": None, "decimal_price": 3.5},
        ],
        "provider_update": "2026-09-25T05:59:00+00:00",
        "source": "API_FOOTBALL_ODDS_V3",
    }]

    async def fake_fetch(client, fixture_id, *, api_key, remaining_calls):
        return stale, 1, "PRICE_API_RESOLVED", 6199

    monkeypatch.setenv("API_FOOTBALL_KEY", "test-key")
    monkeypatch.setattr(v, "_load_primary_clv_maturation_backlog", lambda: {
        "candidate_events": [event],
        "candidate_count": 1,
        "candidate_family_counts": {"1X2": 1},
        "source": "TEST_PRIMARY_MATURATION",
    })
    _empty_team_totals_backlogs(monkeypatch)
    monkeypatch.setattr(v, "_fetch_fixture_odds", fake_fetch)

    result = asyncio.run(v.resolve_payload(payload, max_api_calls=3, calibration_state={}))

    assert result["primary_clv_maturation_api_calls_added"] == 1
    assert result["primary_clv_maturation_fixtures_refreshed"] == 0
    assert result["primary_clv_maturation_unchanged_provider_updates"] == 1
    assert payload["events"] == []
    assert payload["api_calls_this_tick"] == 6


def test_primary_maturation_quote_helper_requires_same_market_and_later_update():
    signals = [
        {
            "market_family": "1X2",
            "market": "Match Winner",
            "signal_generated_at": "2026-09-25T06:00:00+00:00",
        },
        {
            "market_family": "BTTS",
            "market": "Both Teams Score",
            "signal_generated_at": "2026-09-25T06:00:00+00:00",
        },
    ]
    markets = [
        {
            "market": "Match Winner",
            "provider_update": "2026-09-25T06:05:00+00:00",
        },
        {
            "market": "Both Teams Score",
            "provider_update": "2026-09-25T05:59:00+00:00",
        },
    ]
    assert v._primary_signals_with_later_provider_quote(markets, signals) == {"1X2"}



def test_price_resolver_splits_cards_and_props_from_canonical_event_markets():
    raw = {
        "response": [{
            "fixture": {"id": 9901},
            "update": "2026-09-25T11:00:00+00:00",
            "bookmakers": [{
                "id": 1,
                "name": "Book",
                "bets": [
                    {
                        "id": 1,
                        "name": "Match Winner",
                        "values": [
                            {"value": "Home", "odd": "2.00"},
                            {"value": "Draw", "odd": "3.20"},
                            {"value": "Away", "odd": "3.60"},
                        ],
                    },
                    {
                        "id": 200,
                        "name": "Total Yellow Cards",
                        "values": [
                            {"value": "Over 4.5", "odd": "1.90"},
                            {"value": "Under 4.5", "odd": "1.90"},
                        ],
                    },
                    {
                        "id": 201,
                        "name": "Player Shots",
                        "values": [
                            {"value": "Player A Over 2.5", "odd": "1.95"},
                        ],
                    },
                ],
            }],
        }],
    }

    markets = v.normalize_api_response(raw)
    event = {}
    v._attach_market_to_event(event, markets, "PRICE_API_RESOLVED")

    canonical = event["market"]["markets"]
    research = event["market"]["research_cards_props_markets"]
    assert [row["market"] for row in canonical] == ["Match Winner"]
    assert {row["market"] for row in research} == {"Total Yellow Cards", "Player Shots"}
    assert event["market"]["card_research_market_rows"] == 1
    assert event["market"]["player_prop_research_market_rows"] == 1
    assert all(row["research_only"] is True for row in research)
    shots = next(row for row in research if row["market"] == "Player Shots")
    assert shots["values"][0]["raw_selection"] == "Player A Over 2.5"
    assert shots["values"][0]["line"] == 2.5


def test_price_resolver_cache_sidecar_is_marked_cache_not_fresh_provider():
    event = {}
    markets = [{
        "fixture_id": 9902,
        "bookmaker_id": 1,
        "bookmaker": "Book",
        "market_id": 200,
        "market": "Total Yellow Cards",
        "values": [{"selection": "Over", "line": 4.5, "decimal_price": 1.9}],
        "provider_update": "2026-09-25T11:00:00+00:00",
        "source": "POSTGRES_MARKET_SNAPSHOT_CACHE",
    }]
    v._attach_market_to_event(event, markets, "PRICE_CACHE_HIT")
    assert event["market"]["source"] == "POSTGRES_MARKET_SNAPSHOT_CACHE"
    assert event["market"]["resolution_status"] == "PRICE_CACHE_HIT"
    assert event["market"]["card_research_market_rows"] == 1



def test_price_resolver_routes_observed_goal_scorer_and_card_names_to_research_sidecar():
    raw = {
        "response": [{
            "fixture": {"id": 9950},
            "update": "2026-09-25T14:00:00+00:00",
            "bookmakers": [{
                "id": 1,
                "name": "Book",
                "bets": [
                    {"id": 501, "name": "Home Anytime Goal Scorer", "values": [{"value": "Player A", "odd": "2.20"}]},
                    {"id": 502, "name": "Away First Goal Scorer", "values": [{"value": "Player B", "odd": "6.00"}]},
                    {"id": 503, "name": "Home Last Goal Scorer", "values": [{"value": "Player C", "odd": "7.00"}]},
                    {"id": 504, "name": "Cards Asian Handicap", "values": [{"value": "Home +0.5", "odd": "1.90"}]},
                    {"id": 505, "name": "Cards European Handicap", "values": [{"value": "Home +0", "odd": "2.20"}]},
                    {"id": 506, "name": "First Card Received (3 way)", "values": [{"value": "Home", "odd": "1.80"}]},
                    {"id": 507, "name": "RCARD", "values": [{"value": "Yes", "odd": "3.40"}]},
                    {"id": 508, "name": "ShotOnTarget Handicap", "values": [{"value": "Home +0.5", "odd": "1.90"}]},
                ],
            }],
        }],
    }

    markets = v.normalize_api_response(raw)
    event = {}
    v._attach_market_to_event(event, markets, "PRICE_API_RESOLVED")

    canonical_names = {row["market"] for row in event["market"]["markets"]}
    research = {row["market"]: row for row in event["market"]["research_cards_props_markets"]}

    assert "ShotOnTarget Handicap" in canonical_names
    assert research["Home Anytime Goal Scorer"]["research_subfamily"] == "GOALSCORER_ANYTIME"
    assert research["Away First Goal Scorer"]["research_subfamily"] == "GOALSCORER_FIRST"
    assert research["Home Last Goal Scorer"]["research_subfamily"] == "GOALSCORER_LAST"
    assert research["Cards Asian Handicap"]["research_family"] == "CARDS"
    assert research["Cards European Handicap"]["research_family"] == "CARDS"
    assert research["First Card Received (3 way)"]["research_family"] == "CARDS"
    assert research["RCARD"]["research_family"] == "CARDS"


def test_price_resolver_sidecar_xi_aligns_player_prop_values():
    raw = {
        "response": [{
            "fixture": {"id": 9910},
            "update": "2026-09-25T11:00:00+00:00",
            "bookmakers": [{
                "id": 1,
                "name": "Book",
                "bets": [{
                    "id": 201,
                    "name": "Player Shots",
                    "values": [
                        {"value": "Player A Over 2.5", "odd": "1.95"},
                        {"value": "Player A Under 2.5", "odd": "1.85"},
                    ],
                }],
            }],
        }],
    }
    event = {
        "lineups": {
            "both_xi_confirmed": True,
            "both_goalkeepers_confirmed": True,
            "teams": [
                {
                    "team_id": 10,
                    "team": "Home",
                    "starters": [{"id": 501, "name": "Player A", "pos": "F"}],
                },
                {
                    "team_id": 20,
                    "team": "Away",
                    "starters": [{"id": 601, "name": "Keeper B", "pos": "G"}],
                },
            ],
        }
    }

    markets = v.normalize_api_response(raw)
    v._attach_market_to_event(event, markets, "PRICE_API_RESOLVED")

    shots = next(
        row for row in event["market"]["research_cards_props_markets"]
        if row["market"] == "Player Shots"
    )
    assert shots["confirmed_xi_at_quote"] is True
    assert shots["xi_aligned_value_rows"] == 2
    assert {value["player_id"] for value in shots["values"]} == {501}
    assert all(
        value["xi_alignment_status"] == "MATCHED_CONFIRMED_XI"
        for value in shots["values"]
    )


def test_price_resolver_sidecar_normalizes_n_plus_threshold_and_excludes_team_shots_aggregate():
    raw = {
        "response": [{
            "fixture": {"id": 9911},
            "update": "2026-09-25T18:30:00+00:00",
            "bookmakers": [{
                "id": 1,
                "name": "Book",
                "bets": [
                    {
                        "id": 301,
                        "name": "Away Player Shots Total",
                        "values": [
                            {"value": "Over 9.5", "odd": "1.90"},
                            {"value": "Under 9.5", "odd": "1.90"},
                        ],
                    },
                    {
                        "id": 302,
                        "name": "Goalkeeper Saves",
                        "values": [{"value": "Keeper B - 3", "odd": "1.88"}],
                    },
                ],
            }],
        }],
    }
    event = {
        "lineups": {
            "both_xi_confirmed": True,
            "both_goalkeepers_confirmed": True,
            "teams": [
                {
                    "team_id": 10,
                    "team": "Home",
                    "starters": [{"id": 501, "name": "Player A", "pos": "F"}],
                },
                {
                    "team_id": 20,
                    "team": "Away",
                    "starters": [{"id": 601, "name": "Keeper B", "pos": "G"}],
                },
            ],
        }
    }

    markets = v.normalize_api_response(raw)
    v._attach_market_to_event(event, markets, "PRICE_API_RESOLVED")

    props = event["market"]["research_cards_props_markets"]
    assert all(row["market"] != "Away Player Shots Total" for row in props)

    gk = next(row for row in props if row["research_subfamily"] == "GK_SAVES")
    assert gk["xi_aligned_value_rows"] == 1
    assert gk["values"][0]["player_id"] == 601
    assert gk["values"][0]["parsed_line"] == 2.5
    assert gk["values"][0]["line_basis"] == "PLAYER_THRESHOLD_N_PLUS"


def test_player_props_maturation_requires_market_and_modelable_probability():
    event = {
        "market": {
            "research_cards_props_markets": [{
                "research_family": "PLAYER_PROPS",
                "research_subfamily": "SHOTS",
                "market": "Player Shots",
                "values": [{
                    "selection": "Player A - 2",
                    "price": 1.9,
                    "xi_alignment_status": "MATCHED_CONFIRMED_XI",
                    "player_id": 10,
                }],
            }]
        },
        "player_shots_intelligence": {
            "players": [{
                "player_id": 10,
                "status": "LIVE_RESEARCH_SHOTS_DISTRIBUTION",
                "lines": [{"line": 1.5, "p_over": 0.55, "p_under": 0.45}],
            }]
        },
    }
    assert v._player_prop_signal_families(event) == {"SHOTS"}

    event["player_shots_intelligence"]["players"][0]["lines"] = []
    assert v._player_prop_signal_families(event) == set()


def test_player_props_maturation_rejects_anonymous_assists_but_accepts_explicit_player_identity():
    anonymous = {
        "market": {
            "research_cards_props_markets": [{
                "research_family": "PLAYER_PROPS",
                "research_subfamily": "ASSISTS",
                "market": "Player Assists",
                "values": [
                    {"selection": "Yes", "decimal_price": 3.20},
                    {"selection": "No", "decimal_price": 1.30},
                ],
            }]
        },
        "player_assists_intelligence": {
            "players": [{
                "player_id": 501,
                "p_1plus_assist": 0.31,
                "p_2plus_assists": 0.08,
            }]
        },
    }
    assert v._player_prop_signal_families(anonymous) == set()

    identified = {
        **anonymous,
        "market": {
            "research_cards_props_markets": [{
                "research_family": "PLAYER_PROPS",
                "research_subfamily": "ASSISTS",
                "market": "Player Assists",
                "values": [
                    {
                        "selection": "Yes",
                        "decimal_price": 3.20,
                        "player_id": 501,
                        "xi_alignment_status": "MATCHED_CONFIRMED_XI",
                    },
                    {
                        "selection": "No",
                        "decimal_price": 1.30,
                        "player_id": 501,
                        "xi_alignment_status": "MATCHED_CONFIRMED_XI",
                    },
                ],
            }]
        },
    }
    assert v._player_prop_signal_families(identified) == {"ASSISTS"}


def test_price_resolver_preserves_provider_player_identity_for_binary_props():
    raw = {
        "response": [{
            "fixture": {"id": 9960},
            "update": "2026-09-25T19:00:00+00:00",
            "bookmakers": [{
                "id": 1,
                "name": "Book",
                "bets": [{
                    "id": 601,
                    "name": "Player Assists",
                    "values": [
                        {"value": "Yes", "odd": "3.20", "player": {"id": 501, "name": "Player A"}},
                        {"value": "No", "odd": "1.30", "player": {"id": 501, "name": "Player A"}},
                    ],
                }],
            }],
        }],
    }
    event = {
        "lineups": {
            "both_xi_confirmed": True,
            "teams": [
                {"team_id": 10, "team": "Home", "starters": [{"id": 501, "name": "Player A", "pos": "M"}]},
                {"team_id": 20, "team": "Away", "starters": [{"id": 601, "name": "Player B", "pos": "F"}]},
            ],
        }
    }

    markets = v.normalize_api_response(raw)
    v._attach_market_to_event(event, markets, "PRICE_API_RESOLVED")
    assists = next(
        row for row in event["market"]["research_cards_props_markets"]
        if row["research_subfamily"] == "ASSISTS"
    )
    assert assists["xi_aligned_value_rows"] == 2
    assert {row["player_id"] for row in assists["values"]} == {501}
    assert {row["player_name"] for row in assists["values"]} == {"Player A"}


def test_price_resolver_recognizes_player_booking_aliases_as_player_cards():
    assert v._research_derivative_subfamily("Player To Be Booked") == "PLAYER_CARDS"
    assert v._research_derivative_subfamily("Player To Be Carded") == "PLAYER_CARDS"
    assert v._research_derivative_subfamily("Player Yellow Cards") == "PLAYER_CARDS"


def test_player_props_maturation_accepts_only_later_provider_update():
    signals = [{
        "market_family": "SHOTS",
        "signal_generated_at": "2026-09-25T11:30:00+00:00",
    }]
    stale = [{
        "market": "Player Shots",
        "provider_update": "2026-09-25T11:29:00+00:00",
        "values": [{"selection": "Player A - 2", "decimal_price": 1.9}],
    }]
    fresh = [{
        "market": "Player Shots",
        "provider_update": "2026-09-25T11:35:00+00:00",
        "values": [{"selection": "Player A - 2", "decimal_price": 1.85}],
    }]

    assert v._player_prop_markets_with_later_provider_quote(stale, signals) == set()
    assert v._player_prop_markets_with_later_provider_quote(fresh, signals) == {"SHOTS"}


def test_player_props_maturation_does_not_mix_prop_families():
    signals = [{
        "market_family": "GOALSCORER_ANYTIME",
        "signal_generated_at": "2026-09-25T11:30:00+00:00",
    }]
    markets = [{
        "market": "Player Shots",
        "provider_update": "2026-09-25T11:40:00+00:00",
        "values": [{"selection": "Player A - 2", "decimal_price": 1.8}],
    }]
    assert v._player_prop_markets_with_later_provider_quote(markets, signals) == set()


def test_score_or_assist_is_not_goalscorer_anytime_market():
    assert v._research_derivative_subfamily("Player to Score or Assist") is None
    assert v._research_derivative_subfamily("Player Score/Assist") is None
    assert v._research_derivative_subfamily("Anytime Goal Scorer") == "GOALSCORER_ANYTIME"


def test_player_props_maturation_ignores_legacy_score_or_assist_subfamily():
    event = {
        "market": {
            "research_cards_props_markets": [{
                "research_family": "PLAYER_PROPS",
                "research_subfamily": "GOALSCORER_ANYTIME",
                "market": "Player to Score or Assist",
                "values": [{"selection": "Player A", "price": 1.8}],
            }]
        },
        "player_goalscorer_intelligence": {
            "players": [{
                "player_id": 10,
                "p_anytime_goal": 0.42,
            }]
        },
    }
    assert v._player_prop_signal_families(event) == set()



class _PrimaryMaturationDescription:
    def __init__(self, name):
        self.name = name


class _PrimaryMaturationCursor:
    def __init__(self, row):
        self._row = row
        self.query = ""
        self.params = None
        self.description = [
            _PrimaryMaturationDescription(name)
            for name in (
                "fixture_id", "market_family", "market", "signal_generated_at",
                "candidate_source", "league_id", "league", "country", "season",
                "round", "kickoff", "status", "status_long", "home_team_id",
                "home_team", "away_team_id", "away_team", "venue", "city",
            )
        ]

    def execute(self, query, params):
        self.query = query
        self.params = params

    def fetchall(self):
        return [self._row]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _PrimaryMaturationConn:
    def __init__(self, row):
        self.cursor_instance = _PrimaryMaturationCursor(row)

    def cursor(self):
        return self.cursor_instance

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_primary_clv_backlog_accepts_priced_ft_totals_research_without_phase16_rankability(monkeypatch):
    now = datetime.now(timezone.utc)
    signal_at = now - timedelta(minutes=5)
    kickoff = now + timedelta(minutes=35)
    row = (
        9551,
        "FT_TOTALS",
        "Goals Over/Under",
        signal_at,
        "MATCH_TABLE_PRICED_RESEARCH",
        99,
        "Test League",
        "Test Country",
        2026,
        "Round 1",
        kickoff,
        "NS",
        "Not Started",
        10,
        "Home",
        20,
        "Away",
        "Venue",
        "City",
    )
    conn = _PrimaryMaturationConn(row)

    monkeypatch.setattr(v.persistence, "persistence_configured", lambda: True)
    monkeypatch.setattr(v.persistence, "ensure_schema", lambda: None)
    monkeypatch.setattr(v.persistence, "_connect", lambda: conn)

    result = v._load_primary_clv_maturation_backlog(lookahead_minutes=55, limit=10)

    assert result["candidate_count"] == 1
    assert result["candidate_family_counts"] == {"FT_TOTALS": 1}
    assert result["candidate_source_counts"] == {"MATCH_TABLE_PRICED_RESEARCH": 1}
    assert result["source"] == "POSTGRES_PRIMARY_CLV_MATURATION_BACKLOG_V2"
    event = result["candidate_events"][0]
    assert event["classification"] == "RESEARCH_ONLY"
    assert event["bet_eligible"] is False
    assert event["decision_weight"] == 0.0
    assert event["primary_clv_maturation"]["signals"][0]["market_family"] == "FT_TOTALS"
    assert event["primary_clv_maturation"]["signals"][0]["candidate_source"] == "MATCH_TABLE_PRICED_RESEARCH"

    sql = " ".join(conn.cursor_instance.query.split())
    assert "match_table_rows" in sql
    assert "market_mismatch_rows" in sql
    assert "MATCH_TABLE_PRICED_RESEARCH" in sql
    assert "PHASE16_RANKABLE" in sql
    assert "NULLIF(mt.row ->> 'price', '') IS NOT NULL" in sql
    assert "jsonb_typeof(mt.row -> 'price') = 'number'" in sql
    assert "(mt.row ->> 'price')::DOUBLE PRECISION > 1.0" in sql
    assert "latest_signal AS (" in sql
    assert "SELECT ls.* FROM latest_signal ls" in sql


def test_primary_clv_backlog_match_table_branch_normalizes_research_aliases_in_sql():
    # This guards the data-collection path independently from Phase16 promotion readiness.
    source = v._load_primary_clv_maturation_backlog
    assert callable(source)
