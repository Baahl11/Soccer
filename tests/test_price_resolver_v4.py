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


def test_1x2_research_can_select_draw_and_apply_temperature_calibration():
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
    assert row["price_resolution_calibrated_probability_added"] is True
    assert 0 < row["p_model_calibrated"] < 1
    assert row["phase16_calibration_source"] == "CURRENT_MODEL_OOS_TEMPERATURE:1X2"
