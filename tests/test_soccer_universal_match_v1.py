from mcp_gateway import soccer_universal_match_v1 as universal


def fixture_row(*, raw=False, sport=0, markets=0, prices=None, classification=None):
    return {
        "fixture": {"fixture_id": 991, "home_team": "Local", "away_team": "Visita"},
        "coverage": {
            "analysis_rows": 1 if raw else 0,
            "sport_evidence_count": sport,
            "market_evidence_count": markets,
            "raw_sport_projection_present": raw,
            "verified_price_markets": prices or [],
            "evidence_inventory": {"model_runs": 1 if raw else 0, "feature_snapshots": sport},
        },
        "screening": {"raw_projection_recorded": raw, "missing_evidence_codes": []},
        "state": {"classification": classification, "reason_display": "Persisted reasoning"},
    }


def test_pass_with_real_sporting_analysis_remains_visible_even_if_market_not_verified():
    row = universal.add_to_slate_row(fixture_row(raw=True, sport=2, classification="PASS"))
    assert row["sport_analysis"]["status"] == "MODELED_LIMITED"
    assert row["sport_analysis"]["raw_projection_recorded"] is True
    assert row["sport_analysis"]["probabilities"] is None
    assert row["opportunity"]["status"] == "SPORT_SCREEN_PASS"
    assert row["opportunity"]["canonical_classification"] == "PASS"
    assert row["opportunity"]["verified_price_market_count"] == 0
    assert row["opportunity"]["recommendation_executable"] is False


def test_pass_with_verified_market_is_separate_from_sport_data():
    row = universal.add_to_slate_row(fixture_row(raw=True, sport=2, markets=1, prices=["Goals"], classification="PASS"))
    assert row["opportunity"]["status"] == "SIN_VALOR"
    assert row["sport_analysis"]["status"] == "MODELED_LIMITED"


def test_prices_cannot_create_a_sporting_forecast():
    row = universal.add_to_slate_row(fixture_row(markets=9))
    assert row["sport_analysis"]["status"] == "DATA_INSUFFICIENT"
    assert row["sport_analysis"]["probabilities"] is None
    assert row["opportunity"]["status"] == "MERCADO_NO_EVALUADO"


def test_an_unscreened_fixture_is_not_silently_marked_data_insufficient():
    row = universal.add_to_slate_row(fixture_row())
    assert row["sport_analysis"]["status"] == "NOT_ANALYZED"
    assert row["opportunity"]["status"] == "MERCADO_NO_EVALUADO"


def test_deep_analyst_layer_requires_a_real_complete_1x2():
    packet = {
        "sport_context": {"outcome_probabilities": {"home": .48, "draw": .28, "away": .24},
                          "expected_goals": {"home": 1.41, "away": 1.03, "xg_verified": False}},
        "model_context": {"model_version": "test"},
        "decision_summary": {"classification": "PASS"},
        "analyst_review": {"analysis_rows": 1, "verified_price_markets": []},
        "market_context": {"candidate_count": 0},
    }
    layers = universal.detail_layers(packet)
    assert layers["sport_analysis"]["status"] == "MODELED_LIMITED"
    assert layers["sport_analysis"]["tracks"]["GOALS_BTTS"]["goal_rates_not_xg"]["xg_verified"] is False
    assert layers["opportunity"]["status"] == "SPORT_SCREEN_PASS"
    assert layers["opportunity"]["canonical_classification"] == "PASS"
    assert layers["opportunity"]["price_currently_verified"] is False
    packet["sport_context"]["outcome_probabilities"]["away"] = None
    assert universal.detail_layers(packet)["sport_analysis"]["status"] != "MODELED_LIMITED"


def test_bet_not_marked_executable_without_verified_price():
    row = universal.add_to_slate_row(fixture_row(raw=True, sport=2, classification="BET"))
    assert row["opportunity"]["status"] == "VERIFICACION_PENDIENTE"
    assert row["opportunity"]["recommendation_executable"] is False


def test_genuine_p0_queue_proves_a_limited_projection_without_republishing_model():
    row = fixture_row()
    row["screen_queue"] = {"status": "MODELED_LIMITED", "attempt_count": 1}
    observed = universal.add_to_slate_row(row)
    assert observed["sport_analysis"]["status"] == "MODELED_LIMITED"
    assert observed["sport_analysis"]["raw_projection_recorded"] is True
    assert observed["sport_analysis"]["probabilities"] is None
    assert observed["sport_analysis"]["model_run_count"] == 0
    assert observed["opportunity"]["status"] == "MERCADO_NO_EVALUADO"


def test_queue_input_failure_is_explained_not_reported_as_never_analyzed():
    row = fixture_row()
    row["screen_queue"] = {"status": "DATA_INSUFFICIENT", "attempt_count": 1}
    observed = universal.add_to_slate_row(row)
    assert observed["sport_analysis"]["status"] == "DATA_INSUFFICIENT"
    assert observed["sport_analysis"]["screen_attempt_count"] == 1
    assert observed["sport_analysis"]["probabilities"] is None
