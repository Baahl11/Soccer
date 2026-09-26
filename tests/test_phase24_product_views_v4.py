from mcp_gateway import product_views_v4 as v


def _row(**overrides):
    row = {
        "fixture_id": 1,
        "home": "A",
        "away": "B",
        "league": "League",
        "status": "NS",
        "stage": "T-20",
        "market_family": "TOTAL",
        "market": "Goals Over/Under",
        "selection": "Over",
        "model_signal": "VERY_STRONG",
        "model_signal_score": 90.0,
        "execution_status": "WAIT_PRICE",
    }
    row.update(overrides)
    return row


def _payload():
    return {
        "status": "ok",
        "version": "4.32.9-test",
        "model_version": "SOCCER EDGE ENGINE test",
        "generated_at_utc": "2026-09-26T00:53:06Z",
        "generated_at_local": "2026-09-25T18:53:06-06:00",
        "match_table_rows": [
            _row(phase16_binary_calibration_diagnostics={"rows": 407}),
            _row(
                fixture_id=2,
                market_family="BTTS",
                market="Both Teams To Score",
                model_signal="MODERATE",
                execution_status="WAIT_XI",
            ),
            _row(
                fixture_id=3,
                market_family="CORNERS",
                market="Corners Over/Under",
                model_signal="STRONG",
                execution_status="READY",
            ),
        ],
        "market_mismatch_rows": [
            {"fixture_id": 1, "market_family": "FT_TOTALS", "mismatch_score": 88.0}
        ],
        "phase17_clv_engine": {"status": "ENGINE_IMPLEMENTED"},
        "phase18_oos_backtest_framework": {"status": "FRAMEWORK_IMPLEMENTED"},
        "phase19_promotion_framework": {"status": "FRAMEWORK_IMPLEMENTED"},
        "phase20_bankroll_risk_engine": {"status": "ENGINE_IMPLEMENTED_LOCKED_UNTIL_PRODUCTION_VALIDATION"},
        "phase14_cards_referee_validation": {"status": "VALIDATION_GATE_IMPLEMENTED"},
        "phase15_player_props_validation": {
            "status": "VALIDATION_GATE_IMPLEMENTED",
            "minimum_prop_true_clv_rows": 50,
        },
        "phase16_market_mismatch_finder": {"status": "RESEARCH_ENGINE_IMPLEMENTED"},
        "phase21_alert_engine": {"status": "ENGINE_IMPLEMENTED_AUDIT_ONLY"},
        "phase22_live_inplay_engine": {"status": "INPUT_FRAMEWORK_IMPLEMENTED_RESEARCH_ONLY"},
        "phase23_mlops_model_registry": {"status": "FRAMEWORK_IMPLEMENTED_REGISTRY_POPULATION_PENDING"},
        "phase24_final_product_experience": {"status": "READ_ONLY_DASHBOARD_IMPLEMENTED"},
        "v4_013_calibration_engine_v1": {"status": "RESEARCH_FRAMEWORK_IMPLEMENTED"},
        "v4_014_model_disagreement_engine": {"status": "RESEARCH_FRAMEWORK_IMPLEMENTED"},
        "v4_015_confidence_engine_v1": {"status": "RESEARCH_FRAMEWORK_IMPLEMENTED"},
        "v4_017_1x2_calibration_validation": {
            "status": "VALIDATION_GATE_IMPLEMENTED",
            "minimum_calibration_sample": 300,
            "minimum_family_specific_true_clv_rows": 50,
        },
        "v4_018_btts_calibration_validation": {
            "minimum_family_specific_true_clv_rows": 50,
        },
        "v4_019_team_totals_oos_validation": {
            "minimum_family_specific_true_clv_rows": 50,
        },
        "v4_020_1h_oos_validation": {
            "minimum_family_specific_true_clv_rows": 50,
        },
        "v4_022_corners_oos_validation": {
            "minimum_formation_adjusted": 100,
        },
        "database_persisted": True,
        "database_error": None,
        "api_calls_this_tick": 12,
        "max_api_calls_per_tick": 35,
        "effective_max_api_calls_per_tick": 40,
        "last_daily_remaining": 3000,
        "daily_budget_mode": "NORMAL",
        "fixture_scan_count": 200,
        "due_fixture_count": 20,
        "event_count": 7,
        "deep_dive_processed_count": 5,
        "research_visible_count": 9,
        "bet_candidate_count": 0,
    }


def test_phase24_builds_all_master_dashboard_views_and_control_tower():
    result = v.build_views(_payload())
    assert result["status"] == "CONTROL_TOWER_V1_CONTRACT"
    assert set(result["view_names"]) == set(v.VIEW_NAMES)
    assert result["views"]["strong_sport_signals"]["total"] == 2
    assert result["views"]["value_plays"]["total"] == 1
    assert result["views"]["waiting_for_price"]["total"] == 1
    assert result["views"]["waiting_for_xi"]["total"] == 1
    assert result["views"]["corners"]["total"] == 1

    tower = result["views"]["control_tower"]
    assert tower["status"] == "LIVE"
    assert tower["system_health"]["postgres"] == "HEALTHY"
    assert tower["system_health"]["api_football_remaining"] == 3000
    assert tower["pipeline"]["fixtures_scanned"] == 200
    assert tower["pipeline"]["api_call_cap"] == 40
    assert tower["errors"]["count"] == 0
    assert len(tower["phases"]) == 11
    assert tower["production_promotion_allowed"] is False

    gates = {gate["key"]: gate for gate in tower["validation_gates"]}
    assert gates["phase16_calibration_sample"]["current"] == 407
    assert gates["phase16_calibration_sample"]["target"] == 300
    assert gates["1x2_true_clv"]["current"] is None
    assert gates["1x2_true_clv"]["target"] == 50


def test_control_tower_uses_authoritative_top_level_data_health_and_detects_errors():
    payload = _payload()
    payload["database_persisted"] = True
    payload["match_table_rows"].append(
        _row(
            fixture_id=99,
            home="Broken",
            away="Fixture",
            classification="PIPELINE_ERROR",
            reason="_compact_odds failed",
        )
    )
    result = v.build_views(payload)
    assert result["views"]["data_health"]["database_persisted"] is True
    tower = result["views"]["control_tower"]
    assert tower["status"] == "DEGRADED"
    assert tower["errors"]["count"] == 1
    assert tower["errors"]["rows"][0]["fixture_id"] == 99


def test_phase24_excludes_postgame_from_todays_active_slate():
    result = v.build_views({
        "match_table_rows": [
            _row(fixture_id=1, status="FT", stage="POSTGAME"),
            _row(fixture_id=2, status="NS", stage="T-40"),
        ]
    })
    assert result["views"]["todays_slate"]["total"] == 1
    assert result["views"]["todays_slate"]["rows"][0]["fixture_id"] == 2


def test_phase24_view_limits_are_bounded():
    rows = [_row(fixture_id=i) for i in range(50)]
    result = v.build_views({"match_table_rows": rows}, limit=10)
    assert result["views"]["todays_slate"]["total"] == 50
    assert len(result["views"]["todays_slate"]["rows"]) == 10
