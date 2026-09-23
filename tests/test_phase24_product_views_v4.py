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


def test_phase24_builds_all_master_dashboard_views():
    payload = {
        "match_table_rows": [
            _row(),
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
            {
                "fixture_id": 1,
                "market_family": "FT_TOTALS",
                "mismatch_score": 88.0,
            }
        ],
        "phase17_clv_engine": {"status": "ENGINE_IMPLEMENTED"},
        "phase18_oos_backtest_framework": {"status": "FRAMEWORK_IMPLEMENTED"},
        "phase19_promotion_framework": {"status": "FRAMEWORK_IMPLEMENTED"},
        "phase20_bankroll_risk_engine": {"status": "RISK_ENGINE_LOCKED"},
        "v4_013_calibration_engine_v1": {"status": "RESEARCH_FRAMEWORK_IMPLEMENTED"},
        "v4_014_model_disagreement_engine": {"status": "RESEARCH_FRAMEWORK_IMPLEMENTED"},
        "v4_015_confidence_engine_v1": {"status": "RESEARCH_FRAMEWORK_IMPLEMENTED"},
        "phase23_mlops_model_registry": {"status": "FRAMEWORK_IMPLEMENTED_REGISTRY_POPULATION_PENDING"},
        "database_persisted": True,
        "api_calls_this_tick": 12,
        "max_api_calls_per_tick": 35,
        "last_daily_remaining": 3000,
        "fixture_scan_count": 200,
        "due_fixture_count": 20,
    }
    result = v.build_views(payload)
    assert result["status"] == "API_VIEW_CONTRACT_IMPLEMENTED_UI_PENDING"
    assert set(result["view_names"]) == set(v.VIEW_NAMES)
    assert result["views"]["strong_sport_signals"]["total"] == 2
    assert result["views"]["value_plays"]["total"] == 1
    assert result["views"]["waiting_for_price"]["total"] == 1
    assert result["views"]["waiting_for_xi"]["total"] == 1
    assert result["views"]["corners"]["total"] == 1
    assert result["views"]["data_health"]["database_persisted"] is True


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
