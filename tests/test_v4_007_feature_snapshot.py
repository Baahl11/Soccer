from mcp_gateway import feature_snapshot_v4


def _tick():
    return {
        "generated_at_utc": "2026-09-23T04:55:00+00:00",
        "model_version": "SOCCER EDGE ENGINE v1.0",
    }


def _event():
    return {
        "event_type": "SOCCER_REFRESH",
        "stage": "T-40",
        "model_version": "SOCCER EDGE ENGINE v1.0",
        "fixture": {
            "fixture_id": 12345,
            "league_id": 128,
            "season": 2026,
            "home_team_id": 10,
            "away_team_id": 20,
            "venue": "Test Stadium",
            "city": "Test City",
        },
        "coverage": {"data_tier": "A"},
        "availability_confidence": 0.9,
        "lineups": {
            "both_xi_confirmed": True,
            "both_goalkeepers_confirmed": True,
            "teams": [
                {"team_id": 10, "formation": "4-3-3"},
                {"team_id": 20, "formation": "4-2-3-1"},
            ],
        },
        "injuries": [],
        "raw_projection": {
            "status": "MODELED_LIMITED",
            "sample": {
                "home_home_played": 8,
                "away_away_played": 7,
                "minimum_split_sample": 7,
            },
            "raw_home_goal_rate": 1.72,
            "raw_away_goal_rate": 1.03,
            "raw_total_goals": 2.75,
        },
    }


def test_v4_007_snapshot_is_versioned_and_valid():
    snapshot = feature_snapshot_v4.build(_tick(), _event())
    assert snapshot["schema_version"] == "4.0.0"
    assert snapshot["fixture_id"] == 12345
    assert snapshot["sport_first"] is True
    assert snapshot["market_fields_included"] is False
    assert feature_snapshot_v4.validate(snapshot) == []


def test_v4_007_tick_model_version_wins_over_refresh_event_version():
    tick = _tick()
    tick["model_version"] = "SOCCER EDGE ENGINE v1.7"
    event = _event()
    event["model_version"] = "SOCCER EDGE ENGINE v1.0"

    snapshot = feature_snapshot_v4.build(tick, event)

    assert snapshot["model_version"] == "SOCCER EDGE ENGINE v1.7"


def test_v4_007_refresh_event_model_version_is_fallback_when_tick_version_missing():
    tick = _tick()
    tick["model_version"] = None
    event = _event()
    event["model_version"] = "SOCCER EDGE ENGINE v1.0"

    snapshot = feature_snapshot_v4.build(tick, event)

    assert snapshot["model_version"] == "SOCCER EDGE ENGINE v1.0"


def test_v4_007_feature_envelope_has_required_metadata():
    snapshot = feature_snapshot_v4.build(_tick(), _event())
    row = snapshot["features"]["team_performance.home_goal_rate_blend"]
    assert row["value"] == 1.72
    assert row["source"] == "SOCCER_EDGE_VERIFIED_GOAL_RATE_BASELINE"
    assert row["captured_at"] == "2026-09-23T04:55:00+00:00"
    assert row["sample_n"] == 7
    assert row["missing_reason"] is None


def test_v4_007_advanced_metrics_are_explicitly_missing_not_imputed():
    snapshot = feature_snapshot_v4.build(_tick(), _event())
    for key in ("xg.xgf", "xg.npxg", "territory.ppda", "territory.field_tilt"):
        row = snapshot["features"][key]
        assert row["value"] is None
        assert row["missing_reason"]
        assert row["source"] == "NO_VERIFIED_RUNTIME_SOURCE"


def test_v4_007_snapshot_does_not_store_market_fields():
    event = _event()
    event["market"] = {"markets": [{"market": "Match Winner", "price": 1.8}]}
    event["market_decision"] = {"best_decision": {"decimal_price": 1.8}}
    snapshot = feature_snapshot_v4.build(_tick(), event)
    serialized = str(snapshot).lower()
    assert "decimal_price" not in serialized
    assert "match winner" not in serialized
    assert snapshot["market_fields_included"] is False
