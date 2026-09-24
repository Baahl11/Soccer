from mcp_gateway import oos_history_backfill_v4 as v


def _prediction(fixture_id, generated, kickoff, stage, home_prob):
    return {
        "fixture_id": fixture_id,
        "generated_at_utc": generated,
        "kickoff_local": kickoff,
        "stage": stage,
        "fixture_status": "NS",
        "model_version": "M1",
        "raw_projection": {
            "raw_home_win_prob": home_prob,
            "raw_draw_prob": 0.25,
            "raw_away_win_prob": 1.0 - home_prob - 0.25,
            "raw_btts_yes_prob": 0.52,
            "raw_over_2_5_prob": 0.48,
        },
    }


def test_history_backfill_uses_latest_strictly_pre_kickoff_prediction():
    rows = [
        _prediction(1, "2026-09-20T16:00:00+00:00", "2026-09-20T18:00:00+00:00", "T-90", 0.40),
        _prediction(1, "2026-09-20T17:50:00+00:00", "2026-09-20T18:00:00+00:00", "T-10", 0.55),
        _prediction(1, "2026-09-20T18:05:00+00:00", "2026-09-20T18:00:00+00:00", "T-10", 0.99),
        {
            "fixture_id": 1,
            "generated_at_utc": "2026-09-20T20:00:00+00:00",
            "kickoff_local": "2026-09-20T18:00:00+00:00",
            "stage": "POSTGAME",
            "fixture_status": "FT",
            "result": {"goals": {"home": 2, "away": 1}},
        },
    ]
    out = v.historical_rows(rows)
    assert len(out) == 1
    assert out[0]["run_type"] == "T-10"
    assert out[0]["predictions"]["home_win"] == 0.55
    assert out[0]["outcomes"]["home_win"] == 1
    assert out[0]["source"] == "HISTORICAL_SIGNAL_LEDGER"


def test_history_backfill_rejects_result_without_final_or_postgame_state():
    rows = [
        _prediction(1, "2026-09-20T17:00:00+00:00", "2026-09-20T18:00:00+00:00", "T-60", 0.45),
        {
            "fixture_id": 1,
            "generated_at_utc": "2026-09-20T17:30:00+00:00",
            "kickoff_local": "2026-09-20T18:00:00+00:00",
            "stage": "T-30",
            "fixture_status": "NS",
            "result": {"goals": {"home": 2, "away": 1}},
        },
    ]
    assert v.historical_rows(rows) == []


def test_merge_prefers_latest_valid_pre_kickoff_row():
    history = [{
        "fixture_id": 1,
        "run_timestamp": "2026-09-20T17:00:00+00:00",
        "kickoff": "2026-09-20T18:00:00+00:00",
        "source": "HISTORICAL_SIGNAL_LEDGER",
    }]
    postgres = [{
        "fixture_id": 1,
        "run_timestamp": "2026-09-20T17:50:00+00:00",
        "kickoff": "2026-09-20T18:00:00+00:00",
    }]
    merged = v.merge_rows(postgres, history)
    assert len(merged) == 1
    assert merged[0]["source"] == "POSTGRES_NATIVE"
    assert merged[0]["run_timestamp"] == "2026-09-20T17:50:00+00:00"


def test_summary_remains_research_only():
    rows = []
    for fixture_id in range(1, 6):
        rows.append({
            "fixture_id": fixture_id,
            "run_timestamp": "2026-09-20T17:00:00+00:00",
            "kickoff": "2026-09-20T18:00:00+00:00",
            "run_type": "T-60",
            "model_version": "M1",
            "source": "HISTORICAL_SIGNAL_LEDGER",
            "predictions": {
                "home_win": 0.5,
                "draw": 0.25,
                "away_win": 0.25,
                "btts": 0.5,
                "over_2_5": 0.5,
            },
            "outcomes": {
                "home_win": 1,
                "draw": 0,
                "away_win": 0,
                "btts": 1,
                "over_2_5": 1,
            },
        })
    report = v.summarize(rows, postgres_n=0, historical_n=5)
    assert report["fixture_rows"] == 5
    assert report["provider_requests_added"] == 0
    assert report["production_promotion_allowed"] is False
