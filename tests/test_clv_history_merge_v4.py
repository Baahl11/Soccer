from mcp_gateway import clv_history_merge_v4 as v


def _hist(fid, family_market, signal_ts, close_ts, kickoff, clv=0.5, selection="home"):
    return {
        "fixture_id": fid,
        "market": family_market,
        "selection": selection,
        "signal_timestamp_local": signal_ts,
        "close_timestamp_local": close_ts,
        "kickoff_local": kickoff,
        "signal_price": 2.0,
        "close_price": 1.95,
        "signal_fair_probability": 0.50,
        "close_fair_probability": 0.505,
        "clv_probability_pp": clv,
        "is_true_closing_line": True,
        "closing_line_status": "TRUE_PREKICKOFF_CLOSE_SAME_BOOK_DE_VIG",
        "stage": "T-10",
    }


def test_history_backfill_keeps_only_latest_row_per_fixture_family():
    rows = [
        _hist(1, "Match Winner", "2026-09-20T16:00:00+00:00", "2026-09-20T17:50:00+00:00", "2026-09-20T18:00:00+00:00"),
        _hist(1, "Match Winner", "2026-09-20T17:40:00+00:00", "2026-09-20T17:50:00+00:00", "2026-09-20T18:00:00+00:00"),
    ]
    out = v.normalize_history_rows(rows, [])
    assert len(out) == 1
    assert out[0]["entry_timestamp"] == "2026-09-20T17:40:00+00:00"
    assert out[0]["market_family"] == "1X2"


def test_postgres_fixture_family_coverage_wins_over_history():
    current = [{"fixture_id": 1, "market_family": "1X2", "signal_source": "PIPELINE_MATCH_TABLE"}]
    history = [
        _hist(1, "Match Winner", "2026-09-20T17:40:00+00:00", "2026-09-20T17:50:00+00:00", "2026-09-20T18:00:00+00:00"),
        _hist(1, "Both Teams To Score", "2026-09-20T17:40:00+00:00", "2026-09-20T17:50:00+00:00", "2026-09-20T18:00:00+00:00", selection="Yes"),
    ]
    out = v.normalize_history_rows(history, current)
    assert len(out) == 1
    assert out[0]["market_family"] == "BTTS"


def test_history_backfill_rejects_non_strict_close_timestamps():
    rows = [
        _hist(1, "Match Winner", "2026-09-20T17:55:00+00:00", "2026-09-20T17:50:00+00:00", "2026-09-20T18:00:00+00:00"),
        _hist(2, "Match Winner", "2026-09-20T17:40:00+00:00", "2026-09-20T18:01:00+00:00", "2026-09-20T18:00:00+00:00"),
    ]
    assert v.normalize_history_rows(rows, []) == []


def test_merge_report_recomputes_family_and_unique_fixture_counts():
    postgres = {
        "schema_version": "1.0.0",
        "model_version": "POSTGRES",
        "rows": [
            {
                "fixture_id": 1,
                "market_family": "1X2",
                "signal_source": "PIPELINE_MATCH_TABLE",
                "probability_comparable_same_line": True,
            }
        ],
        "notes": [],
        "provider_requests_added": 0,
    }
    history = [
        _hist(2, "Match Winner", "2026-09-20T17:40:00+00:00", "2026-09-20T17:50:00+00:00", "2026-09-20T18:00:00+00:00"),
        _hist(3, "Both Teams To Score", "2026-09-20T17:40:00+00:00", "2026-09-20T17:50:00+00:00", "2026-09-20T18:00:00+00:00", selection="Yes"),
    ]
    report = v.merge_report(postgres, history)
    assert report["historical_backfill_rows_added"] == 2
    assert report["unique_fixtures_by_family"]["1X2"] == 2
    assert report["unique_fixtures_by_family"]["BTTS"] == 1
    assert report["provider_requests_added"] == 0
