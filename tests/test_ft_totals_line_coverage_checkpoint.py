from datetime import datetime, timezone

from mcp_gateway import ft_totals_line_coverage_checkpoint as v


def test_build_report_tracks_quarter_lines_and_focus_without_gate_effect():
    rows = [
        {
            "line": "2.25",
            "value_rows": 40,
            "unique_fixtures": 8,
            "snapshot_rows": 20,
            "bookmaker_count": 4,
            "first_captured_at": "2026-09-01T00:00:00+00:00",
            "last_captured_at": "2026-09-27T00:00:00+00:00",
        },
        {
            "line": "2.50",
            "value_rows": 100,
            "unique_fixtures": 20,
            "snapshot_rows": 50,
            "bookmaker_count": 6,
        },
        {
            "line": "2.75",
            "value_rows": 30,
            "unique_fixtures": 6,
            "snapshot_rows": 15,
            "bookmaker_count": 3,
        },
    ]
    report = v._build_report(
        rows,
        {"market_snapshot_rows": 85, "unique_fixtures": 24, "unique_bookmakers": 7},
        lookback_days=180,
    )

    assert report["status"] == "HISTORICAL_FT_TOTALS_LINE_COVERAGE_READY"
    assert report["unique_fixtures"] == 24
    assert report["focus_quarter_lines"]["2.25"]["unique_fixtures"] == 8
    assert report["focus_quarter_lines"]["2.75"]["unique_fixtures"] == 6
    assert report["focus_quarter_lines"]["3.25"]["unique_fixtures"] == 0
    assert report["focus_quarter_lines"]["2.25"]["quarter_line"] is True
    assert report["focus_quarter_lines"]["2.25"]["validation_supported"] is True
    assert report["counts_as_model_settled"] is False
    assert report["counts_as_oos"] is False
    assert report["counts_as_true_clv"] is False
    assert report["provider_requests_added"] == 0


def test_build_uses_memory_cache_without_second_postgres_refresh(monkeypatch):
    calls = {"count": 0}

    def fake_query(*, lookback_days):
        calls["count"] += 1
        return {
            "schema_version": v.SCHEMA_VERSION,
            "model_version": v.MODEL_VERSION,
            "status": "HISTORICAL_FT_TOTALS_LINE_COVERAGE_READY",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "lookback_days": lookback_days,
            "focus_quarter_lines": {},
            "provider_requests_added": 0,
        }

    monkeypatch.setattr(v, "_query_postgres", fake_query)
    monkeypatch.setattr(v, "_load_persisted_cache", lambda **kwargs: None)
    v._CACHE = None
    v._CACHE_MONOTONIC = None

    first = v.build(lookback_days=90)
    second = v.build(lookback_days=90)

    assert calls["count"] == 1
    assert first["cache_status"] == "POSTGRES_REFRESH"
    assert second["cache_status"] == "MEMORY_CACHE_HIT"
    assert second["provider_requests_added"] == 0


def test_build_reuses_persisted_pipeline_cache_across_worker_processes(monkeypatch):
    persisted = {
        "schema_version": v.SCHEMA_VERSION,
        "model_version": v.MODEL_VERSION,
        "status": "HISTORICAL_FT_TOTALS_LINE_COVERAGE_READY",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "lookback_days": 180,
        "focus_quarter_lines": {"2.25": {"unique_fixtures": 12}},
        "provider_requests_added": 0,
    }

    monkeypatch.setattr(v, "_load_persisted_cache", lambda **kwargs: dict(persisted))
    monkeypatch.setattr(
        v,
        "_query_postgres",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("postgres refresh should not run")),
    )
    v._CACHE = None
    v._CACHE_MONOTONIC = None

    report = v.build(lookback_days=180)

    assert report["cache_status"] == "PERSISTED_PIPELINE_CACHE_HIT"
    assert report["focus_quarter_lines"]["2.25"]["unique_fixtures"] == 12
    assert report["provider_requests_added"] == 0


def test_persisted_cache_loader_rejects_expired_or_wrong_lookback(monkeypatch):
    old = datetime(2020, 1, 1, tzinfo=timezone.utc).isoformat()
    monkeypatch.setattr(
        v.persistence,
        "load_latest_pipeline_payload",
        lambda: {
            "ft_totals_settlement_capture": {
                "historical_line_coverage": {
                    "status": "HISTORICAL_FT_TOTALS_LINE_COVERAGE_READY",
                    "generated_at_utc": old,
                    "lookback_days": 180,
                }
            }
        },
    )
    assert v._load_persisted_cache(lookback_days=180) is None

    monkeypatch.setattr(
        v.persistence,
        "load_latest_pipeline_payload",
        lambda: {
            "ft_totals_settlement_capture": {
                "historical_line_coverage": {
                    "status": "HISTORICAL_FT_TOTALS_LINE_COVERAGE_READY",
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "lookback_days": 90,
                }
            }
        },
    )
    assert v._load_persisted_cache(lookback_days=180) is None


def test_build_returns_error_envelope_without_promotion(monkeypatch):
    def fail(*, lookback_days):
        raise RuntimeError("db unavailable")

    monkeypatch.setattr(v, "_query_postgres", fail)
    v._CACHE = None
    v._CACHE_MONOTONIC = None

    report = v.build(force_refresh=True)

    assert report["status"] == "HISTORICAL_FT_TOTALS_LINE_COVERAGE_ERROR"
    assert "db unavailable" in report["detail"]
    assert report["provider_requests_added"] == 0
    assert report["production_promotion_allowed"] is False
