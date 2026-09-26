from mcp_gateway import clv_engine_v4 as v


def _row(**overrides):
    row = {
        "fixture_id": 1,
        "league": "Test League",
        "stage": "T-20",
        "classification": "WATCH",
        "tier": "A",
        "confidence": "HIGH",
        "market_family": "1X2",
        "market": "Match Winner",
        "selection": "home",
        "entry_line": None,
        "entry_price": 2.20,
        "closing_line": None,
        "closing_price": 2.00,
        "probability_clv": 3.0,
        "bookmaker": "Book",
        "model_version": "v-test",
        "signal_source": "PIPELINE_MATCH_TABLE",
        "is_true_closing_line": True,
    }
    row.update(overrides)
    return row


def test_phase17_derives_positive_price_clv():
    normalized = v.normalize_row(_row())
    assert normalized["price_clv"] == 10.0
    assert normalized["probability_clv"] == 3.0
    assert normalized["bookmaker"] == "Book"
    assert normalized["field_status"]["entry_line"] == "NOT_APPLICABLE"
    assert normalized["field_status"]["closing_line"] == "NOT_APPLICABLE"
    assert normalized["field_status"]["line_movement"] == "NOT_APPLICABLE"


def test_phase17_legacy_row_maps_signal_and_close_fields_without_fabricating_attribution():
    normalized = v.normalize_row({
        "fixture_id": 1,
        "league": "League",
        "stage": "T-40",
        "tier": "B",
        "market_family": "1X2",
        "market": "Match Winner",
        "line": None,
        "signal_price": 3.34,
        "close_price": 2.83,
        "clv_probability_pp": 5.1681,
        "bookmaker_at_signal": "Pinnacle",
        "signal_source": "HISTORICAL_DEDICATED_CLOSE",
        "is_true_closing_line": True,
    })
    assert normalized["entry_price"] == 3.34
    assert normalized["closing_price"] == 2.83
    assert normalized["price_clv"] > 18.0
    assert normalized["probability_clv"] == 5.1681
    assert normalized["bookmaker"] == "Pinnacle"
    assert normalized["field_status"]["model_version"] == "LEGACY_UNKNOWN"
    assert normalized["field_status"]["confidence"] == "LEGACY_UNKNOWN"
    assert normalized["fully_attributed"] is False
    assert normalized["capture_complete"] is True


def test_phase17_blocks_missing_closing_line_and_current_model_metadata_for_line_market():
    rows = [_row(
        fixture_id=index,
        market_family="FT_TOTALS",
        market="Goals Over/Under",
        selection="Over",
        entry_line=2.5,
        closing_line=None,
        model_version=None,
        confidence=None,
    ) for index in range(60)]
    report = v.build_report(rows)
    assert report["status"] == "CLV_TRACKING_INCOMPLETE"
    assert "CLOSING_LINE_NOT_FULLY_CAPTURED" in report["blockers"]
    assert "LINE_MOVEMENT_NOT_FULLY_CAPTURED" in report["blockers"]
    assert "MODEL_VERSION_NOT_FULLY_CAPTURED" in report["blockers"]
    assert "CONFIDENCE_NOT_FULLY_CAPTURED" in report["blockers"]
    assert report["production_promotion_allowed"] is False


def test_phase17_lineless_markets_do_not_create_false_line_blockers():
    rows = [_row(fixture_id=index, market_family="1X2") for index in range(30)]
    rows += [
        _row(
            fixture_id=100 + index,
            market_family="BTTS",
            market="Both Teams To Score",
            selection="Yes",
        )
        for index in range(30)
    ]
    report = v.build_report(rows)
    assert report["status"] == "CLV_CAPTURE_COMPLETE_ANALYSIS_AVAILABLE"
    assert "CLOSING_LINE_NOT_FULLY_CAPTURED" not in report["blockers"]
    assert "ENTRY_LINE_NOT_FULLY_CAPTURED" not in report["blockers"]
    assert "LINE_MOVEMENT_NOT_FULLY_CAPTURED" not in report["blockers"]
    assert report["field_coverage"]["closing_line"]["not_applicable"] == 60
    assert report["capture_completeness"]["line_not_applicable_rows"] == 60


def test_phase17_legacy_unknown_attribution_is_warning_not_modern_capture_blocker():
    rows = []
    for index in range(60):
        rows.append(_row(
            fixture_id=index,
            signal_source="HISTORICAL_DEDICATED_CLOSE",
            model_version=None,
            confidence=None,
        ))
    report = v.build_report(rows)
    assert report["status"] == "CLV_CAPTURE_COMPLETE_ANALYSIS_AVAILABLE"
    assert report["blockers"] == []
    assert "LEGACY_MODEL_VERSION_UNKNOWN_60" in report["warnings"]
    assert "LEGACY_CONFIDENCE_UNKNOWN_60" in report["warnings"]
    assert report["capture_completeness"]["legacy_attribution_rows"] == 60
    assert report["capture_completeness"]["fully_attributed_rows"] == 0


def test_phase17_aggregates_all_required_dimensions_when_complete():
    rows = []
    for index in range(60):
        rows.append(_row(
            fixture_id=index,
            market_family="FT_TOTALS",
            market="Goals Over/Under",
            tier="A" if index % 2 == 0 else "B",
            confidence="HIGH" if index % 2 == 0 else "MODERATE",
            stage="T-20",
            model_version="v4.21",
            entry_line=2.5,
            closing_line=2.25,
            probability_clv=1.0,
        ))
    report = v.build_report(rows)
    assert report["status"] == "CLV_CAPTURE_COMPLETE_ANALYSIS_AVAILABLE"
    assert report["true_closing_line_rows"] == 60
    assert "Goals Over/Under" in report["by_market"]
    assert "FT_TOTALS" in report["by_market_family"]
    assert "Test League" in report["by_league"]
    assert "A" in report["by_tier"]
    assert "HIGH" in report["by_confidence"]
    assert "T-20" in report["by_stage"]
    assert "v4.21" in report["by_model_version"]
    assert report["overall"]["avg_probability_clv_pp"] == 1.0
    assert report["capture_completeness"]["complete_rate"] == 1.0
