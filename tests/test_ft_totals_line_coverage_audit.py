from __future__ import annotations

from mcp_gateway import ft_totals_line_coverage_audit as audit


def test_value_side_line_accepts_quarter_lines_from_multiple_shapes():
    assert audit.value_side_line({"selection": "Over 2.25"}) == ("OVER", 2.25)
    assert audit.value_side_line({"value": "Under 2.75", "handicap": "2.75"}) == ("UNDER", 2.75)
    assert audit.value_side_line({"selection": "Over", "line": 3.25}) == ("OVER", 3.25)


def test_line_coverage_dedupes_fixtures_and_keeps_observed_separate_from_model_evidence():
    rows = [
        {
            "fixture_id": 1001,
            "captured_at": "2026-09-27T10:00:00+00:00",
            "stage": "T-40",
            "bookmaker": "Book A",
            "market": "Goals Over/Under",
            "provider_update": "2026-09-27T09:59:00+00:00",
            "pre_kickoff": True,
            "values": [
                {"selection": "Over 2.25", "decimal_price": 1.95},
                {"selection": "Under 2.25", "decimal_price": 1.91},
                {"selection": "Over 2.5", "decimal_price": 2.10},
                {"selection": "Under 2.5", "decimal_price": 1.80},
            ],
        },
        {
            "fixture_id": 1001,
            "captured_at": "2026-09-27T10:20:00+00:00",
            "stage": "T-20",
            "bookmaker": "Book B",
            "market": "Over/Under",
            "provider_update": "2026-09-27T10:19:00+00:00",
            "pre_kickoff": True,
            "values": [
                {"selection": "Over", "line": 2.25, "price": 1.97},
                {"selection": "Under", "line": 2.25, "price": 1.89},
            ],
        },
        {
            "fixture_id": 1002,
            "captured_at": "2026-09-27T11:00:00+00:00",
            "stage": "T-10",
            "bookmaker": "Book A",
            "market": "Goals Over Under",
            "provider_update": "2026-09-27T10:58:00+00:00",
            "pre_kickoff": True,
            "values": [
                {"selection": "Over 2.75", "odd": "2.05"},
                {"selection": "Under 2.75", "odd": "1.82"},
            ],
        },
        {
            "fixture_id": 1003,
            "captured_at": "2026-09-27T12:00:00+00:00",
            "stage": "POSTGAME",
            "bookmaker": "Book C",
            "market": "Goals Over/Under",
            "provider_update": "2026-09-27T11:58:00+00:00",
            "pre_kickoff": False,
            "values": [
                {"selection": "Over 3.25", "odd": "2.00"},
                {"selection": "Under 3.25", "odd": "1.90"},
            ],
        },
        {
            "fixture_id": 9999,
            "captured_at": "2026-09-27T10:00:00+00:00",
            "stage": "T-20",
            "bookmaker": "Book X",
            "market": "Total Corners",
            "provider_update": "2026-09-27T09:59:00+00:00",
            "pre_kickoff": True,
            "values": [
                {"selection": "Over 2.25", "odd": "1.90"},
                {"selection": "Under 2.25", "odd": "1.90"},
            ],
        },
    ]

    result = audit.summarize_rows(rows, lookback_days=180)

    assert result["status"] == "FT_TOTALS_OBSERVED_LINE_COVERAGE_AUDIT"
    assert result["rows_seen"] == 5
    assert result["canonical_market_snapshot_rows"] == 4
    assert result["canonical_pre_kickoff_market_snapshot_rows"] == 3
    assert result["unique_fixtures"] == 3
    assert result["pre_kickoff_unique_fixtures"] == 2
    assert result["bookmaker_count"] == 3
    assert result["observed_lines"] == [2.25, 2.5, 2.75, 3.25]
    assert result["pre_kickoff_observed_lines"] == [2.25, 2.5, 2.75]
    assert result["quarter_lines_observed"] == [2.25, 2.75, 3.25]
    assert result["quarter_lines_pre_kickoff"] == [2.25, 2.75]
    assert result["quarter_line_unique_fixtures"] == 3
    assert result["pre_kickoff_quarter_line_unique_fixtures"] == 2

    line_225 = result["by_line"]["2.25"]
    assert line_225["supported_by_v4_settlement"] is True
    assert line_225["quarter_line"] is True
    assert line_225["unique_fixtures"] == 1
    assert line_225["pre_kickoff_unique_fixtures"] == 1
    assert line_225["market_snapshot_rows"] == 2
    assert line_225["value_rows"] == 4
    assert line_225["priced_value_rows"] == 4
    assert line_225["bookmaker_count"] == 2
    assert line_225["side_counts"] == {"OVER": 2, "UNDER": 2}

    line_275 = result["supported_line_coverage"]["2.75"]
    assert line_275["observed"] is True
    assert line_275["observed_pre_kickoff"] is True
    assert line_275["pre_kickoff_unique_fixtures"] == 1

    line_325 = result["supported_line_coverage"]["3.25"]
    assert line_325["observed"] is True
    assert line_325["observed_pre_kickoff"] is False
    assert 3.25 in result["missing_supported_pre_kickoff_lines"]

    assert result["provider_requests_added"] == 0
    assert result["decision_weight"] == 0.0
    assert result["production_promotion_allowed"] is False
    assert result["canonical_bet_logic_changed"] is False
    assert result["model_settled_sample_changed"] is False
    assert result["actionable_sample_changed"] is False


def test_unsupported_line_is_visible_but_does_not_enter_supported_coverage():
    rows = [
        {
            "fixture_id": 2001,
            "stage": "T-20",
            "bookmaker": "Book A",
            "market": "Goals Over/Under",
            "provider_update": "2026-09-27T10:00:00+00:00",
            "pre_kickoff": True,
            "values": [
                {"selection": "Over 4.25", "odd": "2.40"},
                {"selection": "Under 4.25", "odd": "1.55"},
            ],
        }
    ]

    result = audit.summarize_rows(rows, lookback_days=30)

    assert result["observed_lines"] == [4.25]
    assert result["unsupported_line_value_rows"] == 2
    assert result["supported_line_value_rows"] == 0
    assert result["by_line"]["4.25"]["supported_by_v4_settlement"] is False
    assert "4.25" not in result["supported_line_coverage"]
