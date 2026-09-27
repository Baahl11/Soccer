from __future__ import annotations

from mcp_gateway import ft_totals_settlement_capture as capture
from mcp_gateway.analyze_ft_totals import grade_total
from mcp_gateway.ft_totals_validation_v4 import settlement_return_units


def test_split_total_line_supports_integer_half_and_quarter_lines():
    assert capture.split_total_line(2.0) == [2.0]
    assert capture.split_total_line(2.5) == [2.5]
    assert capture.split_total_line(2.25) == [2.0, 2.5]
    assert capture.split_total_line(2.75) == [2.5, 3.0]
    assert capture.split_total_line(2.3) == []


def test_quarter_line_settlement_preserves_half_push_and_half_loss():
    dist = capture.settlement_from_total_dist({2: 1.0}, "OVER", 2.25)
    assert dist == {
        "win_fraction": 0.0,
        "push_fraction": 0.5,
        "loss_fraction": 0.5,
    }

    under = capture.settlement_from_total_dist({2: 1.0}, "UNDER", 2.25)
    assert under == {
        "win_fraction": 0.5,
        "push_fraction": 0.5,
        "loss_fraction": 0.0,
    }

    over_275 = capture.settlement_from_total_dist({3: 1.0}, "OVER", 2.75)
    assert over_275 == {
        "win_fraction": 0.5,
        "push_fraction": 0.5,
        "loss_fraction": 0.0,
    }


def test_legacy_analyzer_grades_quarter_lines_with_asian_settlement():
    assert grade_total(2, "OVER", 2.25) == "HALF_LOSS"
    assert grade_total(2, "UNDER", 2.25) == "HALF_WIN"
    assert grade_total(3, "OVER", 2.75) == "HALF_WIN"
    assert grade_total(3, "UNDER", 2.75) == "HALF_LOSS"
    assert settlement_return_units("HALF_LOSS", 1.95) == -0.5
    assert settlement_return_units("HALF_WIN", 1.95) == 0.475


def test_capture_event_keeps_quarter_lines_research_only():
    event = {
        "event_type": "SOCCER_REFRESH",
        "stage": "T-20",
        "fixture": {"fixture_id": 12345},
        "raw_projection": {
            "raw_total_goals": 2.65,
            "raw_home_goal_rate": 1.55,
            "raw_away_goal_rate": 1.10,
        },
        "market": {
            "source": "API_FOOTBALL_ODDS_V3",
            "resolution_status": "PRICE_API_RESOLVED",
            "markets": [
                {
                    "market": "Goals Over/Under",
                    "market_id": 5,
                    "bookmaker": "Book A",
                    "bookmaker_id": 1,
                    "provider_update": "2026-09-27T10:00:00+00:00",
                    "values": [
                        {"selection": "Over", "line": 2.25, "decimal_price": 1.95, "fair_probability": 0.49},
                        {"selection": "Under", "line": 2.25, "decimal_price": 1.90, "fair_probability": 0.51},
                        {"selection": "Over", "line": 2.75, "decimal_price": 2.15, "fair_probability": 0.44},
                        {"selection": "Under", "line": 2.75, "decimal_price": 1.72, "fair_probability": 0.56},
                    ],
                },
                {
                    "market": "Goals Over/Under",
                    "market_id": 5,
                    "bookmaker": "Book B",
                    "bookmaker_id": 2,
                    "provider_update": "2026-09-27T10:00:05+00:00",
                    "values": [
                        {"selection": "Over", "line": 2.25, "decimal_price": 1.97},
                        {"selection": "Under", "line": 2.25, "decimal_price": 1.88},
                    ],
                },
            ],
        },
    }

    result = capture.capture_event(event)

    assert result["status"] == "LIVE_SETTLEMENT_AWARE_CAPTURE"
    assert result["observed_lines"] == [2.25, 2.75]
    assert result["observed_row_count"] == 4
    assert result["fresh_provider"] is True
    assert result["provider_requests_added"] == 0
    assert result["research_only"] is True
    assert result["actionable"] is False
    assert result["decision_weight"] == 0.0
    assert result["production_promotion_allowed"] is False

    over_225 = next(
        row for row in result["observed_rows"]
        if row["selection"] == "OVER" and row["line"] == 2.25
    )
    assert over_225["split_components"] == [2.0, 2.5]
    assert over_225["bookmaker_count"] == 2
    assert over_225["market_no_vig_status"] == "NOT_USED_FOR_ASIAN_SETTLEMENT_RESEARCH"
    assert over_225["classification"] == "RESEARCH_ONLY"
    assert over_225["actionable"] is False


def test_attach_adds_capture_without_provider_calls_or_decision_changes():
    payload = {
        "events": [
            {
                "event_type": "SOCCER_REFRESH",
                "stage": "T-10",
                "fixture": {"fixture_id": 77},
                "raw_projection": {"raw_total_goals": 2.4},
                "market": {
                    "source": "POSTGRES_MARKET_SNAPSHOT_CACHE",
                    "resolution_status": "PRICE_CACHE_RESOLVED",
                    "markets": [
                        {
                            "market": "Goals Over/Under",
                            "bookmaker": "Cached Book",
                            "values": [
                                {"selection": "Over", "line": 2.25, "decimal_price": 1.91},
                                {"selection": "Under", "line": 2.25, "decimal_price": 1.93},
                            ],
                        }
                    ],
                },
                "ft_goals_intelligence": {},
            }
        ]
    }

    summary = capture.attach(payload)

    assert summary["provider_requests_added"] == 0
    assert summary["rows_captured"] == 2
    assert summary["fresh_provider_rows"] == 0
    assert summary["cache_replay_rows"] == 2
    assert summary["canonical_bet_logic_changed"] is False
    assert summary["model_weights_changed"] is False
    event = payload["events"][0]
    assert event["ft_totals_settlement_capture"]["cache_replay"] is True
    assert event["ft_goals_intelligence"]["observed_settlement_aware_lines"] == [2.25]
