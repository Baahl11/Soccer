from mcp_gateway import automation_v124 as v124


def _event(fixture_id: int, family: str, kickoff: str) -> dict:
    return {
        "fixture": {
            "fixture_id": fixture_id,
            "kickoff": kickoff,
        },
        "primary_clv_maturation": {
            "signals": [
                {
                    "market_family": family,
                    "signal_generated_at": "2026-09-29T20:00:00+00:00",
                }
            ]
        },
    }


def test_primary_maturation_priority_is_1x2_then_btts_then_ft_totals() -> None:
    source = {
        "candidate_events": [
            _event(3, "FT_TOTALS", "2026-09-29T23:00:00+00:00"),
            _event(2, "FT_BTTS_RESEARCH", "2026-09-29T22:00:00+00:00"),
            _event(1, "FT_1X2_RESEARCH", "2026-09-29T21:00:00+00:00"),
        ],
        "candidate_count": 3,
        "candidate_family_counts": {"1X2": 1, "BTTS": 1, "FT_TOTALS": 1},
        "source": "POSTGRES_PRIMARY_CLV_MATURATION_BACKLOG_V2",
    }

    result, telemetry = v124._prioritize_primary_backlog_result(source)

    assert [event["fixture"]["fixture_id"] for event in result["candidate_events"]] == [1, 2, 3]
    assert result["candidate_count"] == 3
    assert result["candidate_family_counts"] == source["candidate_family_counts"]
    assert telemetry["reordered"] is True
    assert telemetry["first_family_before"] == "FT_TOTALS"
    assert telemetry["first_family_after"] == "1X2"
    assert telemetry["family_order"] == ["1X2", "BTTS", "FT_TOTALS"]


def test_primary_maturation_priority_does_not_create_candidates() -> None:
    result, telemetry = v124._prioritize_primary_backlog_result({
        "candidate_events": [],
        "candidate_count": 0,
        "candidate_family_counts": {},
    })

    assert result["candidate_events"] == []
    assert result["candidate_count"] == 0
    assert telemetry["candidate_count"] == 0
    assert telemetry["reordered"] is False


def test_unknown_family_remains_after_known_primary_families() -> None:
    source = {
        "candidate_events": [
            _event(9, "UNKNOWN", "2026-09-29T20:00:00+00:00"),
            _event(7, "BTTS", "2026-09-29T22:00:00+00:00"),
            _event(8, "1X2", "2026-09-29T23:00:00+00:00"),
        ]
    }

    result, _ = v124._prioritize_primary_backlog_result(source)

    assert [event["fixture"]["fixture_id"] for event in result["candidate_events"]] == [8, 7, 9]
