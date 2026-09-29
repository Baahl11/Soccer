from mcp_gateway import automation_v2 as v2
from mcp_gateway import automation_v4 as v4
from mcp_gateway import automation_v90 as v90
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


def test_v208_hard_provider_cap_cannot_be_raised_mid_tick(monkeypatch) -> None:
    monkeypatch.setattr(v4, "_MONOTONIC_TICK_CAP", None)
    monkeypatch.setattr(v2, "_API_CALLS_THIS_TICK", 0)
    monkeypatch.setattr(v2, "MAX_API_CALLS_PER_TICK", 50)

    assert v4._effective_tick_cap() == 50

    # Quota becomes known and v90 tightens the upstream gate.
    monkeypatch.setattr(v2, "_API_CALLS_THIS_TICK", 1)
    monkeypatch.setattr(v2, "MAX_API_CALLS_PER_TICK", 35)
    assert v4._effective_tick_cap() == 35

    # A later orchestration layer may restore/report the global ceiling, but
    # the real upstream provider gate must keep the already-reserved low-water cap.
    monkeypatch.setattr(v2, "MAX_API_CALLS_PER_TICK", 55)
    assert v4._effective_tick_cap() == 35

    # A new tick resets the low-water mark when the provider-call counter resets.
    monkeypatch.setattr(v2, "_API_CALLS_THIS_TICK", 0)
    monkeypatch.setattr(v2, "MAX_API_CALLS_PER_TICK", 45)
    assert v4._effective_tick_cap() == 45


def test_v208_v90_reserve_survives_later_global_cap_restore(monkeypatch) -> None:
    monkeypatch.setattr(v90, "_REQUEST_CAP_RESERVE_CALLS", 20)
    monkeypatch.setattr(v90, "_REQUEST_CAP_MIN_UPSTREAM_CALLS", 8)
    monkeypatch.setattr(v4, "_MONOTONIC_TICK_CAP", None)
    monkeypatch.setattr(v2, "_API_CALLS_THIS_TICK", 0)
    monkeypatch.setattr(v2, "MAX_API_CALLS_PER_TICK", 55)

    upstream_cap, reserve, reason = v90._enforce_live_request_cap(4541)
    assert upstream_cap == 35
    assert reserve == 20
    assert reason == "GT_4500"
    assert v2.MAX_API_CALLS_PER_TICK == 35

    # Mimic the later global-cap annotation/restore that previously reopened
    # upstream capacity and starved the post-upstream price resolver.
    monkeypatch.setattr(v2, "_API_CALLS_THIS_TICK", 1)
    assert v4._effective_tick_cap() == 35
    monkeypatch.setattr(v2, "MAX_API_CALLS_PER_TICK", 55)
    assert v4._effective_tick_cap() == 35
