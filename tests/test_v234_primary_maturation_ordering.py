from mcp_gateway import price_resolver_v4 as resolver


def _event(fid: int, *families: str) -> dict:
    return {
        "fixture": {"fixture_id": fid},
        "primary_clv_maturation": {
            "signals": [{"market_family": family} for family in families]
        },
    }


def _fid(event: dict) -> int:
    return int(event["fixture"]["fixture_id"])


def test_v234_keeps_mixed_fixture_first_and_reorders_only():
    events = [
        _event(1, "1X2"),
        _event(2, "BTTS"),
        _event(3, "FT_TOTALS"),
        _event(4, "1X2"),
        _event(5, "1H"),
        _event(6, "2H"),
        _event(7, "1X2", "1H"),
    ]

    ordered, telemetry = resolver._fair_order_primary_maturation_events(events)

    assert _fid(ordered[0]) == 7
    assert sorted(_fid(event) for event in ordered) == sorted(_fid(event) for event in events)
    assert telemetry["input_fixture_events"] == telemetry["output_fixture_events"] == len(events)
    assert telemetry["provider_requests_added"] == 0
    assert telemetry["provider_budget_changed"] is False
    assert telemetry["strict_close_semantics_changed"] is False
    assert telemetry["eligibility_changed"] is False


def test_v234_gives_derivatives_two_opportunities_inside_first_eight_when_available():
    events = [
        *[_event(fid, "1X2") for fid in range(1, 7)],
        _event(101, "1H"),
        _event(102, "2H"),
    ]

    ordered, telemetry = resolver._fair_order_primary_maturation_events(events)
    first_eight = ordered[:8]
    derivative_slots = [
        event for event in first_eight
        if resolver._primary_maturation_event_families(event)
        & resolver.PRIMARY_CLV_MATURATION_DERIVATIVE_FAMILIES
    ]

    assert len(derivative_slots) == 2
    assert [_fid(event) for event in ordered] == [1, 2, 3, 101, 4, 5, 6, 102]
    assert telemetry["policy"] == "MIXED_FIRST_THEN_3_PRIMARY_TO_1_DERIVATIVE_WITHIN_EXISTING_CAP"


def test_v234_does_not_displace_primary_when_no_derivative_backlog_exists():
    events = [_event(fid, "1X2") for fid in range(1, 10)]

    ordered, telemetry = resolver._fair_order_primary_maturation_events(events)

    assert [_fid(event) for event in ordered] == list(range(1, 10))
    assert telemetry["derivative_only_fixture_events"] == 0
