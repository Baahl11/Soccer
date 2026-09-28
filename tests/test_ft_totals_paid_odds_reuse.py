import asyncio

from mcp_gateway import ft_totals_paid_odds_reuse as v


def _ft_row(bookmaker_id: int = 1, provider_update: str = "2026-09-27T12:00:00Z"):
    return {
        "bookmaker_id": bookmaker_id,
        "bookmaker": "Book",
        "market_id": 5,
        "market": "Goals Over/Under",
        "provider_update": provider_update,
        "values": [
            {"selection": "Over", "line": 2.25, "decimal_price": 1.91},
            {"selection": "Under", "line": 2.25, "decimal_price": 1.95},
        ],
    }


def test_collect_only_fresh_price_api_resolved_ft_totals():
    captured = {}
    v.collect_fixture_markets(captured, 100, [_ft_row()], "PRICE_CACHE_HIT")
    assert captured == {}

    captured = {}
    v.collect_fixture_markets(captured, 100, [{"market": "Home Team Goals Over/Under"}], "PRICE_API_RESOLVED")
    assert captured == {}

    from collections import defaultdict
    captured = defaultdict(list)
    v.collect_fixture_markets(captured, 100, [_ft_row()], "PRICE_API_RESOLVED")
    assert len(captured[100]) == 1


def test_attach_dedupes_existing_paid_ft_totals_and_adds_to_synthetic_event():
    existing = _ft_row()
    new_row = _ft_row(bookmaker_id=2)
    payload = {
        "events": [
            {
                "event_type": "SOCCER_REFRESH",
                "fixture": {"fixture_id": 100},
                "market": {
                    "source": "API_FOOTBALL_ODDS_V3",
                    "markets": [existing],
                },
            },
            {
                "event_type": "TEAM_TOTALS_RESEARCH_SPILLOVER",
                "fixture": {"fixture_id": 200},
                "market": {
                    "source": "API_FOOTBALL_ODDS_V3",
                    "markets": [{"market": "Home Team Goals Over/Under", "market_id": 16}],
                },
            },
        ]
    }
    report = v.attach(payload, {100: [existing], 200: [new_row]})

    assert report["provider_requests_added"] == 0
    assert report["already_present_market_rows"] == 1
    assert report["attached_market_rows"] == 1
    assert report["attached_fixtures"] == 1
    synthetic = payload["events"][1]
    assert any(row.get("market") == "Goals Over/Under" for row in synthetic["market"]["markets"])
    assert synthetic["market"]["ft_totals_paid_odds_reuse"]["rows_added"] == 1


def test_install_observer_preserves_return_and_captures_without_extra_calls():
    class Resolver:
        pass

    resolver = Resolver()
    calls = {"count": 0}

    async def fake_fetch(client, fixture_id, *, api_key, remaining_calls):
        calls["count"] += 1
        return [_ft_row()], 1, "PRICE_API_RESOLVED", 7000

    resolver._fetch_fixture_odds = fake_fetch
    original, captured = v.install_fetch_observer(resolver)
    try:
        result = asyncio.run(
            resolver._fetch_fixture_odds(None, 321, api_key="x", remaining_calls=3)
        )
    finally:
        v.restore_fetch_observer(resolver, original)

    assert calls["count"] == 1
    assert result[1] == 1
    assert len(captured[321]) == 1
    assert resolver._fetch_fixture_odds is original


def test_settlement_attach_consumes_capture_once_per_tick():
    from mcp_gateway import ft_totals_settlement_capture as settlement

    v._CAPTURED.clear()
    v._CAPTURED[500].append(_ft_row())
    payload = {
        "events": [
            {
                "event_type": "TEAM_TOTALS_RESEARCH_SPILLOVER",
                "fixture": {"fixture_id": 500},
                "market": {
                    "source": "API_FOOTBALL_ODDS_V3",
                    "markets": [{"market": "Home Team Goals Over/Under", "market_id": 16}],
                },
            }
        ]
    }

    first = settlement.attach(payload)
    second = settlement.attach(payload)

    assert first["paid_odds_reuse"]["attached_market_rows"] == 1
    assert first["paid_odds_reuse"]["provider_requests_added"] == 0
    assert second["paid_odds_reuse"]["captured_fixtures"] == 0
    assert second["paid_odds_reuse"]["attached_market_rows"] == 0
    assert v.drain() == {}
