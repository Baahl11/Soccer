from __future__ import annotations

from datetime import datetime, timezone as dt_timezone

from mcp_gateway import automation_v89 as v89
from mcp_gateway import automation_v90 as v90


def _fixture_row(fixture_id: int) -> dict:
    return {
        "fixture": {
            "id": fixture_id,
            "date": "2026-09-22T12:00:00+00:00",
            "status": {"short": "NS"},
        },
        "league": {"id": 1, "season": 2026},
        "teams": {"home": {"id": 10}, "away": {"id": 20}},
    }


def test_raw_date_fixture_slate_excludes_team_recent_calls() -> None:
    assert v89._is_raw_date_fixture_slate(
        "fixtures", {"date": "2026-09-22", "timezone": "America/Mexico_City"}
    )
    assert not v89._is_raw_date_fixture_slate(
        "fixtures", {"team": 10, "last": 8, "timezone": "America/Mexico_City"}
    )
    assert not v89._is_raw_date_fixture_slate("fixtures", {"id": 123})
    assert not v89._is_raw_date_fixture_slate("odds", {"fixture": 123, "page": 1})


def test_merge_fixture_payloads_dedupes_by_fixture_id_and_preserves_order() -> None:
    primary = {"response": [_fixture_row(1), _fixture_row(2)], "quota": {"daily_remaining": "6500"}}
    reconciliation = {
        "response": [_fixture_row(2), _fixture_row(3), _fixture_row(4)],
        "quota": {"daily_remaining": "6499"},
    }

    merged = v89._merge_fixture_payloads(primary, reconciliation)

    assert [v89._raw_fixture_id(row) for row in merged["response"]] == [1, 2, 3, 4]
    assert merged["results"] == 4
    assert merged["quota"] == {"daily_remaining": "6499"}
    assert merged["slate_reconciliation"]["dedupe_key"] == "fixture.id"


def test_should_reconcile_tiny_today_slate_when_budget_allows(monkeypatch) -> None:
    now_utc = datetime(2026, 9, 22, 13, 0, tzinfo=dt_timezone.utc)
    today = now_utc.astimezone(v89.base.TIMEZONE).date().isoformat()

    monkeypatch.setattr(v89.v2, "_API_CALLS_THIS_TICK", 1)
    monkeypatch.setattr(v89.v2, "MAX_API_CALLS_PER_TICK", 35)
    monkeypatch.setattr(v89.v2, "_LAST_DAILY_REMAINING", 6388)

    should, reason = v89._should_reconcile(
        payload={"response": [_fixture_row(1), _fixture_row(2)]},
        params={"date": today, "timezone": "America/Mexico_City"},
        now_utc=now_utc,
        metrics=v89._new_metrics(),
    )

    assert should is True
    assert reason == "PRIMARY_SLATE_BELOW_FLOOR"


def test_should_not_reconcile_when_today_slate_meets_floor(monkeypatch) -> None:
    now_utc = datetime(2026, 9, 22, 13, 0, tzinfo=dt_timezone.utc)
    today = now_utc.astimezone(v89.base.TIMEZONE).date().isoformat()

    monkeypatch.setattr(v89.v2, "_API_CALLS_THIS_TICK", 1)
    monkeypatch.setattr(v89.v2, "MAX_API_CALLS_PER_TICK", 35)
    monkeypatch.setattr(v89.v2, "_LAST_DAILY_REMAINING", 6388)

    should, reason = v89._should_reconcile(
        payload={"response": [_fixture_row(i) for i in range(1, v89.SLATE_FLOOR_MIN_FIXTURES + 1)]},
        params={"date": today, "timezone": "America/Mexico_City"},
        now_utc=now_utc,
        metrics=v89._new_metrics(),
    )

    assert should is False
    assert reason == "PRIMARY_SLATE_MEETS_FLOOR"


def test_should_not_reconcile_when_daily_budget_is_reduced(monkeypatch) -> None:
    now_utc = datetime(2026, 9, 22, 13, 0, tzinfo=dt_timezone.utc)
    today = now_utc.astimezone(v89.base.TIMEZONE).date().isoformat()

    monkeypatch.setattr(v89.v2, "_API_CALLS_THIS_TICK", 1)
    monkeypatch.setattr(v89.v2, "MAX_API_CALLS_PER_TICK", 35)
    monkeypatch.setattr(v89.v2, "_LAST_DAILY_REMAINING", 3999)

    should, reason = v89._should_reconcile(
        payload={"response": [_fixture_row(1)]},
        params={"date": today, "timezone": "America/Mexico_City"},
        now_utc=now_utc,
        metrics=v89._new_metrics(),
    )

    assert should is False
    assert reason == "DAILY_BUDGET_BLOCKED"



def test_v90_high_quota_prefetches_two_future_days():
    local_now = datetime(2026, 9, 25, 18, 0, tzinfo=v90.base.TIMEZONE)
    assert v90._future_prefetch_offsets(local_now, 7200) == [1, 2]


def test_v90_mid_quota_prefetches_only_tomorrow():
    local_now = datetime(2026, 9, 25, 18, 0, tzinfo=v90.base.TIMEZONE)
    assert v90._future_prefetch_offsets(local_now, 5000) == [1]


def test_v90_low_quota_does_not_prefetch_early():
    local_now = datetime(2026, 9, 25, 18, 0, tzinfo=v90.base.TIMEZONE)
    assert v90._future_prefetch_offsets(local_now, 3000) == []


def test_v90_late_day_keeps_tomorrow_even_when_quota_is_unknown():
    local_now = datetime(2026, 9, 25, 22, 30, tzinfo=v90.base.TIMEZONE)
    assert v90._future_prefetch_offsets(local_now, None) == [1]


def test_v90_fixture_merge_has_no_league_allowlist():
    fixtures = []
    seen = set()
    payload = {
        "response": [
            _fixture_row(100 + i) | {
                "league": {"id": 1000 + i, "season": 2026, "name": f"League {i}", "country": f"Country {i}"}
            }
            for i in range(40)
        ]
    }

    added = v90._append_fixture_payload(fixtures, seen, payload)

    assert added == 40
    assert len(fixtures) == 40
    assert len({row["league_id"] for row in fixtures}) == 40


def test_v90_future_slate_cache_avoids_repeat_provider_call(monkeypatch):
    cache = {}
    provider_calls = {"count": 0}

    def fake_cache_get(namespace, key, ttl, now):
        return cache.get((namespace, key))

    def fake_cache_set(namespace, key, value, now):
        cache[(namespace, key)] = value

    async def fake_api_get(endpoint, params):
        provider_calls["count"] += 1
        return {
            "response": [_fixture_row(999)],
            "quota": {"daily_remaining": "7200"},
        }

    monkeypatch.setattr(v90.base, "_cache_get", fake_cache_get)
    monkeypatch.setattr(v90.base, "_cache_set", fake_cache_set)
    monkeypatch.setattr(v90.base, "_api_get", fake_api_get)

    now = datetime(2026, 9, 25, 18, 0, tzinfo=dt_timezone.utc)
    future_date = now.astimezone(v90.base.TIMEZONE).date()

    rows1, _, hit1 = __import__("asyncio").run(v90._future_date_slate(future_date, now))
    rows2, _, hit2 = __import__("asyncio").run(v90._future_date_slate(future_date, now))

    assert len(rows1) == 1
    assert len(rows2) == 1
    assert hit1 is False
    assert hit2 is True
    assert provider_calls["count"] == 1
