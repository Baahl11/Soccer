from __future__ import annotations

import asyncio
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



def test_v89_active_path_prefetches_48h_and_merges_all_leagues(monkeypatch):
    cache = {}
    calls = []

    def fake_cache_get(namespace, key, ttl, now):
        return cache.get((namespace, key))

    def fake_cache_set(namespace, key, value, now):
        cache[(namespace, key)] = value

    async def fake_provider(endpoint, params):
        calls.append(dict(params))
        v89.v2._LAST_DAILY_REMAINING = 7200
        call_no = len(calls)
        start = {1: 1, 2: 101, 3: 201}[call_no]
        count = {1: 2, 2: 3, 3: 4}[call_no]
        rows = []
        for i in range(count):
            row = _fixture_row(start + i)
            row["league"] = {
                "id": 1000 + start + i,
                "season": 2026,
                "name": f"League {start + i}",
                "country": f"Country {start + i}",
            }
            rows.append(row)
        return {"response": rows, "quota": {"daily_remaining": "7200"}}

    async def fake_downstream():
        today = datetime.now(dt_timezone.utc).astimezone(v89.base.TIMEZONE).date().isoformat()
        payload = await v89.v6._ORIGINAL_PACED_API_GET(
            "fixtures", {"date": today, "timezone": v89.base.TIMEZONE_NAME}
        )
        return {
            "fixture_scan_count": len(payload.get("response") or []),
            "last_daily_remaining": 7200,
        }

    monkeypatch.setattr(v89.base, "_cache_get", fake_cache_get)
    monkeypatch.setattr(v89.base, "_cache_set", fake_cache_set)
    monkeypatch.setattr(v89.v6, "_ORIGINAL_PACED_API_GET", fake_provider)
    monkeypatch.setattr(v89.v88, "run_tick", fake_downstream)
    monkeypatch.setattr(v89.v2, "_API_CALLS_THIS_TICK", 1)
    monkeypatch.setattr(v89.v2, "MAX_API_CALLS_PER_TICK", 70)
    monkeypatch.setattr(v89.v2, "_LAST_DAILY_REMAINING", 7200)

    result = asyncio.run(v89.run_tick())
    metrics = result["slate_floor_reconciliation"]

    assert result["fixture_scan_count"] == 9
    assert len(calls) == 3
    assert len(metrics["scan_dates"]) == 3
    assert metrics["primary_slate_count"] == 2
    assert metrics["future_prefetch_fixture_count"] == 7
    assert metrics["future_prefetch_provider_requests_added"] == 2
    assert metrics["merged_slate_count"] == 9
    assert metrics["unique_leagues_scanned"] == 9
    assert metrics["unique_countries_scanned"] == 9
    assert metrics["league_allowlist_applied"] is False
    assert metrics["policy"] == "RAW_API_FOOTBALL_ROLLING_DATE_SLATE_WITH_CACHED_FUTURE_PREFETCH"


def test_v89_future_prefetch_cache_reuses_raw_slate_without_replaying_quota(monkeypatch):
    cache = {}
    provider_calls = {"count": 0}

    def fake_cache_get(namespace, key, ttl, now):
        return cache.get((namespace, key))

    def fake_cache_set(namespace, key, value, now):
        cache[(namespace, key)] = value

    async def fake_provider(endpoint, params):
        provider_calls["count"] += 1
        return {
            "response": [_fixture_row(777)],
            "quota": {"daily_remaining": "7199"},
        }

    monkeypatch.setattr(v89.base, "_cache_get", fake_cache_get)
    monkeypatch.setattr(v89.base, "_cache_set", fake_cache_set)

    now = datetime.now(dt_timezone.utc)
    tomorrow = (now.astimezone(v89.base.TIMEZONE).date()).isoformat()
    params = {"date": tomorrow, "timezone": v89.base.TIMEZONE_NAME}

    first, hit1 = asyncio.run(
        v89._future_date_payload(fake_provider, "fixtures", params, tomorrow, now)
    )
    second, hit2 = asyncio.run(
        v89._future_date_payload(fake_provider, "fixtures", params, tomorrow, now)
    )

    assert hit1 is False
    assert hit2 is True
    assert provider_calls["count"] == 1
    assert first["quota"]["daily_remaining"] == "7199"
    assert "quota" not in second



def test_v89_active_hook_expands_today_into_cached_48h_horizon(monkeypatch):
    cache = {}
    calls = []
    now_utc = datetime(2026, 9, 25, 21, 0, tzinfo=dt_timezone.utc)
    local_now = now_utc.astimezone(v89.base.TIMEZONE)
    today = local_now.date()
    tomorrow = (today + v89.timedelta(days=1)).isoformat()
    day_after = (today + v89.timedelta(days=2)).isoformat()

    def fake_cache_get(namespace, key, ttl, now):
        return cache.get((namespace, key))

    def fake_cache_set(namespace, key, value, now):
        cache[(namespace, key)] = value

    async def fake_api_get(endpoint, params):
        calls.append((endpoint, dict(params)))
        date = str(params.get("date"))
        if date == today.isoformat():
            return {
                "response": [_fixture_row(1), _fixture_row(2)],
                "quota": {"daily_remaining": "7200"},
            }
        if date == tomorrow:
            return {
                "response": [_fixture_row(2), _fixture_row(3)],
                "quota": {"daily_remaining": "7199"},
            }
        if date == day_after:
            return {
                "response": [_fixture_row(4)],
                "quota": {"daily_remaining": "7198"},
            }
        raise AssertionError(date)

    async def fake_downstream_tick():
        payload = await v89.v6._ORIGINAL_PACED_API_GET(
            "fixtures",
            {"date": today.isoformat(), "timezone": "America/Mexico_City"},
        )
        return {
            "fixture_scan_count": v89._payload_fixture_count(payload),
            "events": [],
        }

    monkeypatch.setattr(v89.base, "_cache_get", fake_cache_get)
    monkeypatch.setattr(v89.base, "_cache_set", fake_cache_set)
    monkeypatch.setattr(v89.v6, "_ORIGINAL_PACED_API_GET", fake_api_get)
    monkeypatch.setattr(v89.v88, "run_tick", fake_downstream_tick)
    monkeypatch.setattr(v89.v2, "_LAST_DAILY_REMAINING", 7200)
    monkeypatch.setattr(v89.v2, "_API_CALLS_THIS_TICK", 1)
    monkeypatch.setattr(v89.v2, "MAX_API_CALLS_PER_TICK", 70)

    class _FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            value = now_utc
            return value if tz is None else value.astimezone(tz)

    monkeypatch.setattr(v89, "datetime", _FixedDateTime)

    result = asyncio.run(v89.run_tick())
    metrics = result["slate_floor_reconciliation"]

    assert result["fixture_scan_count"] == 4
    assert metrics["scan_dates"] == [today.isoformat(), tomorrow, day_after]
    assert metrics["scan_date_counts"] == {
        today.isoformat(): 2,
        tomorrow: 2,
        day_after: 1,
    }
    assert metrics["merged_slate_count"] == 4
    assert metrics["future_prefetch_cache_misses"] == 2
    assert metrics["future_prefetch_provider_requests_added"] == 2
    assert metrics["league_allowlist_applied"] is False
    assert [params["date"] for _, params in calls] == [
        today.isoformat(),
        tomorrow,
        day_after,
    ]


def test_v89_future_prefetch_uses_cache_without_replaying_quota(monkeypatch):
    cache = {}
    calls = []
    now_utc = datetime(2026, 9, 25, 21, 0, tzinfo=dt_timezone.utc)
    future_date = "2026-09-26"

    def fake_cache_get(namespace, key, ttl, now):
        return cache.get((namespace, key))

    def fake_cache_set(namespace, key, value, now):
        cache[(namespace, key)] = value

    async def fake_api_get(endpoint, params):
        calls.append((endpoint, dict(params)))
        return {
            "response": [_fixture_row(101)],
            "quota": {"daily_remaining": "7199"},
        }

    monkeypatch.setattr(v89.base, "_cache_get", fake_cache_get)
    monkeypatch.setattr(v89.base, "_cache_set", fake_cache_set)

    first, first_hit = asyncio.run(
        v89._future_date_payload(
            fake_api_get,
            "fixtures",
            {"date": "2026-09-25", "timezone": "America/Mexico_City"},
            future_date,
            now_utc,
        )
    )
    second, second_hit = asyncio.run(
        v89._future_date_payload(
            fake_api_get,
            "fixtures",
            {"date": "2026-09-25", "timezone": "America/Mexico_City"},
            future_date,
            now_utc,
        )
    )

    assert first_hit is False
    assert second_hit is True
    assert len(calls) == 1
    assert "quota" in first
    assert "quota" not in second
