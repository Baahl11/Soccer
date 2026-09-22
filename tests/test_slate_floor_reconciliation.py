from __future__ import annotations

from datetime import datetime, timezone as dt_timezone

from mcp_gateway import automation_v89 as v89


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
