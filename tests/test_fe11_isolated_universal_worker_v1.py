"""Offline independent whole-slate worker; no real provider, DB or wagers."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mcp_gateway import fe11_universal_worker_v1 as w


def fixture(fid, day, league=39):
    return {
        "fixture": {
            "id": fid, "date": day.isoformat(),
            "status": {"short": "NS", "long": "Not Started"},
            "venue": {"name": "Ground", "city": "City"},
        },
        "league": {
            "id": league, "name": "Premier League", "country": "England",
            "season": day.year, "round": "Round 1",
        },
        "teams": {
            "home": {"id": fid + 10, "name": "Home"},
            "away": {"id": fid + 20, "name": "Away"},
        },
        "goals": {"home": None, "away": None},
        "score": {"halftime": None, "fulltime": None},
    }


def test_dates_cover_prior_day_in_morning_and_next_day_after_noon():
    from zoneinfo import ZoneInfo
    d = datetime(2026, 11, 6, 6, 40, tzinfo=ZoneInfo("America/Mexico_City"))
    assert w.requested_dates(d) == ["2026-11-06", "2026-11-05"]
    later = d.replace(hour=13)
    assert w.requested_dates(later) == ["2026-11-06", "2026-11-07"]


def test_exact_approved_league_scope_and_source_day_are_verified():
    from zoneinfo import ZoneInfo
    d = datetime(2026, 11, 7, 18, tzinfo=ZoneInfo("America/Mexico_City"))
    valid = [fixture(n, d, league=39 if n % 2 else 140) for n in range(1, 8)]
    response = {"response": valid + [fixture(8, d, league=999)],
                "results": 8, "paging": {"current": 1, "total": 1}}
    result, count = w._approved_from_date(response, "2026-11-07")
    assert count == 8
    assert len(result) == 7
    assert all(x["league_id"] in w.universal.APPROVED_LEAGUES for x in result)


def test_provider_pagination_or_partial_fixture_count_never_claims_full_slate():
    response = {"response": [], "results": 1, "paging": {"total": 1}}
    with pytest.raises(ValueError, match="INCOMPLETE"):
        w._approved_from_date(response, "2026-11-06")
    response["results"] = 0
    response["paging"]["total"] = 2
    with pytest.raises(ValueError, match="PAGINATION_UNRESOLVED"):
        w._approved_from_date(response, "2026-11-06")


def test_wrong_local_day_is_rejected_before_db():
    from zoneinfo import ZoneInfo
    kickoff = datetime(2026, 11, 6, 9, tzinfo=ZoneInfo("America/Mexico_City"))
    rows = [fixture(12, kickoff)]
    response = {"response": rows, "results": 1, "paging": {"total": 1}}
    with pytest.raises(ValueError, match="DATE_FIXTURE_MISMATCH"):
        w._approved_from_date(response, "2026-11-07")


def test_missing_quota_is_not_promoted_to_an_invented_remaining_balance():
    assert w._remaining({"quota": {"daily_remaining": None}}) is None
    assert w._remaining({"quota": {"daily_remaining": "n/a"}}) is None
    assert w._remaining({"quota": {"daily_remaining": "6000"}}) == 6000


class FakeCursor:
    def __init__(self):
        self.queries = []
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def execute(self, sql, args):
        self.queries.append((sql, args))


class FakeConn:
    def __init__(self):
        self.cursor_obj = FakeCursor()
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def cursor(self):
        return self.cursor_obj


class FakeClient:
    def __init__(self, **kwargs):
        pass
    async def __aenter__(self):
        return self
    async def __aexit__(self, *args):
        return False


def test_worker_can_freeze_more_than_three_without_importing_heavy_tick(monkeypatch):
    now = datetime.now(timezone.utc)
    from zoneinfo import ZoneInfo
    local = now.astimezone(ZoneInfo("America/Mexico_City"))
    today = local.date()
    today_dt = datetime(today.year, today.month, today.day, 23, 30,
                        tzinfo=ZoneInfo("America/Mexico_City"))
    # Ensure future rows even if this test executes after 23:30 local.
    tomorrow = today_dt + timedelta(days=1)
    rows = [fixture(i + 100, tomorrow, league=39) for i in range(9)]
    provider_payload = {
        "response": rows, "results": len(rows), "paging": {"total": 1},
        "quota": {"daily_remaining": None},
    }
    # The test's current date fixture response should never claim the tomorrow
    # rows are today's. Return an empty primary today slate, then reserve-skip
    # optional tomorrow due to unknown quota. No database writes are invented.
    provider_payload["response"] = []
    provider_payload["results"] = 0
    spy = []
    monkeypatch.setenv("API_FOOTBALL_KEY", "OFFLINE_DUMMY_NOT_REAL")
    monkeypatch.setenv("DATABASE_URL", "postgresql://offline.invalid/offline")
    monkeypatch.setattr(w.persistence, "persistence_configured", lambda: True)
    monkeypatch.setattr(w.persistence, "ensure_schema", lambda: None)
    monkeypatch.setattr(w.persistence, "_connect", lambda: FakeConn())
    monkeypatch.setattr(w.writer, "persist", lambda cur, tick, upsert: spy.append(tick))
    monkeypatch.setattr(w.writer, "inventory", lambda cur: {
        "status":"VERIFIED_READ", "eligible_pregame_baselines":0
    })
    monkeypatch.setattr(w.httpx, "AsyncClient", FakeClient)
    async def get(endpoint, params):
        assert endpoint == "fixtures"
        assert params["date"] == today.isoformat()
        return provider_payload
    monkeypatch.setattr(w.provider, "_api_get", get)
    result = asyncio.run(w.run_once(now_utc=now))
    assert result["status"] == "VERIFIED_ALL_ELIGIBLE_LIFECYCLE"
    assert result["provider_requests_added"] == 1
    assert len(result["source_date_audit"]) == 2
    assert result["source_date_audit"][1]["status"] == "DEFERRED_PROVIDER_RESERVE"
    assert len(spy) == 1
    assert spy[0]["fe11_universal_lifecycle"]["bet_eligible"] is False


def test_oidc_route_and_cgroup_isolation_are_structurally_required():
    server = (Path(__file__).resolve().parents[1]/"mcp_gateway"/"server.py").read_text()
    assert 'FE11_UNIVERSAL_WORKFLOW = ".github/workflows/fe11-universal-slate.yml"' in server
    assert '"mcp_gateway.fe11_universal_worker_v1"' in server
    assert '_base_server._github_oidc_claims(' in server
    assert '_base_server._acquire_db_heavy_gate("fe11_official_universal_capture")' in server
    assert "await _run_instrumented_tick(" in server
    assert '"/internal/fe11-universal-slate-v1"' in server
    scanner = (Path(__file__).resolve().parents[1]/"mcp_gateway"/"automation_v7.py").read_text()
    assert "fe11_universal.build_capture(" not in scanner
