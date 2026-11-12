"""FE11 canonical tomorrow preseed: official identity-only selection, fail closed."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from mcp_gateway.fe11_registry_next_day_v1 import (
    VERSION, PERSISTENCE_FIELDS, select_verified_next_day, discovery_event,
)

NOW = datetime(2026, 11, 6, 19, 10, tzinfo=timezone.utc)  # 13:10 CDMX
TOMORROW_KICKOFF = "2026-11-07T16:00:00+00:00"  # 10:00 CDMX


def official(fixture_id=1270001, league_id=39, status="NS", kickoff=TOMORROW_KICKOFF):
    return {
        "fixture_id": fixture_id, "kickoff": kickoff, "status": status,
        "status_long": "Not Started", "league_id": league_id,
        "league": "Premier League", "country": "England",
        "season": 2026, "round": "Regular Season - 12",
        "home_team_id": 44, "home_team": "Example Home",
        "away_team_id": 45, "away_team": "Example Away",
        "venue": "Example Stadium", "city": "Example City",
        "odds": {"price": 1.9}, "market": "THIS_MUST_BE_DROPPED",
        "model_prediction": "THIS_MUST_BE_DROPPED",
    }


def test_can_preseed_only_tomorrow_official_ns_leagues_and_keep_prices_out():
    rows = [
        official(),
        official(fixture_id=1270002, league_id=135, kickoff="2026-11-08T01:00:00Z"),
        official(fixture_id=1270003, league_id=999),
        official(fixture_id=1270004, status="PST"),
        official(fixture_id=1270005, kickoff="2026-11-07T04:00:00Z"),
        official(fixture_id=1270006, kickoff="2026-11-08T10:00:00Z"),
    ]
    out = select_verified_next_day(rows, now_utc=NOW)
    assert out["version"] == VERSION
    assert out["status"] == "VERIFIED_PROVIDER_ROSTER"
    assert out["date"] == "2026-11-07"
    assert out["count"] == 2
    assert [x["fixture_id"] for x in out["fixtures"]] == [1270001, 1270002]
    assert out["market_inputs_used"] is False
    assert out["bet_eligible"] is False
    assert out["source"] == "API_FOOTBALL_FIXTURES_DATE"
    assert set(out["fixtures"][0]) == set(PERSISTENCE_FIELDS)
    assert not any("odds" in x or "market" in x or "model" in x
                   for x in out["fixtures"][0])
    assert rows[0]["odds"] == {"price": 1.9}  # source left untouched


def test_event_uses_existing_daily_discovery_persistence_contract():
    roster = select_verified_next_day([official()], now_utc=NOW)
    evt = discovery_event(roster)
    assert evt["event_type"] == "DAILY_DISCOVERY"
    assert evt["stage"] == "FE11_NEXT_DAY_PRESEED"
    assert evt["date"] == "2026-11-07"
    assert evt["fixtures"] == roster["fixtures"]
    assert evt["source"] == "API_FOOTBALL_FIXTURES_DATE"
    assert evt["market_data_included"] is False
    assert evt["bet_eligible"] is False
    assert evt["calibrated_joint_probability"] is None


def test_tampered_roster_cannot_inject_market_or_forged_identity_into_db_event():
    roster = select_verified_next_day([official()], now_utc=NOW)
    roster["fixtures"][0]["odds"] = 2.1
    assert discovery_event(roster) is None
    roster = select_verified_next_day([official()], now_utc=NOW)
    roster["source"] = "UNTRUSTED_SOURCE"
    assert discovery_event(roster) is None
    roster = select_verified_next_day([official()], now_utc=NOW)
    roster["fixtures"][0]["kickoff"] = "2026-11-08T12:00:00Z"
    assert discovery_event(roster) is None


def test_uses_existing_parameterized_registry_writer_no_db_connect(monkeypatch):
    from mcp_gateway import persistence
    event = discovery_event(select_verified_next_day([official()], now_utc=NOW))
    assert event is not None
    class FakeCursor:
        def __init__(self): self.calls = []
        def execute(self, sql, params): self.calls.append((sql, params))
    cursor = FakeCursor()
    persistence._upsert_fixture(cursor, event["fixtures"][0])
    assert len(cursor.calls) == 1
    sql, params = cursor.calls[0]
    assert "INSERT INTO soccer_fixtures" in sql
    assert "ON CONFLICT (fixture_id) DO UPDATE SET" in sql
    assert params["fixture_id"] == 1270001
    assert params["status"] == "NS"
    assert params["home_team_id"] == 44
    assert params["away_team_id"] == 45
    assert set(params) == set(PERSISTENCE_FIELDS)


def test_conflicting_duplicates_fail_closed_entire_roster():
    a, b = official(), official()
    b["away_team_id"] = 993
    res = select_verified_next_day([a, b], now_utc=NOW)
    assert res["status"] == "NOT_VERIFIED_IDENTITY_CONFLICT"
    assert res["fixtures"] == []
    assert discovery_event(res) is None


def test_identical_duplicates_deduped_not_counted_twice():
    fixture = official()
    out = select_verified_next_day([fixture, dict(fixture)], now_utc=NOW)
    assert out["count"] == 1
    assert out["excluded"]["IDENTICAL_DUPLICATES"] == 1


def test_oversized_roster_returns_no_partial_registration():
    out = select_verified_next_day([official(fixture_id=1270000+i) for i in range(4)],
                                   now_utc=NOW, max_fixtures=3)
    assert out["status"] == "NOT_VERIFIED_TOO_MANY_FIXTURES"
    assert out["fixtures"] == []
    assert discovery_event(out) is None


@pytest.mark.parametrize("mutate", [
    {"fixture_id": True},
    {"fixture_id": 0},
    {"home_team_id": None},
    {"away_team_id": 44},
    {"season": "2026"},
    {"home_team": ""},
    {"kickoff": "2026-11-07T16:00:00"},  # unverified TZ
])
def test_incomplete_identity_cannot_be_upserted(mutate):
    row = official()
    row.update(mutate)
    out = select_verified_next_day([row], now_utc=NOW)
    assert out["count"] == 0
    assert out["status"] == "NO_VERIFIED_NEXT_DAY_FIXTURES"
    assert discovery_event(out) is None


def test_rejected_source_and_invalid_max_never_silent():
    with pytest.raises(ValueError, match="SOURCE_OR_BOUNDS_INVALID"):
        select_verified_next_day({}, now_utc=NOW)
    with pytest.raises(ValueError, match="SOURCE_OR_BOUNDS_INVALID"):
        select_verified_next_day([], now_utc=NOW, max_fixtures=500)
    with pytest.raises(ValueError, match="SOURCE_OR_BOUNDS_INVALID"):
        select_verified_next_day([], now_utc=NOW.replace(tzinfo=None))


def test_integration_only_13h_and_quota_reserve_with_existing_persistence():
    # Contract check. Never invokes provider or scheduler; CI also compiles.
    path = Path(__file__).resolve().parents[1] / "mcp_gateway"
    source = (path / "automation_v7.py").read_text()
    persistence = (path / "persistence.py").read_text()
    assert 'elif 13 <= local_now.hour < 14:' in source
    assert "v2._LAST_DAILY_REMAINING <= 125" in source
    assert "v2._API_CALLS_THIS_TICK > v2.MAX_API_CALLS_PER_TICK - 5" in source
    assert "fe11_registry.discovery_event(fe11_roster)" in source
    assert 'events.append(roster_event)' in source
    assert 'if event.get("event_type") == "DAILY_DISCOVERY":' in persistence
    assert "_upsert_fixture(cur, fx)" in persistence
    assert "ON CONFLICT (fixture_id) DO UPDATE SET" in persistence


def test_live_audited_international_leagues_enter_existing_next_day_registry():
    """New research scope is identity-only and still rejects unknown leagues."""
    from mcp_gateway.fe11_registry_next_day_v1 import (
        RESEARCH_ONLY_INTERNATIONAL_LEAGUES,
        SCANNER_LEGACY_LEAGUES,
        SCANNER_APPROVED_LEAGUES,
    )
    assert RESEARCH_ONLY_INTERNATIONAL_LEAGUES == frozenset({5, 10, 36, 536})
    assert RESEARCH_ONLY_INTERNATIONAL_LEAGUES.isdisjoint(SCANNER_LEGACY_LEAGUES)
    assert SCANNER_APPROVED_LEAGUES == SCANNER_LEGACY_LEAGUES | RESEARCH_ONLY_INTERNATIONAL_LEAGUES
    dates = [official(fixture_id=1300000 + lid, league_id=lid) for lid in sorted(RESEARCH_ONLY_INTERNATIONAL_LEAGUES)]
    data = select_verified_next_day(dates + [official(fixture_id=1300999, league_id=999)], now_utc=NOW)
    assert data["status"] == "VERIFIED_PROVIDER_ROSTER"
    assert data["count"] == 4
    assert data["bet_eligible"] is False
    assert data["market_inputs_used"] is False
    assert {f["league_id"] for f in data["fixtures"]} == RESEARCH_ONLY_INTERNATIONAL_LEAGUES
