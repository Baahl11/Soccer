"""FE11 all-approved-league capture: offline, no network, no database."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mcp_gateway import fe11_universal_lifecycle_v1 as universal
from mcp_gateway import fe11_universal_persistence_v1 as writer

NOW = datetime(2026, 11, 7, 4, 40, tzinfo=timezone.utc)


def fixture(fid: int, *, league: int = 39, status: str = "NS",
            kickoff_delta: int = 180, goals=None, score=None) -> dict:
    return {
        "fixture_id": fid, "league_id": league, "league": "Premier League",
        "country": "England", "season": 2026, "round": "Regular",
        "kickoff": (NOW + timedelta(minutes=kickoff_delta)).isoformat(),
        "status": status, "status_long": "Not Started" if status == "NS" else "Match Finished",
        "home_team_id": 10000 + fid, "home_team": "Verified Home",
        "away_team_id": 20000 + fid, "away_team": "Verified Away",
        "venue": "Ground", "city": "City",
        "goals": goals, "score": score,
    }


def capture(rows):
    return universal.build_capture(rows, now_utc=NOW)


def test_more_than_three_approved_fixture_baselines_without_extra_provider_calls():
    rows = [fixture(i, league=39 if i % 2 else 140) for i in range(1, 17)]
    result = capture(rows + [fixture(23, league=999)])
    assert result["status"] == "VERIFIED_OFFICIAL_ALL_ELIGIBLE_CAPTURE"
    assert result["pregame_count"] == 16
    assert result["eligible_source_count"] == 16
    assert result["provider_requests_added"] == 0
    assert result["bet_eligible"] is False
    assert len(universal.safe_tick_capture(result, now_utc=NOW.isoformat())[0]) == 16


def test_only_before_kickoff_ns_can_create_freezable_baseline():
    rows = [fixture(1), fixture(2, kickoff_delta=-10),
            fixture(3, status="PST"), fixture(4, status="FT", kickoff_delta=-120,
               goals={"home": 1, "away": 1})]
    result = capture(rows)
    assert [x["fixture_id"] for x in result["pregame_fixtures"]] == [1]
    assert [x["fixture_id"] for x in result["official_final_fixtures"]] == [4]
    assert result["excluded_not_pregame_or_final"] == 2


def test_only_verified_goals_and_periods_with_zero_not_missing():
    row = fixture(
        3, status="FT", kickoff_delta=-130, goals={"home": 3, "away": 1},
        score={"halftime": {"home": 1, "away": 0},
               "fulltime": {"home": 3, "away": 1}},
    )
    result = capture([row])
    o = result["official_final_fixtures"][0]
    assert o["goals"] == {"home": 3, "away": 1}
    assert o["halftime"] == {"home": 1, "away": 0}
    assert o["second_half"] == {"home": 2, "away": 1}
    assert o["corners_ft"] is None and o["corners_verified"] is False
    assert o["bet_eligible"] is False


def test_extratime_score_never_inferred_as_regulation_second_half():
    row = fixture(
        33, status="AET", kickoff_delta=-180, goals={"home": 4, "away": 2},
        score={"halftime": {"home": 1, "away": 0}},
    )
    o = capture([row])["official_final_fixtures"][0]
    assert o["second_half"] is None and o["regulation_fulltime"] is None
    assert o["goals"] == {"home": 4, "away": 2}


def test_unverified_goals_excluded_not_coerced_to_zero():
    result = capture([fixture(1, status="FT", kickoff_delta=-120,
                              goals={"home": 2, "away": None})])
    assert result["official_final_count"] == 0
    assert result["final_score_missing"] == 1


def test_duplicate_conflicting_identity_fails_whole_capture_without_partial_records():
    first = fixture(1)
    altered = dict(first, away_team_id=50009)
    result = capture([first, altered])
    assert result["status"] == "NOT_VERIFIED_SOURCE_IDENTITY_CONFLICT"
    assert result["pregame_fixtures"] == []
    assert universal.safe_tick_capture(result, now_utc=NOW.isoformat()) == ([], [])


def test_tampered_bets_market_source_or_asof_are_rejected_before_db():
    clean = capture([fixture(3), fixture(9, status="FT", kickoff_delta=-100,
                                          goals={"home": 2, "away": 1})])
    assert universal.safe_tick_capture(clean, now_utc=NOW.isoformat())[0]
    for key, bad in (("bet_eligible", True), ("market_inputs_used", True),
                     ("source", "UNTRUSTED"), ("provider_requests_added", 1)):
        altered = dict(clean, **{key: bad})
        assert universal.safe_tick_capture(altered, now_utc=NOW.isoformat()) == ([], [])
    altered = dict(clean, source_observed_at_utc=(NOW+timedelta(days=1)).isoformat())
    assert universal.safe_tick_capture(altered, now_utc=NOW.isoformat()) == ([], [])
    altered = dict(clean, pregame_fixtures=[{**clean["pregame_fixtures"][0], "market_price": 2.3}])
    assert universal.safe_tick_capture(altered, now_utc=NOW.isoformat()) == ([], [])


def test_full_next_day_roster_is_extended_from_verified_discovery_event():
    nextday = fixture(21, kickoff_delta=1500)
    roster = {
        "version": "SOCCER_FE11_NEXT_DAY_REGISTRY_PRESEED_v0.1",
        "status": "VERIFIED_PROVIDER_ROSTER",
        "market_inputs_used": False, "bet_eligible": False,
        "count": 1, "fixtures": [
            {key: nextday[key] for key in universal.CANONICAL_FIELDS}
        ],
        "source": "API_FOOTBALL_FIXTURES_DATE",
        "source_observed_at_utc": NOW.isoformat(),
        "date": (NOW+timedelta(minutes=1500)).astimezone(
            universal.registry.LOCAL_TIMEZONE
        ).date().isoformat(),
    }
    # The roster validator requires exactly the next *local* date, not any
    # arbitrary future timestamp. Construct from local CDMX tomorrow.
    start = NOW.astimezone(universal.registry.LOCAL_TIMEZONE).date() + timedelta(days=1)
    from datetime import time
    from zoneinfo import ZoneInfo
    tomorrow = datetime.combine(start, time(14, 0), tzinfo=ZoneInfo("America/Mexico_City"))
    roster["fixtures"][0]["kickoff"] = tomorrow.astimezone(timezone.utc).isoformat()
    roster["date"] = start.isoformat()
    cap = universal.build_capture([fixture(1)], now_utc=NOW, next_day_roster=roster)
    assert cap["pregame_count"] == 2


class CursorSpy:
    def __init__(self):
        self.sql = []
    def execute(self, sql, params):
        self.sql.append((sql, params))


def test_writer_verifies_inputs_and_issues_only_safe_batch_queries():
    result = capture([
        fixture(1),
        fixture(2, status="FT", kickoff_delta=-150, goals={"home": 2, "away": 0}),
    ])
    cur = CursorSpy()
    upserts = []
    writer.persist(cur, {"generated_at_utc": NOW.isoformat(),
                         "fe11_universal_lifecycle": result},
                   lambda c, x: upserts.append(x["fixture_id"]))
    assert upserts == [1]
    assert len(cur.sql) == 2
    assert "b.captured_at < b.kickoff" in cur.sql[1][0]
    assert "JOIN soccer_fe11_pregame_baselines" in cur.sql[1][0]
    assert "ON CONFLICT (fixture_id, observed_at) DO NOTHING" in cur.sql[1][0]
    assert "f.kickoff > %s::timestamptz" in cur.sql[0][0]
    assert "ON CONFLICT (fixture_id) DO NOTHING" in cur.sql[0][0]
    assert "NULL::integer, NULL::integer, FALSE" in cur.sql[1][0]


def test_no_retrospective_pregame_from_official_terminal_only():
    result = capture([fixture(2, status="FT", kickoff_delta=-150,
                              goals={"home": 1, "away": 1})])
    cur, upserts = CursorSpy(), []
    writer.persist(cur, {"generated_at_utc": NOW.isoformat(),
                         "fe11_universal_lifecycle": result},
                   lambda c, x: upserts.append(x))
    assert upserts == []
    assert len(cur.sql) == 1
    assert "JOIN soccer_fe11_pregame_baselines" in cur.sql[0][0]


def test_existing_shortlist_and_betting_logic_not_replaced_by_whole_slate():
    source = (Path(__file__).resolve().parents[1] / "mcp_gateway" / "automation_v7.py").read_text()
    assert "fe11_universal.build_capture(" in source
    assert '"fe11_universal_lifecycle": fe11_all_eligible_capture' in source
    assert "_upcoming_market_capture_fixtures(fixtures, now_utc)" in source
    assert "MAX_DEEP_DIVE_FIXTURES_PER_TICK" in source
    persistence = (Path(__file__).resolve().parents[1] / "mcp_gateway" / "persistence.py").read_text()
    assert "fe11_universal_persistence_v1.persist(cur, tick, _upsert_fixture)" in persistence
    assert '"fe11_universal_lifecycle",' in persistence


def test_schema_has_immutable_pregame_and_append_only_final():
    schema = (Path(__file__).resolve().parents[1] / "mcp_gateway" / "schema.sql").read_text()
    assert "CREATE TABLE IF NOT EXISTS soccer_fe11_pregame_baselines" in schema
    assert "CHECK (captured_at < kickoff)" in schema
    assert "CREATE TABLE IF NOT EXISTS soccer_fe11_official_outcome_snapshots" in schema
    assert "PRIMARY KEY (fixture_id, observed_at)" in schema
    assert "CHECK (corners_home IS NULL AND corners_away IS NULL)" in schema
