"""Real Postgres end-to-end contract for FE11 pregame -> score -> corners.

Only against an EXPLICIT disposable CI Postgres; never production.
Tests synthetic fixtures to prove relational joins. Not training observations.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os

import pytest

from mcp_gateway import persistence
from mcp_gateway import fe11_universal_lifecycle_v1 as lifecycle
from mcp_gateway import fe11_universal_persistence_v1 as writer
from mcp_gateway import fe11_corners_persistence_v1 as corner_writer

pytestmark = pytest.mark.skipif(
    not os.environ.get("FE11_DISPOSABLE_POSTGRES_URL"),
    reason="Only run with an explicitly configured disposable CI PostgreSQL",
)


def _fixture(fid: int, kickoff: datetime, *,
             status: str = "NS", away_team_id: int = 91002,
             goals: dict | None = None, score: dict | None = None) -> dict:
    return {
        "fixture_id": fid, "league_id": 39, "league": "CI EPL Fixture",
        "country": "England", "season": kickoff.year, "round": "CI test",
        "kickoff": kickoff.isoformat(), "status": status,
        "status_long": "Match Finished" if status == "FT" else "Not Started",
        "home_team_id": 91001, "home_team": "Test Home",
        "away_team_id": away_team_id, "away_team": "Test Away",
        "venue": "CI Ground", "city": "CI",
        "goals": goals, "score": score,
    }


def _persist_capture(rows: list[dict], *, observed: datetime) -> None:
    capture = lifecycle.build_capture(rows, now_utc=observed)
    assert capture["status"] == "VERIFIED_OFFICIAL_ALL_ELIGIBLE_CAPTURE"
    with persistence._connect() as conn:
        with conn.cursor() as cur:
            writer.persist(cur, {
                "generated_at_utc": capture["source_observed_at_utc"],
                "fe11_universal_lifecycle": capture,
            }, persistence._upsert_fixture)


def _single(sql: str, params: tuple = ()):
    with persistence._connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()


def test_true_postgres_exact_asof_chain_without_touching_production(monkeypatch):
    db = os.environ["FE11_DISPOSABLE_POSTGRES_URL"]
    assert db.startswith("postgresql://") and ("127.0.0.1" in db or "localhost" in db)
    monkeypatch.setenv("DATABASE_URL", db)
    monkeypatch.setenv("DATABASE_SSLMODE", "disable")
    persistence.ensure_schema()

    observed = datetime.now(timezone.utc).replace(microsecond=300000)
    kickoff = observed + timedelta(minutes=10)
    known_id = 9000001001
    never_registered = 9000001002
    approved = _fixture(known_id, kickoff)
    _persist_capture([approved], observed=observed)

    assert _single(
        "SELECT COUNT(*) FROM soccer_fe11_pregame_baselines WHERE fixture_id=%s",
        (known_id,),
    )[0] == 1
    frozen = _single(
        "SELECT captured_at, kickoff, source_status, bet_eligible FROM "
        "soccer_fe11_pregame_baselines WHERE fixture_id=%s",
        (known_id,),
    )
    assert frozen[0] == observed
    assert frozen[1] == kickoff
    assert frozen[2] == "NS"
    assert frozen[3] is False

    # Re-observing NS at a later pregame timestamp MUST NOT rewrite the first
    # frozen profile, even if the source has updated the surrounding venue.
    _persist_capture([dict(approved, venue="Updated CI Ground")],
                     observed=observed + timedelta(minutes=3))
    assert _single(
        "SELECT captured_at FROM soccer_fe11_pregame_baselines WHERE fixture_id=%s",
        (known_id,),
    )[0] == observed

    finished_at = kickoff + timedelta(hours=2)
    final = _fixture(
        known_id, kickoff, status="FT",
        goals={"home": 1, "away": 2},
        score={
            "halftime": {"home": 0, "away": 1},
            "fulltime": {"home": 1, "away": 2},
        },
    )
    # These simulated future dates exist ONLY inside the disposable CI test.
    # No synthetic result is ever submitted to the connected production DB.
    _persist_capture([final], observed=finished_at)
    label = _single(
        "SELECT home_goals, away_goals, halftime_home_goals, "
        "halftime_away_goals, second_half_home_goals, second_half_away_goals, "
        "corners_home, corners_away, corners_verified, bet_eligible "
        "FROM soccer_fe11_official_outcome_snapshots WHERE fixture_id=%s",
        (known_id,),
    )
    assert label == (1, 2, 0, 1, 1, 1, None, None, False, False)

    # Idempotence at exactly the same source-observed timestamp, not an UPDATE.
    _persist_capture([final], observed=finished_at)
    assert _single(
        "SELECT COUNT(*) FROM soccer_fe11_official_outcome_snapshots WHERE fixture_id=%s",
        (known_id,),
    )[0] == 1

    # Wrong away-team ID cannot be joined onto frozen source truth.
    _persist_capture(
        [dict(final, away_team_id=99991)],
        observed=finished_at + timedelta(minutes=1),
    )
    assert _single(
        "SELECT COUNT(*) FROM soccer_fe11_official_outcome_snapshots WHERE fixture_id=%s",
        (known_id,),
    )[0] == 1

    # A completed result with no earlier official pregame baseline must not be
    # retroactively created/linked. This forbids outcome leakage.
    _persist_capture(
        [_fixture(
            never_registered, kickoff, status="FT",
            goals={"home": 3, "away": 0},
        )],
        observed=finished_at + timedelta(minutes=2),
    )
    assert _single(
        "SELECT COUNT(*) FROM soccer_fe11_official_outcome_snapshots WHERE fixture_id=%s",
        (never_registered,),
    )[0] == 0

    corner_at = finished_at + timedelta(minutes=5)
    attempt = {
        "fixture_id": known_id,
        "kickoff": kickoff.isoformat(),
        "observed_at_utc": corner_at.isoformat(),
        "status": "VERIFIED_FT_CORNERS",
        "provider_requests_added": 1,
    }
    corner = {
        "fixture_id": known_id,
        "kickoff": kickoff.isoformat(),
        "home_team_id": 91001, "away_team_id": 91002,
        "observed_at_utc": corner_at.isoformat(),
        "home_corners": 0, "away_corners": 7,
        "period": "FT_90",
        "source": corner_writer.CORNERS_SOURCE,
        "source_response_verified": True,
        "bet_eligible": False,
    }
    tick = {
        "fe11_postgame_corner_summary": {
            "version": corner_writer.CORNERS_VERSION,
            "status": "VERIFIED_CAPTURES",
            "captured": 1, "bet_eligible": False,
            "market_inputs_used": False,
        },
        "fe11_postgame_corner_attempts": [attempt],
        "fe11_postgame_corner_observations": [corner],
    }
    with persistence._connect() as conn:
        with conn.cursor() as cur:
            corner_writer.persist(cur, tick)

    assert _single(
        "SELECT home_corners, away_corners, period, bet_eligible "
        "FROM soccer_fe11_verified_corner_snapshots WHERE fixture_id=%s",
        (known_id,),
    ) == (0, 7, "FT_90", False)
    assert _single(
        "SELECT attempt_count, last_status FROM soccer_fe11_corner_capture_attempts "
        "WHERE fixture_id=%s",
        (known_id,),
    ) == (1, "VERIFIED_FT_CORNERS")

    # Incorrect participant cannot produce another corner source observation.
    fake_corner_at = corner_at + timedelta(minutes=1)
    bad_attempt = {**attempt, "observed_at_utc": fake_corner_at.isoformat()}
    bad_corner = {**corner, "away_team_id": 99991, "observed_at_utc": fake_corner_at.isoformat()}
    with persistence._connect() as conn:
        with conn.cursor() as cur:
            corner_writer.persist(cur, {
                "fe11_postgame_corner_summary": tick["fe11_postgame_corner_summary"],
                "fe11_postgame_corner_attempts": [bad_attempt],
                "fe11_postgame_corner_observations": [bad_corner],
            })
    assert _single(
        "SELECT COUNT(*) FROM soccer_fe11_verified_corner_snapshots WHERE fixture_id=%s",
        (known_id,),
    )[0] == 1

    with persistence._connect() as conn:
        with conn.cursor() as cur:
            counts = writer.inventory(cur)
    assert counts["eligible_pregame_baselines"] == 1
    assert counts["matched_official_final_fixtures"] == 1
    assert counts["official_period_complete_fixtures"] == 1
    assert counts["official_ft_corners_verified_fixtures"] == 1
    assert counts["official_period_and_corners_complete_fixtures"] == 1
    assert counts["production_bets_authorized"] is False
    # Synthetic CI observations CANNOT be training observations.
    assert counts["joint_goals_corners_handicap_training_rows"] == 0
