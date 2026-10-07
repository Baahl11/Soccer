from __future__ import annotations

import asyncio
import json
import math
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any

from mcp_gateway import automation as automation_base
from mcp_gateway import automation_v2 as v2
from mcp_gateway import persistence

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_FM4_PERSONNEL_HISTORY_BACKFILL_V1.0.0"
MAX_LINEUP_REQUESTS_PER_RUN = 8
MIN_DAILY_REMAINING = 250
MIN_REQUEST_INTERVAL_SECONDS = 0.8
HISTORICAL_FACT_DELAY_HOURS = 4
MIN_PRIOR_MATCHES_TARGET = 3
MAX_TEAM_IDS = 800
MAX_CANDIDATE_ROWS = 5000


def _parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        out = value
    else:
        out = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    if out.tzinfo is None:
        out = out.replace(tzinfo=timezone.utc)
    return out.astimezone(timezone.utc)


def _candidate_rows(
    conn: Any,
    *,
    team_ids: list[int],
    before: datetime,
    lookback_days: int,
) -> list[dict[str, Any]]:
    cutoff = before - timedelta(days=lookback_days)
    latest_kickoff = before - timedelta(hours=HISTORICAL_FACT_DELAY_HOURS)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                f.fixture_id,
                f.kickoff,
                f.league_id,
                f.league,
                f.country,
                f.season,
                f.round,
                f.home_team_id,
                f.home_team,
                f.away_team_id,
                f.away_team,
                r.final_status,
                r.home_goals,
                r.away_goals
            FROM soccer_fixtures f
            JOIN soccer_results r ON r.fixture_id = f.fixture_id
            WHERE f.kickoff >= %s
              AND f.kickoff < %s
              AND UPPER(COALESCE(r.final_status, '')) IN ('FT', 'AET', 'PEN')
              AND (
                    f.home_team_id = ANY(%s)
                 OR f.away_team_id = ANY(%s)
              )
              AND NOT EXISTS (
                    SELECT 1
                    FROM soccer_refresh_events e
                    WHERE e.fixture_id = f.fixture_id
                      AND e.stage IN (
                          'FM4_PERSONNEL_BACKFILL',
                          'FM4_PERSONNEL_BACKFILL_INCOMPLETE'
                      )
              )
            ORDER BY f.kickoff DESC, f.fixture_id DESC
            LIMIT %s
            """,
            (cutoff, latest_kickoff, team_ids, team_ids, MAX_CANDIDATE_ROWS),
        )
        columns = [desc.name for desc in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def _existing_team_counts(conn: Any, team_ids: set[int]) -> Counter[int]:
    counts: Counter[int] = Counter()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                payload->'fixture'->>'home_team_id' AS home_team_id,
                payload->'fixture'->>'away_team_id' AS away_team_id
            FROM soccer_refresh_events
            WHERE stage = 'FM4_PERSONNEL_BACKFILL'
              AND payload ? 'fixture'
            """
        )
        for home_raw, away_raw in cur.fetchall():
            for raw in (home_raw, away_raw):
                try:
                    team_id = int(raw)
                except (TypeError, ValueError):
                    continue
                if team_id in team_ids:
                    counts[team_id] += 1
    return counts


def select_candidates(
    rows: list[dict[str, Any]],
    *,
    team_ids: set[int],
    prior_counts: Counter[int],
    max_fixtures: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used: set[int] = set()
    temp = Counter(prior_counts)

    while len(selected) < max_fixtures:
        ranked: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        for row in rows:
            fixture_id = int(row.get("fixture_id") or 0)
            if not fixture_id or fixture_id in used:
                continue
            home = int(row.get("home_team_id") or 0)
            away = int(row.get("away_team_id") or 0)
            target_home = home in team_ids
            target_away = away in team_ids
            if not (target_home or target_away):
                continue
            home_count = temp[home] if target_home else MIN_PRIOR_MATCHES_TARGET
            away_count = temp[away] if target_away else MIN_PRIOR_MATCHES_TARGET
            if (
                (not target_home or home_count >= MIN_PRIOR_MATCHES_TARGET)
                and (not target_away or away_count >= MIN_PRIOR_MATCHES_TARGET)
            ):
                continue
            undercovered = int(
                target_home and home_count < MIN_PRIOR_MATCHES_TARGET
            ) + int(target_away and away_count < MIN_PRIOR_MATCHES_TARGET)
            both_target = int(target_home and target_away)
            kickoff = row.get("kickoff")
            kickoff_ts = kickoff.timestamp() if isinstance(kickoff, datetime) else 0.0
            ranked.append(
                (
                    (
                        -undercovered,
                        -both_target,
                        min(home_count, away_count),
                        max(home_count, away_count),
                        -kickoff_ts,
                        fixture_id,
                    ),
                    row,
                )
            )
        if not ranked:
            break
        ranked.sort(key=lambda item: item[0])
        row = ranked[0][1]
        fixture_id = int(row["fixture_id"])
        used.add(fixture_id)
        selected.append(row)
        home = int(row.get("home_team_id") or 0)
        away = int(row.get("away_team_id") or 0)
        if home in team_ids:
            temp[home] += 1
        if away in team_ids:
            temp[away] += 1

    return selected


def _sufficient_lineup(lineup: dict[str, Any]) -> bool:
    teams = lineup.get("teams") if isinstance(lineup.get("teams"), list) else []
    return (
        lineup.get("both_xi_confirmed") is True
        and lineup.get("both_goalkeepers_confirmed") is True
        and len(teams) == 2
        and all(
            isinstance(team, dict)
            and team.get("team_id") is not None
            and len(team.get("starters") or []) >= 11
            for team in teams
        )
    )


def _fixture_payload(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "fixture_id": int(candidate["fixture_id"]),
        "kickoff": (
            candidate["kickoff"].isoformat()
            if isinstance(candidate.get("kickoff"), datetime)
            else candidate.get("kickoff")
        ),
        "league_id": candidate.get("league_id"),
        "league": candidate.get("league"),
        "country": candidate.get("country"),
        "season": candidate.get("season"),
        "round": candidate.get("round"),
        "home_team_id": candidate.get("home_team_id"),
        "home_team": candidate.get("home_team"),
        "away_team_id": candidate.get("away_team_id"),
        "away_team": candidate.get("away_team"),
        "status": candidate.get("final_status"),
    }


def make_event(
    candidate: dict[str, Any],
    lineup: dict[str, Any],
    *,
    retrieved_at: datetime,
    provider_daily_remaining: int | None,
) -> dict[str, Any]:
    kickoff = _parse_dt(candidate["kickoff"])
    fact_available_at = kickoff + timedelta(hours=HISTORICAL_FACT_DELAY_HOURS)
    return {
        "event_type": "RESEARCH_BACKFILL",
        "stage": "FM4_PERSONNEL_BACKFILL",
        "fixture": _fixture_payload(candidate),
        "classification": "PASS",
        "bet_eligible": False,
        "actionable": False,
        "decision_weight": 0.0,
        "result": {
            "status": candidate.get("final_status"),
            "goals": {
                "home": candidate.get("home_goals"),
                "away": candidate.get("away_goals"),
            },
        },
        "historical_personnel_fact": {
            "source": "API_FOOTBALL_FIXTURES_LINEUPS_HISTORICAL_FACT",
            "source_fixture_id": int(candidate["fixture_id"]),
            "source_fixture_kickoff": kickoff.isoformat(),
            "historical_fact_available_at": fact_available_at.isoformat(),
            "retrieved_at": retrieved_at.isoformat(),
            "both_xi_confirmed": lineup.get("both_xi_confirmed") is True,
            "both_goalkeepers_confirmed": (
                lineup.get("both_goalkeepers_confirmed") is True
            ),
            "teams": lineup.get("teams") or [],
            "retroactive_current_fixture_allowed": False,
            "prior_history_use_only": True,
        },
        "backfill": {
            "reason": "FM4_PRIOR_PERSONNEL_HISTORY_DEPTH",
            "provider_daily_remaining_after_call": provider_daily_remaining,
            "historical_fact_reconstruction_allowed": True,
            "retroactive_current_fixture_allowed": False,
            "retroactive_prediction_rewrite": False,
            "retroactive_market_created": False,
            "retroactive_bet_created": False,
            "production_decision_weight": 0.0,
        },
    }


def _persist_event(
    conn: Any,
    *,
    candidate: dict[str, Any],
    event: dict[str, Any],
    generated_at: datetime,
    stage: str,
    event_type: str,
) -> None:
    fixture_id = int(candidate["fixture_id"])
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO soccer_refresh_events (
                fixture_id, stage, event_type, classification,
                availability_confidence, bet_eligible, data_tier,
                generated_at, payload
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
            """,
            (
                fixture_id,
                stage,
                event_type,
                "PASS",
                None,
                False,
                None,
                generated_at,
                json.dumps(event),
            ),
        )


def _persist_incomplete(
    conn: Any,
    *,
    candidate: dict[str, Any],
    lineup: dict[str, Any],
    generated_at: datetime,
) -> None:
    diagnostic = {
        "event_type": "RESEARCH_BACKFILL_DIAGNOSTIC",
        "stage": "FM4_PERSONNEL_BACKFILL_INCOMPLETE",
        "fixture": _fixture_payload(candidate),
        "classification": "PASS",
        "bet_eligible": False,
        "decision_weight": 0.0,
        "provider_observation": {
            "status": "PROVIDER_LINEUP_INCOMPLETE",
            "team_rows": len(lineup.get("teams") or []),
            "both_xi_confirmed": lineup.get("both_xi_confirmed") is True,
            "both_goalkeepers_confirmed": (
                lineup.get("both_goalkeepers_confirmed") is True
            ),
            "retrieved_at": generated_at.isoformat(),
        },
        "retroactive_prediction_rewrite": False,
        "retroactive_market_created": False,
        "retroactive_bet_created": False,
        "production_promotion_allowed": False,
    }
    _persist_event(
        conn,
        candidate=candidate,
        event=diagnostic,
        generated_at=generated_at,
        stage="FM4_PERSONNEL_BACKFILL_INCOMPLETE",
        event_type="RESEARCH_BACKFILL_DIAGNOSTIC",
    )


async def run_backfill(
    *,
    team_ids: list[int],
    before: str,
    lookback_days: int = 365,
    max_fixtures: int = MAX_LINEUP_REQUESTS_PER_RUN,
) -> dict[str, Any]:
    unique_team_ids = sorted(
        {int(value) for value in team_ids if value is not None and int(value) > 0}
    )[:MAX_TEAM_IDS]
    if not unique_team_ids:
        raise ValueError("team_ids required")

    before_dt = _parse_dt(before)
    lookback_days = max(30, min(int(lookback_days), 730))
    max_fixtures = max(1, min(int(max_fixtures), MAX_LINEUP_REQUESTS_PER_RUN))

    if not persistence.persistence_configured():
        return {
            "schema_version": SCHEMA_VERSION,
            "model_version": MODEL_VERSION,
            "status": "POSTGRES_NOT_CONFIGURED",
            "provider_requests_added": 0,
            "production_promotion_allowed": False,
        }

    persistence.ensure_schema()
    attempted = captured = incomplete = errors = 0
    daily_remaining = v2._LAST_DAILY_REMAINING
    details: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    last_request_started = 0.0

    with persistence._connect() as conn:
        target_set = set(unique_team_ids)
        prior_counts = _existing_team_counts(conn, target_set)
        pool = _candidate_rows(
            conn,
            team_ids=unique_team_ids,
            before=before_dt,
            lookback_days=lookback_days,
        )
        candidates = select_candidates(
            pool,
            team_ids=target_set,
            prior_counts=prior_counts,
            max_fixtures=max_fixtures,
        )

        for candidate in candidates:
            fixture_id = int(candidate["fixture_id"])
            if daily_remaining is not None and daily_remaining <= MIN_DAILY_REMAINING:
                details.append(
                    {
                        "fixture_id": fixture_id,
                        "status": "SKIPPED_DAILY_RESERVE_GUARD",
                    }
                )
                break

            wait_for = MIN_REQUEST_INTERVAL_SECONDS - (
                time.monotonic() - last_request_started
            )
            if wait_for > 0:
                await asyncio.sleep(wait_for)

            attempted += 1
            last_request_started = time.monotonic()
            try:
                raw = await v2._ORIGINAL_API_GET(
                    "fixtures/lineups",
                    {"fixture": fixture_id},
                )
                remaining = (raw.get("quota") or {}).get("daily_remaining")
                try:
                    if remaining is not None:
                        daily_remaining = int(remaining)
                        v2._LAST_DAILY_REMAINING = daily_remaining
                except (TypeError, ValueError):
                    pass

                lineup = automation_base._compact_lineups(raw)
                retrieved_at = datetime.now(timezone.utc)
                if not _sufficient_lineup(lineup):
                    incomplete += 1
                    _persist_incomplete(
                        conn,
                        candidate=candidate,
                        lineup=lineup,
                        generated_at=retrieved_at,
                    )
                    details.append(
                        {
                            "fixture_id": fixture_id,
                            "status": "PROVIDER_LINEUP_INCOMPLETE",
                            "team_rows": len(lineup.get("teams") or []),
                            "both_xi_confirmed": (
                                lineup.get("both_xi_confirmed") is True
                            ),
                            "both_goalkeepers_confirmed": (
                                lineup.get("both_goalkeepers_confirmed") is True
                            ),
                            "daily_remaining": daily_remaining,
                        }
                    )
                    continue

                event = make_event(
                    candidate,
                    lineup,
                    retrieved_at=retrieved_at,
                    provider_daily_remaining=daily_remaining,
                )
                _persist_event(
                    conn,
                    candidate=candidate,
                    event=event,
                    generated_at=retrieved_at,
                    stage="FM4_PERSONNEL_BACKFILL",
                    event_type="RESEARCH_BACKFILL",
                )
                events.append(event)
                captured += 1
                details.append(
                    {
                        "fixture_id": fixture_id,
                        "status": "CAPTURED",
                        "home_team_id": candidate.get("home_team_id"),
                        "away_team_id": candidate.get("away_team_id"),
                        "kickoff": (
                            candidate["kickoff"].isoformat()
                            if isinstance(candidate.get("kickoff"), datetime)
                            else candidate.get("kickoff")
                        ),
                        "daily_remaining": daily_remaining,
                    }
                )
            except Exception as exc:
                errors += 1
                details.append(
                    {
                        "fixture_id": fixture_id,
                        "status": "PROVIDER_ERROR",
                        "error": str(exc)[:180],
                        "daily_remaining": daily_remaining,
                    }
                )

    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": "FM4_PERSONNEL_HISTORY_BACKFILL_COMPLETE",
        "before": before_dt.isoformat(),
        "lookback_days": lookback_days,
        "historical_fact_delay_hours": HISTORICAL_FACT_DELAY_HOURS,
        "target_team_count": len(unique_team_ids),
        "candidate_pool_rows": len(pool),
        "selected_fixture_count": len(candidates),
        "attempted": attempted,
        "captured": captured,
        "incomplete": incomplete,
        "errors": errors,
        "details": details,
        "events": events,
        "provider_requests_added": attempted,
        "max_provider_requests_per_run": MAX_LINEUP_REQUESTS_PER_RUN,
        "daily_remaining_after_run": daily_remaining,
        "prior_history_use_only": True,
        "retroactive_current_fixture_allowed": False,
        "retroactive_prediction_rewrite": False,
        "retroactive_market_created": False,
        "retroactive_bet_created": False,
        "decision_weight": 0.0,
        "production_promotion_allowed": False,
        "policy": (
            "FINALIZED_PRECOHORT_FIXTURES_ONLY; SOURCE_FIXTURE_MUST_END_AT_LEAST_FOUR_HOURS "
            "BEFORE_TARGET_COHORT; EXACT_API_FOOTBALL_FIXTURES_LINEUPS_ONLY; BOTH_XI_AND_GK_REQUIRED; "
            "MAX_EIGHT_LINEUP_REQUESTS_PER_RUN; KNOWN_INCOMPLETE_FIXTURES_NOT_REQUERIED; "
            "BACKFILLED_XI_CAN_ONLY_BE_USED_AS_PRIOR_PERSONNEL_HISTORY_FOR_LATER_TARGET_FIXTURES; "
            "NEVER_RETROACTIVELY_TREAT_AS_CURRENT_PREGAME_LINEUP_OR_FORMATION; NO_PREDICTION_MARKET_CLV_BET_REWRITE."
        ),
    }
