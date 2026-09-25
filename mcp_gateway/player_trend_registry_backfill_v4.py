from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import json
import time
from typing import Any

from mcp_gateway import automation_v2 as v2
from mcp_gateway import persistence as persistence_base
from mcp_gateway import player_trends

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_PLAYER_TREND_REGISTRY_BACKFILL_V4_1.0.0"
MAX_FIXTURES_PER_RUN = 8
MIN_DAILY_REMAINING = 250
MIN_REQUEST_INTERVAL_SECONDS = 0.8


def _candidate_fixtures(
    conn,
    *,
    lookback_days: int,
    max_fixtures: int,
) -> list[dict[str, Any]]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                m.fixture_id,
                f.kickoff,
                MAX(m.captured_at) AS last_player_prop_market_at
            FROM soccer_market_snapshots m
            JOIN soccer_fixtures f ON f.fixture_id = m.fixture_id
            JOIN soccer_results r ON r.fixture_id = m.fixture_id
            WHERE m.captured_at >= %s
              AND (
                    LOWER(COALESCE(m.market, '')) LIKE '%%player shot%%'
                 OR LOWER(COALESCE(m.market, '')) LIKE '%%shot on target%%'
                 OR LOWER(COALESCE(m.market, '')) LIKE '%%goalscorer%%'
                 OR LOWER(COALESCE(m.market, '')) LIKE '%%scorer%%'
                 OR LOWER(COALESCE(m.market, '')) LIKE '%%assist%%'
                 OR LOWER(COALESCE(m.market, '')) LIKE '%%goalkeeper save%%'
                 OR LOWER(COALESCE(m.market, '')) LIKE '%%keeper save%%'
                 OR LOWER(COALESCE(m.market, '')) LIKE '%%player card%%'
                 OR LOWER(COALESCE(m.market, '')) LIKE '%%booking%%'
              )
              AND NOT EXISTS (
                    SELECT 1
                    FROM soccer_refresh_events e
                    WHERE e.fixture_id = m.fixture_id
                      AND e.stage IN (
                            'POSTGAME',
                            'POSTGAME_BACKFILL',
                            'POSTGAME_REGISTRY_BACKFILL'
                      )
                      AND e.payload ? 'postgame_player_stats'
              )
            GROUP BY m.fixture_id, f.kickoff
            ORDER BY MAX(m.captured_at) DESC
            LIMIT %s
            """,
            (cutoff, max_fixtures),
        )
        cols = [desc.name for desc in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def _load_materialized_captures(
    conn,
    *,
    lookback_days: int,
    limit: int = 500,
) -> list[dict[str, Any]]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                e.fixture_id,
                f.kickoff,
                e.payload->'postgame_player_stats' AS player_stats
            FROM soccer_refresh_events e
            LEFT JOIN soccer_fixtures f ON f.fixture_id = e.fixture_id
            WHERE e.generated_at >= %s
              AND e.stage = 'POSTGAME_REGISTRY_BACKFILL'
              AND e.payload ? 'postgame_player_stats'
            ORDER BY e.generated_at DESC
            LIMIT %s
            """,
            (cutoff, int(limit)),
        )
        rows = cur.fetchall()

    captures: list[dict[str, Any]] = []
    seen: set[int] = set()
    for fixture_id, kickoff, player_stats in rows:
        if fixture_id is None:
            continue
        fid = int(fixture_id)
        if fid in seen:
            continue
        seen.add(fid)
        stats = player_stats if isinstance(player_stats, dict) else {}
        teams = stats.get("teams") if isinstance(stats.get("teams"), list) else []
        if not teams:
            continue
        captures.append({
            "fixture_id": fid,
            "kickoff": kickoff.isoformat() if isinstance(kickoff, datetime) else kickoff,
            "capture_phase": "POSTGAME_REGISTRY_BACKFILL",
            "teams": teams,
        })
    return captures


def _has_players(compact: dict[str, Any]) -> bool:
    return any(
        isinstance(team, dict) and bool(team.get("players"))
        for team in (compact.get("teams") or [])
    )


def make_registry_backfill_event(
    fixture_id: int,
    kickoff: Any,
    compact: dict[str, Any],
    *,
    provider_daily_remaining: int | None,
) -> dict[str, Any]:
    stats = dict(compact)
    stats["fixture_id"] = int(fixture_id)
    stats["capture_phase"] = "POSTGAME_REGISTRY_BACKFILL"
    stats["finalized_fixture_required"] = True
    stats["future_registry_use_only"] = True
    return {
        "event_type": "RESEARCH_BACKFILL",
        "stage": "POSTGAME_REGISTRY_BACKFILL",
        "fixture": {
            "fixture_id": int(fixture_id),
            "kickoff": kickoff.isoformat() if isinstance(kickoff, datetime) else kickoff,
        },
        "classification": "PASS",
        "bet_eligible": False,
        "actionable": False,
        "decision_weight": 0.0,
        "postgame_player_stats": stats,
        "backfill": {
            "reason": "PLAYER_TREND_REGISTRY_MATURATION",
            "future_registry_use_only": True,
            "eligible_for_historical_oos_reconstruction": False,
            "pregame_signal_required": False,
            "finalized_result_required": True,
            "retroactive_pregame_signal_created": False,
            "retroactive_market_created": False,
            "provider_daily_remaining_after_call": provider_daily_remaining,
        },
    }


def _persist_event(conn, event: dict[str, Any], *, generated_at: datetime) -> None:
    fixture_id = int((event.get("fixture") or {}).get("fixture_id"))
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO soccer_refresh_events (
                fixture_id, stage, event_type, classification, availability_confidence,
                bet_eligible, data_tier, generated_at, payload
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
            """,
            (
                fixture_id,
                "POSTGAME_REGISTRY_BACKFILL",
                "RESEARCH_BACKFILL",
                "PASS",
                None,
                False,
                None,
                generated_at,
                json.dumps(event),
            ),
        )


async def run_backfill(
    *,
    lookback_days: int = 180,
    max_fixtures: int = MAX_FIXTURES_PER_RUN,
) -> dict[str, Any]:
    lookback_days = max(1, min(int(lookback_days), 730))
    max_fixtures = max(1, min(int(max_fixtures), MAX_FIXTURES_PER_RUN))
    if not persistence_base.persistence_configured():
        return {
            "schema_version": SCHEMA_VERSION,
            "model_version": MODEL_VERSION,
            "status": "POSTGRES_NOT_CONFIGURED",
            "provider_requests_added": 0,
            "production_promotion_allowed": False,
        }

    persistence_base.ensure_schema()
    attempted = captured = unavailable = 0
    details: list[dict[str, Any]] = []
    newly_captured: list[dict[str, Any]] = []
    daily_remaining = v2._LAST_DAILY_REMAINING
    last_request_started = 0.0

    with persistence_base._connect() as conn:
        candidates = _candidate_fixtures(
            conn,
            lookback_days=lookback_days,
            max_fixtures=max_fixtures,
        )

        for candidate in candidates:
            fixture_id = int(candidate["fixture_id"])
            kickoff = candidate.get("kickoff")
            if daily_remaining is not None and daily_remaining <= MIN_DAILY_REMAINING:
                details.append({
                    "fixture_id": fixture_id,
                    "status": "SKIPPED_DAILY_RESERVE_GUARD",
                })
                break

            wait_for = MIN_REQUEST_INTERVAL_SECONDS - (time.monotonic() - last_request_started)
            if wait_for > 0:
                await asyncio.sleep(wait_for)

            attempted += 1
            last_request_started = time.monotonic()
            try:
                raw = await v2._ORIGINAL_API_GET(
                    "fixtures/players",
                    {"fixture": fixture_id},
                )
                remaining = (raw.get("quota") or {}).get("daily_remaining")
                try:
                    if remaining is not None:
                        daily_remaining = int(remaining)
                        v2._LAST_DAILY_REMAINING = daily_remaining
                except (TypeError, ValueError):
                    pass

                compact = player_trends._compact(raw)
                if not _has_players(compact):
                    unavailable += 1
                    details.append({
                        "fixture_id": fixture_id,
                        "status": "PROVIDER_RETURNED_NO_PLAYER_ROWS",
                        "daily_remaining": daily_remaining,
                    })
                    continue

                event = make_registry_backfill_event(
                    fixture_id,
                    kickoff,
                    compact,
                    provider_daily_remaining=daily_remaining,
                )
                _persist_event(conn, event, generated_at=datetime.now(timezone.utc))
                captured += 1
                player_rows = sum(
                    len(team.get("players") or [])
                    for team in compact.get("teams") or []
                    if isinstance(team, dict)
                )
                capture = {
                    "fixture_id": fixture_id,
                    "kickoff": kickoff.isoformat() if isinstance(kickoff, datetime) else kickoff,
                    "capture_phase": "POSTGAME_REGISTRY_BACKFILL",
                    "teams": compact.get("teams") or [],
                }
                newly_captured.append(capture)
                details.append({
                    "fixture_id": fixture_id,
                    "status": "CAPTURED_FOR_FUTURE_REGISTRY",
                    "player_rows": player_rows,
                    "daily_remaining": daily_remaining,
                })
            except Exception as exc:
                unavailable += 1
                details.append({
                    "fixture_id": fixture_id,
                    "status": "PROVIDER_ERROR",
                    "error": str(exc)[:180],
                    "daily_remaining": daily_remaining,
                })

    with persistence_base._connect() as conn:
        materialized_captures = _load_materialized_captures(
            conn,
            lookback_days=lookback_days,
            limit=500,
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": "PLAYER_TREND_REGISTRY_BACKFILL_COMPLETE",
        "candidate_fixtures": [int(row["fixture_id"]) for row in candidates],
        "candidate_fixture_count": len(candidates),
        "attempted": attempted,
        "captured": captured,
        "unavailable": unavailable,
        "details": details,
        "captures": materialized_captures,
        "newly_captured_count": len(newly_captured),
        "materialized_capture_count": len(materialized_captures),
        "provider_requests_added": attempted,
        "max_provider_requests_per_run": MAX_FIXTURES_PER_RUN,
        "daily_remaining_after_run": daily_remaining,
        "future_registry_use_only": True,
        "retroactive_pregame_signal_created": False,
        "eligible_for_historical_oos_reconstruction": False,
        "decision_weight": 0.0,
        "production_promotion_allowed": False,
        "policy": (
            "FINALIZED FIXTURES WITH OBSERVED PLAYER-PROP MARKETS ONLY; OUTCOME/STATS BACKFILL "
            "MAY FEED FUTURE PLAYER TREND/ROLE/MODEL REGISTRIES ONLY; NEVER CREATE OR RECONSTRUCT "
            "HISTORICAL PREGAME SIGNALS, XI, PRICES, CLV, OOS PREDICTIONS, OR BET ELIGIBILITY."
        ),
    }
