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
MODEL_VERSION = "SOCCER_PLAYER_TREND_REGISTRY_BACKFILL_V4_1.2.0"
MAX_FIXTURES_PER_RUN = 16
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
                f.home_team_id,
                f.away_team_id,
                r.home_goals,
                r.away_goals,
                MAX(m.captured_at) AS last_player_prop_market_at,
                COUNT(*) FILTER (
                    WHERE LOWER(COALESCE(m.market, '')) LIKE '%%goalkeeper save%%'
                       OR LOWER(COALESCE(m.market, '')) LIKE '%%keeper save%%'
                       OR LOWER(COALESCE(m.market, '')) LIKE '%%gk save%%'
                ) AS gk_saves_market_rows,
                BOOL_OR(
                    LOWER(COALESCE(m.market, '')) LIKE '%%goalkeeper save%%'
                    OR LOWER(COALESCE(m.market, '')) LIKE '%%keeper save%%'
                    OR LOWER(COALESCE(m.market, '')) LIKE '%%gk save%%'
                ) AS has_gk_saves_market,
                COUNT(*) AS player_prop_market_rows
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
            GROUP BY m.fixture_id, f.kickoff, f.home_team_id, f.away_team_id, r.home_goals, r.away_goals
            ORDER BY
                BOOL_OR(
                    LOWER(COALESCE(m.market, '')) LIKE '%%goalkeeper save%%'
                    OR LOWER(COALESCE(m.market, '')) LIKE '%%keeper save%%'
                    OR LOWER(COALESCE(m.market, '')) LIKE '%%gk save%%'
                ) DESC,
                MAX(m.captured_at) DESC
            LIMIT %s
            """,
            (cutoff, max_fixtures),
        )
        cols = [desc.name for desc in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def _enrich_full_match_goalkeeper_conceded(
    teams: list[dict[str, Any]],
    *,
    home_team_id: Any,
    away_team_id: Any,
    home_goals: Any,
    away_goals: Any,
) -> list[dict[str, Any]]:
    try:
        home_id = int(home_team_id) if home_team_id is not None else None
        away_id = int(away_team_id) if away_team_id is not None else None
        home_score = float(home_goals) if home_goals is not None else None
        away_score = float(away_goals) if away_goals is not None else None
    except (TypeError, ValueError):
        return teams

    if home_id is None or away_id is None or home_score is None or away_score is None:
        return teams

    enriched: list[dict[str, Any]] = []
    for team in teams:
        if not isinstance(team, dict):
            continue
        row = dict(team)
        team_id = row.get("team_id")
        opponent_goals = None
        try:
            tid = int(team_id) if team_id is not None else None
        except (TypeError, ValueError):
            tid = None
        if tid == home_id:
            opponent_goals = away_score
        elif tid == away_id:
            opponent_goals = home_score

        players: list[dict[str, Any]] = []
        for player in row.get("players") or []:
            if not isinstance(player, dict):
                continue
            p = dict(player)
            position = str(p.get("position") or "").upper()
            try:
                minutes = float(p.get("minutes")) if p.get("minutes") is not None else None
            except (TypeError, ValueError):
                minutes = None
            if (
                opponent_goals is not None
                and position in {"G", "GK", "GOALKEEPER"}
                and minutes is not None
                and minutes >= 89.0
                and p.get("goals_conceded") is None
            ):
                p["goals_conceded"] = opponent_goals
                p["goals_conceded_source"] = "FINAL_SCORE_FULL_MATCH_GK_FALLBACK"
                p["goals_conceded_inferred_for_registry_only"] = True
            players.append(p)
        row["players"] = players
        enriched.append(row)
    return enriched


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
                f.home_team_id,
                f.away_team_id,
                r.home_goals,
                r.away_goals,
                e.payload->'postgame_player_stats' AS player_stats
            FROM soccer_refresh_events e
            LEFT JOIN soccer_fixtures f ON f.fixture_id = e.fixture_id
            LEFT JOIN soccer_results r ON r.fixture_id = e.fixture_id
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
    for fixture_id, kickoff, home_team_id, away_team_id, home_goals, away_goals, player_stats in rows:
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
        teams = _enrich_full_match_goalkeeper_conceded(
            teams,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            home_goals=home_goals,
            away_goals=away_goals,
        )
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
    priority_family: str | None = None,
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
            "priority_family": priority_family,
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
    captured_gk_priority = 0
    captured_goalkeeper_rows = 0
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
                compact["teams"] = _enrich_full_match_goalkeeper_conceded(
                    compact.get("teams") or [],
                    home_team_id=candidate.get("home_team_id"),
                    away_team_id=candidate.get("away_team_id"),
                    home_goals=candidate.get("home_goals"),
                    away_goals=candidate.get("away_goals"),
                )
                if not _has_players(compact):
                    unavailable += 1
                    details.append({
                        "fixture_id": fixture_id,
                        "status": "PROVIDER_RETURNED_NO_PLAYER_ROWS",
                        "daily_remaining": daily_remaining,
                    })
                    continue

                priority_family = "GK_SAVES" if candidate.get("has_gk_saves_market") is True else None
                event = make_registry_backfill_event(
                    fixture_id,
                    kickoff,
                    compact,
                    provider_daily_remaining=daily_remaining,
                    priority_family=priority_family,
                )
                _persist_event(conn, event, generated_at=datetime.now(timezone.utc))
                captured += 1
                if priority_family == "GK_SAVES":
                    captured_gk_priority += 1
                player_rows = sum(
                    len(team.get("players") or [])
                    for team in compact.get("teams") or []
                    if isinstance(team, dict)
                )
                goalkeeper_rows = sum(
                    1
                    for team in compact.get("teams") or []
                    if isinstance(team, dict)
                    for player in (team.get("players") or [])
                    if isinstance(player, dict)
                    and str(player.get("position") or "").upper() in {"G", "GK", "GOALKEEPER"}
                    and (_num(player.get("minutes")) or 0.0) > 0.0
                )
                captured_goalkeeper_rows += goalkeeper_rows
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
                    "priority_family": priority_family,
                    "gk_saves_market_rows": int(candidate.get("gk_saves_market_rows") or 0),
                    "player_rows": player_rows,
                    "goalkeeper_rows": goalkeeper_rows,
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
        "gk_saves_priority_candidate_count": sum(
            1 for row in candidates if row.get("has_gk_saves_market") is True
        ),
        "attempted": attempted,
        "captured": captured,
        "captured_gk_saves_priority_fixtures": captured_gk_priority,
        "captured_goalkeeper_rows": captured_goalkeeper_rows,
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
            "FINALIZED FIXTURES WITH OBSERVED PLAYER-PROP MARKETS ONLY; GK SAVES MARKET FIXTURES ARE PRIORITIZED; OUTCOME/STATS BACKFILL "
            "MAY FEED FUTURE PLAYER TREND/ROLE/MODEL REGISTRIES ONLY; NEVER CREATE OR RECONSTRUCT "
            "HISTORICAL PREGAME SIGNALS, XI, PRICES, CLV, OOS PREDICTIONS, OR BET ELIGIBILITY."
        ),
    }
