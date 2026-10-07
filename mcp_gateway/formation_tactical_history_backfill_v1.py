from __future__ import annotations

import asyncio
import json
import math
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any

from mcp_gateway import automation_v2 as v2
from mcp_gateway import persistence as persistence_base

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_FM4_TACTICAL_HISTORY_BACKFILL_V1.0.0"
MAX_FIXTURES_PER_RUN = 8
MIN_DAILY_REMAINING = 250
MIN_REQUEST_INTERVAL_SECONDS = 0.8
MIN_PRIOR_MATCHES_TARGET = 3
MAX_TEAM_IDS = 800
MAX_CANDIDATE_ROWS = 5000

_STAT_MAP = {
    "shots on goal": "shots_on_goal",
    "shots off goal": "shots_off_goal",
    "total shots": "total_shots",
    "blocked shots": "blocked_shots",
    "shots insidebox": "shots_inside_box",
    "shots inside box": "shots_inside_box",
    "shots outsidebox": "shots_outside_box",
    "shots outside box": "shots_outside_box",
    "fouls": "fouls",
    "corner kicks": "corners",
    "offsides": "offsides",
    "ball possession": "possession",
    "yellow cards": "yellow_cards",
    "red cards": "red_cards",
    "goalkeeper saves": "goalkeeper_saves",
    "passes total": "passes_total",
    "passes accurate": "passes_accurate",
    "passes %": "passes_pct",
}


def _num(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip().replace("%", "")
        if not value:
            return None
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


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
                r.home_goals,
                r.away_goals
            FROM soccer_fixtures f
            JOIN soccer_results r ON r.fixture_id = f.fixture_id
            WHERE f.kickoff >= %s
              AND f.kickoff < %s
              AND (
                    f.home_team_id = ANY(%s)
                 OR f.away_team_id = ANY(%s)
              )
              AND NOT EXISTS (
                    SELECT 1
                    FROM soccer_refresh_events e
                    WHERE e.fixture_id = f.fixture_id
                      AND e.stage = 'FM4_TACTICAL_BACKFILL'
              )
            ORDER BY f.kickoff DESC, f.fixture_id DESC
            LIMIT %s
            """,
            (cutoff, before, team_ids, team_ids, MAX_CANDIDATE_ROWS),
        )
        cols = [desc.name for desc in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def _existing_team_counts(conn: Any, team_ids: set[int]) -> Counter[int]:
    counts: Counter[int] = Counter()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                payload->'fixture'->>'home_team_id' AS home_team_id,
                payload->'fixture'->>'away_team_id' AS away_team_id
            FROM soccer_refresh_events
            WHERE stage = 'FM4_TACTICAL_BACKFILL'
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
    pool = [dict(row) for row in rows]
    selected: list[dict[str, Any]] = []
    temp = Counter(prior_counts)
    used: set[int] = set()

    while pool and len(selected) < max_fixtures:
        ranked: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        for row in pool:
            fid = int(row.get("fixture_id") or 0)
            if not fid or fid in used:
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
            both_target = int(target_home and target_away)
            undercovered = int(target_home and home_count < MIN_PRIOR_MATCHES_TARGET) + int(
                target_away and away_count < MIN_PRIOR_MATCHES_TARGET
            )
            kickoff = row.get("kickoff")
            kickoff_ts = kickoff.timestamp() if isinstance(kickoff, datetime) else 0.0
            score = (
                -undercovered,
                -both_target,
                min(home_count, away_count),
                max(home_count, away_count),
                -kickoff_ts,
                fid,
            )
            ranked.append((score, row))
        if not ranked:
            break
        ranked.sort(key=lambda item: item[0])
        row = ranked[0][1]
        fid = int(row["fixture_id"])
        used.add(fid)
        selected.append(row)
        home = int(row.get("home_team_id") or 0)
        away = int(row.get("away_team_id") or 0)
        if home in team_ids:
            temp[home] += 1
        if away in team_ids:
            temp[away] += 1

    return selected


def compact_tactical_stats(raw: dict[str, Any]) -> dict[str, Any]:
    teams: list[dict[str, Any]] = []
    for team_row in raw.get("response") or []:
        if not isinstance(team_row, dict):
            continue
        team = team_row.get("team") if isinstance(team_row.get("team"), dict) else {}
        team_id = team.get("id")
        if team_id is None:
            continue
        compact: dict[str, Any] = {
            "team_id": int(team_id),
            "team": team.get("name"),
        }
        for stat in team_row.get("statistics") or []:
            if not isinstance(stat, dict):
                continue
            key = _STAT_MAP.get(str(stat.get("type") or "").strip().lower())
            if not key:
                continue
            compact[key] = _num(stat.get("value"))
        teams.append(compact)

    totals: dict[str, float] = {}
    numeric_keys = {
        key
        for team in teams
        for key, value in team.items()
        if key not in {"team_id", "team"} and _num(value) is not None
    }
    for key in numeric_keys:
        values = [_num(team.get(key)) for team in teams]
        if len(values) == 2 and all(value is not None for value in values):
            totals[key] = float(values[0]) + float(values[1])

    return {
        "teams": teams,
        "totals": totals,
        "provider": "API_FOOTBALL",
        "endpoint": "fixtures/statistics",
    }


def _sufficient_stats(compact: dict[str, Any]) -> bool:
    teams = compact.get("teams") if isinstance(compact.get("teams"), list) else []
    if len(teams) != 2:
        return False
    required = (
        "total_shots",
        "shots_on_goal",
        "blocked_shots",
        "possession",
        "fouls",
        "yellow_cards",
    )
    return all(
        all(_num(team.get(field)) is not None for field in required)
        for team in teams
        if isinstance(team, dict)
    )


def make_event(
    candidate: dict[str, Any],
    tactical: dict[str, Any],
    *,
    retrieved_at: datetime,
    provider_daily_remaining: int | None,
) -> dict[str, Any]:
    return {
        "event_type": "RESEARCH_BACKFILL",
        "stage": "FM4_TACTICAL_BACKFILL",
        "fixture": {
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
        },
        "classification": "PASS",
        "bet_eligible": False,
        "actionable": False,
        "decision_weight": 0.0,
        "postgame_tactical_stats": {
            **tactical,
            "fixture_id": int(candidate["fixture_id"]),
            "retrieved_at": retrieved_at.isoformat(),
            "historical_fact_time": (
                candidate["kickoff"].isoformat()
                if isinstance(candidate.get("kickoff"), datetime)
                else candidate.get("kickoff")
            ),
            "capture_phase": "FM4_TACTICAL_HISTORY_BACKFILL",
        },
        "backfill": {
            "reason": "FM4_PRIOR_STYLE_HISTORY_DEPTH",
            "provider_daily_remaining_after_call": provider_daily_remaining,
            "historical_research_feature_reconstruction_allowed": True,
            "retroactive_prediction_rewrite": False,
            "retroactive_market_created": False,
            "retroactive_bet_created": False,
            "production_decision_weight": 0.0,
        },
    }


def _persist_event(conn: Any, event: dict[str, Any], *, generated_at: datetime) -> None:
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
                "FM4_TACTICAL_BACKFILL",
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
    team_ids: list[int],
    before: str,
    lookback_days: int = 365,
    max_fixtures: int = MAX_FIXTURES_PER_RUN,
) -> dict[str, Any]:
    unique_team_ids = sorted(
        {
            int(value)
            for value in team_ids
            if value is not None and int(value) > 0
        }
    )[:MAX_TEAM_IDS]
    if not unique_team_ids:
        raise ValueError("team_ids required")
    before_dt = _parse_dt(before)
    lookback_days = max(30, min(int(lookback_days), 730))
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
    daily_remaining = v2._LAST_DAILY_REMAINING
    attempted = captured = incomplete = errors = 0
    details: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    last_request_started = 0.0

    with persistence_base._connect() as conn:
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
                    "fixtures/statistics",
                    {"fixture": fixture_id},
                )
                remaining = (raw.get("quota") or {}).get("daily_remaining")
                try:
                    if remaining is not None:
                        daily_remaining = int(remaining)
                        v2._LAST_DAILY_REMAINING = daily_remaining
                except (TypeError, ValueError):
                    pass

                compact = compact_tactical_stats(raw)
                if not _sufficient_stats(compact):
                    incomplete += 1
                    details.append(
                        {
                            "fixture_id": fixture_id,
                            "status": "PROVIDER_STATS_INCOMPLETE",
                            "team_rows": len(compact.get("teams") or []),
                            "daily_remaining": daily_remaining,
                        }
                    )
                    continue

                retrieved_at = datetime.now(timezone.utc)
                event = make_event(
                    candidate,
                    compact,
                    retrieved_at=retrieved_at,
                    provider_daily_remaining=daily_remaining,
                )
                _persist_event(conn, event, generated_at=retrieved_at)
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
        "status": "FM4_TACTICAL_HISTORY_BACKFILL_COMPLETE",
        "before": before_dt.isoformat(),
        "lookback_days": lookback_days,
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
        "max_provider_requests_per_run": MAX_FIXTURES_PER_RUN,
        "daily_remaining_after_run": daily_remaining,
        "historical_research_feature_reconstruction_allowed": True,
        "retroactive_prediction_rewrite": False,
        "retroactive_market_created": False,
        "retroactive_bet_created": False,
        "decision_weight": 0.0,
        "production_promotion_allowed": False,
        "policy": (
            "ONLY FINALIZED FIXTURES STRICTLY BEFORE THE FM4 TARGET COHORT; "
            "EXACT PROVIDER FIXTURE STATISTICS ONLY; MAX EIGHT PROVIDER REQUESTS PER RUN; "
            "STOP AT DAILY RESERVE GUARD; PERSIST RETRIEVAL PROVENANCE; "
            "MAY EXPAND OFFLINE HISTORICAL FEATURE RESEARCH BUT NEVER REWRITE ORIGINAL "
            "PREGAME PREDICTIONS, MARKETS, CLV, BETS, OR PRODUCTION DECISIONS."
        ),
    }
