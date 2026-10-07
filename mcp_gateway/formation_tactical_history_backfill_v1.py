from __future__ import annotations

import asyncio
import json
import math
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any

from mcp_gateway import automation as base
from mcp_gateway import automation_v2 as v2
from mcp_gateway import persistence as persistence_base

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_FM4_TACTICAL_HISTORY_BACKFILL_V1.1.0"
RESEARCH_KEY = "FM4_TACTICAL_HISTORY_V1"
MAX_FIXTURES_PER_RUN = 8
MAX_DISCOVERY_TEAM_CALLS_PER_RUN = 8
MIN_DAILY_REMAINING = 250
MIN_REQUEST_INTERVAL_SECONDS = 0.8
MIN_PRIOR_MATCHES_TARGET = 3
MAX_TEAM_IDS = 800
MAX_CANDIDATE_ROWS = 5000
FINAL_STATUSES = {"FT", "AET", "PEN"}

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


def _local_prior_fixture_counts(
    conn: Any,
    *,
    team_ids: list[int],
    before: datetime,
    lookback_days: int,
) -> Counter[int]:
    cutoff = before - timedelta(days=lookback_days)
    counts: Counter[int] = Counter()
    target = set(team_ids)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT f.home_team_id, f.away_team_id
            FROM soccer_fixtures f
            JOIN soccer_results r ON r.fixture_id = f.fixture_id
            WHERE f.kickoff >= %s
              AND f.kickoff < %s
              AND r.final_status = ANY(%s)
              AND (
                    f.home_team_id = ANY(%s)
                 OR f.away_team_id = ANY(%s)
              )
            """,
            (cutoff, before, list(FINAL_STATUSES), team_ids, team_ids),
        )
        for home_raw, away_raw in cur.fetchall():
            for raw in (home_raw, away_raw):
                try:
                    team_id = int(raw)
                except (TypeError, ValueError):
                    continue
                if team_id in target:
                    counts[team_id] += 1
    return counts


def _source_season_by_team(
    conn: Any,
    *,
    team_ids: list[int],
    before: datetime,
) -> dict[int, int]:
    """Resolve each target team's nearest canonical cohort season."""
    target = set(team_ids)
    rows: list[tuple[Any, Any, Any, Any]] = []
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT home_team_id, away_team_id, season, kickoff
            FROM soccer_fixtures
            WHERE season IS NOT NULL
              AND kickoff >= %s
              AND kickoff < %s
              AND (
                    home_team_id = ANY(%s)
                 OR away_team_id = ANY(%s)
              )
            ORDER BY kickoff ASC
            """,
            (
                before - timedelta(days=14),
                before + timedelta(days=45),
                team_ids,
                team_ids,
            ),
        )
        rows = cur.fetchall()

    best: dict[int, tuple[float, int]] = {}
    for home_raw, away_raw, season_raw, kickoff_raw in rows:
        try:
            season = int(season_raw)
            kickoff = _parse_dt(kickoff_raw)
        except (TypeError, ValueError):
            continue
        distance = abs((kickoff - before).total_seconds())
        for raw in (home_raw, away_raw):
            try:
                team_id = int(raw)
            except (TypeError, ValueError):
                continue
            if team_id not in target:
                continue
            current = best.get(team_id)
            if current is None or distance < current[0]:
                best[team_id] = (distance, season)
    return {team_id: season for team_id, (_distance, season) in best.items()}


def _already_discovered_teams(conn: Any, *, before: datetime) -> set[int]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT team_id
            FROM soccer_research_backfill_discovery
            WHERE research_key = %s
              AND cutoff = %s
              AND provider_status = 'OK'
            """,
            (RESEARCH_KEY, before),
        )
        return {int(row[0]) for row in cur.fetchall()}


def _select_discovery_teams(
    team_ids: list[int],
    *,
    local_counts: Counter[int],
    already_discovered: set[int],
    max_calls: int,
) -> list[int]:
    candidates = [
        int(team_id)
        for team_id in team_ids
        if int(team_id) not in already_discovered
        and local_counts[int(team_id)] < MIN_PRIOR_MATCHES_TARGET
    ]
    candidates.sort(key=lambda team_id: (local_counts[team_id], team_id))
    return candidates[: max(0, int(max_calls))]


def _eligible_discovered_fixture(
    row: dict[str, Any],
    *,
    team_id: int,
    before: datetime,
    lookback_days: int,
) -> dict[str, Any] | None:
    compact = base._compact_fixture(row)
    try:
        fixture_id = int(compact.get("fixture_id") or 0)
        kickoff = _parse_dt(compact.get("kickoff"))
        home_team_id = int(compact.get("home_team_id") or 0)
        away_team_id = int(compact.get("away_team_id") or 0)
    except (TypeError, ValueError):
        return None
    if not fixture_id:
        return None
    if str(compact.get("status") or "").upper() not in FINAL_STATUSES:
        return None
    if not (before - timedelta(days=lookback_days) <= kickoff < before):
        return None
    if team_id not in {home_team_id, away_team_id}:
        return None
    compact["kickoff"] = kickoff
    return compact


def _persist_discovered_fixture(
    conn: Any,
    *,
    fixture: dict[str, Any],
    retrieved_at: datetime,
) -> None:
    fixture_id = int(fixture["fixture_id"])
    goals = fixture.get("goals") if isinstance(fixture.get("goals"), dict) else {}
    score = fixture.get("score") if isinstance(fixture.get("score"), dict) else {}
    with conn.cursor() as cur:
        persistence_base._upsert_fixture(cur, fixture)
        cur.execute(
            """
            INSERT INTO soccer_results (
                fixture_id, final_status, home_goals, away_goals,
                final_score, match_stats, graded_at, payload
            ) VALUES (%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s::jsonb)
            ON CONFLICT (fixture_id) DO NOTHING
            """,
            (
                fixture_id,
                fixture.get("status"),
                goals.get("home"),
                goals.get("away"),
                json.dumps(score),
                json.dumps([]),
                retrieved_at,
                json.dumps(
                    {
                        "source": "API_FOOTBALL_FM4_HISTORICAL_DISCOVERY",
                        "fixture": {
                            key: (
                                value.isoformat()
                                if isinstance(value, datetime)
                                else value
                            )
                            for key, value in fixture.items()
                        },
                        "retrieved_at": retrieved_at.isoformat(),
                    }
                ),
            ),
        )


def _record_discovery(
    conn: Any,
    *,
    team_id: int,
    before: datetime,
    lookback_days: int,
    provider_status: str,
    provider_fixture_count: int,
    persisted_fixture_count: int,
    payload: dict[str, Any],
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO soccer_research_backfill_discovery (
                research_key, team_id, cutoff, lookback_days, attempted_at,
                provider_status, provider_fixture_count, persisted_fixture_count, payload
            ) VALUES (%s,%s,%s,%s,NOW(),%s,%s,%s,%s::jsonb)
            ON CONFLICT (research_key, team_id, cutoff) DO UPDATE SET
                lookback_days = EXCLUDED.lookback_days,
                attempted_at = NOW(),
                provider_status = EXCLUDED.provider_status,
                provider_fixture_count = EXCLUDED.provider_fixture_count,
                persisted_fixture_count = EXCLUDED.persisted_fixture_count,
                payload = EXCLUDED.payload
            """,
            (
                RESEARCH_KEY,
                team_id,
                before,
                lookback_days,
                provider_status,
                provider_fixture_count,
                persisted_fixture_count,
                json.dumps(payload),
            ),
        )


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
    max_discovery_teams: int = MAX_DISCOVERY_TEAM_CALLS_PER_RUN,
    team_seasons: dict[Any, Any] | None = None,
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
    max_discovery_teams = max(
        0,
        min(int(max_discovery_teams), MAX_DISCOVERY_TEAM_CALLS_PER_RUN),
    )
    provided_team_seasons: dict[int, int] = {}
    for team_raw, season_raw in (team_seasons or {}).items():
        try:
            team_id = int(team_raw)
            season = int(season_raw)
        except (TypeError, ValueError):
            continue
        if team_id in unique_team_ids and season > 0:
            provided_team_seasons[team_id] = season

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
    discovery_attempted = discovery_success = discovery_errors = 0
    discovered_fixture_rows = 0
    details: list[dict[str, Any]] = []
    discovery_details: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    last_request_started = 0.0

    async def provider_get(endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        nonlocal daily_remaining, last_request_started
        if daily_remaining is not None and daily_remaining <= MIN_DAILY_REMAINING:
            raise RuntimeError("DAILY_PROVIDER_RESERVE_GUARD")
        wait_for = MIN_REQUEST_INTERVAL_SECONDS - (
            time.monotonic() - last_request_started
        )
        if wait_for > 0:
            await asyncio.sleep(wait_for)
        last_request_started = time.monotonic()
        raw = await v2._ORIGINAL_API_GET(endpoint, params)
        remaining = (raw.get("quota") or {}).get("daily_remaining")
        try:
            if remaining is not None:
                daily_remaining = int(remaining)
                v2._LAST_DAILY_REMAINING = daily_remaining
        except (TypeError, ValueError):
            pass
        return raw

    with persistence_base._connect() as conn:
        target_set = set(unique_team_ids)
        prior_counts = _existing_team_counts(conn, target_set)
        pool = _candidate_rows(
            conn,
            team_ids=unique_team_ids,
            before=before_dt,
            lookback_days=lookback_days,
        )

        # If the local canonical fixture/result store is too shallow, discover
        # finalized historical fixtures directly from the same provider. This is
        # fixture identity/result ingestion only; no sporting feature is inferred.
        if len(pool) < max_fixtures and max_discovery_teams > 0:
            local_counts = _local_prior_fixture_counts(
                conn,
                team_ids=unique_team_ids,
                before=before_dt,
                lookback_days=lookback_days,
            )
            already_discovered = _already_discovered_teams(conn, before=before_dt)
            source_seasons = _source_season_by_team(
                conn,
                team_ids=unique_team_ids,
                before=before_dt,
            )
            # Canonical formation research state may carry the verified fixture
            # season even when the live Postgres fixture table is sparse. Prefer
            # those explicit values over inference.
            source_seasons.update(provided_team_seasons)
            discovery_team_ids = _select_discovery_teams(
                unique_team_ids,
                local_counts=local_counts,
                already_discovered=already_discovered,
                max_calls=max_discovery_teams,
            )
            discovery_from = (before_dt - timedelta(days=lookback_days)).date().isoformat()
            discovery_to = (before_dt - timedelta(seconds=1)).date().isoformat()

            for team_id in discovery_team_ids:
                if daily_remaining is not None and daily_remaining <= MIN_DAILY_REMAINING:
                    discovery_details.append(
                        {
                            "team_id": team_id,
                            "status": "SKIPPED_DAILY_RESERVE_GUARD",
                        }
                    )
                    break
                season = source_seasons.get(team_id)
                if season is None:
                    discovery_details.append(
                        {
                            "team_id": team_id,
                            "status": "SKIPPED_SOURCE_SEASON_NOT_VERIFIED",
                        }
                    )
                    continue
                discovery_attempted += 1
                try:
                    raw = await provider_get(
                        "fixtures",
                        {
                            "team": team_id,
                            "season": season,
                            "from": discovery_from,
                            "to": discovery_to,
                            "timezone": "UTC",
                        },
                    )
                    provider_rows = [
                        row for row in (raw.get("response") or [])
                        if isinstance(row, dict)
                    ]
                    persisted_for_team = 0
                    for row in provider_rows:
                        fixture = _eligible_discovered_fixture(
                            row,
                            team_id=team_id,
                            before=before_dt,
                            lookback_days=lookback_days,
                        )
                        if fixture is None:
                            continue
                        _persist_discovered_fixture(
                            conn,
                            fixture=fixture,
                            retrieved_at=datetime.now(timezone.utc),
                        )
                        persisted_for_team += 1
                    discovered_fixture_rows += persisted_for_team
                    discovery_success += 1
                    _record_discovery(
                        conn,
                        team_id=team_id,
                        before=before_dt,
                        lookback_days=lookback_days,
                        provider_status="OK",
                        provider_fixture_count=len(provider_rows),
                        persisted_fixture_count=persisted_for_team,
                        payload={
                            "endpoint": "fixtures",
                            "params": {
                                "team": team_id,
                                "season": season,
                                "from": discovery_from,
                                "to": discovery_to,
                                "timezone": "UTC",
                            },
                            "daily_remaining_after_call": daily_remaining,
                        },
                    )
                    discovery_details.append(
                        {
                            "team_id": team_id,
                            "season": season,
                            "status": "DISCOVERED",
                            "provider_fixture_count": len(provider_rows),
                            "persisted_fixture_count": persisted_for_team,
                            "daily_remaining": daily_remaining,
                        }
                    )
                except Exception as exc:
                    discovery_errors += 1
                    discovery_details.append(
                        {
                            "team_id": team_id,
                            "season": season,
                            "status": "DISCOVERY_ERROR",
                            "error": str(exc)[:180],
                            "daily_remaining": daily_remaining,
                        }
                    )

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

            attempted += 1
            try:
                raw = await provider_get(
                    "fixtures/statistics",
                    {"fixture": fixture_id},
                )
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
        "provided_team_season_count": len(provided_team_seasons),
        "candidate_pool_rows": len(pool),
        "selected_fixture_count": len(candidates),
        "fixture_discovery": {
            "max_team_calls_per_run": MAX_DISCOVERY_TEAM_CALLS_PER_RUN,
            "attempted_team_calls": discovery_attempted,
            "successful_team_calls": discovery_success,
            "errors": discovery_errors,
            "persisted_fixture_rows": discovered_fixture_rows,
            "details": discovery_details,
        },
        "attempted": attempted,
        "captured": captured,
        "incomplete": incomplete,
        "errors": errors,
        "details": details,
        "events": events,
        "fixture_discovery_provider_requests_added": discovery_attempted,
        "statistics_provider_requests_added": attempted,
        "provider_requests_added": discovery_attempted + attempted,
        "max_statistics_provider_requests_per_run": MAX_FIXTURES_PER_RUN,
        "max_discovery_provider_requests_per_run": MAX_DISCOVERY_TEAM_CALLS_PER_RUN,
        "daily_remaining_after_run": daily_remaining,
        "historical_research_feature_reconstruction_allowed": True,
        "retroactive_prediction_rewrite": False,
        "retroactive_market_created": False,
        "retroactive_bet_created": False,
        "decision_weight": 0.0,
        "production_promotion_allowed": False,
        "policy": (
            "DISCOVER ONLY VERIFIED FINALIZED FIXTURES STRICTLY BEFORE THE FM4 TARGET COHORT; "
            "PERSIST FIXTURE/RESULT IDENTITY IN CANONICAL POSTGRES; "
            "EXACT PROVIDER FIXTURE STATISTICS ONLY; MAX EIGHT STATISTICS REQUESTS PER RUN; "
            "MAX EIGHT FIXTURE-DISCOVERY TEAM REQUESTS PER RUN; STOP AT DAILY RESERVE GUARD; "
            "PERSIST DISCOVERY AND RETRIEVAL PROVENANCE; NO SYNTHETIC TACTICAL VALUES; "
            "MAY EXPAND OFFLINE HISTORICAL FEATURE RESEARCH BUT NEVER REWRITE ORIGINAL "
            "PREGAME PREDICTIONS, MARKETS, CLV, BETS, OR PRODUCTION DECISIONS."
        ),
    }
