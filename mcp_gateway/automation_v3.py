from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone as dt_timezone
from typing import Any

import httpx

from mcp_gateway import automation as base
from mcp_gateway import automation_v2 as v2

_BULK_COVERAGE: dict[str, dict[str, Any]] | None = None


def _coverage_flags_from_season(season_row: dict[str, Any]) -> dict[str, Any]:
    coverage = season_row.get("coverage") or {}
    fixtures = coverage.get("fixtures") or {}
    lineups = bool(fixtures.get("lineups"))
    statistics = bool(fixtures.get("statistics_fixtures"))
    injuries = bool(coverage.get("injuries"))
    odds = bool(coverage.get("odds"))
    if lineups and statistics and odds and injuries:
        tier = "A"
    elif lineups and statistics and odds:
        tier = "B"
    elif statistics or lineups:
        tier = "C"
    else:
        tier = "D"
    return {
        "known": True,
        "events": bool(fixtures.get("events")),
        "lineups": lineups,
        "statistics_fixtures": statistics,
        "statistics_players": bool(fixtures.get("statistics_players")),
        "injuries": injuries,
        "predictions": bool(coverage.get("predictions")),
        "odds": odds,
        "data_tier": tier,
    }


async def _bulk_coverage(now: datetime) -> dict[str, dict[str, Any]]:
    global _BULK_COVERAGE
    if _BULK_COVERAGE is not None:
        return _BULK_COVERAGE

    cached = base._cache_get("bulk_coverage", "all", timedelta(days=3), now)
    if isinstance(cached, dict) and cached:
        _BULK_COVERAGE = cached
        return cached

    payload = await base._api_get("leagues", {})
    mapping: dict[str, dict[str, Any]] = {}
    for row in payload.get("response", []) or []:
        league = row.get("league") or {}
        league_id = league.get("id")
        if not league_id:
            continue
        for season_row in row.get("seasons") or []:
            year = season_row.get("year")
            if year is None:
                continue
            mapping[f"{league_id}:{year}"] = _coverage_flags_from_season(season_row)

    base._cache_set("bulk_coverage", "all", mapping, now)
    _BULK_COVERAGE = mapping
    return mapping


async def _coverage_fast(league_id: int, season: int, now: datetime) -> dict[str, Any]:
    key = f"{league_id}:{season}"
    cached = base._cache_get("coverage", key, timedelta(days=7), now)
    if isinstance(cached, dict):
        return cached
    mapping = await _bulk_coverage(now)
    flags = mapping.get(key) or {"known": False, "data_tier": "D"}
    base._cache_set("coverage", key, flags, now)
    return flags


def _had_pregame_interest(fixture_id: int, now: datetime) -> bool:
    conn = base._cache_conn()
    cutoff = (now - timedelta(hours=12)).timestamp()
    row = conn.execute(
        """
        SELECT 1 FROM processed_stages
        WHERE fixture_id=?
          AND stage IN ('T-90','T-60','T-40','T-30','T-20','T-10','CLOSE')
          AND updated_at >= ?
        LIMIT 1
        """,
        (fixture_id, cutoff),
    ).fetchone()
    return bool(row)


def _daily_discovery_done(run_date: str, now: datetime) -> bool:
    cached = base._cache_get("daily_discovery", run_date, timedelta(days=2), now)
    return cached == "done"


def _mark_daily_discovery(run_date: str, now: datetime) -> None:
    base._cache_set("daily_discovery", run_date, "done", now)


async def _daily_discovery_event(
    fixtures: list[dict[str, Any]], now_utc: datetime, local_now: datetime
) -> dict[str, Any] | None:
    run_date = local_now.date().isoformat()
    if _daily_discovery_done(run_date, now_utc):
        return None

    upcoming: list[dict[str, Any]] = []
    tier_counts: Counter[str] = Counter()
    league_counts: Counter[str] = Counter()

    for fx in fixtures:
        kickoff = base._dt(fx["kickoff"])
        if kickoff < now_utc:
            continue
        if fx.get("status") in base.CANCELLED_STATUSES | base.POSTPONED_STATUSES:
            continue
        coverage = await _coverage_fast(fx["league_id"], fx["season"], now_utc)
        tier = coverage.get("data_tier") or "D"
        tier_counts[tier] += 1
        league_counts[f"{fx.get('country') or ''} | {fx.get('league') or ''}"] += 1
        if tier in {"A", "B", "C"}:
            item = dict(fx)
            item["data_tier"] = tier
            upcoming.append(item)

    upcoming.sort(key=lambda x: x.get("kickoff") or "")
    # Compact state remains bounded. Counts describe the complete slate; the
    # attached universe prioritizes data-eligible A/B/C fixtures by kickoff.
    attached = upcoming[:300]
    _mark_daily_discovery(run_date, now_utc)

    return {
        "event_type": "DAILY_DISCOVERY",
        "stage": "MORNING" if local_now.hour < 12 else "LATE_DISCOVERY",
        "date": run_date,
        "timezone": base.TIMEZONE_NAME,
        "upcoming_count": sum(tier_counts.values()),
        "data_tier_counts": dict(tier_counts),
        "analysis_universe_count": len(upcoming),
        "attached_fixture_count": len(attached),
        "truncated": len(upcoming) > len(attached),
        "fixtures": attached,
        "top_competitions_by_fixture_count": [
            {"competition": name, "fixtures": count}
            for name, count in league_counts.most_common(30)
        ],
        "classification": "PRE-FINAL",
        "sport_first": True,
        "market_data_included": False,
        "model_version": "SOCCER EDGE ENGINE v1.0",
        "notes": [
            "Complete current slate counted using API-Football fixtures and bulk competition coverage.",
            "Only Data Tier A/B/C fixtures are attached for sporting analysis; detailed markets remain downstream of the sporting screen.",
            "Daily discovery is emitted once per Mexico City calendar date even if the service starts after 06:00.",
        ],
    }


async def run_tick() -> dict[str, Any]:
    v2._API_CALLS_THIS_TICK = 0
    v2._LAST_DAILY_REMAINING = None

    now_utc = datetime.now(dt_timezone.utc)
    local_now = now_utc.astimezone(base.TIMEZONE)
    base._prune_cache(now_utc)

    dates = [local_now.date()]
    if local_now.hour >= 22:
        dates.append((local_now + timedelta(days=1)).date())

    fixtures: list[dict[str, Any]] = []
    quota: dict[str, Any] = {}

    # Replace per-league coverage calls with a single cached bulk map.
    original_coverage = base._coverage
    base._coverage = _coverage_fast

    base._HTTP_CLIENT = httpx.AsyncClient(
        timeout=base.TIMEOUT,
        limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
    )
    try:
        for d in dates:
            payload = await base._api_get("fixtures", {"date": d.isoformat(), "timezone": base.TIMEZONE_NAME})
            quota = payload.get("quota", quota)
            for row in payload.get("response", []):
                fx = base._compact_fixture(row)
                if fx.get("fixture_id") and fx.get("kickoff") and fx.get("league_id") and fx.get("season"):
                    fixtures.append(fx)

        events: list[dict[str, Any]] = []

        discovery = await _daily_discovery_event(fixtures, now_utc, local_now)
        if discovery is not None:
            events.append(discovery)

        due: list[tuple[int, datetime, dict[str, Any], str]] = []
        priority = {"T-40": 0, "T-20": 1, "T-10": 2, "CLOSE": 3, "T-60": 4, "T-90": 5, "T-30": 6, "POSTGAME": 7}
        for fx in fixtures:
            kickoff = base._dt(fx["kickoff"])
            minutes_to = (kickoff - now_utc).total_seconds() / 60.0
            stage = base._stage_for(minutes_to, fx.get("status") or "")
            if not stage:
                continue
            if stage == "POSTGAME":
                minutes_since = -minutes_to
                if minutes_since < 95 or minutes_since > 240:
                    continue
                # Grade only matches that actually entered our pregame funnel.
                if not _had_pregame_interest(fx["fixture_id"], now_utc):
                    continue
            due.append((priority.get(stage, 99), kickoff, fx, stage))

        due.sort(key=lambda item: (item[0], item[1]))
        deferred_due_to_budget = 0
        for _, _, fx, stage in due:
            if not base._dedupe_stage(fx["fixture_id"], stage, now_utc):
                continue
            try:
                events.append(await v2._event_for_fixture(fx, stage, now_utc))
            except v2.TickBudgetExceeded as exc:
                deferred_due_to_budget += 1
                events.append({
                    "event_type": "QUOTA_GUARD",
                    "stage": stage,
                    "fixture": fx,
                    "model_version": "SOCCER EDGE ENGINE v1.0",
                    "classification": "WATCH",
                    "error": str(exc),
                })
                break
            except Exception as exc:
                events.append({
                    "event_type": "PIPELINE_ERROR",
                    "stage": stage,
                    "fixture": fx,
                    "model_version": "SOCCER EDGE ENGINE v1.0",
                    "classification": "WATCH",
                    "error": str(exc)[:500],
                })

        actionable = [e for e in events if e.get("stage") in {"T-40", "T-20", "T-10", "CLOSE"}]
        bets = [e for e in events if e.get("classification") == "BET"]
        return {
            "service": "soccer-edge-automation",
            "version": "1.3.0",
            "model_version": "SOCCER EDGE ENGINE v1.0",
            "generated_at_utc": now_utc.isoformat(),
            "generated_at_local": local_now.isoformat(),
            "timezone": base.TIMEZONE_NAME,
            "fixture_scan_count": len(fixtures),
            "due_fixture_count": len(due),
            "event_count": len(events),
            "actionable_refresh_count": len(actionable),
            "bet_candidate_count": len(bets),
            "api_calls_this_tick": v2._API_CALLS_THIS_TICK,
            "max_api_calls_per_tick": v2.MAX_API_CALLS_PER_TICK,
            "last_daily_remaining": v2._LAST_DAILY_REMAINING,
            "deferred_due_to_budget": deferred_due_to_budget,
            "events": events,
            "quota": quota,
            "database_persistence": "OPTIONAL_NOT_REQUIRED_FOR_SCHEDULER",
        }
    finally:
        base._coverage = original_coverage
        if base._HTTP_CLIENT is not None:
            await base._HTTP_CLIENT.aclose()
            base._HTTP_CLIENT = None
        if base._CACHE_CONN is not None:
            base._CACHE_CONN.commit()
