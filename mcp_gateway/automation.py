import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta, timezone as dt_timezone
from zoneinfo import ZoneInfo
from typing import Any

import httpx

API_BASE_URL = os.getenv("API_FOOTBALL_BASE_URL", "https://v3.football.api-sports.io").rstrip("/")
TIMEZONE_NAME = os.getenv("SOCCER_TIMEZONE", "America/Mexico_City")
TIMEZONE = ZoneInfo(TIMEZONE_NAME)
TIMEOUT = float(os.getenv("API_FOOTBALL_TIMEOUT", "20"))
CACHE_DB_PATH = os.getenv("SOCCER_EDGE_CACHE_PATH", "/tmp/soccer_edge_cache.sqlite3")

# The heavy scheduler runs in a short-lived child process. Persistent local SQLite
# keeps API caches and stage de-duplication across child-process runs without
# growing the long-lived web process RSS.
_CACHE_CONN: sqlite3.Connection | None = None
_HTTP_CLIENT: httpx.AsyncClient | None = None

logging.getLogger("httpx").setLevel(logging.WARNING)

FINISHED_STATUSES = {"FT", "AET", "PEN"}
CANCELLED_STATUSES = {"CANC", "ABD", "AWD", "WO"}
POSTPONED_STATUSES = {"PST", "SUSP", "INT"}


def _api_key() -> str:
    key = os.getenv("API_FOOTBALL_KEY", "").strip()
    if not key:
        raise RuntimeError("API_FOOTBALL_KEY is not configured")
    return key


def _cache_conn() -> sqlite3.Connection:
    global _CACHE_CONN
    if _CACHE_CONN is None:
        conn = sqlite3.connect(CACHE_DB_PATH, timeout=15)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_entries (
                namespace TEXT NOT NULL,
                cache_key TEXT NOT NULL,
                updated_at REAL NOT NULL,
                value_json TEXT NOT NULL,
                PRIMARY KEY(namespace, cache_key)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS processed_stages (
                fixture_id INTEGER NOT NULL,
                stage TEXT NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY(fixture_id, stage)
            )
            """
        )
        conn.commit()
        _CACHE_CONN = conn
    return _CACHE_CONN


def _cache_get(namespace: str, key: str, ttl: timedelta, now: datetime) -> Any | None:
    conn = _cache_conn()
    row = conn.execute(
        "SELECT updated_at, value_json FROM cache_entries WHERE namespace=? AND cache_key=?",
        (namespace, key),
    ).fetchone()
    if not row:
        return None
    if now.timestamp() - float(row[0]) > ttl.total_seconds():
        conn.execute("DELETE FROM cache_entries WHERE namespace=? AND cache_key=?", (namespace, key))
        conn.commit()
        return None
    try:
        return json.loads(row[1])
    except (TypeError, json.JSONDecodeError):
        conn.execute("DELETE FROM cache_entries WHERE namespace=? AND cache_key=?", (namespace, key))
        conn.commit()
        return None


def _cache_set(namespace: str, key: str, value: Any, now: datetime) -> None:
    conn = _cache_conn()
    conn.execute(
        """
        INSERT INTO cache_entries(namespace, cache_key, updated_at, value_json)
        VALUES(?,?,?,?)
        ON CONFLICT(namespace, cache_key) DO UPDATE SET
            updated_at=excluded.updated_at,
            value_json=excluded.value_json
        """,
        (namespace, key, now.timestamp(), json.dumps(value, separators=(",", ":"))),
    )
    conn.commit()


def _prune_cache(now: datetime) -> None:
    conn = _cache_conn()
    cutoff = (now - timedelta(days=8)).timestamp()
    conn.execute("DELETE FROM cache_entries WHERE updated_at < ?", (cutoff,))
    conn.execute("DELETE FROM processed_stages WHERE updated_at < ?", ((now - timedelta(hours=12)).timestamp(),))
    conn.commit()


async def _api_get(endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
    headers = {"x-apisports-key": _api_key(), "Accept": "application/json"}
    client = _HTTP_CLIENT
    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(timeout=TIMEOUT)
    assert client is not None
    try:
        response = await client.get(f"{API_BASE_URL}/{endpoint.lstrip('/')}", headers=headers, params=params)
        response.raise_for_status()
        payload = response.json()
    finally:
        if owns_client:
            await client.aclose()
    if payload.get("errors"):
        raise RuntimeError(f"API-Football error on {endpoint}: {payload['errors']}")
    return {
        "response": payload.get("response", []),
        "results": payload.get("results", 0),
        "paging": payload.get("paging", {}),
        "quota": {
            "daily_limit": response.headers.get("x-ratelimit-requests-limit"),
            "daily_remaining": response.headers.get("x-ratelimit-requests-remaining"),
            "per_minute_remaining": response.headers.get("X-RateLimit-Remaining"),
        },
    }


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _compact_fixture(item: dict[str, Any]) -> dict[str, Any]:
    fixture = item.get("fixture", {})
    league = item.get("league", {})
    teams = item.get("teams", {})
    status = fixture.get("status", {})
    venue = fixture.get("venue", {}) or {}
    return {
        "fixture_id": fixture.get("id"),
        "kickoff": fixture.get("date"),
        "timestamp": fixture.get("timestamp"),
        "status": status.get("short"),
        "status_long": status.get("long"),
        "elapsed": status.get("elapsed"),
        "league_id": league.get("id"),
        "league": league.get("name"),
        "country": league.get("country"),
        "season": league.get("season"),
        "round": league.get("round"),
        "home_team_id": (teams.get("home") or {}).get("id"),
        "home_team": (teams.get("home") or {}).get("name"),
        "away_team_id": (teams.get("away") or {}).get("id"),
        "away_team": (teams.get("away") or {}).get("name"),
        "venue": venue.get("name"),
        "city": venue.get("city"),
        "goals": item.get("goals"),
        "score": item.get("score"),
    }


def _coverage_flags(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("response") or []
    if not rows:
        return {"known": False, "data_tier": "D"}
    seasons = rows[0].get("seasons") or []
    coverage = seasons[0].get("coverage", {}) if seasons else {}
    fixtures = coverage.get("fixtures", {}) or {}
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


async def _coverage(league_id: int, season: int, now: datetime) -> dict[str, Any]:
    key = f"{league_id}:{season}"
    cached = _cache_get("coverage", key, timedelta(days=7), now)
    if isinstance(cached, dict):
        return cached
    payload = await _api_get("leagues", {"id": league_id, "season": season})
    flags = _coverage_flags(payload)
    _cache_set("coverage", key, flags, now)
    return flags


def _compact_team_stats(payload: dict[str, Any]) -> dict[str, Any]:
    r = payload.get("response") or {}
    fixtures = r.get("fixtures") or {}
    goals = r.get("goals") or {}
    return {
        "form": r.get("form"),
        "fixtures": fixtures,
        "goals": {"for": goals.get("for"), "against": goals.get("against")},
        "clean_sheet": r.get("clean_sheet") or {},
        "failed_to_score": r.get("failed_to_score") or {},
        "biggest": r.get("biggest") or {},
        "lineups": (r.get("lineups") or [])[:20],
    }


async def _team_stats(team_id: int, league_id: int, season: int, now: datetime) -> dict[str, Any]:
    key = f"{team_id}:{league_id}:{season}"
    cached = _cache_get("team_stats", key, timedelta(hours=6), now)
    if isinstance(cached, dict):
        return cached
    payload = await _api_get("teams/statistics", {"team": team_id, "league": league_id, "season": season})
    compact = _compact_team_stats(payload)
    _cache_set("team_stats", key, compact, now)
    return compact


async def _recent(team_id: int, now: datetime) -> list[dict[str, Any]]:
    key = str(team_id)
    cached = _cache_get("recent", key, timedelta(hours=6), now)
    if isinstance(cached, list):
        return cached
    payload = await _api_get("fixtures", {"team": team_id, "last": 8, "timezone": TIMEZONE_NAME})
    matches = [_compact_fixture(x) for x in payload.get("response", [])]
    _cache_set("recent", key, matches, now)
    return matches


def _compact_injuries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for row in payload.get("response", []):
        player = row.get("player") or {}
        team = row.get("team") or {}
        out.append({
            "player_id": player.get("id"),
            "player": player.get("name"),
            "team_id": team.get("id"),
            "team": team.get("name"),
            "type": player.get("type"),
            "reason": player.get("reason"),
        })
    return out


def _compact_lineups(payload: dict[str, Any]) -> dict[str, Any]:
    teams = []
    for row in payload.get("response", []):
        team = row.get("team") or {}
        coach = row.get("coach") or {}
        starters = []
        goalkeepers = []
        for p in row.get("startXI") or []:
            player = p.get("player") or {}
            compact = {
                "id": player.get("id"),
                "name": player.get("name"),
                "number": player.get("number"),
                "pos": player.get("pos"),
                "grid": player.get("grid"),
            }
            starters.append(compact)
            if player.get("pos") == "G":
                goalkeepers.append(compact)
        teams.append({
            "team_id": team.get("id"),
            "team": team.get("name"),
            "formation": row.get("formation"),
            "coach_id": coach.get("id"),
            "coach": coach.get("name"),
            "starters": starters,
            "goalkeepers": goalkeepers,
            "substitutes_count": len(row.get("substitutes") or []),
        })
    both_xi = len(teams) == 2 and all(len(t.get("starters", [])) >= 11 for t in teams)
    both_gk = len(teams) == 2 and all(len(t.get("goalkeepers", [])) >= 1 for t in teams)
    return {
        "teams": teams,
        "both_xi_confirmed": both_xi,
        "both_goalkeepers_confirmed": both_gk,
        "lineup_state": "CONFIRMED_API" if both_xi else "PENDING",
    }


def _wanted_market(name: str) -> bool:
    n = (name or "").lower()
    keys = (
        "match winner", "winner", "goals over/under", "over/under",
        "asian handicap", "handicap", "both teams score", "both teams to score",
        "team total", "corners", "corner",
    )
    return any(k in n for k in keys)


def _is_ft_team_total_bet(bet: dict[str, Any]) -> bool:
    try:
        market_id = int(bet.get("id")) if bet.get("id") is not None else None
    except (TypeError, ValueError):
        market_id = None
    if market_id in {16, 17}:
        return True
    n = " ".join(str(bet.get("name") or "").strip().lower().split())
    return n in {
        "total - home",
        "total home",
        "total - away",
        "total away",
        "home team total goals",
        "away team total goals",
        "home team goals over/under",
        "away team goals over/under",
    }


def _compact_odds(payload: dict[str, Any]) -> dict[str, Any]:
    primary_rows: list[dict[str, Any]] = []
    team_total_rows: list[dict[str, Any]] = []
    for fixture_row in payload.get("response", []):
        update = fixture_row.get("update")
        for book in fixture_row.get("bookmakers") or []:
            for bet in book.get("bets") or []:
                name = bet.get("name") or ""
                is_team_total = _is_ft_team_total_bet(bet)
                if not is_team_total and not _wanted_market(name):
                    continue
                values = [
                    {"selection": value.get("value"), "price": value.get("odd")}
                    for value in (bet.get("values") or [])
                ]
                row = {
                    "bookmaker_id": book.get("id"),
                    "bookmaker": book.get("name"),
                    "market_id": bet.get("id"),
                    "market": name,
                    "values": values,
                    "provider_update": update,
                }
                if is_team_total:
                    team_total_rows.append(row)
                else:
                    primary_rows.append(row)

    # Preserve the original primary-market cap while guaranteeing a bounded
    # Team Totals sidecar from the very same paid /odds response. This costs
    # zero additional provider requests and cannot evict 1X2/FT totals/BTTS.
    kept_primary = primary_rows[:60]
    kept_team_totals = team_total_rows[:20]
    rows = kept_primary + kept_team_totals
    total_relevant = len(primary_rows) + len(team_total_rows)
    return {
        "markets": rows,
        "market_count": total_relevant,
        "truncated": total_relevant > len(rows),
        "primary_market_rows": len(kept_primary),
        "ft_team_total_rows": len(kept_team_totals),
        "ft_team_totals_reused_from_same_provider_response": bool(kept_team_totals),
    }


def _stage_for(minutes_to_kickoff: float, status: str) -> str | None:
    if status in FINISHED_STATUSES:
        return "POSTGAME"
    if status in CANCELLED_STATUSES or status in POSTPONED_STATUSES:
        return None
    targets = [
        (90, "T-90"), (60, "T-60"), (40, "T-40"), (30, "T-30"),
        (20, "T-20"), (10, "T-10"), (0, "CLOSE"),
    ]
    # Scheduler runs every 10 minutes; the wider window tolerates normal GitHub
    # scheduled-job delay while persistent stage de-duplication prevents repeats.
    for target, name in targets:
        if abs(minutes_to_kickoff - target) <= 7.5:
            return name
    return None


def _dedupe_stage(fixture_id: int, stage: str, now: datetime) -> bool:
    conn = _cache_conn()
    row = conn.execute(
        "SELECT updated_at FROM processed_stages WHERE fixture_id=? AND stage=?",
        (fixture_id, stage),
    ).fetchone()
    if row and now.timestamp() - float(row[0]) < timedelta(hours=6).total_seconds():
        return False
    conn.execute(
        """
        INSERT INTO processed_stages(fixture_id, stage, updated_at)
        VALUES(?,?,?)
        ON CONFLICT(fixture_id, stage) DO UPDATE SET updated_at=excluded.updated_at
        """,
        (fixture_id, stage, now.timestamp()),
    )
    conn.commit()
    return True


async def _sport_bundle(fx: dict[str, Any], coverage: dict[str, Any], now: datetime) -> dict[str, Any]:
    if coverage.get("data_tier") not in {"A", "B", "C"}:
        return {"sport_data": "INSUFFICIENT", "reason": "DATA_TIER_D"}
    home_stats = await _team_stats(fx["home_team_id"], fx["league_id"], fx["season"], now)
    away_stats = await _team_stats(fx["away_team_id"], fx["league_id"], fx["season"], now)
    home_recent = await _recent(fx["home_team_id"], now)
    away_recent = await _recent(fx["away_team_id"], now)
    return {
        "sport_data": "AVAILABLE",
        "home_stats": home_stats,
        "away_stats": away_stats,
        "home_recent": home_recent,
        "away_recent": away_recent,
        "raw_projection_status": "REQUIRES_SOCCER_ENGINE",
    }


async def _event_for_fixture(fx: dict[str, Any], stage: str, now: datetime) -> dict[str, Any]:
    coverage = await _coverage(fx["league_id"], fx["season"], now)
    event: dict[str, Any] = {
        "event_type": "SOCCER_REFRESH",
        "stage": stage,
        "fixture": fx,
        "coverage": coverage,
        "classification": "WATCH",
        "bet_eligible": False,
        "availability_confidence": None,
        "notes": [],
    }

    # SPORT FIRST: gather sporting inputs before market data.
    if stage in {"T-90", "T-60", "T-40"}:
        event["sporting"] = await _sport_bundle(fx, coverage, now)

    if stage in {"T-90", "T-60", "T-40", "T-20"} and coverage.get("injuries"):
        event["injuries"] = _compact_injuries(await _api_get("injuries", {"fixture": fx["fixture_id"]}))
    elif stage in {"T-90", "T-60", "T-40", "T-20"}:
        event["injuries"] = "NOT VERIFIED"

    if stage in {"T-60", "T-40", "T-30", "T-20", "T-10"} and coverage.get("lineups"):
        lineup = _compact_lineups(await _api_get("fixtures/lineups", {"fixture": fx["fixture_id"]}))
        event["lineups"] = lineup
        if lineup["both_xi_confirmed"] and lineup["both_goalkeepers_confirmed"]:
            event["availability_confidence"] = 0.90
        else:
            event["availability_confidence"] = 0.70 if stage in {"T-60", "T-40"} else 0.60
            event["notes"].append("Material lineup/goalkeeper information remains NOT VERIFIED.")
    elif stage in {"T-60", "T-40", "T-30", "T-20", "T-10"}:
        event["lineups"] = "NOT VERIFIED"
        event["availability_confidence"] = 0.60

    # Market snapshots are stored only after sporting data collection and never
    # define the raw sporting projection.
    if stage in {"T-40", "T-20", "T-10", "CLOSE"} and coverage.get("odds"):
        event["market"] = _compact_odds(await _api_get("odds", {"fixture": fx["fixture_id"], "page": 1}))
        event["market_use"] = "HISTORY_AND_LATER_MARKET_COMPARISON_ONLY"
    elif stage in {"T-40", "T-20", "T-10", "CLOSE"}:
        event["market"] = "NOT VERIFIED"

    if stage == "POSTGAME":
        if coverage.get("statistics_fixtures"):
            stats = await _api_get("fixtures/statistics", {"fixture": fx["fixture_id"]})
            event["match_stats"] = stats.get("response", [])
        event["result"] = {"goals": fx.get("goals"), "score": fx.get("score"), "status": fx.get("status")}
        event["classification"] = "POSTGAME"

    if stage == "T-20":
        lineup = event.get("lineups")
        if isinstance(lineup, dict) and not lineup.get("both_xi_confirmed"):
            event["notes"].append("T-20 lineup gate failed: BET eligibility blocked.")

    return event


async def run_tick() -> dict[str, Any]:
    global _HTTP_CLIENT
    now_utc = datetime.now(dt_timezone.utc)
    local_now = now_utc.astimezone(TIMEZONE)
    _prune_cache(now_utc)

    # Most of the day only today's slate is needed. Tomorrow is added late at
    # night so T-90/T-60 windows around local midnight are still covered.
    dates = [local_now.date()]
    if local_now.hour >= 22:
        dates.append((local_now + timedelta(days=1)).date())

    fixtures: list[dict[str, Any]] = []
    quota: dict[str, Any] = {}

    _HTTP_CLIENT = httpx.AsyncClient(timeout=TIMEOUT, limits=httpx.Limits(max_connections=8, max_keepalive_connections=4))
    try:
        for d in dates:
            payload = await _api_get("fixtures", {"date": d.isoformat(), "timezone": TIMEZONE_NAME})
            quota = payload.get("quota", quota)
            for row in payload.get("response", []):
                fx = _compact_fixture(row)
                if fx.get("fixture_id") and fx.get("kickoff"):
                    fixtures.append(fx)

        events: list[dict[str, Any]] = []

        if local_now.hour == 6 and local_now.minute < 15:
            upcoming = []
            for fx in fixtures:
                kickoff = _dt(fx["kickoff"])
                if kickoff >= now_utc and fx.get("status") not in CANCELLED_STATUSES | POSTPONED_STATUSES:
                    upcoming.append(fx)
            events.append({
                "event_type": "DAILY_DISCOVERY",
                "stage": "MORNING",
                "date": local_now.date().isoformat(),
                "timezone": TIMEZONE_NAME,
                "upcoming_count": len(upcoming),
                "fixtures": upcoming,
                "classification": "PRE-FINAL",
                "sport_first": True,
                "market_data_included": False,
            })

        for fx in fixtures:
            kickoff = _dt(fx["kickoff"])
            minutes_to = (kickoff - now_utc).total_seconds() / 60.0
            stage = _stage_for(minutes_to, fx.get("status") or "")
            if not stage:
                continue
            if stage == "POSTGAME":
                minutes_since = -minutes_to
                if minutes_since < 95 or minutes_since > 240:
                    continue
            if not _dedupe_stage(fx["fixture_id"], stage, now_utc):
                continue
            try:
                events.append(await _event_for_fixture(fx, stage, now_utc))
            except Exception as exc:
                events.append({
                    "event_type": "PIPELINE_ERROR",
                    "stage": stage,
                    "fixture": fx,
                    "classification": "WATCH",
                    "error": str(exc)[:500],
                })

        actionable = [e for e in events if e.get("stage") in {"T-40", "T-20", "T-10", "CLOSE"}]
        return {
            "service": "soccer-edge-automation",
            "version": "1.1.0",
            "generated_at_utc": now_utc.isoformat(),
            "generated_at_local": local_now.isoformat(),
            "timezone": TIMEZONE_NAME,
            "fixture_scan_count": len(fixtures),
            "event_count": len(events),
            "actionable_refresh_count": len(actionable),
            "events": events,
            "quota": quota,
            "database_persistence": "OPTIONAL_NOT_REQUIRED_FOR_SCHEDULER",
        }
    finally:
        if _HTTP_CLIENT is not None:
            await _HTTP_CLIENT.aclose()
            _HTTP_CLIENT = None
        if _CACHE_CONN is not None:
            _CACHE_CONN.commit()
