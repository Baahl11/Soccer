import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def _database_url() -> str | None:
    value = os.getenv("DATABASE_URL", "").strip()
    return value or None


def persistence_configured() -> bool:
    return _database_url() is not None


def _ssl_database_url(url: str) -> str:
    """Require TLS for Postgres unless DATABASE_SSLMODE explicitly overrides it."""
    sslmode = os.getenv("DATABASE_SSLMODE", "").strip() or "require"
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.setdefault("sslmode", sslmode)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def _connect():
    import psycopg
    url = _database_url()
    if not url:
        raise RuntimeError("DATABASE_URL is not configured")
    return psycopg.connect(_ssl_database_url(url), autocommit=True)


def ensure_schema() -> None:
    if not persistence_configured():
        return
    schema = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(schema)


def _quota_remaining(tick: dict[str, Any]) -> int | None:
    value = (tick.get("quota") or {}).get("daily_remaining")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _upsert_fixture(cur, fx: dict[str, Any]) -> None:
    if not fx or not fx.get("fixture_id"):
        return
    cur.execute(
        """
        INSERT INTO soccer_fixtures (
            fixture_id, league_id, league, country, season, round, kickoff, status, status_long,
            home_team_id, home_team, away_team_id, away_team, venue, city, last_seen_at
        ) VALUES (
            %(fixture_id)s, %(league_id)s, %(league)s, %(country)s, %(season)s, %(round)s,
            %(kickoff)s, %(status)s, %(status_long)s, %(home_team_id)s, %(home_team)s,
            %(away_team_id)s, %(away_team)s, %(venue)s, %(city)s, NOW()
        )
        ON CONFLICT (fixture_id) DO UPDATE SET
            league_id = EXCLUDED.league_id,
            league = EXCLUDED.league,
            country = EXCLUDED.country,
            season = EXCLUDED.season,
            round = EXCLUDED.round,
            kickoff = EXCLUDED.kickoff,
            status = EXCLUDED.status,
            status_long = EXCLUDED.status_long,
            home_team_id = EXCLUDED.home_team_id,
            home_team = EXCLUDED.home_team,
            away_team_id = EXCLUDED.away_team_id,
            away_team = EXCLUDED.away_team,
            venue = EXCLUDED.venue,
            city = EXCLUDED.city,
            last_seen_at = NOW()
        """,
        fx,
    )


def _persist_refresh_event(cur, tick: dict[str, Any], event: dict[str, Any]) -> None:
    fx = event.get("fixture") or {}
    fixture_id = fx.get("fixture_id")
    if fixture_id:
        _upsert_fixture(cur, fx)
    coverage = event.get("coverage") or {}
    cur.execute(
        """
        INSERT INTO soccer_refresh_events (
            fixture_id, stage, event_type, classification, availability_confidence,
            bet_eligible, data_tier, generated_at, payload
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
        """,
        (
            fixture_id,
            event.get("stage"),
            event.get("event_type"),
            event.get("classification"),
            event.get("availability_confidence"),
            bool(event.get("bet_eligible", False)),
            coverage.get("data_tier"),
            tick.get("generated_at_utc"),
            json.dumps(event),
        ),
    )

    lineup = event.get("lineups")
    if fixture_id and isinstance(lineup, dict):
        cur.execute(
            """
            INSERT INTO soccer_lineup_snapshots (
                fixture_id, captured_at, stage, lineup_state,
                both_xi_confirmed, both_goalkeepers_confirmed, payload
            ) VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb)
            """,
            (
                fixture_id,
                tick.get("generated_at_utc"),
                event.get("stage"),
                lineup.get("lineup_state"),
                lineup.get("both_xi_confirmed"),
                lineup.get("both_goalkeepers_confirmed"),
                json.dumps(lineup),
            ),
        )

    injuries = event.get("injuries")
    if fixture_id and injuries is not None:
        cur.execute(
            """
            INSERT INTO soccer_availability_snapshots (
                fixture_id, captured_at, stage, availability_confidence, payload
            ) VALUES (%s,%s,%s,%s,%s::jsonb)
            """,
            (
                fixture_id,
                tick.get("generated_at_utc"),
                event.get("stage"),
                event.get("availability_confidence"),
                json.dumps({"injuries": injuries, "notes": event.get("notes", [])}),
            ),
        )

    market = event.get("market")
    if fixture_id and isinstance(market, dict):
        for row in market.get("markets") or []:
            cur.execute(
                """
                INSERT INTO soccer_market_snapshots (
                    fixture_id, captured_at, stage, bookmaker_id, bookmaker,
                    market_id, market, values, provider_update
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s)
                """,
                (
                    fixture_id,
                    tick.get("generated_at_utc"),
                    event.get("stage"),
                    row.get("bookmaker_id"),
                    row.get("bookmaker"),
                    row.get("market_id"),
                    row.get("market"),
                    json.dumps(row.get("values") or []),
                    row.get("provider_update"),
                ),
            )

    if fixture_id and event.get("stage") == "POSTGAME":
        result = event.get("result") or {}
        goals = result.get("goals") or {}
        cur.execute(
            """
            INSERT INTO soccer_results (
                fixture_id, final_status, home_goals, away_goals, final_score,
                match_stats, graded_at, payload
            ) VALUES (%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s::jsonb)
            ON CONFLICT (fixture_id) DO UPDATE SET
                final_status = EXCLUDED.final_status,
                home_goals = EXCLUDED.home_goals,
                away_goals = EXCLUDED.away_goals,
                final_score = EXCLUDED.final_score,
                match_stats = EXCLUDED.match_stats,
                graded_at = EXCLUDED.graded_at,
                payload = EXCLUDED.payload
            """,
            (
                fixture_id,
                result.get("status"),
                goals.get("home"),
                goals.get("away"),
                json.dumps(result.get("score") or {}),
                json.dumps(event.get("match_stats") or []),
                tick.get("generated_at_utc"),
                json.dumps(event),
            ),
        )



def load_latest_pipeline_payload() -> dict[str, Any] | None:
    """Return the most recent compact persisted pipeline payload."""
    if not persistence_configured():
        return None
    ensure_schema()
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT payload
                FROM soccer_pipeline_runs
                ORDER BY generated_at_utc DESC
                LIMIT 1
                """
            )
            row = cur.fetchone()
    if not row:
        return None
    payload = row[0]
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        try:
            value = json.loads(payload)
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None
    return None

def persist_tick(tick: dict[str, Any]) -> bool:
    if not persistence_configured():
        return False
    ensure_schema()
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO soccer_pipeline_runs (
                    generated_at_utc, generated_at_local, timezone, fixture_scan_count,
                    event_count, actionable_refresh_count, quota_remaining, payload
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                """,
                (
                    tick.get("generated_at_utc"),
                    tick.get("generated_at_local"),
                    tick.get("timezone"),
                    tick.get("fixture_scan_count", 0),
                    tick.get("event_count", 0),
                    tick.get("actionable_refresh_count", 0),
                    _quota_remaining(tick),
                    # The relational tables below already persist the detailed
                    # per-event payloads. Keeping the entire tick (including all
                    # events) again in soccer_pipeline_runs temporarily creates a
                    # very large JSON string on the 512 MB worker. Persist a
                    # compact run envelope here instead.
                    json.dumps({key: value for key, value in tick.items() if key not in {"events", "shortlist_state"}}),
                ),
            )
            for event in tick.get("events") or []:
                if event.get("event_type") == "DAILY_DISCOVERY":
                    for fx in event.get("fixtures") or []:
                        _upsert_fixture(cur, fx)
                    continue
                _persist_refresh_event(cur, tick, event)
    return True
