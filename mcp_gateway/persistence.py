import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from mcp_gateway import feature_snapshot_v4


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


def _persist_feature_snapshot(cur, tick: dict[str, Any], event: dict[str, Any], fixture_id: Any) -> None:
    """Persist the point-in-time v4 feature envelope with each pregame refresh."""
    if not fixture_id:
        return
    if str(event.get("event_type") or "").upper() != "SOCCER_REFRESH":
        return
    if str(event.get("stage") or "").upper() == "POSTGAME":
        return

    snapshot = feature_snapshot_v4.build(tick, event)
    if feature_snapshot_v4.validate(snapshot):
        return

    cur.execute(
        """
        INSERT INTO soccer_feature_snapshots (
            fixture_id, captured_at, stage, schema_version,
            model_version, data_tier, feature_count,
            missing_feature_count, payload
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
        ON CONFLICT (fixture_id, captured_at, stage, schema_version) DO NOTHING
        """,
        (
            snapshot.get("fixture_id"),
            snapshot.get("captured_at"),
            snapshot.get("stage"),
            snapshot.get("schema_version"),
            snapshot.get("model_version"),
            snapshot.get("data_tier"),
            snapshot.get("feature_count", 0),
            snapshot.get("missing_feature_count", 0),
            json.dumps(snapshot),
        ),
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

    _persist_feature_snapshot(cur, tick, event, fixture_id)

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
    market_source = str((market or {}).get("source") or "").upper() if isinstance(market, dict) else ""
    market_resolution_status = str((market or {}).get("resolution_status") or "").upper() if isinstance(market, dict) else ""
    market_is_cache_replay = "CACHE" in market_source or "CACHE" in market_resolution_status
    if fixture_id and isinstance(market, dict) and not market_is_cache_replay:
        canonical_rows = [
            row for row in (market.get("markets") or [])
            if isinstance(row, dict)
        ]
        research_rows = [
            row for row in (market.get("research_cards_props_markets") or [])
            if isinstance(row, dict)
        ]
        for row in canonical_rows + research_rows:
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
            # Multiple research/maturation events for the same fixture and
            # stage can legitimately be produced within one scheduler tick. The
            # relational schema keeps (fixture_id, stage, generated_at) unique,
            # so preserve every event by assigning a deterministic microsecond
            # offset only when that key repeats inside this tick. This does not
            # alter the event payload's prediction point; it only disambiguates
            # persistence ordering.
            persisted_key_counts: dict[tuple[Any, Any], int] = {}
            base_generated_at = tick.get("generated_at_utc")
            try:
                parsed_generated_at = datetime.fromisoformat(
                    str(base_generated_at).replace("Z", "+00:00")
                ) if base_generated_at else None
            except ValueError:
                parsed_generated_at = None

            for event in tick.get("events") or []:
                if event.get("event_type") == "DAILY_DISCOVERY":
                    for fx in event.get("fixtures") or []:
                        _upsert_fixture(cur, fx)
                    continue

                fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
                key = (fixture.get("fixture_id"), event.get("stage"))
                ordinal = persisted_key_counts.get(key, 0)
                persisted_key_counts[key] = ordinal + 1

                persist_tick_view = tick
                if ordinal > 0 and parsed_generated_at is not None:
                    persist_tick_view = dict(tick)
                    persist_tick_view["generated_at_utc"] = (
                        parsed_generated_at + timedelta(microseconds=ordinal)
                    ).isoformat()
                    event.setdefault("persistence", {})
                    if isinstance(event["persistence"], dict):
                        event["persistence"].update({
                            "same_fixture_stage_ordinal": ordinal,
                            "generated_at_microsecond_offset": ordinal,
                            "reason": "DISAMBIGUATE_SAME_TICK_FIXTURE_STAGE_EVENTS",
                        })

                _persist_refresh_event(cur, persist_tick_view, event)
    return True
