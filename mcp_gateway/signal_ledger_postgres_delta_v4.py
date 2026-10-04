from __future__ import annotations

from bisect import bisect_right
from datetime import datetime, timedelta, timezone
from typing import Any

from mcp_gateway import persistence, signal_ledger_postgres_v4

SCHEMA_VERSION = "1.1.0"
MODEL_VERSION = "SOCCER_SIGNAL_LEDGER_POSTGRES_DELTA_V4_1.1.0"
MAX_PAGE_ROWS = 200


def _as_utc(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value or "").strip().replace("Z", "+00:00")
        if not text:
            raise ValueError("since is required")
        parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _pipeline_matches(cur, page: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Resolve the same-tick pipeline envelope in one indexed range query.

    The previous implementation executed a LATERAL ORDER BY/LIMIT lookup once
    per refresh event.  That became prohibitively expensive as the ledger grew.
    This keeps the exact point-in-time rule (latest pipeline run at-or-before the
    event, less than one second apart) while scanning the relevant pipeline
    window only once per page.
    """
    timed_events = [
        row for row in page
        if isinstance(row.get("generated_at"), datetime)
    ]
    if not timed_events:
        return {}

    lower = min(row["generated_at"] for row in timed_events) - timedelta(seconds=1)
    upper = max(row["generated_at"] for row in timed_events)
    cur.execute(
        """
        SELECT
            run_id,
            generated_at_utc,
            generated_at_local,
            timezone,
            payload
        FROM soccer_pipeline_runs
        WHERE generated_at_utc >= %s
          AND generated_at_utc <= %s
        ORDER BY generated_at_utc ASC, run_id ASC
        """,
        (lower, upper),
    )
    columns = [desc.name for desc in cur.description]
    pipeline_rows = [dict(zip(columns, values)) for values in cur.fetchall()]
    if not pipeline_rows:
        return {}

    pipeline_times = [row["generated_at_utc"] for row in pipeline_rows]
    matches: dict[int, dict[str, Any]] = {}
    for event in timed_events:
        event_time = event["generated_at"]
        pos = bisect_right(pipeline_times, event_time) - 1
        if pos < 0:
            continue
        candidate = pipeline_rows[pos]
        delta = event_time - candidate["generated_at_utc"]
        if timedelta(0) <= delta < timedelta(seconds=1):
            matches[int(event["event_id"])] = candidate
    return matches


def build_delta(*, since: str, after_event_id: int = 0, max_rows: int = 500) -> dict[str, Any]:
    since_dt = _as_utc(since)
    after_event_id = max(0, int(after_event_id))
    max_rows = max(1, min(int(max_rows), MAX_PAGE_ROWS))

    persistence.ensure_schema()
    with persistence._connect() as conn:
        with conn.cursor() as cur:
            effective_after_event_id = after_event_id
            if after_event_id == 0:
                # Jump directly to the first event newer than the materialized
                # timestamp instead of walking the event_id PK from the oldest
                # row and filtering thousands of historical events.
                cur.execute(
                    """
                    SELECT MIN(event_id)
                    FROM soccer_refresh_events
                    WHERE generated_at > %s
                    """,
                    (since_dt,),
                )
                first_new = cur.fetchone()[0]
                if first_new is not None:
                    effective_after_event_id = max(0, int(first_new) - 1)

            cur.execute(
                """
                SELECT
                    e.event_id,
                    e.fixture_id,
                    e.generated_at,
                    e.stage,
                    e.event_type,
                    e.classification,
                    e.availability_confidence,
                    e.bet_eligible,
                    e.data_tier,
                    e.payload AS event_payload,
                    f.kickoff,
                    f.league_id,
                    f.league,
                    f.country,
                    f.season,
                    f.home_team_id,
                    f.home_team,
                    f.away_team_id,
                    f.away_team
                FROM soccer_refresh_events e
                LEFT JOIN soccer_fixtures f ON f.fixture_id = e.fixture_id
                WHERE e.generated_at > %s
                  AND e.event_id > %s
                ORDER BY e.event_id ASC
                LIMIT %s
                """,
                (since_dt, effective_after_event_id, max_rows + 1),
            )
            columns = [desc.name for desc in cur.description]
            raw_rows = [dict(zip(columns, values)) for values in cur.fetchall()]

            has_more = len(raw_rows) > max_rows
            page = raw_rows[:max_rows]
            pipeline_matches = _pipeline_matches(cur, page)

    rows: list[dict[str, Any]] = []
    invalid_rows = 0
    rows_without_pipeline_match = 0
    for raw in page:
        pipeline = pipeline_matches.get(int(raw["event_id"]))
        if pipeline:
            raw["pipeline_run_id"] = pipeline.get("run_id")
            raw["pipeline_generated_at_local"] = pipeline.get("generated_at_local")
            raw["pipeline_timezone"] = pipeline.get("timezone")
            raw["pipeline_payload"] = pipeline.get("payload")
        else:
            raw["pipeline_run_id"] = None
            raw["pipeline_generated_at_local"] = None
            raw["pipeline_timezone"] = None
            raw["pipeline_payload"] = None
            rows_without_pipeline_match += 1

        row = signal_ledger_postgres_v4._ledger_row(raw)
        if row is None:
            invalid_rows += 1
            continue
        rows.append(row)

    next_event_id = int(page[-1]["event_id"]) if page else effective_after_event_id
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": "OK",
        "source": signal_ledger_postgres_v4.SOURCE,
        "since": since_dt.isoformat(),
        "after_event_id": after_event_id,
        "effective_after_event_id": effective_after_event_id,
        "max_rows": max_rows,
        "raw_refresh_events_loaded": len(page),
        "row_count": len(rows),
        "invalid_rows": invalid_rows,
        "rows_without_pipeline_run_match": rows_without_pipeline_match,
        "first_event_id": int(page[0]["event_id"]) if page else None,
        "last_event_id": int(page[-1]["event_id"]) if page else None,
        "next_event_id": next_event_id,
        "has_more": has_more,
        "rows": rows,
        "provider_requests_added": 0,
        "canonical_bet_logic_changed": False,
        "model_weights_changed": False,
        "historical_probabilities_recomputed": False,
        "historical_rows_recalibrated": False,
        "synthetic_close_rows_added": 0,
        "production_promotion_allowed": False,
    }
