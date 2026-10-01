from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from mcp_gateway import persistence, signal_ledger_postgres_v4

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_SIGNAL_LEDGER_POSTGRES_DELTA_V4_1.0.0"
MAX_PAGE_ROWS = 1000


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


def build_delta(*, since: str, after_event_id: int = 0, max_rows: int = 500) -> dict[str, Any]:
    since_dt = _as_utc(since)
    after_event_id = max(0, int(after_event_id))
    max_rows = max(1, min(int(max_rows), MAX_PAGE_ROWS))

    persistence.ensure_schema()
    with persistence._connect() as conn:
        with conn.cursor() as cur:
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
                    f.away_team,
                    p.run_id AS pipeline_run_id,
                    p.generated_at_local AS pipeline_generated_at_local,
                    p.timezone AS pipeline_timezone,
                    p.payload AS pipeline_payload
                FROM soccer_refresh_events e
                LEFT JOIN soccer_fixtures f ON f.fixture_id = e.fixture_id
                LEFT JOIN LATERAL (
                    SELECT
                        pr.run_id,
                        pr.generated_at_local,
                        pr.timezone,
                        pr.payload
                    FROM soccer_pipeline_runs pr
                    WHERE pr.generated_at_utc <= e.generated_at
                      AND e.generated_at - pr.generated_at_utc < INTERVAL '1 second'
                    ORDER BY pr.generated_at_utc DESC
                    LIMIT 1
                ) p ON TRUE
                WHERE e.generated_at > %s
                  AND e.event_id > %s
                ORDER BY e.event_id ASC
                LIMIT %s
                """,
                (since_dt, after_event_id, max_rows + 1),
            )
            columns = [desc.name for desc in cur.description]
            raw_rows = [dict(zip(columns, values)) for values in cur.fetchall()]

    has_more = len(raw_rows) > max_rows
    page = raw_rows[:max_rows]
    rows: list[dict[str, Any]] = []
    invalid_rows = 0
    rows_without_pipeline_match = 0
    for raw in page:
        if not raw.get("pipeline_run_id"):
            rows_without_pipeline_match += 1
        row = signal_ledger_postgres_v4._ledger_row(raw)
        if row is None:
            invalid_rows += 1
            continue
        rows.append(row)

    next_event_id = int(page[-1]["event_id"]) if page else after_event_id
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": "OK",
        "source": signal_ledger_postgres_v4.SOURCE,
        "since": since_dt.isoformat(),
        "after_event_id": after_event_id,
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
