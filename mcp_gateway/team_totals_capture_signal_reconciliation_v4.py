from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from mcp_gateway import persistence

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_TEAM_TOTALS_CAPTURE_SIGNAL_RECON_V226_1.0.0"


def build(*, lookback_days: int = 60) -> dict[str, Any]:
    persistence.ensure_schema()
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, min(int(lookback_days), 180)))
    with persistence._connect() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT e.fixture_id
            FROM soccer_refresh_events e
            JOIN soccer_fixtures f ON f.fixture_id=e.fixture_id
            WHERE e.fixture_id IS NOT NULL
              AND e.generated_at >= %s
              AND e.generated_at < f.kickoff
              AND COALESCE(e.payload->'team_totals_diversity_capture'->>'qualifies','false')='true'
        """, (cutoff,))
        captured_ids = [int(row[0]) for row in cur.fetchall()]

        exact_ids: set[int] = set()
        if captured_ids:
            cur.execute("""
                SELECT DISTINCT e.fixture_id
                FROM soccer_refresh_events e
                JOIN soccer_fixtures f ON f.fixture_id=e.fixture_id
                WHERE e.fixture_id = ANY(%s)
                  AND e.generated_at >= %s
                  AND e.generated_at < f.kickoff
                  AND jsonb_typeof(e.payload->'team_totals_intelligence'->'observed_exact_market_rows')='array'
                  AND jsonb_array_length(e.payload->'team_totals_intelligence'->'observed_exact_market_rows')>0
            """, (captured_ids, cutoff))
            exact_ids = {int(row[0]) for row in cur.fetchall()}

        missing_ids = [fixture_id for fixture_id in captured_ids if fixture_id not in exact_ids]
        missing_stats: dict[int, tuple[int, int]] = {}
        if missing_ids:
            cur.execute("""
                SELECT e.fixture_id,
                  COUNT(*) FILTER (WHERE e.payload->'team_totals_intelligence' IS NOT NULL)::int AS intel_events,
                  COUNT(*) FILTER (WHERE jsonb_typeof(e.payload->'team_totals_intelligence'->'unsupported_observed_rows')='array'
                                    AND jsonb_array_length(e.payload->'team_totals_intelligence'->'unsupported_observed_rows')>0)::int AS unsupported_events
                FROM soccer_refresh_events e
                JOIN soccer_fixtures f ON f.fixture_id=e.fixture_id
                WHERE e.fixture_id = ANY(%s)
                  AND e.generated_at >= %s
                  AND e.generated_at < f.kickoff
                GROUP BY e.fixture_id
            """, (missing_ids, cutoff))
            missing_stats = {int(row[0]): (int(row[1] or 0), int(row[2] or 0)) for row in cur.fetchall()}

    no_intel = sum(1 for fixture_id in missing_ids if missing_stats.get(fixture_id, (0, 0))[0] == 0)
    no_exact_or_unsupported = sum(1 for fixture_id in missing_ids if missing_stats.get(fixture_id, (0, 0))[0] > 0 and missing_stats.get(fixture_id, (0, 0))[1] == 0)
    unsupported_only = sum(1 for fixture_id in missing_ids if missing_stats.get(fixture_id, (0, 0))[0] > 0 and missing_stats.get(fixture_id, (0, 0))[1] > 0)
    row=(len(captured_ids), no_intel, no_exact_or_unsupported, unsupported_only, len(exact_ids))
    labels=("strict_capture_unique_fixtures","no_team_totals_intelligence","intelligence_no_exact_or_unsupported_market","unsupported_team_total_market_only","has_exact_observed_market")
    counts={k:int(v or 0) for k,v in zip(labels,row)}
    counts["capture_without_exact_observed_market"] = counts["strict_capture_unique_fixtures"]-counts["has_exact_observed_market"]
    return {"schema_version":SCHEMA_VERSION,"model_version":MODEL_VERSION,"generated_at_utc":datetime.now(timezone.utc).isoformat(),"lookback_days":int(lookback_days),"counts":counts,"provider_requests_added":0,"historical_rows_mutated":False,"strict_close_semantics_changed":False,"production_promotion_allowed":False}
