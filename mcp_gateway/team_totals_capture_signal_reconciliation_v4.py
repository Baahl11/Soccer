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
            WITH fixture_stats AS (
              SELECT e.fixture_id,
                BOOL_OR(COALESCE(e.payload->'team_totals_diversity_capture'->>'qualifies','false')='true') AS captured,
                COUNT(*) FILTER (WHERE e.payload->'team_totals_intelligence' IS NOT NULL) AS intel_events,
                COUNT(*) FILTER (WHERE jsonb_typeof(e.payload->'team_totals_intelligence'->'observed_exact_market_rows')='array'
                                  AND jsonb_array_length(e.payload->'team_totals_intelligence'->'observed_exact_market_rows')>0) AS exact_events,
                COUNT(*) FILTER (WHERE jsonb_typeof(e.payload->'team_totals_intelligence'->'unsupported_observed_rows')='array'
                                  AND jsonb_array_length(e.payload->'team_totals_intelligence'->'unsupported_observed_rows')>0) AS unsupported_events
              FROM soccer_refresh_events e
              JOIN soccer_fixtures f ON f.fixture_id=e.fixture_id
              WHERE e.fixture_id IS NOT NULL
                AND e.generated_at >= %s
                AND e.generated_at < f.kickoff
              GROUP BY e.fixture_id
            )
            SELECT COUNT(*) FILTER (WHERE captured)::int,
              COUNT(*) FILTER (WHERE captured AND intel_events=0)::int,
              COUNT(*) FILTER (WHERE captured AND intel_events>0 AND exact_events=0 AND unsupported_events=0)::int,
              COUNT(*) FILTER (WHERE captured AND intel_events>0 AND exact_events=0 AND unsupported_events>0)::int,
              COUNT(*) FILTER (WHERE captured AND exact_events>0)::int
            FROM fixture_stats
        """, (cutoff,))
        row=cur.fetchone() or (0,0,0,0,0)
    labels=("strict_capture_unique_fixtures","no_team_totals_intelligence","intelligence_no_exact_or_unsupported_market","unsupported_team_total_market_only","has_exact_observed_market")
    counts={k:int(v or 0) for k,v in zip(labels,row)}
    counts["capture_without_exact_observed_market"] = counts["strict_capture_unique_fixtures"]-counts["has_exact_observed_market"]
    return {"schema_version":SCHEMA_VERSION,"model_version":MODEL_VERSION,"generated_at_utc":datetime.now(timezone.utc).isoformat(),"lookback_days":int(lookback_days),"counts":counts,"provider_requests_added":0,"historical_rows_mutated":False,"strict_close_semantics_changed":False,"production_promotion_allowed":False}
