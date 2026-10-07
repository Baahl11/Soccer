from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from mcp_gateway import clv_postgres_v4 as clv, price_resolver_v4 as price

MODEL_VERSION = "SOCCER_PRIMARY_CLV_ANCHOR_V4_1.1.0"
ANCHOR_POLICY = "OLDEST_UNRESOLVED_POINT_IN_TIME_SIGNAL"
DIAGNOSTIC_SCHEMA_VERSION = "1.0.0"


def _load_exclusion_diagnostics(
    cur: Any,
    *,
    cutoff: datetime,
    now: datetime,
    lookahead: datetime,
) -> dict[str, dict[str, int]]:
    """Count where real priced primary signals fall out of the live backlog.

    This is a Postgres-only observability query. It does not call a provider,
    mutate history, relax strict-close chronology, or change candidate selection.
    Counts are fixture-level so recycled scheduler rows cannot inflate progress.
    """
    cur.execute(
        """
        WITH raw_signal AS (
            SELECT
                (mm.row ->> 'fixture_id')::BIGINT AS fixture_id,
                UPPER(mm.row ->> 'market_family') AS market_family,
                mm.row ->> 'market' AS market,
                p.generated_at_utc AS signal_generated_at,
                f.kickoff,
                f.status
            FROM soccer_pipeline_runs p
            CROSS JOIN LATERAL jsonb_array_elements(
                CASE
                    WHEN jsonb_typeof(COALESCE(p.payload -> 'market_mismatch_rows', '[]'::jsonb)) = 'array'
                    THEN COALESCE(p.payload -> 'market_mismatch_rows', '[]'::jsonb)
                    ELSE '[]'::jsonb
                END
            ) AS mm(row)
            JOIN soccer_fixtures f
              ON f.fixture_id = (mm.row ->> 'fixture_id')::BIGINT
            WHERE p.generated_at_utc >= %s
              AND UPPER(mm.row ->> 'market_family') IN ('1X2','FT_TOTALS','BTTS')
              AND COALESCE((mm.row ->> 'rankable')::boolean, false) = true
              AND NULLIF(mm.row ->> 'market', '') IS NOT NULL
              AND NULLIF(mm.row ->> 'selection', '') IS NOT NULL
              AND NULLIF(mm.row ->> 'price', '') IS NOT NULL

            UNION ALL

            SELECT
                (mt.row ->> 'fixture_id')::BIGINT AS fixture_id,
                CASE
                    WHEN UPPER(mt.row ->> 'market_family') IN ('TOTAL','FT_TOTALS_RESEARCH') THEN 'FT_TOTALS'
                    WHEN UPPER(mt.row ->> 'market_family') IN ('FT_BTTS','FT_BTTS_RESEARCH') THEN 'BTTS'
                    WHEN UPPER(mt.row ->> 'market_family') IN ('FT_1X2','FT_1X2_RESEARCH','MATCH_WINNER') THEN '1X2'
                    ELSE UPPER(mt.row ->> 'market_family')
                END AS market_family,
                mt.row ->> 'market' AS market,
                p.generated_at_utc AS signal_generated_at,
                f.kickoff,
                f.status
            FROM soccer_pipeline_runs p
            CROSS JOIN LATERAL jsonb_array_elements(
                CASE
                    WHEN jsonb_typeof(COALESCE(p.payload -> 'match_table_rows', '[]'::jsonb)) = 'array'
                    THEN COALESCE(p.payload -> 'match_table_rows', '[]'::jsonb)
                    ELSE '[]'::jsonb
                END
            ) AS mt(row)
            JOIN soccer_fixtures f
              ON f.fixture_id = (mt.row ->> 'fixture_id')::BIGINT
            WHERE p.generated_at_utc >= %s
              AND CASE
                    WHEN UPPER(mt.row ->> 'market_family') IN ('TOTAL','FT_TOTALS_RESEARCH') THEN 'FT_TOTALS'
                    WHEN UPPER(mt.row ->> 'market_family') IN ('FT_BTTS','FT_BTTS_RESEARCH') THEN 'BTTS'
                    WHEN UPPER(mt.row ->> 'market_family') IN ('FT_1X2','FT_1X2_RESEARCH','MATCH_WINNER') THEN '1X2'
                    ELSE UPPER(mt.row ->> 'market_family')
                  END IN ('1X2','FT_TOTALS','BTTS')
              AND NULLIF(mt.row ->> 'market', '') IS NOT NULL
              AND NULLIF(mt.row ->> 'selection', '') IS NOT NULL
              AND NULLIF(mt.row ->> 'price', '') IS NOT NULL
              AND jsonb_typeof(mt.row -> 'price') = 'number'
              AND (mt.row ->> 'price')::DOUBLE PRECISION > 1.0
        ),
        classified AS (
            SELECT
                rs.*,
                EXISTS (
                    SELECT 1
                    FROM soccer_market_snapshots m
                    WHERE m.fixture_id = rs.fixture_id
                      AND m.captured_at > rs.signal_generated_at
                      AND m.captured_at < rs.kickoff
                      AND m.provider_update IS NOT NULL
                      AND m.provider_update > rs.signal_generated_at
                      AND LOWER(TRIM(COALESCE(m.market, ''))) = LOWER(TRIM(COALESCE(rs.market, '')))
                ) AS has_strict_later_quote
            FROM raw_signal rs
        )
        SELECT
            market_family,
            COUNT(*)::BIGINT AS priced_signal_rows,
            COUNT(DISTINCT fixture_id)::BIGINT AS priced_fixtures,
            COUNT(DISTINCT fixture_id) FILTER (
                WHERE signal_generated_at < kickoff
            )::BIGINT AS prekickoff_signal_fixtures,
            COUNT(DISTINCT fixture_id) FILTER (
                WHERE signal_generated_at < kickoff
                  AND kickoff > %s
                  AND COALESCE(status, 'NS') NOT IN ('FT','AET','PEN','CANC','PST','ABD','AWD','WO')
            )::BIGINT AS future_active_fixtures,
            COUNT(DISTINCT fixture_id) FILTER (
                WHERE signal_generated_at < kickoff
                  AND kickoff > %s
                  AND kickoff <= %s
                  AND COALESCE(status, 'NS') NOT IN ('FT','AET','PEN','CANC','PST','ABD','AWD','WO')
            )::BIGINT AS within_lookahead_fixtures,
            COUNT(DISTINCT fixture_id) FILTER (
                WHERE signal_generated_at < kickoff
                  AND kickoff > %s
                  AND kickoff <= %s
                  AND COALESCE(status, 'NS') NOT IN ('FT','AET','PEN','CANC','PST','ABD','AWD','WO')
                  AND has_strict_later_quote
            )::BIGINT AS strict_later_quote_fixtures,
            COUNT(DISTINCT fixture_id) FILTER (
                WHERE signal_generated_at < kickoff
                  AND kickoff > %s
                  AND kickoff <= %s
                  AND COALESCE(status, 'NS') NOT IN ('FT','AET','PEN','CANC','PST','ABD','AWD','WO')
                  AND NOT has_strict_later_quote
            )::BIGINT AS unresolved_within_lookahead_fixtures
        FROM classified
        GROUP BY market_family
        ORDER BY market_family
        """,
        (
            cutoff,
            cutoff,
            now,
            now,
            lookahead,
            now,
            lookahead,
            now,
            lookahead,
        ),
    )
    columns = [desc.name for desc in cur.description]
    diagnostics: dict[str, dict[str, int]] = {}
    for raw_row in cur.fetchall():
        row = dict(zip(columns, raw_row))
        family = str(row.get("market_family") or "").upper()
        if family not in {"1X2", "BTTS", "FT_TOTALS"}:
            continue
        metrics = {
            key: int(row.get(key) or 0)
            for key in (
                "priced_signal_rows",
                "priced_fixtures",
                "prekickoff_signal_fixtures",
                "future_active_fixtures",
                "within_lookahead_fixtures",
                "strict_later_quote_fixtures",
                "unresolved_within_lookahead_fixtures",
            )
        }
        metrics["outside_lookahead_active_fixtures"] = max(
            0,
            metrics["future_active_fixtures"] - metrics["within_lookahead_fixtures"],
        )
        diagnostics[family] = metrics
    for family in ("1X2", "BTTS", "FT_TOTALS"):
        diagnostics.setdefault(
            family,
            {
                "priced_signal_rows": 0,
                "priced_fixtures": 0,
                "prekickoff_signal_fixtures": 0,
                "future_active_fixtures": 0,
                "within_lookahead_fixtures": 0,
                "strict_later_quote_fixtures": 0,
                "unresolved_within_lookahead_fixtures": 0,
                "outside_lookahead_active_fixtures": 0,
            },
        )
    return diagnostics


def load_primary_clv_maturation_backlog(
    *,
    lookback_days: int | None = None,
    lookahead_minutes: int | None = None,
    limit: int | None = None,
    include_diagnostics: bool = True,
) -> dict[str, Any]:
    """Load primary CLV work using the oldest still-unresolved real signal.

    Repeated scheduler ticks can re-emit an unchanged priced research row from
    Postgres cache. Treating each pipeline timestamp as the newest signal anchor
    moves the strict-close threshold forward every tick and can prevent a later
    real provider quote from ever materializing. This loader keeps strict-close
    unchanged and instead selects, per fixture/family, the oldest point-in-time
    signal that still has no strictly later pre-kickoff provider update.

    The query is provider-call free. It creates no signal, changes no model,
    threshold, gate, budget, or promotion decision, and does not mutate history.
    """
    lookback_days = (
        price.TEAM_TOTALS_DIVERSITY_LOOKBACK_DAYS
        if lookback_days is None
        else max(1, int(lookback_days))
    )
    lookahead_minutes = (
        price.PRIMARY_CLV_MATURATION_LOOKAHEAD_MINUTES
        if lookahead_minutes is None
        else max(20, int(lookahead_minutes))
    )
    limit = (
        price.PRIMARY_CLV_MATURATION_BACKLOG_LIMIT
        if limit is None
        else max(1, int(limit))
    )

    empty = {
        "candidate_events": [],
        "candidate_count": 0,
        "candidate_family_counts": {},
        "candidate_source_counts": {},
        "source": "POSTGRES_NOT_CONFIGURED",
        "signal_anchor_policy": ANCHOR_POLICY,
        "diagnostic_schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "diagnostic_family_counts": {},
        "provider_requests_added": 0,
        "selection_logic_changed": False,
    }
    if not price.persistence.persistence_configured():
        return empty

    price.persistence.ensure_schema()
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=lookback_days)
    lookahead = now + timedelta(minutes=lookahead_minutes)

    with price.persistence._connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH candidate_signal AS (
                    SELECT
                        (mm.row ->> 'fixture_id')::BIGINT AS fixture_id,
                        UPPER(mm.row ->> 'market_family') AS market_family,
                        mm.row ->> 'market' AS market,
                        p.generated_at_utc AS signal_generated_at,
                        'PHASE16_RANKABLE'::TEXT AS candidate_source,
                        0::INT AS source_priority,
                        f.league_id,
                        f.league,
                        f.country,
                        f.season,
                        f.round,
                        f.kickoff,
                        f.status,
                        f.status_long,
                        f.home_team_id,
                        f.home_team,
                        f.away_team_id,
                        f.away_team,
                        f.venue,
                        f.city
                    FROM soccer_pipeline_runs p
                    CROSS JOIN LATERAL jsonb_array_elements(
                        CASE
                            WHEN jsonb_typeof(COALESCE(p.payload -> 'market_mismatch_rows', '[]'::jsonb)) = 'array'
                            THEN COALESCE(p.payload -> 'market_mismatch_rows', '[]'::jsonb)
                            ELSE '[]'::jsonb
                        END
                    ) AS mm(row)
                    JOIN soccer_fixtures f
                      ON f.fixture_id = (mm.row ->> 'fixture_id')::BIGINT
                    WHERE p.generated_at_utc >= %s
                      AND p.generated_at_utc < f.kickoff
                      AND f.kickoff > %s
                      AND f.kickoff <= %s
                      AND COALESCE(f.status, 'NS') NOT IN ('FT','AET','PEN','CANC','PST','ABD','AWD','WO')
                      AND UPPER(mm.row ->> 'market_family') IN ('1X2','FT_TOTALS','BTTS')
                      AND COALESCE((mm.row ->> 'rankable')::boolean, false) = true
                      AND NULLIF(mm.row ->> 'market', '') IS NOT NULL
                      AND NULLIF(mm.row ->> 'selection', '') IS NOT NULL
                      AND NULLIF(mm.row ->> 'price', '') IS NOT NULL

                    UNION ALL

                    SELECT
                        (mt.row ->> 'fixture_id')::BIGINT AS fixture_id,
                        CASE
                            WHEN UPPER(mt.row ->> 'market_family') IN ('TOTAL','FT_TOTALS_RESEARCH') THEN 'FT_TOTALS'
                            WHEN UPPER(mt.row ->> 'market_family') IN ('FT_BTTS','FT_BTTS_RESEARCH') THEN 'BTTS'
                            WHEN UPPER(mt.row ->> 'market_family') IN ('FT_1X2','FT_1X2_RESEARCH','MATCH_WINNER') THEN '1X2'
                            ELSE UPPER(mt.row ->> 'market_family')
                        END AS market_family,
                        mt.row ->> 'market' AS market,
                        p.generated_at_utc AS signal_generated_at,
                        'MATCH_TABLE_PRICED_RESEARCH'::TEXT AS candidate_source,
                        1::INT AS source_priority,
                        f.league_id,
                        f.league,
                        f.country,
                        f.season,
                        f.round,
                        f.kickoff,
                        f.status,
                        f.status_long,
                        f.home_team_id,
                        f.home_team,
                        f.away_team_id,
                        f.away_team,
                        f.venue,
                        f.city
                    FROM soccer_pipeline_runs p
                    CROSS JOIN LATERAL jsonb_array_elements(
                        CASE
                            WHEN jsonb_typeof(COALESCE(p.payload -> 'match_table_rows', '[]'::jsonb)) = 'array'
                            THEN COALESCE(p.payload -> 'match_table_rows', '[]'::jsonb)
                            ELSE '[]'::jsonb
                        END
                    ) AS mt(row)
                    JOIN soccer_fixtures f
                      ON f.fixture_id = (mt.row ->> 'fixture_id')::BIGINT
                    WHERE p.generated_at_utc >= %s
                      AND p.generated_at_utc < f.kickoff
                      AND f.kickoff > %s
                      AND f.kickoff <= %s
                      AND COALESCE(f.status, 'NS') NOT IN ('FT','AET','PEN','CANC','PST','ABD','AWD','WO')
                      AND CASE
                            WHEN UPPER(mt.row ->> 'market_family') IN ('TOTAL','FT_TOTALS_RESEARCH') THEN 'FT_TOTALS'
                            WHEN UPPER(mt.row ->> 'market_family') IN ('FT_BTTS','FT_BTTS_RESEARCH') THEN 'BTTS'
                            WHEN UPPER(mt.row ->> 'market_family') IN ('FT_1X2','FT_1X2_RESEARCH','MATCH_WINNER') THEN '1X2'
                            ELSE UPPER(mt.row ->> 'market_family')
                          END IN ('1X2','FT_TOTALS','BTTS')
                      AND NULLIF(mt.row ->> 'market', '') IS NOT NULL
                      AND NULLIF(mt.row ->> 'selection', '') IS NOT NULL
                      AND NULLIF(mt.row ->> 'price', '') IS NOT NULL
                      AND jsonb_typeof(mt.row -> 'price') = 'number'
                      AND (mt.row ->> 'price')::DOUBLE PRECISION > 1.0
                ),
                oldest_unresolved_signal AS (
                    SELECT DISTINCT ON (cs.fixture_id, cs.market_family)
                        cs.fixture_id,
                        cs.market_family,
                        cs.market,
                        cs.signal_generated_at,
                        cs.candidate_source,
                        cs.league_id,
                        cs.league,
                        cs.country,
                        cs.season,
                        cs.round,
                        cs.kickoff,
                        cs.status,
                        cs.status_long,
                        cs.home_team_id,
                        cs.home_team,
                        cs.away_team_id,
                        cs.away_team,
                        cs.venue,
                        cs.city
                    FROM candidate_signal cs
                    WHERE NOT EXISTS (
                        SELECT 1
                        FROM soccer_market_snapshots m
                        WHERE m.fixture_id = cs.fixture_id
                          AND m.captured_at > cs.signal_generated_at
                          AND m.captured_at < cs.kickoff
                          AND m.provider_update IS NOT NULL
                          AND m.provider_update > cs.signal_generated_at
                          AND LOWER(TRIM(COALESCE(m.market, ''))) = LOWER(TRIM(COALESCE(cs.market, '')))
                    )
                    ORDER BY
                        cs.fixture_id,
                        cs.market_family,
                        cs.signal_generated_at ASC,
                        cs.source_priority ASC
                )
                SELECT ous.*
                FROM oldest_unresolved_signal ous
                LEFT JOIN LATERAL (
                    SELECT COUNT(DISTINCT m.captured_at)::BIGINT AS later_capture_visits
                    FROM soccer_market_snapshots m
                    WHERE m.fixture_id = ous.fixture_id
                      AND m.captured_at > ous.signal_generated_at
                      AND m.captured_at < ous.kickoff
                      AND LOWER(TRIM(COALESCE(m.market, ''))) = LOWER(TRIM(COALESCE(ous.market, '')))
                ) visit ON TRUE
                ORDER BY
                    COALESCE(visit.later_capture_visits, 0) ASC,
                    ous.kickoff ASC,
                    ous.fixture_id ASC,
                    ous.market_family ASC
                LIMIT %s
                """,
                (cutoff, now, lookahead, cutoff, now, lookahead, limit),
            )
            rows = cur.fetchall()
            columns = [desc.name for desc in cur.description]
            diagnostics = (
                _load_exclusion_diagnostics(
                    cur,
                    cutoff=cutoff,
                    now=now,
                    lookahead=lookahead,
                )
                if include_diagnostics
                else {}
            )

    grouped: dict[int, dict[str, Any]] = {}
    family_counts: dict[str, int] = defaultdict(int)
    source_counts: dict[str, int] = defaultdict(int)
    for raw_row in rows:
        row = dict(zip(columns, raw_row))
        try:
            fixture_id = int(row.get("fixture_id"))
        except (TypeError, ValueError):
            continue
        family = str(row.get("market_family") or "").upper()
        if family not in {"1X2", "FT_TOTALS", "BTTS"}:
            continue
        signal_at = row.get("signal_generated_at")
        kickoff = row.get("kickoff")
        family_counts[family] += 1
        candidate_source = str(row.get("candidate_source") or "UNKNOWN")
        source_counts[candidate_source] += 1
        record = grouped.setdefault(
            fixture_id,
            {
                "fixture": {
                    "fixture_id": fixture_id,
                    "league_id": row.get("league_id"),
                    "league": row.get("league"),
                    "country": row.get("country"),
                    "season": row.get("season"),
                    "round": row.get("round"),
                    "kickoff": kickoff.isoformat() if isinstance(kickoff, datetime) else kickoff,
                    "status": row.get("status"),
                    "status_long": row.get("status_long"),
                    "home_team_id": row.get("home_team_id"),
                    "home_team": row.get("home_team"),
                    "away_team_id": row.get("away_team_id"),
                    "away_team": row.get("away_team"),
                    "venue": row.get("venue"),
                    "city": row.get("city"),
                },
                "kickoff": kickoff,
                "signals": [],
            },
        )
        record["signals"].append(
            {
                "market_family": family,
                "market": row.get("market"),
                "signal_generated_at": (
                    signal_at.isoformat() if isinstance(signal_at, datetime) else signal_at
                ),
                "candidate_source": candidate_source,
                "signal_anchor_policy": ANCHOR_POLICY,
            }
        )

    events: list[dict[str, Any]] = []
    for record in grouped.values():
        events.append(
            {
                "event_type": price.PRIMARY_CLV_MATURATION_EVENT_TYPE,
                "stage": price._maturation_stage(record.get("kickoff"), now),
                "fixture": record["fixture"],
                "classification": "RESEARCH_ONLY",
                "bet_eligible": False,
                "research_only": True,
                "decision_weight": 0.0,
                "primary_clv_maturation": {
                    "candidate_source": "POSTGRES_PHASE16_OLDEST_UNRESOLVED_SIGNAL_NO_LATER_REAL_QUOTE",
                    "signals": list(record["signals"]),
                    "signal_anchor_policy": ANCHOR_POLICY,
                    "provider_requests_before_price_resolver": 0,
                    "primary_markets_preempted": False,
                    "requires_provider_update_after_signal": True,
                    "strict_close_semantics_changed": False,
                    "historical_signal_mutated": False,
                },
            }
        )

    events.sort(
        key=lambda event: (
            str(((event.get("fixture") or {}).get("kickoff") or "")),
            int(((event.get("fixture") or {}).get("fixture_id") or 0)),
        )
    )
    return {
        "candidate_events": events,
        "candidate_count": len(events),
        "candidate_family_counts": dict(sorted(family_counts.items())),
        "candidate_source_counts": dict(sorted(source_counts.items())),
        "source": "POSTGRES_PRIMARY_CLV_MATURATION_BACKLOG_V3_OLDEST_UNRESOLVED",
        "signal_anchor_policy": ANCHOR_POLICY,
        "diagnostic_schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "diagnostic_status": "INLINE" if include_diagnostics else "DEFERRED_OFFLINE",
        "diagnostic_family_counts": diagnostics,
        "diagnostic_window": {
            "lookback_days": lookback_days,
            "lookahead_minutes": lookahead_minutes,
            "cutoff_utc": cutoff.isoformat(),
            "observed_at_utc": now.isoformat(),
            "lookahead_utc": lookahead.isoformat(),
        },
        "provider_requests_added": 0,
        "provider_budget_changed": False,
        "strict_close_semantics_changed": False,
        "historical_rows_mutated": False,
        "selection_logic_changed": False,
        "backlog_priority_policy": "FEWEST_LATER_CAPTURE_VISITS_THEN_KICKOFF",
    }


POST_V223_LIVE_AT = datetime(2026, 10, 4, 7, 13, 43, tzinfo=timezone.utc)

def build_post_v223_visit_matrix(*, limit: int = 500) -> dict[str, Any]:
    """DB-only canonical-equivalent primary CLV visit audit.

    This audit intentionally mirrors Phase17 primary-signal eligibility:
    match_table_rows only, canonical stages/classes, priced entries, exact market
    name, strictly later captured_at/provider_update, same-book preference when
    available, and the latest strict provider quote before kickoff.

    It remains observability-only and never mutates the canonical CLV ledger.
    """
    if not price.persistence.persistence_configured():
        return {
            "status": "POSTGRES_NOT_CONFIGURED",
            "rows": [],
            "provider_requests_added": 0,
        }

    price.persistence.ensure_schema()
    bounded_limit = max(1, min(int(limit), 2000))
    with price.persistence._connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH raw_signal AS (
                    SELECT
                        (mt.row->>'fixture_id')::BIGINT AS fixture_id,
                        CASE
                            WHEN UPPER(mt.row->>'market_family') IN ('TOTAL','FT_TOTALS_RESEARCH') THEN 'FT_TOTALS'
                            WHEN UPPER(mt.row->>'market_family') IN ('FT_BTTS','FT_BTTS_RESEARCH') THEN 'BTTS'
                            WHEN UPPER(mt.row->>'market_family') IN ('FT_1X2','FT_1X2_RESEARCH','MATCH_WINNER') THEN '1X2'
                            ELSE UPPER(mt.row->>'market_family')
                        END AS market_family,
                        mt.row->>'market' AS market,
                        mt.row->>'selection' AS selection,
                        mt.row->>'line' AS line,
                        mt.row->>'price' AS entry_price,
                        mt.row->>'p_market_fair' AS entry_fair_probability,
                        mt.row->>'bookmaker' AS bookmaker,
                        COALESCE(NULLIF(mt.row->>'stage',''), e.stage) AS stage,
                        COALESCE(NULLIF(mt.row->>'classification',''), e.classification) AS classification,
                        p.generated_at_utc AS signal_generated_at,
                        f.kickoff
                    FROM soccer_pipeline_runs p
                    CROSS JOIN LATERAL jsonb_array_elements(
                        CASE
                            WHEN jsonb_typeof(COALESCE(p.payload->'match_table_rows','[]'::jsonb))='array'
                            THEN COALESCE(p.payload->'match_table_rows','[]'::jsonb)
                            ELSE '[]'::jsonb
                        END
                    ) mt(row)
                    JOIN soccer_fixtures f
                      ON f.fixture_id=(mt.row->>'fixture_id')::BIGINT
                    LEFT JOIN soccer_refresh_events e
                      ON e.fixture_id=f.fixture_id
                     AND e.generated_at=p.generated_at_utc
                    WHERE p.generated_at_utc >= %s
                      AND p.generated_at_utc < f.kickoff
                      AND COALESCE(NULLIF(mt.row->>'stage',''), e.stage) = ANY(%s)
                      AND COALESCE(NULLIF(mt.row->>'classification',''), e.classification) = ANY(%s)
                      AND CASE
                            WHEN UPPER(mt.row->>'market_family') IN ('TOTAL','FT_TOTALS_RESEARCH') THEN 'FT_TOTALS'
                            WHEN UPPER(mt.row->>'market_family') IN ('FT_BTTS','FT_BTTS_RESEARCH') THEN 'BTTS'
                            WHEN UPPER(mt.row->>'market_family') IN ('FT_1X2','FT_1X2_RESEARCH','MATCH_WINNER') THEN '1X2'
                            ELSE UPPER(mt.row->>'market_family')
                          END IN ('1X2','FT_TOTALS','BTTS')
                      AND NULLIF(mt.row->>'market','') IS NOT NULL
                      AND NULLIF(mt.row->>'selection','') IS NOT NULL
                      AND NULLIF(mt.row->>'price','') IS NOT NULL
                      AND (mt.row->>'price') ~ '^[0-9]+([.][0-9]+)?$'
                      AND (mt.row->>'price')::DOUBLE PRECISION > 1.0
                ),
                anchor AS (
                    SELECT DISTINCT ON (fixture_id, market_family)
                        fixture_id,
                        market_family,
                        market,
                        selection,
                        line,
                        entry_price,
                        entry_fair_probability,
                        bookmaker,
                        stage,
                        classification,
                        signal_generated_at,
                        kickoff
                    FROM raw_signal
                    ORDER BY fixture_id, market_family, signal_generated_at ASC
                ),
                anchor_limited AS (
                    SELECT *
                    FROM anchor
                    ORDER BY kickoff DESC, fixture_id ASC, market_family ASC
                    LIMIT %s
                )
                SELECT
                    a.fixture_id,
                    a.market_family,
                    a.market,
                    a.selection,
                    a.line,
                    a.entry_price,
                    a.entry_fair_probability,
                    a.bookmaker,
                    a.stage,
                    a.classification,
                    a.signal_generated_at,
                    a.kickoff,
                    m.captured_at,
                    m.provider_update,
                    m.bookmaker_id,
                    m.bookmaker AS snapshot_bookmaker,
                    m.values
                FROM anchor_limited a
                LEFT JOIN soccer_market_snapshots m
                  ON m.fixture_id=a.fixture_id
                 AND m.captured_at>a.signal_generated_at
                 AND m.captured_at<a.kickoff
                 AND LOWER(TRIM(COALESCE(m.market,'')))=LOWER(TRIM(COALESCE(a.market,'')))
                ORDER BY
                    a.kickoff DESC,
                    a.fixture_id,
                    a.market_family,
                    m.captured_at,
                    m.bookmaker_id
                """,
                (
                    POST_V223_LIVE_AT,
                    list(clv.SIGNAL_STAGES),
                    list(clv.SIGNAL_CLASSES),
                    bounded_limit,
                ),
            )
            cols = [d.name for d in cur.description]
            raw_rows = [dict(zip(cols, row)) for row in cur.fetchall()]

    grouped: dict[tuple[int, str], dict[str, Any]] = {}
    for raw in raw_rows:
        fixture_id = int(raw["fixture_id"])
        family = str(raw.get("market_family") or "").upper()
        key = (fixture_id, family)
        signal_at = raw.get("signal_generated_at")
        record = grouped.setdefault(
            key,
            {
                "fixture_id": fixture_id,
                "market_family": family,
                "market": raw.get("market"),
                "selection": raw.get("selection"),
                "line": price._num(raw.get("line")),
                "entry_price": price._num(raw.get("entry_price")),
                "entry_fair_probability": price._num(raw.get("entry_fair_probability")),
                "bookmaker": raw.get("bookmaker"),
                "stage": raw.get("stage"),
                "classification": raw.get("classification"),
                "signal_generated_at": signal_at,
                "kickoff": raw.get("kickoff"),
                "_snapshots": [],
            },
        )
        if record["line"] is None:
            _, parsed_line = price._parse_value(record.get("selection"))
            record["line"] = parsed_line
        if raw.get("captured_at") is None:
            continue
        record["_snapshots"].append(
            {
                "captured_at": raw.get("captured_at"),
                "provider_update": raw.get("provider_update"),
                "bookmaker_id": raw.get("bookmaker_id"),
                "bookmaker": raw.get("snapshot_bookmaker"),
                "values": raw.get("values") if isinstance(raw.get("values"), list) else [],
            }
        )

    def _iso(value: Any) -> Any:
        return value.isoformat() if isinstance(value, datetime) else value

    clean: list[dict[str, Any]] = []
    for record in grouped.values():
        signal_at = record["signal_generated_at"]
        selection = record.get("selection")
        line = record.get("line")
        entry_price = record.get("entry_price")
        entry_fair = record.get("entry_fair_probability")
        if entry_fair is None and entry_price is not None and entry_price > 1.0:
            entry_fair = 1.0 / entry_price

        snapshots = sorted(
            record.pop("_snapshots"),
            key=lambda snap: (
                clv._as_utc_datetime(snap.get("captured_at"))
                or datetime.min.replace(tzinfo=timezone.utc),
                str(snap.get("bookmaker_id") or ""),
            ),
        )
        by_capture: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for snap in snapshots:
            by_capture[snap.get("captured_at")].append(snap)

        visits: list[dict[str, Any]] = []
        any_exact_later = False
        for captured_at in sorted(
            by_capture,
            key=lambda value: value or datetime.min.replace(tzinfo=timezone.utc),
        ):
            snaps = by_capture[captured_at]
            strict_rows = [
                snap
                for snap in snaps
                if clv._is_strictly_later_provider_quote(snap, signal_at)
            ]
            exact_rows = 0
            for snap in snaps:
                fair, _ = clv._group_fair_probability(
                    snap.get("values") if isinstance(snap.get("values"), list) else [],
                    selection,
                    line,
                )
                if fair is not None:
                    exact_rows += 1
            any_exact_later = any_exact_later or exact_rows > 0
            visits.append(
                {
                    "captured_at": _iso(captured_at),
                    "snapshot_rows": len(snaps),
                    "strict_provider_rows": len(strict_rows),
                    "exact_instrument_rows": exact_rows,
                }
            )

        strict_candidates = [
            snap
            for snap in snapshots
            if clv._is_strictly_later_provider_quote(snap, signal_at)
        ]
        same_book = [
            snap
            for snap in strict_candidates
            if clv._norm(snap.get("bookmaker")) == clv._norm(record.get("bookmaker"))
        ]
        pool = same_book or strict_candidates
        close_at = (
            max(snap["captured_at"] for snap in pool if snap.get("captured_at") is not None)
            if pool
            else None
        )
        close_groups = [
            snap for snap in pool
            if close_at is not None and snap.get("captured_at") == close_at
        ]

        fair_values: list[float] = []
        price_values: list[float] = []
        close_provider_updates: list[datetime] = []
        for snap in close_groups:
            provider_update = clv._as_utc_datetime(snap.get("provider_update"))
            if provider_update is not None:
                close_provider_updates.append(provider_update)
            fair, closing_price = clv._group_fair_probability(
                snap.get("values") if isinstance(snap.get("values"), list) else [],
                selection,
                line,
            )
            if fair is not None:
                fair_values.append(float(fair))
                if closing_price is not None:
                    price_values.append(float(closing_price))

        closing_fair = (
            sorted(fair_values)[len(fair_values) // 2] if fair_values else None
        )
        closing_price = (
            sorted(price_values)[len(price_values) // 2] if price_values else None
        )
        close_provider_update = max(close_provider_updates) if close_provider_updates else None
        exact_comparable = closing_fair is not None
        if not snapshots:
            skip_reason = "NO_LATER_PREKICKOFF_MARKET_SNAPSHOT"
        elif not strict_candidates:
            skip_reason = "NO_LATER_PROVIDER_UPDATE"
        elif not exact_comparable:
            skip_reason = "NO_SELECTION_MATCH_AT_CLOSE"
        else:
            skip_reason = None

        probability_clv = (
            round((closing_fair - entry_fair) * 100.0, 6)
            if exact_comparable and entry_fair is not None
            else None
        )
        price_clv = (
            round((entry_price / closing_price - 1.0) * 100.0, 6)
            if exact_comparable
            and entry_price is not None
            and closing_price is not None
            and closing_price > 1.0
            else None
        )

        record.update(
            {
                "entry_fair_probability": (
                    round(entry_fair, 8) if entry_fair is not None else None
                ),
                "signal_generated_at": _iso(signal_at),
                "kickoff": _iso(record.get("kickoff")),
                "later_market_snapshot_rows": len(snapshots),
                "distinct_capture_visits": len(by_capture),
                "with_exact_instrument_later_quote": any_exact_later,
                "strict_provider_candidate_rows": len(strict_candidates),
                "same_book_strict_rows": len(same_book),
                "same_book_preferred": bool(same_book),
                "canonical_close_at": _iso(close_at),
                "closing_provider_update": _iso(close_provider_update),
                "closing_fair_probability": (
                    round(closing_fair, 8) if closing_fair is not None else None
                ),
                "closing_price": (
                    round(closing_price, 6) if closing_price is not None else None
                ),
                "probability_clv_pp": probability_clv,
                "price_clv_pct": price_clv,
                "strict_close_eligible": exact_comparable,
                "true_clv_comparable": bool(
                    exact_comparable
                    and (probability_clv is not None or price_clv is not None)
                ),
                "skip_reason": skip_reason,
                "visits": visits,
            }
        )
        clean.append(record)

    clean.sort(
        key=lambda row: (
            str(row.get("kickoff") or ""),
            int(row.get("fixture_id") or 0),
            str(row.get("market_family") or ""),
        ),
        reverse=True,
    )

    families: dict[str, dict[str, Any]] = {}
    for fam in ("1X2", "BTTS", "FT_TOTALS"):
        family_rows = [row for row in clean if row["market_family"] == fam]
        comparable = [row for row in family_rows if bool(row["true_clv_comparable"])]
        skip_counts: dict[str, int] = defaultdict(int)
        for row in family_rows:
            reason = row.get("skip_reason")
            if reason:
                skip_counts[str(reason)] += 1
        families[fam] = {
            "signals": len(family_rows),
            "unique_fixtures": len({row["fixture_id"] for row in family_rows}),
            "with_any_later_snapshot": sum(
                int(row["later_market_snapshot_rows"] or 0) > 0
                for row in family_rows
            ),
            "with_2plus_capture_visits": sum(
                int(row["distinct_capture_visits"] or 0) >= 2
                for row in family_rows
            ),
            "with_provider_update_advance": sum(
                int(row["strict_provider_candidate_rows"] or 0) > 0
                for row in family_rows
            ),
            "with_exact_instrument_later_quote": sum(
                bool(row["with_exact_instrument_later_quote"])
                for row in family_rows
            ),
            "same_book_preferred_rows": sum(
                bool(row["same_book_preferred"]) for row in family_rows
            ),
            "strict_close_eligible": len(comparable),
            "true_clv_comparable": len(comparable),
            "skip_reason_counts": dict(sorted(skip_counts.items())),
            "strict_close_conversion_rate": (
                round(len(comparable) / len(family_rows), 6)
                if family_rows
                else None
            ),
        }

    return {
        "schema_version": "3.0.0",
        "model_version": "SOCCER_POST_V223_VISIT_MATRIX_V3_CANONICAL_EQUIVALENT",
        "status": "OBSERVABILITY_ONLY",
        "cohort_start_utc": POST_V223_LIVE_AT.isoformat(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "signal_policy": {
            "source": "PIPELINE_MATCH_TABLE_ONLY",
            "stages": list(clv.SIGNAL_STAGES),
            "classifications": list(clv.SIGNAL_CLASSES),
            "anchor": "OLDEST_PRICED_ELIGIBLE_SIGNAL_PER_FIXTURE_FAMILY",
            "same_book_preferred": True,
            "cross_book_fallback": True,
            "close": "LATEST_STRICT_PROVIDER_QUOTE_BEFORE_KICKOFF",
            "exact_instrument_required": True,
        },
        "family_summary": families,
        "rows": clean,
        "provider_requests_added": 0,
        "strict_close_semantics_changed": False,
        "models_changed": False,
        "thresholds_changed": False,
        "gates_changed": False,
        "provider_budget_changed": False,
        "canonical_bet_logic_changed": False,
        "production_promotion_allowed": False,
        "note": (
            "This is a DB-only canonical-equivalent coverage audit. It uses the same "
            "primary signal stages/classes and strict close matcher as Phase17, but it "
            "does not mutate canonical Phase17 counts. Any excess comparable rows must "
            "be reconciled through a separate exact historical backfill before they can "
            "enter canonical True CLV."
        ),
    }

