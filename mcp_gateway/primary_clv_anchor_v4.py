from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from mcp_gateway import price_resolver_v4 as price

MODEL_VERSION = "SOCCER_PRIMARY_CLV_ANCHOR_V4_1.0.0"
ANCHOR_POLICY = "OLDEST_UNRESOLVED_POINT_IN_TIME_SIGNAL"


def load_primary_clv_maturation_backlog(
    *,
    lookback_days: int | None = None,
    lookahead_minutes: int | None = None,
    limit: int | None = None,
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
                ORDER BY ous.kickoff ASC, ous.fixture_id ASC, ous.market_family ASC
                LIMIT %s
                """,
                (cutoff, now, lookahead, cutoff, now, lookahead, limit),
            )
            rows = cur.fetchall()
            columns = [desc.name for desc in cur.description]

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
        "provider_requests_added": 0,
        "provider_budget_changed": False,
        "strict_close_semantics_changed": False,
        "historical_rows_mutated": False,
    }
