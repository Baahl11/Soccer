from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from mcp_gateway import price_resolver_v4 as price

MODEL_VERSION = "SOCCER_TEAM_TOTALS_CLV_ANCHOR_V4_1.0.0"
ANCHOR_POLICY = "OLDEST_UNRESOLVED_POINT_IN_TIME_SIGNAL_PER_EXACT_TEAM_TOTAL"
DEFAULT_LOOKBACK_HOURS = max(
    48,
    int(price.TEAM_TOTALS_DIVERSITY_LOOKAHEAD_HOURS) + 12,
)


def load_team_totals_maturation_backlog(
    *,
    lookback_days: int | None = None,
    lookahead_minutes: int = price.TEAM_TOTALS_MATURATION_LOOKAHEAD_MINUTES,
    limit: int = price.TEAM_TOTALS_MATURATION_BACKLOG_LIMIT,
) -> dict[str, Any]:
    empty = {
        "candidate_events": [],
        "candidate_count": 0,
        "candidate_signal_count": 0,
        "source": "POSTGRES_NOT_CONFIGURED",
        "signal_anchor_policy": ANCHOR_POLICY,
        "lookback_hours": DEFAULT_LOOKBACK_HOURS,
        "provider_requests_added": 0,
        "provider_budget_changed": False,
        "strict_close_semantics_changed": False,
        "historical_rows_mutated": False,
    }
    if not price.persistence.persistence_configured():
        return empty

    price.persistence.ensure_schema()
    now = datetime.now(timezone.utc)
    if lookback_days is None:
        lookback_cutoff = now - timedelta(hours=DEFAULT_LOOKBACK_HOURS)
        lookback_hours = DEFAULT_LOOKBACK_HOURS
    else:
        lookback_cutoff = now - timedelta(days=max(1, int(lookback_days)))
        lookback_hours = max(1, int(lookback_days)) * 24
    lookahead_cutoff = now + timedelta(minutes=max(20, int(lookahead_minutes)))

    query = """
        WITH bounds AS (
            SELECT
                %s::timestamptz AS cutoff,
                %s::timestamptz AS now_utc,
                %s::timestamptz AS lookahead
        ),
        upcoming_fixtures AS MATERIALIZED (
            SELECT f.*
            FROM soccer_fixtures f
            CROSS JOIN bounds b
            WHERE f.kickoff > b.now_utc
              AND f.kickoff <= b.lookahead
              AND COALESCE(f.status, 'NS') NOT IN ('FT','AET','PEN','CANC','PST','ABD','AWD','WO')
        ),
        base_events AS MATERIALIZED (
            SELECT
                e.fixture_id,
                e.generated_at,
                e.payload,
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
            FROM soccer_refresh_events e
            JOIN upcoming_fixtures f ON f.fixture_id = e.fixture_id
            CROSS JOIN bounds b
            WHERE e.generated_at >= b.cutoff
              AND e.generated_at < f.kickoff
        ),
        strict_capture AS (
            SELECT DISTINCT be.fixture_id
            FROM base_events be
            WHERE COALESCE(
                      be.payload -> 'team_totals_diversity_capture' ->> 'qualifies',
                      'false'
                  ) = 'true'
        ),
        candidate_signal AS (
            SELECT
                be.fixture_id,
                sig.row ->> 'market' AS market,
                sig.row ->> 'selection' AS selection,
                sig.row ->> 'line' AS line,
                be.generated_at AS signal_generated_at,
                be.league_id,
                be.league,
                be.country,
                be.season,
                be.round,
                be.kickoff,
                be.status,
                be.status_long,
                be.home_team_id,
                be.home_team,
                be.away_team_id,
                be.away_team,
                be.venue,
                be.city
            FROM base_events be
            JOIN strict_capture sc ON sc.fixture_id = be.fixture_id
            CROSS JOIN LATERAL jsonb_array_elements(
                CASE
                    WHEN jsonb_typeof(
                        COALESCE(
                            be.payload -> 'team_totals_intelligence' -> 'observed_exact_market_rows',
                            '[]'::jsonb
                        )
                    ) = 'array'
                    THEN COALESCE(
                        be.payload -> 'team_totals_intelligence' -> 'observed_exact_market_rows',
                        '[]'::jsonb
                    )
                    ELSE '[]'::jsonb
                END
            ) AS sig(row)
            WHERE NULLIF(sig.row ->> 'market', '') IS NOT NULL
              AND NULLIF(sig.row ->> 'selection', '') IS NOT NULL
              AND COALESCE(sig.row ->> 'line', '') ~ '^[0-9]+([.][0-9]+)?$'
              AND COALESCE(sig.row ->> 'decimal_price', sig.row ->> 'price', '') ~ '^[0-9]+([.][0-9]+)?$'
              AND COALESCE(sig.row ->> 'decimal_price', sig.row ->> 'price')::DOUBLE PRECISION > 1.0
        ),
        oldest_unresolved_signal AS (
            SELECT DISTINCT ON (
                cs.fixture_id,
                LOWER(TRIM(COALESCE(cs.market, ''))),
                CASE
                    WHEN LOWER(TRIM(COALESCE(cs.selection, ''))) LIKE 'over%%' THEN 'over'
                    WHEN LOWER(TRIM(COALESCE(cs.selection, ''))) LIKE 'under%%' THEN 'under'
                    ELSE LOWER(TRIM(COALESCE(cs.selection, '')))
                END,
                (cs.line)::NUMERIC
            )
                cs.*
            FROM candidate_signal cs
            WHERE NOT EXISTS (
                SELECT 1
                FROM soccer_market_snapshots m
                CROSS JOIN LATERAL jsonb_array_elements(
                    CASE
                        WHEN jsonb_typeof(COALESCE(m.values, '[]'::jsonb)) = 'array'
                        THEN COALESCE(m.values, '[]'::jsonb)
                        ELSE '[]'::jsonb
                    END
                ) AS q(value)
                WHERE m.fixture_id = cs.fixture_id
                  AND m.captured_at > cs.signal_generated_at
                  AND m.captured_at < cs.kickoff
                  AND m.provider_update IS NOT NULL
                  AND m.provider_update > cs.signal_generated_at
                  AND LOWER(TRIM(COALESCE(m.market, ''))) = LOWER(TRIM(COALESCE(cs.market, '')))
                  AND (
                        CASE
                            WHEN LOWER(TRIM(COALESCE(q.value ->> 'selection', ''))) LIKE 'over%%' THEN 'over'
                            WHEN LOWER(TRIM(COALESCE(q.value ->> 'selection', ''))) LIKE 'under%%' THEN 'under'
                            ELSE LOWER(TRIM(COALESCE(q.value ->> 'selection', '')))
                        END
                      ) = (
                        CASE
                            WHEN LOWER(TRIM(COALESCE(cs.selection, ''))) LIKE 'over%%' THEN 'over'
                            WHEN LOWER(TRIM(COALESCE(cs.selection, ''))) LIKE 'under%%' THEN 'under'
                            ELSE LOWER(TRIM(COALESCE(cs.selection, '')))
                        END
                      )
                  AND COALESCE(q.value ->> 'line', '') ~ '^[0-9]+([.][0-9]+)?$'
                  AND ABS(
                        (q.value ->> 'line')::NUMERIC
                        - (cs.line)::NUMERIC
                      ) < 0.000001
            )
            ORDER BY
                cs.fixture_id,
                LOWER(TRIM(COALESCE(cs.market, ''))),
                CASE
                    WHEN LOWER(TRIM(COALESCE(cs.selection, ''))) LIKE 'over%%' THEN 'over'
                    WHEN LOWER(TRIM(COALESCE(cs.selection, ''))) LIKE 'under%%' THEN 'under'
                    ELSE LOWER(TRIM(COALESCE(cs.selection, '')))
                END,
                (cs.line)::NUMERIC,
                cs.signal_generated_at ASC
        )
        SELECT ous.*
        FROM oldest_unresolved_signal ous
        ORDER BY
            ous.kickoff ASC,
            ous.fixture_id ASC,
            ous.signal_generated_at ASC,
            ous.market ASC,
            ous.selection ASC,
            (ous.line)::NUMERIC ASC
        LIMIT %s
    """

    with price.persistence._connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                query,
                (
                    lookback_cutoff,
                    now,
                    lookahead_cutoff,
                    max(1, int(limit)),
                ),
            )
            rows = cur.fetchall()
            columns = [desc.name for desc in cur.description]

    grouped: dict[int, dict[str, Any]] = {}
    signal_count = 0
    for raw_row in rows:
        row = dict(zip(columns, raw_row))
        try:
            fixture_id = int(row.get("fixture_id"))
        except (TypeError, ValueError):
            continue
        line = price._num(row.get("line"))
        if line is None:
            continue
        kickoff = row.get("kickoff")
        signal_at = row.get("signal_generated_at")
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
                "signal_times": [],
                "signals": [],
            },
        )
        if signal_at is not None:
            record["signal_times"].append(signal_at)
        record["signals"].append(
            {
                "market_family": "TEAM_TOTALS",
                "market": row.get("market"),
                "selection": row.get("selection"),
                "line": line,
                "signal_generated_at": (
                    signal_at.isoformat() if isinstance(signal_at, datetime) else signal_at
                ),
                "candidate_source": "DERIVATIVE_INTELLIGENCE:team_totals_intelligence",
                "signal_anchor_policy": ANCHOR_POLICY,
            }
        )
        signal_count += 1

    candidate_events: list[dict[str, Any]] = []
    for record in grouped.values():
        signal_times = [value for value in record["signal_times"] if value is not None]
        anchor_at = min(signal_times) if signal_times else None
        candidate_events.append(
            {
                "event_type": price.TEAM_TOTALS_SPILLOVER_EVENT_TYPE,
                "stage": price._maturation_stage(record.get("kickoff"), now),
                "fixture": record["fixture"],
                "classification": "RESEARCH_ONLY",
                "bet_eligible": False,
                "research_only": True,
                "decision_weight": 0.0,
                "team_totals_clv_maturation": {
                    "candidate_source": "POSTGRES_TEAM_TOTALS_OLDEST_UNRESOLVED_EXACT_SIGNAL",
                    "signal_generated_at": (
                        anchor_at.isoformat() if isinstance(anchor_at, datetime) else anchor_at
                    ),
                    "signals": list(record["signals"]),
                    "signal_anchor_policy": ANCHOR_POLICY,
                    "provider_requests_before_price_resolver": 0,
                    "primary_markets_preempted": False,
                    "requires_provider_update_after_signal": True,
                    "requires_same_selection_and_line": True,
                    "strict_close_semantics_changed": False,
                    "historical_signal_mutated": False,
                },
            }
        )

    candidate_events.sort(
        key=lambda event: (
            str(((event.get("fixture") or {}).get("kickoff") or "")),
            int(((event.get("fixture") or {}).get("fixture_id") or 0)),
        )
    )
    return {
        "candidate_events": candidate_events,
        "candidate_count": len(candidate_events),
        "candidate_signal_count": signal_count,
        "source": "POSTGRES_TEAM_TOTALS_CLV_MATURATION_BACKLOG_V2_OLDEST_UNRESOLVED_EXACT",
        "signal_anchor_policy": ANCHOR_POLICY,
        "lookback_hours": lookback_hours,
        "provider_requests_added": 0,
        "provider_budget_changed": False,
        "strict_close_semantics_changed": False,
        "historical_rows_mutated": False,
    }
