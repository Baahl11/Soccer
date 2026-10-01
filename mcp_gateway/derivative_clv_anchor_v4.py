from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from mcp_gateway import price_resolver_v4 as price

MODEL_VERSION = "SOCCER_DERIVATIVE_CLV_ANCHOR_V4_1.0.0"
ANCHOR_POLICY = "OLDEST_UNRESOLVED_POINT_IN_TIME_SIGNAL_PER_EXACT_DERIVATIVE"

FAMILY_CONFIG: dict[str, tuple[str, str]] = {
    "1H": (
        "one_h_goals_intelligence",
        "DERIVATIVE_INTELLIGENCE:one_h_goals_intelligence",
    ),
    "2H": (
        "two_h_goals_intelligence",
        "DERIVATIVE_INTELLIGENCE:two_h_goals_intelligence",
    ),
    "FT_CORNERS": (
        "corners_intelligence",
        "DERIVATIVE_INTELLIGENCE:corners_intelligence",
    ),
    "TEAM_CORNERS": (
        "team_corners_intelligence",
        "DERIVATIVE_INTELLIGENCE:team_corners_intelligence",
    ),
}


def _load_exact_derivative_backlog(
    *,
    families: tuple[str, ...],
    lookback_days: int | None = None,
    lookahead_minutes: int | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    normalized_families = tuple(
        family
        for family in dict.fromkeys(str(value).upper() for value in families)
        if family in FAMILY_CONFIG
    )
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
        "provider_requests_added": 0,
        "provider_budget_changed": False,
        "strict_close_semantics_changed": False,
        "historical_rows_mutated": False,
    }
    if not normalized_families:
        empty["source"] = "NO_SUPPORTED_DERIVATIVE_FAMILIES"
        return empty
    if not price.persistence.persistence_configured():
        return empty

    price.persistence.ensure_schema()
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=lookback_days)
    lookahead = now + timedelta(minutes=lookahead_minutes)

    fragments: list[str] = []
    for family in normalized_families:
        container_key, source = FAMILY_CONFIG[family]
        fragments.append(
            f"""
            SELECT
                e.fixture_id,
                '{family}'::TEXT AS market_family,
                sig.row ->> 'market' AS market,
                sig.row ->> 'selection' AS selection,
                sig.row ->> 'line' AS line,
                e.generated_at AS signal_generated_at,
                '{source}'::TEXT AS candidate_source,
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
            JOIN soccer_fixtures f ON f.fixture_id = e.fixture_id
            CROSS JOIN bounds b
            CROSS JOIN LATERAL jsonb_array_elements(
                CASE
                    WHEN jsonb_typeof(
                        COALESCE(
                            e.payload -> '{container_key}' -> 'observed_market_rows',
                            '[]'::jsonb
                        )
                    ) = 'array'
                    THEN COALESCE(
                        e.payload -> '{container_key}' -> 'observed_market_rows',
                        '[]'::jsonb
                    )
                    ELSE '[]'::jsonb
                END
            ) AS sig(row)
            WHERE e.generated_at >= b.cutoff
              AND e.generated_at < f.kickoff
              AND f.kickoff > b.now_utc
              AND f.kickoff <= b.lookahead
              AND COALESCE(f.status, 'NS') NOT IN ('FT','AET','PEN','CANC','PST','ABD','AWD','WO')
              AND NULLIF(sig.row ->> 'market', '') IS NOT NULL
              AND NULLIF(sig.row ->> 'selection', '') IS NOT NULL
              AND COALESCE(sig.row ->> 'line', '') ~ '^[0-9]+([.][0-9]+)?$'
              AND COALESCE(sig.row ->> 'decimal_price', sig.row ->> 'price', '') ~ '^[0-9]+([.][0-9]+)?$'
              AND COALESCE(sig.row ->> 'decimal_price', sig.row ->> 'price')::DOUBLE PRECISION > 1.0
            """
        )

    candidate_sql = "\nUNION ALL\n".join(fragments)
    query = f"""
        WITH bounds AS (
            SELECT
                %s::timestamptz AS cutoff,
                %s::timestamptz AS now_utc,
                %s::timestamptz AS lookahead
        ),
        candidate_signal AS (
            {candidate_sql}
        ),
        oldest_unresolved_signal AS (
            SELECT DISTINCT ON (
                cs.fixture_id,
                cs.market_family,
                LOWER(TRIM(COALESCE(cs.market, ''))),
                CASE
                    WHEN LOWER(TRIM(COALESCE(cs.selection, ''))) LIKE 'over%%' THEN 'over'
                    WHEN LOWER(TRIM(COALESCE(cs.selection, ''))) LIKE 'under%%' THEN 'under'
                    ELSE LOWER(TRIM(COALESCE(cs.selection, '')))
                END,
                (cs.line)::NUMERIC
            )
                cs.fixture_id,
                cs.market_family,
                cs.market,
                cs.selection,
                cs.line,
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
                cs.market_family,
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
            ous.market_family ASC,
            ous.market ASC,
            ous.selection ASC,
            (ous.line)::NUMERIC ASC
        LIMIT %s
    """

    with price.persistence._connect() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (cutoff, now, lookahead, limit))
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
        if family not in normalized_families:
            continue
        line = price._num(row.get("line"))
        if line is None:
            continue
        signal_at = row.get("signal_generated_at")
        kickoff = row.get("kickoff")
        source = str(row.get("candidate_source") or "UNKNOWN")
        family_counts[family] += 1
        source_counts[source] += 1
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
                "selection": row.get("selection"),
                "line": line,
                "signal_generated_at": (
                    signal_at.isoformat() if isinstance(signal_at, datetime) else signal_at
                ),
                "candidate_source": source,
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
                    "candidate_source": "POSTGRES_DERIVATIVE_OLDEST_UNRESOLVED_SIGNAL_NO_LATER_EXACT_QUOTE",
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
        "source": "POSTGRES_DERIVATIVE_CLV_MATURATION_BACKLOG_V2_OLDEST_UNRESOLVED",
        "signal_anchor_policy": ANCHOR_POLICY,
        "provider_requests_added": 0,
        "provider_budget_changed": False,
        "strict_close_semantics_changed": False,
        "historical_rows_mutated": False,
    }


def load_one_h_clv_maturation_backlog(**kwargs: Any) -> dict[str, Any]:
    return _load_exact_derivative_backlog(families=("1H",), **kwargs)


def load_two_h_clv_maturation_backlog(**kwargs: Any) -> dict[str, Any]:
    return _load_exact_derivative_backlog(families=("2H",), **kwargs)


def load_corners_clv_maturation_backlog(**kwargs: Any) -> dict[str, Any]:
    return _load_exact_derivative_backlog(
        families=("FT_CORNERS", "TEAM_CORNERS"),
        **kwargs,
    )
