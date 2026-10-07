from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from mcp_gateway import primary_clv_anchor_v4 as legacy
from mcp_gateway import price_resolver_v4 as price

MODEL_VERSION = "SOCCER_PRIMARY_CLV_ANCHOR_NORMALIZED_V4_1.0.0"


def _flatten_candidate_keys(report: dict[str, Any]) -> set[tuple[str, ...]]:
    keys: set[tuple[str, ...]] = set()
    for event in report.get("candidate_events") or []:
        if not isinstance(event, dict):
            continue
        fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
        fixture_id = str(fixture.get("fixture_id") or "")
        meta = (
            event.get("primary_clv_maturation")
            if isinstance(event.get("primary_clv_maturation"), dict)
            else {}
        )
        for signal in meta.get("signals") or []:
            if not isinstance(signal, dict):
                continue
            keys.add(
                (
                    fixture_id,
                    str(signal.get("market_family") or "").upper(),
                    str(signal.get("market") or "").strip().lower(),
                    str(signal.get("signal_generated_at") or ""),
                    str(signal.get("candidate_source") or ""),
                )
            )
    return keys


def load_primary_clv_maturation_backlog(
    *,
    lookback_days: int | None = None,
    lookahead_minutes: int | None = None,
    limit: int | None = None,
    include_diagnostics: bool = False,
) -> dict[str, Any]:
    """Read primary CLV candidates from normalized signal anchors.

    Strict-close chronology and market matching are identical to the legacy
    loader. The only change is replacing repeated JSON-array expansion from
    soccer_pipeline_runs with a normalized relational source.
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
        "signal_anchor_policy": legacy.ANCHOR_POLICY,
        "diagnostic_schema_version": legacy.DIAGNOSTIC_SCHEMA_VERSION,
        "diagnostic_status": "DEFERRED_OFFLINE",
        "diagnostic_family_counts": {},
        "provider_requests_added": 0,
        "selection_logic_changed": False,
        "normalized_storage": True,
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
                        a.fixture_id,
                        a.market_family,
                        a.market,
                        a.signal_generated_at,
                        a.candidate_source,
                        a.source_priority,
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
                    FROM soccer_primary_clv_signal_anchors a
                    JOIN soccer_fixtures f
                      ON f.fixture_id = a.fixture_id
                    WHERE a.signal_generated_at >= %s
                      AND a.signal_generated_at < f.kickoff
                      AND f.kickoff > %s
                      AND f.kickoff <= %s
                      AND COALESCE(f.status, 'NS') NOT IN (
                          'FT','AET','PEN','CANC','PST','ABD','AWD','WO'
                      )
                      AND a.market_family IN ('1X2','FT_TOTALS','BTTS')
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
                          AND LOWER(TRIM(COALESCE(m.market, ''))) =
                              LOWER(TRIM(COALESCE(cs.market, '')))
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
                      AND LOWER(TRIM(COALESCE(m.market, ''))) =
                          LOWER(TRIM(COALESCE(ous.market, '')))
                ) visit ON TRUE
                ORDER BY
                    COALESCE(visit.later_capture_visits, 0) ASC,
                    ous.kickoff ASC,
                    ous.fixture_id ASC,
                    ous.market_family ASC
                LIMIT %s
                """,
                (cutoff, now, lookahead, limit),
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
                "signal_anchor_policy": legacy.ANCHOR_POLICY,
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
                    "candidate_source": (
                        "POSTGRES_NORMALIZED_OLDEST_UNRESOLVED_SIGNAL_NO_LATER_REAL_QUOTE"
                    ),
                    "signals": list(record["signals"]),
                    "signal_anchor_policy": legacy.ANCHOR_POLICY,
                    "provider_requests_before_price_resolver": 0,
                    "primary_markets_preempted": False,
                    "requires_provider_update_after_signal": True,
                    "strict_close_semantics_changed": False,
                    "historical_signal_mutated": False,
                    "normalized_storage": True,
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
        "source": "POSTGRES_PRIMARY_CLV_MATURATION_BACKLOG_V4_NORMALIZED",
        "signal_anchor_policy": legacy.ANCHOR_POLICY,
        "diagnostic_schema_version": legacy.DIAGNOSTIC_SCHEMA_VERSION,
        "diagnostic_status": "DEFERRED_OFFLINE",
        "diagnostic_family_counts": {},
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
        "normalized_storage": True,
        "legacy_json_expansion_used": False,
        "include_diagnostics_requested": bool(include_diagnostics),
    }


def compare_with_legacy(
    *,
    lookback_days: int = 180,
    lookahead_minutes: int = 180,
    limit: int = 80,
) -> dict[str, Any]:
    legacy_report = legacy.load_primary_clv_maturation_backlog(
        lookback_days=lookback_days,
        lookahead_minutes=lookahead_minutes,
        limit=limit,
        include_diagnostics=False,
    )
    normalized_report = load_primary_clv_maturation_backlog(
        lookback_days=lookback_days,
        lookahead_minutes=lookahead_minutes,
        limit=limit,
        include_diagnostics=False,
    )
    legacy_keys = _flatten_candidate_keys(legacy_report)
    normalized_keys = _flatten_candidate_keys(normalized_report)
    missing = sorted(legacy_keys - normalized_keys)
    extra = sorted(normalized_keys - legacy_keys)
    return {
        "model_version": MODEL_VERSION,
        "status": "EQUIVALENT" if not missing and not extra else "MISMATCH",
        "equivalent": not missing and not extra,
        "legacy_source": legacy_report.get("source"),
        "normalized_source": normalized_report.get("source"),
        "legacy_candidate_events": int(legacy_report.get("candidate_count") or 0),
        "normalized_candidate_events": int(normalized_report.get("candidate_count") or 0),
        "legacy_signal_keys": len(legacy_keys),
        "normalized_signal_keys": len(normalized_keys),
        "missing_from_normalized_count": len(missing),
        "extra_in_normalized_count": len(extra),
        "missing_from_normalized_examples": [list(value) for value in missing[:20]],
        "extra_in_normalized_examples": [list(value) for value in extra[:20]],
        "provider_requests_added": 0,
        "strict_close_semantics_changed": False,
        "models_changed": False,
        "thresholds_changed": False,
        "gates_changed": False,
        "provider_budget_changed": False,
        "canonical_bet_logic_changed": False,
        "production_promotion_allowed": False,
    }
