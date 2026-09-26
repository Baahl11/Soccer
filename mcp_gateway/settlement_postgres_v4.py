from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from mcp_gateway import evaluate_postgame as ep
from mcp_gateway import persistence

SCHEMA_VERSION = "1.1.0"
MODEL_VERSION = "SOCCER_SETTLEMENT_POSTGRES_V4_1.1.0"
ACTIONABLE_CLASSES = {"BET", "LEAN"}


def _parse_dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not value:
        return None
    try:
        out = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return out if out.tzinfo is not None else out.replace(tzinfo=timezone.utc)


def _result_from_row(row: dict[str, Any]) -> dict[str, Any] | None:
    if row.get("home_goals") is None or row.get("away_goals") is None:
        return None
    event_payload = row.get("result_event_payload")
    event_payload = event_payload if isinstance(event_payload, dict) else {}
    raw_result = event_payload.get("result")
    result = dict(raw_result) if isinstance(raw_result, dict) else {}

    goals = result.get("goals")
    if not isinstance(goals, dict):
        goals = {}
    goals.setdefault("home", row.get("home_goals"))
    goals.setdefault("away", row.get("away_goals"))
    result["goals"] = goals

    if not isinstance(result.get("score"), dict) and isinstance(row.get("final_score"), dict):
        result["score"] = row.get("final_score")
    result.setdefault("status", row.get("final_status"))

    tactical = event_payload.get("postgame_tactical_stats")
    if not isinstance(tactical, dict):
        tactical = event_payload.get("tactical_stats")
    if isinstance(tactical, dict):
        result.setdefault("tactical_stats", tactical)
    return result


def _best_market(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    best = payload.get("best_market")
    return best if isinstance(best, dict) and best else None


def _decision_key(row: dict[str, Any]) -> tuple[Any, ...]:
    best = row.get("best_market") or {}
    return (
        int(row["fixture_id"]),
        str(row.get("classification") or "").upper(),
        ep.market_family(best),
        ep.norm(best.get("market")),
        ep.norm(best.get("selection")),
        ep.fnum(best.get("line")),
    )


def _normalized_actionable_rows(raw_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple[Any, ...], tuple[datetime, dict[str, Any]]] = {}
    for raw in raw_rows:
        if not isinstance(raw, dict):
            continue
        classification = str(raw.get("classification") or "").upper()
        if classification not in ACTIONABLE_CLASSES:
            continue
        try:
            fixture_id = int(raw.get("fixture_id"))
        except (TypeError, ValueError):
            continue
        generated_at = _parse_dt(raw.get("generated_at"))
        kickoff = _parse_dt(raw.get("kickoff"))
        if generated_at is None or kickoff is None or generated_at >= kickoff:
            continue
        best = _best_market(raw.get("event_payload"))
        result = _result_from_row(raw)
        if best is None or result is None:
            continue

        payload = raw.get("event_payload") if isinstance(raw.get("event_payload"), dict) else {}
        row = {
            "fixture_id": fixture_id,
            "generated_at": generated_at,
            "kickoff": kickoff,
            "stage": raw.get("stage"),
            "classification": classification,
            "availability_confidence": raw.get("availability_confidence"),
            "bet_eligible": raw.get("bet_eligible"),
            "data_tier": raw.get("data_tier"),
            "league": raw.get("league"),
            "home_team_id": raw.get("home_team_id"),
            "home_team": raw.get("home_team"),
            "away_team_id": raw.get("away_team_id"),
            "away_team": raw.get("away_team"),
            "tier": payload.get("tier"),
            "stake_units": ep.fnum(payload.get("stake_units")) or 1.0,
            "quote_timestamp": _parse_dt(raw.get("quote_timestamp")),
            "quote_provider_update": _parse_dt(raw.get("quote_provider_update")),
            "lineup_timestamp": _parse_dt(raw.get("lineup_timestamp")),
            "lineup_observation_timestamp": (
                _parse_dt(raw.get("lineup_timestamp"))
                or _parse_dt(raw.get("feature_timestamp"))
            ),
            "feature_timestamp": _parse_dt(raw.get("feature_timestamp")),
            "best_market": best,
            "result": result,
        }
        key = _decision_key(row)
        prior = latest.get(key)
        if prior is None or generated_at > prior[0]:
            latest[key] = (generated_at, row)

    return [value[1] for _, value in sorted(latest.items(), key=lambda item: item[0])]


def _settlement_row(row: dict[str, Any]) -> dict[str, Any]:
    best = row["best_market"]
    result = row["result"]
    outcome = ep.grade_market(
        best,
        result,
        row.get("home_team") or "",
        row.get("away_team") or "",
        row.get("home_team_id"),
        row.get("away_team_id"),
    )
    price = ep.fnum(best.get("decimal_price"))
    stake = ep.fnum(row.get("stake_units")) or 1.0
    generated_at = row.get("generated_at")
    kickoff = row.get("kickoff")
    return {
        "schema_version": SCHEMA_VERSION,
        "event_key": (
            f"postgres:{row['fixture_id']}:{generated_at.isoformat()}"
            if isinstance(generated_at, datetime)
            else f"postgres:{row['fixture_id']}"
        ),
        "fixture_id": row["fixture_id"],
        "kickoff_local": kickoff.isoformat() if isinstance(kickoff, datetime) else kickoff,
        "generated_at_local": generated_at.isoformat() if isinstance(generated_at, datetime) else generated_at,
        "stage": row.get("stage"),
        "league": row.get("league"),
        "home_team_id": row.get("home_team_id"),
        "home_team": row.get("home_team"),
        "away_team_id": row.get("away_team_id"),
        "away_team": row.get("away_team"),
        "classification": row.get("classification"),
        "tier": row.get("tier"),
        "data_tier": row.get("data_tier"),
        "availability_confidence": row.get("availability_confidence"),
        "bet_eligible": row.get("bet_eligible"),
        "period": ep.period_type(ep.norm(best.get("market"))),
        "market_family": ep.market_family(best),
        "canonical_ft_market": ep.canonical_full_match(best),
        "market": best.get("market"),
        "selection": best.get("selection"),
        "line": ep.fnum(best.get("line")),
        "decimal_price": price,
        "bookmaker": best.get("bookmaker") or best.get("book"),
        "quote_timestamp": row.get("quote_timestamp").isoformat() if isinstance(row.get("quote_timestamp"), datetime) else row.get("quote_timestamp"),
        "bookmaker_timestamp": row.get("quote_timestamp").isoformat() if isinstance(row.get("quote_timestamp"), datetime) else row.get("quote_timestamp"),
        "quote_provider_update": row.get("quote_provider_update").isoformat() if isinstance(row.get("quote_provider_update"), datetime) else row.get("quote_provider_update"),
        "lineup_timestamp": row.get("lineup_timestamp").isoformat() if isinstance(row.get("lineup_timestamp"), datetime) else row.get("lineup_timestamp"),
        "lineup_captured_at": row.get("lineup_timestamp").isoformat() if isinstance(row.get("lineup_timestamp"), datetime) else row.get("lineup_timestamp"),
        "lineup_observation_timestamp": row.get("lineup_observation_timestamp").isoformat() if isinstance(row.get("lineup_observation_timestamp"), datetime) else row.get("lineup_observation_timestamp"),
        "feature_timestamp": row.get("feature_timestamp").isoformat() if isinstance(row.get("feature_timestamp"), datetime) else row.get("feature_timestamp"),
        "feature_captured_at": row.get("feature_timestamp").isoformat() if isinstance(row.get("feature_timestamp"), datetime) else row.get("feature_timestamp"),
        "stake_units": stake,
        "settlement_status": outcome,
        "settled": outcome in {"WIN", "LOSS", "PUSH"},
        "roi_units": ep.roi_units(outcome, price, stake),
        "has_tactical_stats": bool(ep.tactical_stats(result)),
        "result": result,
        "source": "POSTGRES_REFRESH_EVENT",
    }


def _market_performance_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_market: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_market_class: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        family = str(row.get("market_family") or "UNKNOWN")
        classification = str(row.get("classification") or "UNKNOWN")
        by_market[family].append(row)
        by_market_class[family][classification].append(row)

    by_market_family = {
        family: ep.performance_summary(group)
        for family, group in sorted(by_market.items())
    }
    by_market_family_and_classification = {
        family: {
            classification: ep.performance_summary(group)
            for classification, group in sorted(class_groups.items())
        }
        for family, class_groups in sorted(by_market_class.items())
    }
    promotion_gate_review = {
        family: ep.tier_gate(summary)
        for family, summary in by_market_family.items()
    }
    promotion_gate_by_market_family_and_classification = {
        family: {
            classification: ep.tier_gate(summary)
            for classification, summary in class_summaries.items()
        }
        for family, class_summaries in by_market_family_and_classification.items()
    }
    return {
        "schema_version": "1.2.0",
        "source": "POSTGRES_SOCCER_REFRESH_EVENTS_AND_RESULTS",
        "timezone_basis": "UTC_TIMESTAMPTZ; FIXTURE_LOCAL_LABELS_PRESERVED_IN_PAYLOAD",
        "settlement_decisions": len(rows),
        "by_market_family": by_market_family,
        "by_market_family_and_classification": by_market_family_and_classification,
        "promotion_gate_review": promotion_gate_review,
        "promotion_gate_by_market_family_and_classification": promotion_gate_by_market_family_and_classification,
        "minimum_sample_policy": {
            "directional_read": 20,
            "tier_b_candidate": 20,
            "tier_a_candidate": 50,
            "tier_s_candidate": 100,
        },
        "safety_note": (
            "Postgres-native settlement is read-only. It reuses the canonical evaluate_postgame "
            "grading logic and never changes runtime classifications, tiers, stakes, model weights, "
            "or historical decisions."
        ),
    }


def build_report_from_rows(raw_rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    raw_list = [row for row in raw_rows if isinstance(row, dict)]
    normalized = _normalized_actionable_rows(raw_list)
    settlement_rows = [_settlement_row(row) for row in normalized]
    market_summary = _market_performance_summary(settlement_rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": "POSTGRES_SETTLEMENT_ACTIVE",
        "source": "soccer_refresh_events + soccer_fixtures + soccer_results",
        "raw_actionable_rows_loaded": len(raw_list),
        "deduped_actionable_decisions": len(settlement_rows),
        "settled_decisions": sum(1 for row in settlement_rows if row.get("settled")),
        "market_performance_summary": market_summary,
        "rows": settlement_rows,
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
        "runtime_logic_changed": False,
        "notes": [
            "Only persisted BET/LEAN refresh events captured strictly before kickoff are eligible.",
            "The latest snapshot per fixture/classification/market family/market/selection/line is retained, matching legacy settlement dedupe semantics.",
            "Final results are joined from soccer_results solely for grading; they never feed pre-kickoff candidate generation.",
            "This source remains complete after Git history switched to compact operational envelopes without events.",
            "Quote, lineup and feature timestamps are recovered only from persisted snapshots captured at or before the decision timestamp; no post-decision provenance is allowed.",
        ],
    }


def _load_diagnostics(conn, *, lookback_days: int) -> dict[str, Any]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, int(lookback_days)))
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                COUNT(*) AS refresh_events,
                COUNT(*) FILTER (WHERE e.generated_at < f.kickoff) AS prekickoff_events,
                COUNT(*) FILTER (WHERE r.fixture_id IS NOT NULL) AS events_with_result_fixture,
                COUNT(*) FILTER (
                    WHERE e.generated_at < f.kickoff
                      AND r.fixture_id IS NOT NULL
                ) AS prekickoff_events_with_result,
                COUNT(*) FILTER (
                    WHERE jsonb_typeof(e.payload -> 'best_market') = 'object'
                      AND e.payload -> 'best_market' <> '{}'::jsonb
                ) AS events_with_best_market,
                COUNT(*) FILTER (
                    WHERE e.generated_at < f.kickoff
                      AND jsonb_typeof(e.payload -> 'best_market') = 'object'
                      AND e.payload -> 'best_market' <> '{}'::jsonb
                ) AS prekickoff_events_with_best_market,
                COUNT(*) FILTER (
                    WHERE UPPER(COALESCE(e.classification, '')) IN ('BET','LEAN')
                ) AS actionable_classification_events,
                COUNT(*) FILTER (
                    WHERE e.generated_at < f.kickoff
                      AND UPPER(COALESCE(e.classification, '')) IN ('BET','LEAN')
                ) AS prekickoff_actionable_events,
                COUNT(*) FILTER (
                    WHERE e.generated_at < f.kickoff
                      AND r.fixture_id IS NOT NULL
                      AND UPPER(COALESCE(e.classification, '')) IN ('BET','LEAN')
                      AND jsonb_typeof(e.payload -> 'best_market') = 'object'
                      AND e.payload -> 'best_market' <> '{}'::jsonb
                ) AS fully_eligible_rows
            FROM soccer_refresh_events e
            JOIN soccer_fixtures f ON f.fixture_id = e.fixture_id
            LEFT JOIN soccer_results r ON r.fixture_id = e.fixture_id
            WHERE e.generated_at >= %s
            """,
            (cutoff,),
        )
        row = cur.fetchone()
        columns = [desc.name for desc in cur.description]
        totals = dict(zip(columns, row)) if row else {}

        cur.execute(
            """
            SELECT COALESCE(NULLIF(e.classification, ''), '(NULL)') AS classification, COUNT(*) AS n
            FROM soccer_refresh_events e
            JOIN soccer_fixtures f ON f.fixture_id = e.fixture_id
            WHERE e.generated_at >= %s
              AND e.generated_at < f.kickoff
            GROUP BY 1
            ORDER BY n DESC, classification ASC
            LIMIT 25
            """,
            (cutoff,),
        )
        classification_counts = {str(row[0]): int(row[1]) for row in cur.fetchall()}

        cur.execute(
            """
            SELECT COALESCE(NULLIF(e.event_type, ''), '(NULL)') AS event_type, COUNT(*) AS n
            FROM soccer_refresh_events e
            JOIN soccer_fixtures f ON f.fixture_id = e.fixture_id
            WHERE e.generated_at >= %s
              AND e.generated_at < f.kickoff
            GROUP BY 1
            ORDER BY n DESC, event_type ASC
            LIMIT 25
            """,
            (cutoff,),
        )
        event_type_counts = {str(row[0]): int(row[1]) for row in cur.fetchall()}

    return {
        **{key: int(value or 0) for key, value in totals.items()},
        "prekickoff_classification_counts": classification_counts,
        "prekickoff_event_type_counts": event_type_counts,
    }


def _load_rows(conn, *, lookback_days: int, max_rows: int) -> list[dict[str, Any]]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, int(lookback_days)))
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                e.fixture_id,
                e.generated_at,
                e.stage,
                e.classification,
                e.availability_confidence,
                e.bet_eligible,
                e.data_tier,
                e.payload AS event_payload,
                (
                    SELECT m.captured_at
                    FROM soccer_market_snapshots m
                    WHERE m.fixture_id = e.fixture_id
                      AND m.captured_at <= e.generated_at
                      AND LOWER(TRIM(COALESCE(m.market, ''))) = LOWER(TRIM(COALESCE(e.payload -> 'best_market' ->> 'market', '')))
                      AND (
                            NULLIF(COALESCE(e.payload -> 'best_market' ->> 'bookmaker', e.payload -> 'best_market' ->> 'book'), '') IS NULL
                            OR LOWER(TRIM(COALESCE(m.bookmaker, ''))) = LOWER(TRIM(COALESCE(e.payload -> 'best_market' ->> 'bookmaker', e.payload -> 'best_market' ->> 'book', '')))
                          )
                    ORDER BY m.captured_at DESC, m.snapshot_id DESC
                    LIMIT 1
                ) AS quote_timestamp,
                (
                    SELECT m.provider_update
                    FROM soccer_market_snapshots m
                    WHERE m.fixture_id = e.fixture_id
                      AND m.captured_at <= e.generated_at
                      AND LOWER(TRIM(COALESCE(m.market, ''))) = LOWER(TRIM(COALESCE(e.payload -> 'best_market' ->> 'market', '')))
                      AND (
                            NULLIF(COALESCE(e.payload -> 'best_market' ->> 'bookmaker', e.payload -> 'best_market' ->> 'book'), '') IS NULL
                            OR LOWER(TRIM(COALESCE(m.bookmaker, ''))) = LOWER(TRIM(COALESCE(e.payload -> 'best_market' ->> 'bookmaker', e.payload -> 'best_market' ->> 'book', '')))
                          )
                    ORDER BY m.captured_at DESC, m.snapshot_id DESC
                    LIMIT 1
                ) AS quote_provider_update,
                (
                    SELECT l.captured_at
                    FROM soccer_lineup_snapshots l
                    WHERE l.fixture_id = e.fixture_id
                      AND l.captured_at <= e.generated_at
                    ORDER BY l.captured_at DESC, l.snapshot_id DESC
                    LIMIT 1
                ) AS lineup_timestamp,
                (
                    SELECT s.captured_at
                    FROM soccer_feature_snapshots s
                    WHERE s.fixture_id = e.fixture_id
                      AND s.captured_at <= e.generated_at
                    ORDER BY s.captured_at DESC, s.snapshot_id DESC
                    LIMIT 1
                ) AS feature_timestamp,
                f.kickoff,
                f.league,
                f.home_team_id,
                f.home_team,
                f.away_team_id,
                f.away_team,
                r.final_status,
                r.home_goals,
                r.away_goals,
                r.final_score,
                r.payload AS result_event_payload
            FROM soccer_refresh_events e
            JOIN soccer_fixtures f
              ON f.fixture_id = e.fixture_id
            JOIN soccer_results r
              ON r.fixture_id = e.fixture_id
            WHERE e.generated_at >= %s
              AND e.generated_at < f.kickoff
              AND UPPER(COALESCE(e.classification, '')) = ANY(%s)
              AND jsonb_typeof(e.payload -> 'best_market') = 'object'
              AND e.payload -> 'best_market' <> '{}'::jsonb
            ORDER BY e.generated_at ASC
            LIMIT %s
            """,
            (cutoff, sorted(ACTIONABLE_CLASSES), max(100, int(max_rows))),
        )
        columns = [desc.name for desc in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def build_from_postgres(*, lookback_days: int = 180, max_rows: int = 50000) -> dict[str, Any]:
    persistence.ensure_schema()
    with persistence._connect() as conn:
        diagnostics = _load_diagnostics(conn, lookback_days=lookback_days)
        raw_rows = _load_rows(conn, lookback_days=lookback_days, max_rows=max_rows)
    report = build_report_from_rows(raw_rows)
    report["lookback_days"] = int(lookback_days)
    report["source_diagnostics"] = diagnostics
    return report
