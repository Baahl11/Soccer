from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from mcp_gateway import build_signal_ledger as legacy
from mcp_gateway import persistence

SCHEMA_VERSION = "2.1.0"
SOURCE = "POSTGRES_SOCCER_REFRESH_EVENTS_POINT_IN_TIME"
DEFAULT_TIMEZONE = "America/Mexico_City"


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _local_iso(value: Any, timezone_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, datetime):
        try:
            value = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return str(value)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    try:
        zone = ZoneInfo(timezone_name or DEFAULT_TIMEZONE)
    except Exception:
        zone = ZoneInfo(DEFAULT_TIMEZONE)
    return value.astimezone(zone).isoformat()


def _fixture_from_row(row: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    """Return point-in-time fixture data with stable metadata fallbacks only.

    soccer_fixtures is a latest-state table. It is safe for stable identity and
    schedule/team metadata, but its status/result fields must never be copied
    into a historical ledger row because that would leak future knowledge.
    """
    point_in_time = event.get("fixture")
    fixture = dict(point_in_time) if isinstance(point_in_time, dict) else {}
    stable_fallbacks = {
        "fixture_id": row.get("fixture_id"),
        "kickoff": _iso(row.get("kickoff")),
        "league_id": row.get("league_id"),
        "league": row.get("league"),
        "country": row.get("country"),
        "season": row.get("season"),
        "home_team_id": row.get("home_team_id"),
        "home_team": row.get("home_team"),
        "away_team_id": row.get("away_team_id"),
        "away_team": row.get("away_team"),
    }
    for key, value in stable_fallbacks.items():
        if fixture.get(key) is None and value is not None:
            fixture[key] = value
    return fixture


def _pipeline_tick_view(row: dict[str, Any], generated_at: Any, timezone_name: str) -> dict[str, Any]:
    payload = row.get("pipeline_payload")
    payload = payload if isinstance(payload, dict) else {}
    generated_at_local = row.get("pipeline_generated_at_local")
    return {
        "generated_at_utc": _iso(generated_at),
        "generated_at_local": _iso(generated_at_local) or _local_iso(generated_at, timezone_name),
        "timezone": timezone_name,
        "version": payload.get("version"),
        "model_version": payload.get("model_version"),
        "match_table_rows": payload.get("match_table_rows") if isinstance(payload.get("match_table_rows"), list) else [],
    }


def _ledger_row(row: dict[str, Any]) -> dict[str, Any] | None:
    event = row.get("event_payload")
    if not isinstance(event, dict):
        return None
    fixture = _fixture_from_row(row, event)
    fid = fixture.get("fixture_id") or row.get("fixture_id")
    if fid is None:
        return None
    try:
        fid = int(fid)
    except (TypeError, ValueError):
        return None

    timezone_name = str(row.get("pipeline_timezone") or DEFAULT_TIMEZONE)
    generated_at = row.get("generated_at")
    tick = _pipeline_tick_view(row, generated_at, timezone_name)
    md = event.get("market_decision") or {}
    coverage = event.get("coverage") or {}
    best_market = legacy._market_snapshot(event.get("best_market"))

    result = {
        "generated_at_local": tick.get("generated_at_local"),
        "generated_at_utc": tick.get("generated_at_utc"),
        "timezone": timezone_name,
        "service_version": tick.get("version"),
        "model_version": tick.get("model_version"),
        "fixture_id": fid,
        "kickoff_local": fixture.get("kickoff") or _iso(row.get("kickoff")),
        "league_id": fixture.get("league_id") or row.get("league_id"),
        "league": fixture.get("league") or row.get("league"),
        "country": fixture.get("country") or row.get("country"),
        "season": fixture.get("season") or row.get("season"),
        "home_team_id": fixture.get("home_team_id") or row.get("home_team_id"),
        "home_team": fixture.get("home_team") or row.get("home_team"),
        "away_team_id": fixture.get("away_team_id") or row.get("away_team_id"),
        "away_team": fixture.get("away_team") or row.get("away_team"),
        # Never fall back to soccer_fixtures.status here. Only the status that
        # existed inside the persisted point-in-time event is admissible.
        "fixture_status": fixture.get("status"),
        "event_type": event.get("event_type") or row.get("event_type"),
        "stage": event.get("stage") or row.get("stage"),
        "classification": event.get("classification") if event.get("classification") is not None else row.get("classification"),
        "tier": event.get("tier"),
        "stake_units": legacy._f(event.get("stake_units")),
        "bet_eligible": event.get("bet_eligible") if event.get("bet_eligible") is not None else row.get("bet_eligible"),
        "availability_confidence": legacy._f(
            event.get("availability_confidence")
            if event.get("availability_confidence") is not None
            else row.get("availability_confidence")
        ),
        "data_tier": coverage.get("data_tier") if isinstance(coverage, dict) else row.get("data_tier"),
        "sporting_shortlist": legacy._shortlist_snapshot(event.get("sporting_shortlist")),
        "sporting_screen_initial": legacy._shortlist_snapshot(event.get("sporting_screen_initial")),
        "sporting_screen_refined": legacy._shortlist_snapshot(event.get("sporting_screen_refined")),
        "raw_projection": legacy._raw_snapshot(event.get("raw_projection")),
        "lineups": legacy._lineup_snapshot(event.get("lineups")),
        "best_market": best_market,
        "phase16_calibration_provenance": legacy._phase16_calibration_snapshot(tick, fid, best_market),
        "market_decision": {
            "status": md.get("status"),
            "reason": md.get("reason"),
            "first_half_markets_ignored": md.get("first_half_markets_ignored"),
            "period_markets_ignored": md.get("period_markets_ignored"),
            "btts_market_mode": md.get("btts_market_mode"),
        } if isinstance(md, dict) and md else None,
        "notes": event.get("notes") or [],
        "result": legacy._result_snapshot(event, fixture),
        "error": event.get("error"),
        "ledger_source": SOURCE,
        "postgres_event_id": row.get("event_id"),
        "pipeline_run_matched": bool(row.get("pipeline_run_id")),
    }
    result["event_key"] = legacy._event_key(result)
    return result


def summarize_rows(
    rows: list[dict[str, Any]],
    *,
    source: str,
    postgres_rows: int | None = None,
    legacy_rows_loaded: int | None = None,
    duplicate_event_keys_replaced: int | None = None,
) -> dict[str, Any]:
    rows = sorted(
        rows,
        key=lambda r: (str(r.get("generated_at_local") or ""), int(r.get("fixture_id") or 0), str(r.get("stage") or "")),
    )
    stage_counts = Counter(str(r.get("stage") or "UNKNOWN") for r in rows)
    class_counts = Counter(str(r.get("classification") or "UNKNOWN") for r in rows)
    signal_counts: Counter[str] = Counter()
    fixtures: set[int] = set()
    finals: set[int] = set()
    with_market = with_raw = with_lineups = with_shadow = with_tactical = 0
    with_phase16_provenance = phase16_promotion_shadow_eligible = 0
    phase16_calibrated = phase16_frozen_artifact = 0
    postgres_source_rows = 0

    for row in rows:
        try:
            fixtures.add(int(row.get("fixture_id")))
        except (TypeError, ValueError):
            pass
        if row.get("result"):
            try:
                finals.add(int(row.get("fixture_id")))
            except (TypeError, ValueError):
                pass
        with_market += bool(row.get("best_market"))
        with_raw += bool(row.get("raw_projection"))
        with_lineups += bool(row.get("lineups"))
        with_shadow += bool((row.get("raw_projection") or {}).get("relative_strength_shadow"))
        with_tactical += bool((row.get("result") or {}).get("tactical_stats"))
        postgres_source_rows += str(row.get("ledger_source") or "") == SOURCE
        provenance = row.get("phase16_calibration_provenance")
        has_provenance = isinstance(provenance, dict) and bool(provenance)
        with_phase16_provenance += has_provenance
        phase16_promotion_shadow_eligible += has_provenance and provenance.get("promotion_shadow_eligible") is True
        calibrated = has_provenance and bool(provenance.get("calibrated_probability_fields"))
        phase16_calibrated += calibrated
        phase16_frozen_artifact += calibrated and bool(
            (provenance.get("phase16_calibrator_artifact") or {}).get("fingerprint_sha256")
        )
        for track in (row.get("sporting_shortlist") or {}).get("tracks") or []:
            signal_counts[str(track)] += 1

    first_local = rows[0].get("generated_at_local") if rows else None
    last_local = rows[-1].get("generated_at_local") if rows else None
    first_utc = rows[0].get("generated_at_utc") if rows else None
    last_utc = rows[-1].get("generated_at_utc") if rows else None
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "timezone_basis": DEFAULT_TIMEZONE,
        "rows": len(rows),
        "unique_fixtures": len(fixtures),
        "fixtures_with_final_result": len(finals),
        "rows_with_raw_projection": with_raw,
        "rows_with_relative_strength_shadow": with_shadow,
        "rows_with_market": with_market,
        "rows_with_lineups": with_lineups,
        "rows_with_tactical_stats": with_tactical,
        "rows_with_phase16_calibration_provenance": with_phase16_provenance,
        "rows_phase16_promotion_shadow_eligible": phase16_promotion_shadow_eligible,
        "rows_with_applied_phase16_calibration": phase16_calibrated,
        "rows_with_frozen_phase16_calibrator_artifact": phase16_frozen_artifact,
        "stage_counts": dict(sorted(stage_counts.items())),
        "classification_counts": dict(sorted(class_counts.items())),
        "sport_signal_counts": dict(sorted(signal_counts.items())),
        "first_generated_at_local": first_local,
        "first_generated_at_utc": first_utc,
        "last_generated_at_local": last_local,
        "last_generated_at_utc": last_utc,
        "last_materialized_ledger_row_at_local": last_local,
        "last_materialized_ledger_row_at_utc": last_utc,
        "postgres_source_rows": postgres_source_rows,
        "provider_requests_added": 0,
        "canonical_bet_logic_changed": False,
        "model_weights_changed": False,
        "historical_probabilities_recomputed": False,
        "historical_rows_recalibrated": False,
        "synthetic_close_rows_added": 0,
        "point_in_time_payloads_reused": True,
    }
    if postgres_rows is not None:
        summary["postgres_rows"] = int(postgres_rows)
    if legacy_rows_loaded is not None:
        summary["legacy_rows_loaded"] = int(legacy_rows_loaded)
    if duplicate_event_keys_replaced is not None:
        summary["duplicate_event_keys_replaced_by_postgres"] = int(duplicate_event_keys_replaced)
    return summary


def build_from_postgres(*, lookback_days: int = 60, max_rows: int = 50000) -> dict[str, Any]:
    lookback_days = max(1, int(lookback_days))
    max_rows = max(1, int(max_rows))
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

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
                WHERE e.generated_at >= %s
                ORDER BY e.generated_at ASC, e.event_id ASC
                LIMIT %s
                """,
                (cutoff, max_rows),
            )
            columns = [desc.name for desc in cur.description]
            raw_rows = [dict(zip(columns, values)) for values in cur.fetchall()]

    by_key: dict[str, dict[str, Any]] = {}
    rows_without_pipeline_match = 0
    for raw in raw_rows:
        if not raw.get("pipeline_run_id"):
            rows_without_pipeline_match += 1
        ledger_row = _ledger_row(raw)
        if ledger_row is None:
            continue
        by_key[ledger_row["event_key"]] = ledger_row

    rows = sorted(
        by_key.values(),
        key=lambda r: (str(r.get("generated_at_local") or ""), int(r.get("fixture_id") or 0), str(r.get("stage") or "")),
    )
    summary = summarize_rows(rows, source=SOURCE, postgres_rows=len(rows))
    summary.update({
        "lookback_days": lookback_days,
        "max_rows": max_rows,
        "raw_refresh_events_loaded": len(raw_rows),
        "rows_without_pipeline_run_match": rows_without_pipeline_match,
        "event_key_deduplicated_rows": len(raw_rows) - len(rows),
        "strict_point_in_time_policy": (
            "REUSE_SOCCER_REFRESH_EVENTS_PAYLOAD_AT_GENERATED_AT; "
            "MATCH_ONLY_SAME_TICK_PIPELINE_ENVELOPE_WITHIN_1_SECOND; "
            "NEVER_FILL_HISTORICAL_STATUS_OR_RESULT_FROM_MUTABLE_SOCCER_FIXTURES; "
            "DO_NOT_RECALCULATE_HISTORICAL_PROBABILITIES_OR_CALIBRATORS"
        ),
        "production_promotion_allowed": False,
    })
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "OK",
        "source": SOURCE,
        "summary": summary,
        "rows": rows,
        "provider_requests_added": 0,
        "canonical_bet_logic_changed": False,
        "model_weights_changed": False,
        "historical_probabilities_recomputed": False,
        "historical_rows_recalibrated": False,
        "synthetic_close_rows_added": 0,
        "production_promotion_allowed": False,
    }
