from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
import re
from typing import Any

from mcp_gateway import persistence

# Observability only: this module never changes price selection, provider budget, or strict-close eligibility.
MODEL_VERSION = "SOCCER_TEAM_TOTALS_CLOSE_PROVENANCE_V4_1.1.0"
SCHEMA_VERSION = "1.1.0"
MAX_SAMPLES = 80
HISTORICAL_LOOKBACK_HOURS = 12
HISTORICAL_SIGNAL_LIMIT = 2000
TEAM_TOTALS_RESEARCH_STAGES = ("EARLY_RESEARCH", "T-90", "T-60", "T-40", "T-30", "T-20", "T-10", "CLOSE")


def _parse_time(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif value:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _selection_norm(value: Any) -> str:
    normalized = _norm(value)
    if normalized.startswith("over"):
        return "over"
    if normalized.startswith("under"):
        return "under"
    return normalized


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _line_from_selection(value: Any) -> float | None:
    match = re.search(r"(?:over|under)\s*([0-9]+(?:\.[0-9]+)?)", str(value or ""), re.I)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _signal_line(signal: dict[str, Any]) -> float | None:
    line = _num(signal.get("line"))
    if line is None:
        line = _num(signal.get("handicap"))
    if line is None:
        line = _line_from_selection(signal.get("selection"))
    return line


def _value_line(value: dict[str, Any]) -> float | None:
    line = _num(value.get("line"))
    if line is None:
        line = _num(value.get("handicap"))
    if line is None:
        line = _line_from_selection(value.get("selection") if value.get("selection") is not None else value.get("value"))
    return line


def _value_selection(value: dict[str, Any]) -> Any:
    return value.get("selection") if value.get("selection") is not None else value.get("value")


def _event_market_rows(event: dict[str, Any]) -> list[dict[str, Any]]:
    market = event.get("market") if isinstance(event.get("market"), dict) else {}
    rows: list[dict[str, Any]] = []
    for key in ("markets", "research_cards_props_markets"):
        for row in market.get(key) or []:
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _quote_values(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [value for value in (row.get("values") or []) if isinstance(value, dict)]


def _classify_signal(
    *,
    fixture_id: int,
    stage: Any,
    kickoff_raw: Any,
    signal_at_raw: Any,
    market: Any,
    selection: Any,
    line_raw: Any,
    snapshots: list[dict[str, Any]],
) -> dict[str, Any]:
    kickoff = _parse_time(kickoff_raw)
    signal_at = _parse_time(signal_at_raw)
    market_norm = _norm(market)
    selection_norm = _selection_norm(selection)
    line = _num(line_raw)
    if line is None:
        line = _line_from_selection(selection)

    later_market: list[dict[str, Any]] = []
    for snap in snapshots:
        captured_at = _parse_time(snap.get("captured_at"))
        if _norm(snap.get("market")) != market_norm:
            continue
        if signal_at is None or kickoff is None or captured_at is None:
            continue
        if not (signal_at < captured_at < kickoff):
            continue
        later_market.append(snap)

    strict_market: list[dict[str, Any]] = []
    for snap in later_market:
        provider_update = _parse_time(snap.get("provider_update"))
        if signal_at is None or provider_update is None or provider_update <= signal_at:
            continue
        if kickoff is not None and provider_update >= kickoff:
            continue
        strict_market.append(snap)

    exact_matches: list[tuple[dict[str, Any], dict[str, Any]]] = []
    side_matches: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for snap in strict_market:
        for value in _quote_values(snap):
            if _selection_norm(_value_selection(value)) != selection_norm:
                continue
            side_matches.append((snap, value))
            value_line = _value_line(value)
            if line is not None and value_line is not None and abs(value_line - line) < 0.000001:
                exact_matches.append((snap, value))

    best_snap: dict[str, Any] | None = None
    best_value: dict[str, Any] | None = None
    best_provider_update: datetime | None = None
    best_captured_at: datetime | None = None
    for snap, value in exact_matches:
        provider_update = _parse_time(snap.get("provider_update"))
        captured_at = _parse_time(snap.get("captured_at"))
        rank = (captured_at or datetime.min.replace(tzinfo=timezone.utc), provider_update or datetime.min.replace(tzinfo=timezone.utc))
        best_rank = (best_captured_at or datetime.min.replace(tzinfo=timezone.utc), best_provider_update or datetime.min.replace(tzinfo=timezone.utc))
        if best_snap is None or rank > best_rank:
            best_snap = snap
            best_value = value
            best_provider_update = provider_update
            best_captured_at = captured_at

    if signal_at is None or kickoff is None:
        reason = "MISSING_TIMESTAMPS"
    elif not later_market:
        reason = "NO_LATER_PREKICKOFF_MARKET_SNAPSHOT"
    elif not strict_market:
        reason = "NO_LATER_PROVIDER_UPDATE"
    elif exact_matches:
        reason = "STRICT_LATER_EXACT_QUOTE"
    elif side_matches:
        reason = "SELECTION_MATCH_LINE_MOVED"
    else:
        reason = "NO_SELECTION_MATCH_AT_CLOSE"

    return {
        "fixture_id": fixture_id,
        "stage": stage,
        "kickoff": kickoff.isoformat() if kickoff is not None else kickoff_raw,
        "market": market,
        "selection": selection,
        "line": line,
        "signal_generated_at": signal_at.isoformat() if signal_at is not None else signal_at_raw,
        "captured_at": best_captured_at.isoformat() if best_captured_at is not None else None,
        "provider_update": best_provider_update.isoformat() if best_provider_update is not None else None,
        "bookmaker_id": best_snap.get("bookmaker_id") if best_snap else None,
        "bookmaker": best_snap.get("bookmaker") if best_snap else None,
        "decimal_price": (
            _num(best_value.get("decimal_price"))
            if best_value and best_value.get("decimal_price") is not None
            else _num(best_value.get("price")) if best_value else None
        ),
        "later_same_market_snapshot_count": len(later_market),
        "strict_later_provider_snapshot_count": len(strict_market),
        "same_side_quote_count": len(side_matches),
        "exact_side_line_quote_count": len(exact_matches),
        "result": reason,
    }


def build_report(
    *,
    candidate_events: list[dict[str, Any]],
    resolved_events: list[dict[str, Any]],
    captured_at: Any,
    max_samples: int = MAX_SAMPLES,
) -> dict[str, Any]:
    captured = _parse_time(captured_at)
    resolved_by_fixture: dict[int, list[dict[str, Any]]] = {}
    for event in resolved_events or []:
        fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
        try:
            fixture_id = int(fixture.get("fixture_id"))
        except (TypeError, ValueError):
            continue
        rows = _event_market_rows(event)
        if rows:
            resolved_by_fixture.setdefault(fixture_id, []).extend(rows)

    samples: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}
    fixture_results: dict[int, set[str]] = {}
    signal_count = 0

    for candidate in candidate_events or []:
        fixture = candidate.get("fixture") if isinstance(candidate.get("fixture"), dict) else {}
        meta = candidate.get("team_totals_clv_maturation") if isinstance(candidate.get("team_totals_clv_maturation"), dict) else {}
        try:
            fixture_id = int(fixture.get("fixture_id"))
        except (TypeError, ValueError):
            continue
        rows = resolved_by_fixture.get(fixture_id, [])

        for signal in meta.get("signals") or []:
            if not isinstance(signal, dict):
                continue
            signal_count += 1
            signal_at = _parse_time(signal.get("signal_generated_at"))
            line = _signal_line(signal)
            same_market = [row for row in rows if _norm(row.get("market")) == _norm(signal.get("market"))]
            exact_quotes: list[tuple[dict[str, Any], dict[str, Any]]] = []
            for row in same_market:
                for value in _quote_values(row):
                    value_line = _value_line(value)
                    if (
                        _selection_norm(_value_selection(value)) == _selection_norm(signal.get("selection"))
                        and line is not None
                        and value_line is not None
                        and abs(value_line - line) < 0.000001
                    ):
                        exact_quotes.append((row, value))

            best_row: dict[str, Any] | None = None
            best_value: dict[str, Any] | None = None
            best_provider_update: datetime | None = None
            for row, value in exact_quotes:
                provider_update = _parse_time(row.get("provider_update"))
                if best_row is None or (
                    provider_update is not None
                    and (best_provider_update is None or provider_update > best_provider_update)
                ):
                    best_row = row
                    best_value = value
                    best_provider_update = provider_update

            kickoff = _parse_time(fixture.get("kickoff"))
            if not same_market:
                reason = "NO_EXACT_MARKET"
            elif not exact_quotes:
                reason = "NO_EXACT_SIDE_LINE_MATCH"
            elif captured is None or signal_at is None or captured <= signal_at:
                reason = "CAPTURE_NOT_AFTER_SIGNAL"
            elif kickoff is not None and captured >= kickoff:
                reason = "CAPTURE_NOT_PREKICKOFF"
            elif best_provider_update is None:
                reason = "PROVIDER_UPDATE_MISSING"
            elif best_provider_update <= signal_at:
                reason = "PROVIDER_UPDATE_NOT_AFTER_SIGNAL"
            elif kickoff is not None and best_provider_update >= kickoff:
                reason = "PROVIDER_UPDATE_NOT_PREKICKOFF"
            else:
                reason = "STRICT_LATER_EXACT_QUOTE"

            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            fixture_results.setdefault(fixture_id, set()).add(reason)
            if len(samples) < max(1, int(max_samples)):
                samples.append({
                    "fixture_id": fixture_id,
                    "stage": candidate.get("stage"),
                    "kickoff": fixture.get("kickoff"),
                    "market": signal.get("market"),
                    "selection": signal.get("selection"),
                    "line": line,
                    "signal_generated_at": signal.get("signal_generated_at"),
                    "captured_at": captured.isoformat() if captured is not None else captured_at,
                    "provider_update": best_provider_update.isoformat() if best_provider_update is not None else None,
                    "bookmaker_id": best_row.get("bookmaker_id") if best_row else None,
                    "bookmaker": best_row.get("bookmaker") if best_row else None,
                    "decimal_price": (
                        _num(best_value.get("decimal_price"))
                        if best_value and best_value.get("decimal_price") is not None
                        else _num(best_value.get("price")) if best_value else None
                    ),
                    "result": reason,
                })

    strict_fixture_ids = sorted(
        fixture_id for fixture_id, reasons in fixture_results.items()
        if "STRICT_LATER_EXACT_QUOTE" in reasons
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": "OBSERVABILITY_ONLY",
        "candidate_fixture_count": len({
            int((event.get("fixture") or {}).get("fixture_id"))
            for event in candidate_events or []
            if isinstance(event.get("fixture"), dict)
            and str((event.get("fixture") or {}).get("fixture_id") or "").isdigit()
        }),
        "candidate_signal_count": signal_count,
        "strict_later_exact_quote_fixture_count": len(strict_fixture_ids),
        "strict_later_exact_quote_fixture_ids": strict_fixture_ids,
        "reason_counts": dict(sorted(reason_counts.items())),
        "samples": samples,
        "provider_requests_added": 0,
        "selection_logic_changed": False,
        "strict_close_semantics_changed": False,
        "historical_rows_mutated": False,
        "decision_weight": 0.0,
        "production_promotion_allowed": False,
    }


def build_historical_report(
    *,
    signal_rows: list[dict[str, Any]],
    snapshot_rows: list[dict[str, Any]],
    now: Any = None,
    max_samples: int = MAX_SAMPLES,
) -> dict[str, Any]:
    now_dt = _parse_time(now) or datetime.now(timezone.utc)
    snapshots_by_fixture: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for snap in snapshot_rows or []:
        try:
            fixture_id = int(snap.get("fixture_id"))
        except (TypeError, ValueError):
            continue
        snapshots_by_fixture[fixture_id].append(snap)

    reason_counts: dict[str, int] = {}
    fixture_results: dict[int, set[str]] = defaultdict(set)
    samples: list[dict[str, Any]] = []
    signal_count = 0
    for row in signal_rows or []:
        try:
            fixture_id = int(row.get("fixture_id"))
        except (TypeError, ValueError):
            continue
        kickoff = _parse_time(row.get("kickoff"))
        if kickoff is None or kickoff > now_dt:
            continue
        candidate = row.get("market_candidate") if isinstance(row.get("market_candidate"), dict) else {}
        sample = _classify_signal(
            fixture_id=fixture_id,
            stage=row.get("stage"),
            kickoff_raw=row.get("kickoff"),
            signal_at_raw=row.get("signal_generated_at") or row.get("generated_at"),
            market=candidate.get("market"),
            selection=candidate.get("selection"),
            line_raw=candidate.get("line"),
            snapshots=snapshots_by_fixture.get(fixture_id, []),
        )
        signal_count += 1
        reason = str(sample.get("result") or "UNKNOWN")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        fixture_results[fixture_id].add(reason)
        if len(samples) < max(1, int(max_samples)):
            samples.append(sample)

    strict_fixture_ids = sorted(
        fixture_id for fixture_id, reasons in fixture_results.items()
        if "STRICT_LATER_EXACT_QUOTE" in reasons
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": "OBSERVABILITY_ONLY_CAP_INDEPENDENT_HISTORY",
        "historical_signal_count": signal_count,
        "historical_fixture_count": len(fixture_results),
        "strict_later_exact_quote_fixture_count": len(strict_fixture_ids),
        "strict_later_exact_quote_fixture_ids": strict_fixture_ids,
        "fixture_reason_map": {
            str(fixture_id): sorted(reasons)
            for fixture_id, reasons in sorted(fixture_results.items())
        },
        "reason_counts": dict(sorted(reason_counts.items())),
        "samples": samples,
        "provider_requests_added": 0,
        "selection_logic_changed": False,
        "strict_close_semantics_changed": False,
        "historical_rows_mutated": False,
        "global_signal_cap_dependency": False,
        "decision_weight": 0.0,
        "production_promotion_allowed": False,
    }


def load_historical_report(
    *,
    lookback_hours: int = HISTORICAL_LOOKBACK_HOURS,
    signal_limit: int = HISTORICAL_SIGNAL_LIMIT,
    max_samples: int = MAX_SAMPLES,
) -> dict[str, Any]:
    if not persistence.persistence_configured():
        report = build_historical_report(signal_rows=[], snapshot_rows=[], max_samples=max_samples)
        report.update({"status": "NO_DATABASE", "lookback_hours": lookback_hours, "signal_limit": signal_limit})
        return report

    persistence.ensure_schema()
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=max(1, int(lookback_hours)))
    with persistence._connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH oldest_exact AS (
                    SELECT DISTINCT ON (
                        e.fixture_id,
                        COALESCE(t.row_value ->> 'market', ''),
                        COALESCE(t.row_value ->> 'selection', ''),
                        COALESCE(t.row_value ->> 'line', '')
                    )
                        e.fixture_id,
                        e.generated_at AS signal_generated_at,
                        e.stage,
                        f.kickoff,
                        t.row_value AS market_candidate
                    FROM soccer_refresh_events e
                    JOIN soccer_fixtures f ON f.fixture_id = e.fixture_id
                    CROSS JOIN LATERAL jsonb_array_elements(
                        COALESCE(e.payload -> 'team_totals_intelligence' -> 'observed_exact_market_rows', '[]'::jsonb)
                    ) AS t(row_value)
                    WHERE e.generated_at >= %s
                      AND e.generated_at < f.kickoff
                      AND f.kickoff > %s
                      AND f.kickoff <= %s
                      AND e.stage = ANY(%s)
                    ORDER BY
                        e.fixture_id,
                        COALESCE(t.row_value ->> 'market', ''),
                        COALESCE(t.row_value ->> 'selection', ''),
                        COALESCE(t.row_value ->> 'line', ''),
                        e.generated_at ASC
                )
                SELECT fixture_id, signal_generated_at, stage, kickoff, market_candidate
                FROM oldest_exact
                ORDER BY signal_generated_at DESC
                LIMIT %s
                """,
                (cutoff, cutoff, now, list(TEAM_TOTALS_RESEARCH_STAGES), max(1, int(signal_limit))),
            )
            signal_columns = [desc.name for desc in cur.description]
            signal_rows = [dict(zip(signal_columns, row)) for row in cur.fetchall()]

        fixture_ids = sorted({int(row["fixture_id"]) for row in signal_rows if row.get("fixture_id") is not None})
        snapshot_rows: list[dict[str, Any]] = []
        if fixture_ids:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT fixture_id, captured_at, stage, bookmaker_id, bookmaker,
                           market_id, market, values, provider_update
                    FROM soccer_market_snapshots
                    WHERE fixture_id = ANY(%s)
                      AND captured_at >= %s
                      AND captured_at <= %s
                    ORDER BY fixture_id ASC, captured_at ASC
                    """,
                    (fixture_ids, cutoff, now),
                )
                snapshot_columns = [desc.name for desc in cur.description]
                snapshot_rows = [dict(zip(snapshot_columns, row)) for row in cur.fetchall()]

    report = build_historical_report(
        signal_rows=signal_rows,
        snapshot_rows=snapshot_rows,
        now=now,
        max_samples=max_samples,
    )
    report.update({
        "lookback_hours": max(1, int(lookback_hours)),
        "signal_limit": max(1, int(signal_limit)),
        "snapshot_rows_loaded": len(snapshot_rows),
        "source": "POSTGRES_REFRESH_EVENTS_TEAM_TOTALS_PLUS_MARKET_SNAPSHOTS",
    })
    return report
