from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

MODEL_VERSION = "SOCCER_TEAM_TOTALS_CLOSE_PROVENANCE_V4_1.0.0"
SCHEMA_VERSION = "1.0.0"
MAX_SAMPLES = 80


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
        kickoff = _parse_time(fixture.get("kickoff"))
        rows = resolved_by_fixture.get(fixture_id, [])

        for signal in meta.get("signals") or []:
            if not isinstance(signal, dict):
                continue
            signal_count += 1
            signal_at = _parse_time(signal.get("signal_generated_at"))
            market_norm = _norm(signal.get("market"))
            selection_norm = _selection_norm(signal.get("selection"))
            line = _num(signal.get("line"))

            same_market = [row for row in rows if _norm(row.get("market")) == market_norm]
            exact_quotes: list[tuple[dict[str, Any], dict[str, Any]]] = []
            for row in same_market:
                for value in _quote_values(row):
                    value_line = _num(value.get("line"))
                    if value_line is None:
                        value_line = _num(value.get("handicap"))
                    if (
                        _selection_norm(value.get("selection") if value.get("selection") is not None else value.get("value"))
                        == selection_norm
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
                samples.append(
                    {
                        "fixture_id": fixture_id,
                        "stage": candidate.get("stage"),
                        "kickoff": fixture.get("kickoff"),
                        "market": signal.get("market"),
                        "selection": signal.get("selection"),
                        "line": line,
                        "signal_generated_at": signal.get("signal_generated_at"),
                        "captured_at": captured.isoformat() if captured is not None else captured_at,
                        "provider_update": (
                            best_provider_update.isoformat() if best_provider_update is not None else None
                        ),
                        "bookmaker_id": best_row.get("bookmaker_id") if best_row else None,
                        "bookmaker": best_row.get("bookmaker") if best_row else None,
                        "decimal_price": (
                            _num(best_value.get("decimal_price"))
                            if best_value and best_value.get("decimal_price") is not None
                            else _num(best_value.get("odd")) if best_value else None
                        ),
                        "result": reason,
                    }
                )

    strict_fixture_ids = sorted(
        fixture_id
        for fixture_id, reasons in fixture_results.items()
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
