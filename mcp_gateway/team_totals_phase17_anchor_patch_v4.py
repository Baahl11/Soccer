from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from mcp_gateway import clv_postgres_v4

MODEL_VERSION = "SOCCER_PHASE17_TEAM_TOTALS_OLDEST_EXACT_ANCHOR_PATCH_V4_1.0.0"
PATCH_STATUS = "RESEARCH_ONLY_OLDEST_EXACT_TEAM_TOTALS_SIGNAL"

_ORIGINAL_LOADER = clv_postgres_v4._load_derivative_signals
_INSTALLED = False


def _signal_time(signal: dict[str, Any]) -> datetime:
    parsed = clv_postgres_v4._as_utc_datetime(signal.get("generated_at"))
    return parsed if parsed is not None else datetime.max.replace(tzinfo=timezone.utc)


def _team_total_exact_key(signal: dict[str, Any]) -> tuple[Any, ...] | None:
    if str(signal.get("signal_source") or "") != "DERIVATIVE_INTELLIGENCE:team_totals_intelligence":
        return None
    candidate = signal.get("market_candidate")
    if not isinstance(candidate, dict):
        return None
    fixture_id = signal.get("fixture_id")
    if fixture_id is None:
        return None
    line = clv_postgres_v4._num(candidate.get("line"))
    if line is None:
        line = clv_postgres_v4._line_from_selection(candidate.get("selection"))
    if line is None:
        return None
    market = clv_postgres_v4._norm(candidate.get("market"))
    side = clv_postgres_v4._selection_side(candidate.get("selection"))
    if not market or not side:
        return None
    return (int(fixture_id), market, side, float(line))


def collapse_oldest_exact_team_totals(signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse only Team Totals recycled timestamps to the oldest exact point-in-time signal."""
    oldest_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    first_team_total_position: int | None = None

    for index, signal in enumerate(signals):
        key = _team_total_exact_key(signal)
        if key is None:
            continue
        if first_team_total_position is None:
            first_team_total_position = index
        current = oldest_by_key.get(key)
        if current is None or _signal_time(signal) < _signal_time(current):
            oldest_by_key[key] = signal

    if not oldest_by_key:
        return list(signals)

    collapsed_team_totals = sorted(
        oldest_by_key.values(),
        key=lambda signal: (
            _signal_time(signal),
            int(signal.get("fixture_id") or 0),
            clv_postgres_v4._norm((signal.get("market_candidate") or {}).get("market")),
            clv_postgres_v4._selection_side((signal.get("market_candidate") or {}).get("selection")),
            float(
                clv_postgres_v4._num((signal.get("market_candidate") or {}).get("line"))
                or clv_postgres_v4._line_from_selection((signal.get("market_candidate") or {}).get("selection"))
                or 0.0
            ),
        ),
    )

    output: list[dict[str, Any]] = []
    inserted = False
    for signal in signals:
        if _team_total_exact_key(signal) is not None:
            if not inserted:
                output.extend(collapsed_team_totals)
                inserted = True
            continue
        output.append(signal)
    return output


def _patched_load_derivative_signals(
    conn: Any,
    *,
    lookback_days: int,
    max_rows: int,
) -> list[dict[str, Any]]:
    raw = _ORIGINAL_LOADER(conn, lookback_days=lookback_days, max_rows=max_rows)
    return collapse_oldest_exact_team_totals(raw)


def install() -> dict[str, Any]:
    global _INSTALLED
    if not _INSTALLED:
        clv_postgres_v4._load_derivative_signals = _patched_load_derivative_signals
        _INSTALLED = True
    return {
        "model_version": MODEL_VERSION,
        "status": PATCH_STATUS,
        "installed": _INSTALLED,
        "scope": ["HOME_TT", "AWAY_TT"],
        "provider_requests_added": 0,
        "global_signal_cap_changed": False,
        "strict_close_semantics_changed": False,
        "selection_logic_changed": False,
        "historical_rows_mutated": False,
        "decision_weight": 0.0,
        "production_promotion_allowed": False,
        "policy": (
            "COLLAPSE_RECYCLED_TEAM_TOTALS_TIMESTAMPS_TO_OLDEST_EXACT_FIXTURE_MARKET_SIDE_LINE_SIGNAL_"
            "BEFORE_PHASE17_MERGE; PRESERVE_EXISTING_GLOBAL_CAP_AND_STRICT_CLOSE_RULES"
        ),
    }
