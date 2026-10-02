from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from mcp_gateway import player_props_clv_postgres_v4 as base

MODEL_VERSION = "SOCCER_PLAYER_PROPS_SHOTS_ANCHOR_PATCH_V4_1.0.0"
RECENT_CLOSED_HOURS = 12
MAX_RECENT_EVENT_IDS = 600

_ORIGINAL_LOAD_EVENT_SIGNALS = base._load_event_signals
_INSTALLED = False


def _shots_key(signal: dict[str, Any]) -> tuple[int, str, str, float | None] | None:
    if str(signal.get("market_family") or "").upper() != "SHOTS":
        return None
    try:
        fixture_id = int(signal.get("fixture_id"))
    except (TypeError, ValueError):
        return None
    player_id = signal.get("player_id")
    side = str(signal.get("side") or signal.get("selection") or "").upper()
    if player_id is None or side not in {"OVER", "UNDER"}:
        return None
    return (
        fixture_id,
        str(player_id),
        side,
        base._line_key(base._num(signal.get("line"))),
    )


def _signal_at(signal: dict[str, Any]):
    return base._dt(signal.get("signal_timestamp"))


def _oldest_by_key(signals: list[dict[str, Any]]) -> dict[tuple[int, str, str, float | None], dict[str, Any]]:
    out: dict[tuple[int, str, str, float | None], dict[str, Any]] = {}
    for signal in signals:
        if not isinstance(signal, dict):
            continue
        key = _shots_key(signal)
        signal_at = _signal_at(signal)
        if key is None or signal_at is None:
            continue
        current = out.get(key)
        current_at = _signal_at(current) if current is not None else None
        if current is None or current_at is None or signal_at < current_at:
            out[key] = signal
    return out


def _merge_oldest_shots(
    original_signals: list[dict[str, Any]],
    recent_signals: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    non_shots = [row for row in original_signals if _shots_key(row) is None]
    original_shots = [row for row in original_signals if _shots_key(row) is not None]
    anchors = _oldest_by_key(original_shots)
    recent_anchors = _oldest_by_key(recent_signals)

    replaced = 0
    matched_recent = 0
    for key, candidate in recent_anchors.items():
        # v217 is a bridge, not a cohort expansion: a historical anchor may only
        # replace an instrument already present in the canonical bounded cohort.
        if key not in anchors:
            continue
        matched_recent += 1
        candidate_at = _signal_at(candidate)
        anchor_at = _signal_at(anchors[key])
        if candidate_at is not None and (anchor_at is None or candidate_at < anchor_at):
            anchors[key] = candidate
            replaced += 1

    ordered_anchors = sorted(
        anchors.values(),
        key=lambda row: (
            _signal_at(row) or datetime.max.replace(tzinfo=timezone.utc),
            int(row.get("fixture_id") or 0),
            str(row.get("player_id") or ""),
            str(row.get("side") or ""),
            float(row.get("line") or 0.0),
        ),
    )
    merged = non_shots + ordered_anchors
    return merged, {
        "original_shots_signals": len(original_shots),
        "unique_shots_instruments": len(anchors),
        "shots_duplicates_collapsed": max(0, len(original_shots) - len(anchors)),
        "recent_shots_signals": len([row for row in recent_signals if _shots_key(row) is not None]),
        "recent_matching_instruments": matched_recent,
        "older_anchors_replaced": replaced,
    }


def _load_recent_closed_shots(
    conn,
    *,
    hours: int = RECENT_CLOSED_HOURS,
    max_event_ids: int = MAX_RECENT_EVENT_IDS,
    batch_size: int = 100,
) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
    now = datetime.now(timezone.utc)
    kickoff_cutoff = now - timedelta(hours=max(1, int(hours)))
    event_cutoff = kickoff_cutoff - timedelta(hours=2)
    bounded_limit = max(1, min(int(max_event_ids), MAX_RECENT_EVENT_IDS))
    bounded_batch = max(1, min(int(batch_size), 200))

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT e.event_id
            FROM soccer_refresh_events e
            JOIN soccer_fixtures f ON f.fixture_id = e.fixture_id
            WHERE f.kickoff >= %s
              AND f.kickoff <= %s
              AND e.generated_at >= %s
              AND e.generated_at < f.kickoff
              AND e.stage IN ('T-40','T-30','T-20','T-10')
              AND e.payload ? 'market'
              AND e.payload ? 'player_shots_intelligence'
            ORDER BY e.generated_at ASC, e.event_id ASC
            LIMIT %s
            """,
            (kickoff_cutoff, now, event_cutoff, bounded_limit),
        )
        event_ids = [int(row[0]) for row in cur.fetchall()]

    recent_signals: list[dict[str, Any]] = []
    extraction_diagnostics: dict[str, Any] = {}
    for offset in range(0, len(event_ids), bounded_batch):
        events = base._hydrate_event_batch(conn, event_ids[offset:offset + bounded_batch])
        extracted = base.extract_shadow_signals(events, extraction_diagnostics)
        recent_signals.extend(
            row for row in extracted
            if str(row.get("market_family") or "").upper() == "SHOTS"
        )
    return recent_signals, len(event_ids), extraction_diagnostics


def _compact_recent_shots_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    family = ((diagnostics.get("families") or {}).get("SHOTS") or {})
    keys = (
        "intelligence_event_rows",
        "market_overlap_event_rows",
        "modelable_player_rows",
        "market_rows",
        "raw_market_values",
        "aligned_market_values",
        "player_id_overlap_values",
        "priced_overlap_values",
        "model_probability_values",
        "signal_rows",
        "sidecar_capture_status",
        "probability_failure_reasons",
    )
    return {
        "eligible_event_rows": int(diagnostics.get("eligible_event_rows") or 0),
        **{key: family.get(key) for key in keys},
    }


def _patched_load_event_signals(
    conn,
    *,
    lookback_days: int,
    max_rows: int,
    batch_size: int = 100,
):
    signals, diagnostics, loaded_rows = _ORIGINAL_LOAD_EVENT_SIGNALS(
        conn,
        lookback_days=lookback_days,
        max_rows=max_rows,
        batch_size=batch_size,
    )

    try:
        recent_shots, recent_event_ids, recent_extraction_diagnostics = _load_recent_closed_shots(
            conn,
            batch_size=batch_size,
        )
        merged, patch_diag = _merge_oldest_shots(signals, recent_shots)
        diagnostics["v217_shots_anchor_patch"] = {
            "model_version": MODEL_VERSION,
            "status": "RESEARCH_ONLY",
            "recent_closed_hours": RECENT_CLOSED_HOURS,
            "recent_event_ids_loaded": recent_event_ids,
            "recent_extraction": _compact_recent_shots_diagnostics(recent_extraction_diagnostics),
            **patch_diag,
            "provider_requests_added": 0,
            "strict_close_semantics_changed": False,
            "selection_logic_changed": False,
            "cohort_expansion_allowed": False,
        }
        return merged, diagnostics, loaded_rows
    except Exception as exc:
        # Observability/anchor recovery must never take down the canonical builder.
        diagnostics["v217_shots_anchor_patch"] = {
            "model_version": MODEL_VERSION,
            "status": "FALLBACK_TO_CANONICAL",
            "error": str(exc)[:300],
            "provider_requests_added": 0,
            "strict_close_semantics_changed": False,
            "selection_logic_changed": False,
            "cohort_expansion_allowed": False,
        }
        return signals, diagnostics, loaded_rows


def install() -> bool:
    global _INSTALLED
    if _INSTALLED:
        return False
    base._load_event_signals = _patched_load_event_signals
    _INSTALLED = True
    return True
