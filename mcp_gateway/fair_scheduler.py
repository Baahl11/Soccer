from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta, timezone as dt_timezone
from typing import Any

from mcp_gateway import automation as base

SCHEMA_VERSION = "1.0.0"
STATE_NAMESPACE = "fair_scheduler_state"
STATE_TTL = timedelta(hours=72)
EXPORT_MAX_AGE = timedelta(hours=72)
MAX_STATE_ITEMS = 2500

CATEGORY_WEIGHTS = {
    "actionable": 60,
    "unseen": 25,
    "exploratory": 15,
}


def get_state(fixture_id: int, now: datetime) -> dict[str, Any]:
    value = base._cache_get(STATE_NAMESPACE, str(fixture_id), STATE_TTL, now)
    if isinstance(value, dict):
        return value
    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": int(fixture_id),
        "deep_dive_count": 0,
        "first_deep_dive_at": None,
        "last_deep_dive_at": None,
        "first_deep_dive_stage": None,
        "last_deep_dive_stage": None,
        "stage_counts": {},
    }


def record_deep_dive(fixture_id: int, stage: str, now: datetime) -> dict[str, Any]:
    state = dict(get_state(fixture_id, now))
    count = int(state.get("deep_dive_count") or 0) + 1
    stage_counts = dict(state.get("stage_counts") or {})
    stage_counts[str(stage)] = int(stage_counts.get(str(stage)) or 0) + 1

    state.update(
        {
            "schema_version": SCHEMA_VERSION,
            "fixture_id": int(fixture_id),
            "deep_dive_count": count,
            "first_deep_dive_at": state.get("first_deep_dive_at") or now.isoformat(),
            "last_deep_dive_at": now.isoformat(),
            "first_deep_dive_stage": state.get("first_deep_dive_stage") or str(stage),
            "last_deep_dive_stage": str(stage),
            "stage_counts": stage_counts,
        }
    )
    base._cache_set(STATE_NAMESPACE, str(fixture_id), state, now)
    return state


def category(prior_shortlisted: bool, state: dict[str, Any]) -> str:
    if prior_shortlisted:
        return "actionable"
    if int(state.get("deep_dive_count") or 0) <= 0:
        return "unseen"
    return "exploratory"


def _last_deep_dive_sort_value(state: dict[str, Any]) -> str:
    # ISO-8601 UTC strings sort chronologically. Missing means oldest and should
    # be serviced first when a category is otherwise tied.
    return str(state.get("last_deep_dive_at") or "")


def _sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    priority = tuple(item.get("priority") or ())
    state = item.get("fairness_state") if isinstance(item.get("fairness_state"), dict) else {}
    cat = str(item.get("fairness_category") or "exploratory")
    last = _last_deep_dive_sort_value(state)

    if cat == "actionable":
        # Preserve lifecycle urgency first, then favor the least-recently
        # refreshed shortlist within an otherwise equivalent gate.
        head = priority[:3]
        tail = priority[3:]
        return (*head, last, *tail)
    if cat == "unseen":
        return priority
    # Repeat research should age into service instead of monopolizing merely
    # because the same fixture keeps hitting T-40/T-20/T-10 windows.
    return (last, *priority)


def fair_order(
    items: list[dict[str, Any]],
    max_slots: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    queues: dict[str, list[dict[str, Any]]] = {key: [] for key in CATEGORY_WEIGHTS}
    for item in items:
        cat = str(item.get("fairness_category") or "exploratory")
        if cat not in queues:
            cat = "exploratory"
            item["fairness_category"] = cat
        queues[cat].append(item)

    for queue in queues.values():
        queue.sort(key=_sort_key)

    initial_counts = {key: len(value) for key, value in queues.items()}
    selected: list[dict[str, Any]] = []
    current = {key: 0 for key in CATEGORY_WEIGHTS}
    limit = max(int(max_slots or 0), 0)

    while len(selected) < limit:
        available = [key for key, queue in queues.items() if queue]
        if not available:
            break
        total_weight = sum(CATEGORY_WEIGHTS[key] for key in available)
        for key in available:
            current[key] += CATEGORY_WEIGHTS[key]
        chosen = max(available, key=lambda key: (current[key], CATEGORY_WEIGHTS[key]))
        current[chosen] -= total_weight
        selected.append(queues[chosen].pop(0))

    deferred: list[dict[str, Any]] = []
    for queue in queues.values():
        deferred.extend(queue)
    deferred.sort(key=lambda item: tuple(item.get("priority") or ()))

    planned_counts = Counter(str(item.get("fairness_category") or "exploratory") for item in selected)
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "policy": "WEIGHTED_FAIR_60_ACTIONABLE_25_UNSEEN_15_EXPLORATORY",
        "weights_pct": dict(CATEGORY_WEIGHTS),
        "eligible_queue_counts": initial_counts,
        "planned_slot_counts": {key: int(planned_counts.get(key, 0)) for key in CATEGORY_WEIGHTS},
        "planned_slot_count": len(selected),
        "deferred_after_slot_plan": len(deferred),
    }
    return selected, deferred, metrics


def import_state(seed: Any) -> int:
    if not isinstance(seed, dict):
        return 0
    now = datetime.now(dt_timezone.utc)
    now_ts = now.timestamp()
    min_ts = (now - EXPORT_MAX_AGE).timestamp()
    conn = base._cache_conn()
    imported = 0

    for cache_key, item in list(seed.items())[:MAX_STATE_ITEMS]:
        if not isinstance(item, dict):
            continue
        value = item.get("value")
        updated_at = item.get("updated_at")
        if not isinstance(value, dict):
            continue
        try:
            ts = float(updated_at)
        except (TypeError, ValueError):
            continue
        if ts < min_ts or ts > now_ts + 300:
            continue
        conn.execute(
            """
            INSERT INTO cache_entries(namespace, cache_key, updated_at, value_json)
            VALUES(?,?,?,?)
            ON CONFLICT(namespace, cache_key) DO UPDATE SET
                updated_at=CASE
                    WHEN excluded.updated_at > cache_entries.updated_at THEN excluded.updated_at
                    ELSE cache_entries.updated_at
                END,
                value_json=CASE
                    WHEN excluded.updated_at > cache_entries.updated_at THEN excluded.value_json
                    ELSE cache_entries.value_json
                END
            """,
            (STATE_NAMESPACE, str(cache_key), ts, json.dumps(value, separators=(",", ":"))),
        )
        imported += 1
    conn.commit()
    return imported


def export_state() -> dict[str, Any]:
    now = datetime.now(dt_timezone.utc)
    cutoff = (now - EXPORT_MAX_AGE).timestamp()
    conn = base._cache_conn()
    rows = conn.execute(
        """
        SELECT cache_key, updated_at, value_json
        FROM cache_entries
        WHERE namespace=? AND updated_at >= ?
        ORDER BY updated_at DESC
        LIMIT ?
        """,
        (STATE_NAMESPACE, cutoff, MAX_STATE_ITEMS),
    ).fetchall()

    out: dict[str, Any] = {}
    for cache_key, updated_at, value_json in rows:
        try:
            value = json.loads(value_json)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            out[str(cache_key)] = {
                "updated_at": float(updated_at),
                "value": value,
            }
    return out


def coverage_metrics(
    due_items: list[dict[str, Any]],
    processed_counts: dict[str, int],
    api_calls_this_tick: int,
    deep_dive_processed: int,
    now: datetime,
) -> dict[str, Any]:
    eligible = [item for item in due_items if str(item.get("tier")) in {"A", "B"}]
    unique: dict[int, dict[str, Any]] = {}
    for item in eligible:
        fx = item.get("fx") if isinstance(item.get("fx"), dict) else {}
        fixture_id = fx.get("fixture_id")
        if fixture_id is None:
            continue
        unique[int(fixture_id)] = item

    states = [get_state(fixture_id, now) for fixture_id in unique]
    analyzed = [state for state in states if int(state.get("deep_dive_count") or 0) > 0]
    first_t90 = [state for state in analyzed if state.get("first_deep_dive_stage") == "T-90"]
    first_by_t40 = [
        state
        for state in analyzed
        if state.get("first_deep_dive_stage") in {"T-90", "T-60", "T-40"}
    ]

    actionable_due = sum(
        1
        for item in unique.values()
        if str(item.get("fairness_category") or "") == "actionable"
        and str(item.get("stage") or "") in {"T-40", "T-30", "T-20", "T-10", "CLOSE"}
    )
    actionable_processed = int(processed_counts.get("actionable") or 0)

    total = len(states)
    analyzed_count = len(analyzed)
    avg_deep = (
        round(sum(int(state.get("deep_dive_count") or 0) for state in states) / total, 3)
        if total
        else None
    )
    due_pct = round(analyzed_count / total * 100.0, 2) if total else None
    t90_pct = round(len(first_t90) / total * 100.0, 2) if total else None
    t40_pct = round(len(first_by_t40) / total * 100.0, 2) if total else None
    actionable_pct = (
        round(min(actionable_processed, actionable_due) / actionable_due * 100.0, 2)
        if actionable_due
        else None
    )

    return {
        "eligible_tier_ab_due_count": total,
        "due_with_any_deep_dive_count": analyzed_count,
        "due_analyzed_pct": due_pct,
        "first_deep_dive_at_t90_count": len(first_t90),
        "analyzed_by_t90_pct": t90_pct,
        "first_deep_dive_by_t40_count": len(first_by_t40),
        "analyzed_by_t40_pct": t40_pct,
        "starvation_count": max(total - analyzed_count, 0),
        "avg_deep_dives_per_due_fixture": avg_deep,
        "provider_calls_per_processed_fixture": (
            round(int(api_calls_this_tick or 0) / deep_dive_processed, 3)
            if deep_dive_processed
            else None
        ),
        "actionable_due_count": actionable_due,
        "actionable_processed_count": min(actionable_processed, actionable_due),
        "actionable_refresh_pct": actionable_pct,
        "tier_ab_due_coverage_target_pct": 90.0,
        "shortlist_actionable_refresh_target_pct": 80.0,
        "tier_ab_due_coverage_target_met": bool(due_pct is not None and due_pct >= 90.0),
        "shortlist_actionable_refresh_target_met": (
            None if actionable_pct is None else bool(actionable_pct >= 80.0)
        ),
    }
