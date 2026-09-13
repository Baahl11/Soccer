from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone as dt_timezone
from typing import Any

from mcp_gateway import automation as base
from mcp_gateway import automation_v2 as v2
from mcp_gateway import automation_v4 as v4
from mcp_gateway import automation_v5 as v5

MODEL_VERSION = "SOCCER EDGE ENGINE v1.0"
AUTOMATION_VERSION = "1.6.0"
SHORTLIST_SEED_MAX_AGE = timedelta(hours=18)
SHORTLIST_EXPORT_MAX_AGE = timedelta(hours=14)
MAX_SEED_ITEMS = 500

_BASE_MAX_API_CALLS_PER_TICK = int(os.getenv("SOCCER_EDGE_MAX_API_CALLS_PER_TICK", str(v2.MAX_API_CALLS_PER_TICK)))
_ORIGINAL_PACED_API_GET = v4._paced_api_get
_BUDGET_MODE = "NORMAL"


def _budget_for_remaining(remaining: int | None) -> tuple[str, int]:
    """Protect daily quota on every slate, not only unusually heavy days."""
    if remaining is None:
        return "NORMAL", _BASE_MAX_API_CALLS_PER_TICK
    if remaining <= 500:
        return "RESERVE", min(_BASE_MAX_API_CALLS_PER_TICK, 4)
    if remaining <= 1500:
        return "EMERGENCY", min(_BASE_MAX_API_CALLS_PER_TICK, 8)
    if remaining <= 2500:
        return "PRIORITY_ONLY", min(_BASE_MAX_API_CALLS_PER_TICK, 12)
    if remaining <= 4000:
        return "REDUCED", min(_BASE_MAX_API_CALLS_PER_TICK, 16)
    return "NORMAL", _BASE_MAX_API_CALLS_PER_TICK


async def _adaptive_paced_api_get(endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
    global _BUDGET_MODE
    payload = await _ORIGINAL_PACED_API_GET(endpoint, params)
    remaining_raw = (payload.get("quota") or {}).get("daily_remaining")
    try:
        remaining = int(remaining_raw) if remaining_raw is not None else None
    except (TypeError, ValueError):
        remaining = None
    mode, cap = _budget_for_remaining(remaining)
    _BUDGET_MODE = mode
    if cap < v2.MAX_API_CALLS_PER_TICK:
        v2.MAX_API_CALLS_PER_TICK = cap
    return payload


def import_shortlist_state(seed: Any) -> int:
    """Restore sporting-shortlist state without refreshing its original TTL."""
    if not isinstance(seed, dict):
        return 0
    now = datetime.now(dt_timezone.utc)
    now_ts = now.timestamp()
    min_ts = (now - SHORTLIST_SEED_MAX_AGE).timestamp()
    conn = base._cache_conn()
    imported = 0
    for cache_key, item in list(seed.items())[:MAX_SEED_ITEMS]:
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
        key = str(cache_key)
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
            ("sport_shortlist", key, ts, json.dumps(value, separators=(",", ":"))),
        )
        imported += 1
    conn.commit()
    return imported


def export_shortlist_state() -> dict[str, Any]:
    """Export only live shortlist entries for durable GitHub-state handoff."""
    now = datetime.now(dt_timezone.utc)
    cutoff = (now - SHORTLIST_EXPORT_MAX_AGE).timestamp()
    conn = base._cache_conn()
    rows = conn.execute(
        """
        SELECT cache_key, updated_at, value_json
        FROM cache_entries
        WHERE namespace='sport_shortlist' AND updated_at >= ?
        ORDER BY updated_at DESC
        LIMIT ?
        """,
        (cutoff, MAX_SEED_ITEMS),
    ).fetchall()
    out: dict[str, Any] = {}
    for cache_key, updated_at, value_json in rows:
        try:
            value = json.loads(value_json)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            out[str(cache_key)] = {"updated_at": float(updated_at), "value": value}
    return out


async def run_tick() -> dict[str, Any]:
    global _BUDGET_MODE
    _BUDGET_MODE = "NORMAL"
    v2.MAX_API_CALLS_PER_TICK = _BASE_MAX_API_CALLS_PER_TICK

    # v1.5 installs the v1.4 paced getter at tick start. Replace that symbol
    # temporarily with the adaptive wrapper so the same pacing/backoff remains
    # active while the daily reserve can tighten the per-tick cap.
    previous_symbol = v4._paced_api_get
    v4._paced_api_get = _adaptive_paced_api_get
    try:
        payload = await v5.run_tick()
    finally:
        v4._paced_api_get = previous_symbol

    shortlist_state = export_shortlist_state()
    payload["version"] = AUTOMATION_VERSION
    payload["daily_budget_mode"] = _BUDGET_MODE
    payload["effective_max_api_calls_per_tick"] = v2.MAX_API_CALLS_PER_TICK
    payload["daily_budget_policy"] = {
        "normal_above": 4000,
        "reduced_at_or_below": 4000,
        "priority_only_at_or_below": 2500,
        "emergency_at_or_below": 1500,
        "reserve_at_or_below": 500,
    }
    payload["shortlist_persistence"] = "GITHUB_STATE_SEEDED"
    payload["shortlist_state_count"] = len(shortlist_state)
    payload["shortlist_state"] = shortlist_state
    return payload
