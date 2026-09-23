from __future__ import annotations

from datetime import datetime, timedelta, timezone as dt_timezone
from typing import Any

import httpx

from mcp_gateway import automation as base

_PROCESS_MEMO: dict[tuple[str, str], dict[str, Any] | None] = {}


def load_json(
    *,
    url: str,
    expected_status: str,
    cache_namespace: str,
    ttl: timedelta = timedelta(hours=6),
) -> dict[str, Any] | None:
    """Load a research artifact once per tick worker process.

    Several player-intelligence layers consume the same large registries. The
    worker is short-lived, so a process-local memo avoids repeated SQLite
    json.loads()/HTTP JSON parses of identical artifacts within the same tick
    without changing model logic or persistence semantics.
    """
    memo_key = (url, expected_status)
    if memo_key in _PROCESS_MEMO:
        return _PROCESS_MEMO[memo_key]

    now = datetime.now(dt_timezone.utc)
    cached = base._cache_get(cache_namespace, "latest", ttl, now)
    if isinstance(cached, dict) and cached.get("status") == expected_status:
        _PROCESS_MEMO[memo_key] = cached
        return cached

    try:
        response = httpx.get(url, timeout=5.0, follow_redirects=True)
        if response.status_code != 200:
            _PROCESS_MEMO[memo_key] = None
            return None
        payload = response.json()
    except Exception:
        _PROCESS_MEMO[memo_key] = None
        return None

    if not isinstance(payload, dict) or payload.get("status") != expected_status:
        _PROCESS_MEMO[memo_key] = None
        return None

    base._cache_set(cache_namespace, "latest", payload, now)
    _PROCESS_MEMO[memo_key] = payload
    return payload
