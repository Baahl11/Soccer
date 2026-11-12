"""Bounded source status lookup for a requested, verified registry fixture.

Read-only source request; never queries prices, team stats or betting markets.
Returns a small derived status instead of any provider body. Source errors fail
closed to the stored fixture status. This does NOT rewrite stored predictions.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from threading import Lock
from time import monotonic
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE_URL = "https://v3.football.api-sports.io"
TTL_SECONDS = 15 * 60
HOURLY_UNIQUE_CAP = 12
DAILY_REQUEST_CAP = 36
MIN_PROVIDER_REMAINING = 1200
CANCEL_STATES = frozenset({"CANC", "PST", "ABD", "AWD", "WO", "SUSP"})
RESULT_STATES = frozenset({"FT", "AET", "PEN"})
ALLOWED_STATES = frozenset({
    "NS", "TBD", "1H", "2H", "HT", "ET", "BT", "P", "LIVE", "INT",
    *CANCEL_STATES, *RESULT_STATES,
})
_lock = Lock()
_cache: dict[int, tuple[float, dict[str, Any]]] = {}
_hour_begin = 0.0
_hourly_count = 0
_day_begin = 0.0
_daily_count = 0

def _key() -> str | None:
    return os.getenv("API_FOOTBALL_KEY", "").strip() or None

def _call_api(fixture_id: int, key: str) -> dict[str, Any]:
    req = Request(
        BASE_URL + "/fixtures?" + urlencode({"id": fixture_id}),
        headers={"x-apisports-key": key, "Accept": "application/json"},
        method="GET",
    )
    with urlopen(req, timeout=5.0) as res:
        if res.status != 200:
            raise RuntimeError("PROVIDER_HTTP_STATUS_NOT_OK")
        body = res.read(192 * 1024 + 1)
        if len(body) > 192 * 1024:
            raise RuntimeError("PROVIDER_RESPONSE_TOO_LARGE")
    return json.loads(body)

def _parse(data: dict[str, Any], fixture_id: int,
           home_id: int, away_id: int, fetched_at: str) -> dict[str, Any] | None:
    rows = data.get("response")
    if not isinstance(rows, list) or len(rows) != 1:
        return None
    row = rows[0]
    if not isinstance(row, dict) or (row.get("fixture") or {}).get("id") != fixture_id:
        return None
    teams = row.get("teams") or {}
    if (teams.get("home") or {}).get("id") != home_id:
        return None
    if (teams.get("away") or {}).get("id") != away_id:
        return None
    status = (row.get("fixture") or {}).get("status") or {}
    short = status.get("short")
    if short not in ALLOWED_STATES:
        return None
    quota = data.get("quota") or {}
    remaining = quota.get("daily_remaining")
    if remaining is not None:
        try:
            if int(remaining) < MIN_PROVIDER_REMAINING:
                return None
        except (ValueError, TypeError):
            return None
    return {
        "status": short,
        "cancelled_or_postponed": short in CANCEL_STATES,
        "completed": short in RESULT_STATES,
        "observed_at_utc": fetched_at,
        "source": "API_FOOTBALL_FIXTURE_STATUS_BY_ID",
        "fixture_id": fixture_id,
        "quote_requested": False,
        "provider_requests_added": 1,
        "official_team_ids_verified": True,
    }

def lookup(fixture_id: Any, home_id: Any, away_id: Any,
           *, fetcher: Any = None, clock: Any = None) -> dict[str, Any]:
    """Call only behind PRO entitlement and known verified fixture registry."""
    if (type(fixture_id) is not int or fixture_id <= 0
        or type(home_id) is not int or home_id <= 0
        or type(away_id) is not int or away_id <= 0):
        return {"status": "NOT_VERIFIED", "provider_requests_added": 0}
    now = float(clock() if clock is not None else monotonic())
    global _hour_begin, _hourly_count, _day_begin, _daily_count
    with _lock:
        saved = _cache.get(fixture_id)
        if saved and now - saved[0] < TTL_SECONDS:
            return {**saved[1], "provider_requests_added": 0, "from_cache": True}
        if now - _hour_begin >= 3600:
            _hour_begin, _hourly_count = now, 0
        if now - _day_begin >= 86400:
            _day_begin, _daily_count = now, 0
        if _hourly_count >= HOURLY_UNIQUE_CAP or _daily_count >= DAILY_REQUEST_CAP:
            return {"status": "NOT_VERIFIED", "reason": "LIVE_STATUS_BUDGET_EXHAUSTED",
                    "provider_requests_added": 0}
        key = _key()
        if fetcher is None and not key:
            return {"status": "NOT_VERIFIED", "reason": "PROVIDER_NOT_CONFIGURED",
                    "provider_requests_added": 0}
        # Reservations are made atomically so concurrent page openings cannot burst.
        _hourly_count += 1
        _daily_count += 1
    try:
        source = (fetcher(fixture_id, key) if fetcher else _call_api(fixture_id, key))
        observed = datetime.now(timezone.utc).isoformat()
        parsed = _parse(source, fixture_id, home_id, away_id, observed)
        if parsed is None:
            return {"status": "NOT_VERIFIED", "reason": "SOURCE_STATUS_IDENTITY_NOT_VERIFIED",
                    "provider_requests_added": 1}
        with _lock:
            _cache[fixture_id] = (now, parsed)
        return parsed
    except Exception:
        return {"status": "NOT_VERIFIED", "reason": "PROVIDER_STATUS_UNAVAILABLE",
                "provider_requests_added": 1}
