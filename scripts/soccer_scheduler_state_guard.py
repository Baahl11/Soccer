"""Read-only canonical Soccer scheduler freshness guard.

Uses only the already-persisted public aggregate health.json snapshot.
No provider requests, database writes, wagers, replays, or model promotion.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

VERSION = "SOCCER_CANONICAL_SCHEDULER_STATE_WATCH_v1.0"


def _dt(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        out = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return out.astimezone(timezone.utc) if out.tzinfo else None
    except ValueError:
        return None


def verify_health(state: Any, *, now_utc: datetime, max_age_minutes: int = 180) -> dict[str, Any]:
    if (not isinstance(now_utc, datetime) or now_utc.tzinfo is None
            or type(max_age_minutes) is not int or not 30 <= max_age_minutes <= 1440):
        raise ValueError("INVALID_WATCH_CONTRACT")
    now = now_utc.astimezone(timezone.utc)
    blockers: list[str] = []
    if not isinstance(state, dict):
        state = {}
        blockers.append("STATE_NOT_OBJECT")
    generated = _dt(state.get("generated_at_utc"))
    age_min = round((now - generated).total_seconds() / 60, 2) if generated else None
    if age_min is None:
        blockers.append("TIMESTAMP_NOT_VERIFIED")
    elif age_min < -5:
        blockers.append("TIMESTAMP_IN_FUTURE")
    elif age_min > max_age_minutes:
        blockers.append("SCHEDULER_SNAPSHOT_STALE")
    if state.get("status") not in (None, "ok"):
        blockers.append("SCHEDULER_STATE_NOT_OK")
    if state.get("database_persisted") is not True or state.get("database_error"):
        blockers.append("DATABASE_PERSISTENCE_NOT_VERIFIED")
    scans = state.get("fixture_scan_count")
    if type(scans) is not int or scans <= 0:
        blockers.append("OFFICIAL_FIXTURE_SCAN_NOT_VERIFIED")
    remaining = state.get("last_daily_remaining")
    if remaining is not None and (type(remaining) is not int or remaining < 0):
        blockers.append("PROVIDER_QUOTA_TELEMETRY_INVALID")
    return {
        "version": VERSION,
        "status": "VERIFIED_RECENT_CANONICAL_TICK" if not blockers else "NOT_VERIFIED",
        "observed_at_utc": now.isoformat(),
        "generated_at_utc": generated.isoformat() if generated else None,
        "snapshot_age_minutes": age_min,
        "max_age_minutes": max_age_minutes,
        "fixture_scan_count": scans if type(scans) is int else None,
        "api_calls_this_tick": state.get("api_calls_this_tick")
            if type(state.get("api_calls_this_tick")) is int else None,
        "last_daily_remaining": remaining if type(remaining) is int else None,
        "database_persisted": state.get("database_persisted") is True,
        "blockers": blockers,
        "provider_requests_added": 0,
        "database_queries_added": 0,
        "market_decisions_changed": False,
        "bets_authorized": False,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--health", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-age-minutes", type=int, default=180)
    args = p.parse_args()
    try:
        obj = json.loads(args.health.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        obj = {}
    result = verify_health(
        obj, now_utc=datetime.now(timezone.utc),
        max_age_minutes=args.max_age_minutes,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "VERIFIED_RECENT_CANONICAL_TICK" else 2


if __name__ == "__main__":
    raise SystemExit(main())
