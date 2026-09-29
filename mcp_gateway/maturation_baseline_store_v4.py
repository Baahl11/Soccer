from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from mcp_gateway import persistence

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_MATURATION_BASELINE_STORE_V4_1.0.0"
STATE_KEY = "maturation_watchdogs_v4"
_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS maturation_watchdog_state (
    state_key TEXT PRIMARY KEY,
    baseline JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""


def _status(status: str, *, reason: str | None = None, updated_at: Any = None) -> dict[str, Any]:
    out = {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": status,
        "source": "POSTGRES:maturation_watchdog_state",
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
    }
    if reason:
        out["reason"] = reason
    if updated_at is not None:
        out["updated_at_utc"] = updated_at.isoformat() if hasattr(updated_at, "isoformat") else str(updated_at)
    return out


def load_baseline() -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not persistence.persistence_configured():
        return None, _status("UNAVAILABLE", reason="DATABASE_URL_NOT_CONFIGURED")
    try:
        with persistence._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(_TABLE_SQL)
                cur.execute(
                    "SELECT baseline, updated_at FROM maturation_watchdog_state WHERE state_key = %s",
                    (STATE_KEY,),
                )
                row = cur.fetchone()
        if not row:
            return None, _status("EMPTY", reason="BASELINE_NOT_INITIALIZED")
        baseline = row[0] if isinstance(row[0], dict) else json.loads(row[0])
        if not isinstance(baseline, dict):
            return None, _status("ERROR", reason="BASELINE_JSON_NOT_OBJECT", updated_at=row[1])
        baseline = dict(baseline)
        baseline["source"] = "POSTGRES:maturation_watchdog_state"
        return baseline, _status("OK", updated_at=row[1])
    except Exception as exc:
        return None, _status("ERROR", reason=f"{type(exc).__name__}: {str(exc)[:160]}")


def save_baseline(baseline: dict[str, Any]) -> dict[str, Any]:
    if not persistence.persistence_configured():
        return _status("UNAVAILABLE", reason="DATABASE_URL_NOT_CONFIGURED")
    if not isinstance(baseline, dict):
        return _status("ERROR", reason="BASELINE_NOT_OBJECT")
    payload = dict(baseline)
    payload.pop("source", None)
    try:
        with persistence._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(_TABLE_SQL)
                cur.execute(
                    """
                    INSERT INTO maturation_watchdog_state (state_key, baseline, updated_at)
                    VALUES (%s, %s::jsonb, NOW())
                    ON CONFLICT (state_key) DO UPDATE SET
                        baseline = EXCLUDED.baseline,
                        updated_at = NOW()
                    RETURNING updated_at
                    """,
                    (STATE_KEY, json.dumps(payload, separators=(",", ":"))),
                )
                row = cur.fetchone()
        return _status("OK", updated_at=row[0] if row else datetime.now(timezone.utc))
    except Exception as exc:
        return _status("ERROR", reason=f"{type(exc).__name__}: {str(exc)[:160]}")
