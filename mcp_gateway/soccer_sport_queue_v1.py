"""Idempotent fixture-level soccer screening coverage; no model or provider calls."""
from __future__ import annotations

from typing import Any
from datetime import datetime, timezone

STATUS = frozenset({"QUEUED", "MODELED_LIMITED", "DATA_INSUFFICIENT", "RETRY_REQUIRED"})
EXCLUDED = ("FT", "AET", "PEN", "CANC", "PST", "SUSP", "INT", "ABD", "AWD", "WO")
MAX_REGISTRY = 5000

def sync_pending(cur: Any) -> None:
    """Discover persistent queue entries for the next 3 days using only registry."""
    cur.execute(
        """
        INSERT INTO soccer_sport_screen_queue (fixture_id, kickoff)
        SELECT fixture_id, kickoff FROM soccer_fixtures
        WHERE kickoff > CURRENT_TIMESTAMP
          AND kickoff <= CURRENT_TIMESTAMP + INTERVAL '3 days'
          AND COALESCE(status, 'NS') NOT IN
              ('FT','AET','PEN','CANC','PST','SUSP','INT','ABD','AWD','WO')
        ORDER BY kickoff, fixture_id
        LIMIT %s
        ON CONFLICT (fixture_id) DO UPDATE
          SET kickoff = EXCLUDED.kickoff,
              updated_at = NOW()
        WHERE soccer_sport_screen_queue.kickoff IS DISTINCT FROM EXCLUDED.kickoff
        """,
        (MAX_REGISTRY,),
    )

def record_event(cur: Any, event: dict[str, Any], *, at: Any) -> None:
    """Only a genuine SPORT_PREVIEW completion changes sport processing state."""
    if event.get("event_type") != "SOCCER_REFRESH" or event.get("stage") != "SPORT_PREVIEW":
        return
    fixture = event.get("fixture")
    fixture = fixture if isinstance(fixture, dict) else {}
    fid = fixture.get("fixture_id")
    kickoff = fixture.get("kickoff")
    if type(fid) is not int or fid <= 0 or not kickoff:
        return
    # Persisted research-only outcome. Missing-input state never overrides a
    # previously recorded RAW sport model on the same fixture.
    modeled = event.get("raw_sport_show_as_forecast") is True
    status = "MODELED_LIMITED" if modeled else "DATA_INSUFFICIENT"
    reason = str(event.get("raw_sport_screen_status") or "NOT_VERIFIED")[:280]
    cur.execute(
        """
        INSERT INTO soccer_sport_screen_queue
            (fixture_id,kickoff,sport_status,attempt_count,last_attempt_at,
             last_completed_at,last_model_version,last_reason)
        VALUES (%s,%s,%s,1,%s,%s,%s,%s)
        ON CONFLICT (fixture_id) DO UPDATE SET
            sport_status = CASE
                WHEN soccer_sport_screen_queue.sport_status='MODELED_LIMITED'
                    THEN 'MODELED_LIMITED'
                ELSE EXCLUDED.sport_status END,
            kickoff = EXCLUDED.kickoff,
            attempt_count = soccer_sport_screen_queue.attempt_count + 1,
            last_attempt_at = EXCLUDED.last_attempt_at,
            last_completed_at = EXCLUDED.last_completed_at,
            last_model_version = EXCLUDED.last_model_version,
            last_reason = EXCLUDED.last_reason,
            updated_at = NOW()
        """,
        (fid, kickoff, status, at, at, event.get("model_version"), reason),
    )

def read_statuses(connect: Any, fixture_ids: list[int]) -> dict[str, dict[str, Any]]:
    """One bounded DB read; missing table remains UNKNOWN not magically zero."""
    ids = sorted(set(x for x in fixture_ids if type(x) is int and x > 0))[:500]
    if not ids:
        return {}
    try:
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT fixture_id,sport_status,attempt_count,last_attempt_at,
                           last_completed_at,last_model_version,last_reason
                    FROM soccer_sport_screen_queue
                    WHERE fixture_id = ANY(%s)
                    """,
                    (ids,),
                )
                rows = cur.fetchall()
    except Exception:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for fid, status, attempts, last_attempt, completed, version, reason in rows:
        if status not in STATUS:
            continue
        out[str(fid)] = {
            "status": status,
            "attempt_count": int(attempts),
            "last_attempt_at": last_attempt.isoformat() if hasattr(last_attempt, "isoformat") else last_attempt,
            "last_completed_at": completed.isoformat() if hasattr(completed, "isoformat") else completed,
            "model_version": version,
            "reason": reason,
            "provider_requests_added": 0,
            "source": "POSTGRES_SOCCER_SPORT_SCREEN_QUEUE",
        }
    return out

def coverage_summary(queue_rows: dict[str, dict[str, Any]], fixtures: list[dict[str, Any]]) -> dict[str, Any]:
    """All eligible fixtures counted; missing per-fixture queue is UNKNOWN."""
    counts = {name: 0 for name in ("QUEUED","MODELED_LIMITED","DATA_INSUFFICIENT","RETRY_REQUIRED","NOT_VERIFIED")}
    for f in fixtures:
        fixture = f.get("fixture") if isinstance(f, dict) else {}
        fixture = fixture if isinstance(fixture, dict) else {}
        q = queue_rows.get(str(fixture.get("fixture_id"))) or {}
        key = q.get("status")
        counts[key if key in STATUS else "NOT_VERIFIED"] += 1
    return {
        "total_eligible_fixtures": len(fixtures),
        "by_sport_queue_status": counts,
        "all_fixtures_accounted_for": sum(counts.values()) == len(fixtures),
        "note": "Registro de estado; modelado completo y comercializacion NO implicados.",
        "no_market_inputs": True,
    }
