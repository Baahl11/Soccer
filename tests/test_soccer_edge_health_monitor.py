from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from soccer_edge_health_monitor import (  # noqa: E402
    FetchResult,
    apply_dedupe,
    evaluate_state,
    fingerprint,
    parse_datetime,
)


def ok_health(**overrides):
    base = {
        "status": "ok",
        "version": "3.60.0",
        "generated_at_local": "2026-09-21T10:56:26-06:00",
        "fixture_scan_count": 150,
        "due_fixture_count": 71,
        "event_count": 0,
        "actionable_refresh_count": 0,
        "bet_candidate_count": 0,
        "api_calls_this_tick": 24,
        "last_daily_remaining": 6513,
        "daily_budget_mode": "NORMAL",
        "database_persisted": False,
    }
    base.update(overrides)
    return base


def decision_for(health, last_error=None, now="2026-09-21T11:15:00-06:00"):
    return evaluate_state(
        FetchResult(ok=True, missing=False, path="health", data=health),
        FetchResult(ok=last_error is not None, missing=last_error is None, path="last_error", data=last_error),
        parse_datetime(now),
    )


def test_current_ok_health_does_not_alert_with_zero_bets_and_database_false():
    decision = decision_for(ok_health())
    assert decision.notify is False
    assert decision.severity == "OK"
    assert decision.health_fields["database_persisted"] is False
    assert decision.health_fields["bet_candidate_count"] == 0


def test_missing_health_alerts_even_if_last_error_is_absent():
    decision = evaluate_state(
        FetchResult(ok=False, missing=True, path="health", error="404 Not Found"),
        FetchResult(ok=False, missing=True, path="last_error", error="404 Not Found"),
        parse_datetime("2026-09-21T11:15:00-06:00"),
    )
    assert decision.notify is True
    assert decision.severity == "P1"
    assert "missing/unreadable" in decision.reason


def test_stale_health_alerts_during_active_window():
    decision = decision_for(
        ok_health(generated_at_local="2026-09-21T08:00:00-06:00"),
        now="2026-09-21T10:00:01-06:00",
    )
    assert decision.notify is True
    assert decision.severity == "P1"
    assert "stale" in decision.reason


def test_version_below_360_alerts():
    decision = decision_for(ok_health(version="3.59.9"))
    assert decision.notify is True
    assert decision.severity == "P2"
    assert "version below" in decision.reason


def test_non_ok_status_alerts():
    decision = decision_for(ok_health(status="degraded"))
    assert decision.notify is True
    assert decision.severity == "P1"
    assert "status is not ok" in decision.reason


def test_last_error_newer_than_heartbeat_alerts_for_http_500_tick_failure():
    last_error = {
        "captured_at": "2026-09-21T17:05:00Z",
        "http_code": "500",
        "response": {
            "detail": "tick_failed: module 'mcp_gateway.automation_v2' has no attribute '_american'"
        },
    }
    decision = decision_for(ok_health(), last_error=last_error)
    assert decision.notify is True
    assert decision.severity == "P2"
    assert "HTTP 5xx" in decision.reason
    assert decision.fingerprint.startswith("http_5xx_tick_failure:")


def test_last_error_older_than_health_does_not_alert():
    last_error = {
        "captured_at": "2026-09-21T16:00:00Z",
        "http_code": "500",
        "response": {"detail": "old failure"},
    }
    decision = decision_for(ok_health(), last_error=last_error)
    assert decision.notify is False
    assert decision.severity == "OK"


def test_rate_limit_is_alert_but_lower_severity():
    last_error = {
        "captured_at": "2026-09-21T17:05:00Z",
        "http_code": "429",
        "response": {"detail": "429 Too Many Requests from API-Football"},
    }
    decision = decision_for(ok_health(), last_error=last_error)
    assert decision.notify is True
    assert decision.severity == "P3"
    assert "rate-limit" in decision.reason


def test_fingerprint_normalizes_timestamps_for_dedupe():
    a = fingerprint("http_5xx_tick_failure", {"detail": "failed at 2026-09-21T17:05:00Z"})
    b = fingerprint("http_5xx_tick_failure", {"detail": "failed at 2026-09-21T18:05:00Z"})
    assert a == b


def test_dedupe_suppresses_same_recent_alert():
    last_error = {
        "captured_at": "2026-09-21T17:05:00Z",
        "http_code": "500",
        "response": {"detail": "tick_failed: same bug"},
    }
    decision = decision_for(ok_health(), last_error=last_error)
    state = {
        "last_alert": {
            "notified_at_local": "2026-09-21T11:05:00-06:00",
            "severity": decision.severity,
            "fingerprint": decision.fingerprint,
        }
    }
    suppressed = apply_dedupe(decision, state, parse_datetime("2026-09-21T11:15:00-06:00"), 4)
    assert suppressed.notify is False
    assert suppressed.suppressed is True


def test_old_latest_timestamp_is_not_part_of_health_evaluation():
    # There is intentionally no latest.json input to evaluate_state. Health is the scheduler heartbeat.
    decision = decision_for(ok_health(event_count=0, bet_candidate_count=0))
    assert decision.notify is False
