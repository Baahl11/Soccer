#!/usr/bin/env python3
"""Health-first monitor for the Soccer Edge scheduler state branch.

This script only reads persisted GitHub state. It does not trigger ticks,
provider requests, deployments, or code changes.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

DEFAULT_REPO = "Baahl11/Soccer"
STATE_REF = "soccer-edge-state"
HEALTH_PATH = "soccer_edge_state/health.json"
LAST_ERROR_PATH = "soccer_edge_state/last_error.json"
LATEST_PATH = "soccer_edge_state/latest.json"
TZ_NAME = "America/Mexico_City"
LOCAL_TZ = ZoneInfo(TZ_NAME)
MIN_VERSION = (3, 60, 0)
ACTIVE_START_HOUR = 7
ACTIVE_END_HOUR = 23
STALE_MINUTES = 75
DEFAULT_DEDUPE_HOURS = 4

HEALTH_FIELDS = [
    "generated_at_local",
    "version",
    "fixture_scan_count",
    "due_fixture_count",
    "event_count",
    "actionable_refresh_count",
    "bet_candidate_count",
    "api_calls_this_tick",
    "last_daily_remaining",
    "daily_budget_mode",
    "database_persisted",
]


@dataclass(frozen=True)
class FetchResult:
    ok: bool
    missing: bool
    path: str
    data: dict[str, Any] | None = None
    error: str | None = None


@dataclass(frozen=True)
class MonitorDecision:
    notify: bool
    severity: str
    reason: str
    fingerprint: str
    health_fields: dict[str, Any]
    details: dict[str, Any]
    suppressed: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Soccer Edge persisted scheduler health.")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="GitHub repo in owner/name form.")
    parser.add_argument("--ref", default=STATE_REF, help="Persisted scheduler state branch/ref.")
    parser.add_argument("--token", default=os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN"))
    parser.add_argument("--state-file", default=".soccer_edge_health_monitor_state.json")
    parser.add_argument("--no-dedupe", action="store_true", help="Always print active alert.")
    parser.add_argument("--dedupe-hours", type=float, default=DEFAULT_DEDUPE_HOURS)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("--now-local", help="Override current local time for tests/manual validation.")
    return parser.parse_args()


def parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=LOCAL_TZ)
    return parsed.astimezone(LOCAL_TZ)


def parse_version(value: Any) -> tuple[int, int, int]:
    if value is None:
        return (0, 0, 0)
    numbers = [int(part) for part in re.findall(r"\d+", str(value))[:3]]
    while len(numbers) < 3:
        numbers.append(0)
    return tuple(numbers[:3])  # type: ignore[return-value]


def normalize_text(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, ensure_ascii=False) if not isinstance(value, str) else value
    text = text.lower()
    text = re.sub(r"\d{4}-\d{2}-\d{2}t\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:z|[+-]\d{2}:\d{2})?", "<ts>", text)
    text = re.sub(r"\b\d{10,}\b", "<num>", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:500]


def fingerprint(category: str, payload: Any) -> str:
    normalized = f"{category}:{normalize_text(payload)}"
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"{category}:{digest}"


def active_window(now_local: datetime) -> bool:
    return ACTIVE_START_HOUR <= now_local.hour < ACTIVE_END_HOUR


def health_field_snapshot(health: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(health, dict):
        return {key: None for key in HEALTH_FIELDS}
    return {key: health.get(key) for key in HEALTH_FIELDS}


def github_content_url(repo: str, path: str, ref: str) -> str:
    owner_repo = urllib.parse.quote(repo, safe="/")
    quoted_path = urllib.parse.quote(path, safe="/")
    quoted_ref = urllib.parse.quote(ref, safe="")
    return f"https://api.github.com/repos/{owner_repo}/contents/{quoted_path}?ref={quoted_ref}"


def fetch_json_file(repo: str, path: str, ref: str, token: str | None = None) -> FetchResult:
    url = github_content_url(repo, path, ref)
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "soccer-edge-health-monitor",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return FetchResult(ok=False, missing=True, path=path, error="404 Not Found")
        return FetchResult(ok=False, missing=False, path=path, error=f"HTTP {exc.code}: {exc.reason}")
    except Exception as exc:  # pragma: no cover - defensive network wrapper
        return FetchResult(ok=False, missing=False, path=path, error=f"{type(exc).__name__}: {exc}")

    try:
        if payload.get("encoding") == "base64":
            raw = base64.b64decode(payload.get("content", "")).decode("utf-8")
        else:
            raw = payload.get("content", "")
        data = json.loads(raw)
    except Exception as exc:
        return FetchResult(ok=False, missing=False, path=path, error=f"Unreadable JSON: {type(exc).__name__}: {exc}")
    if not isinstance(data, dict):
        return FetchResult(ok=False, missing=False, path=path, error="JSON root is not an object")
    return FetchResult(ok=True, missing=False, path=path, data=data)


def extract_error_timestamp(last_error: dict[str, Any]) -> datetime | None:
    candidates = [
        last_error.get("captured_at_local"),
        last_error.get("generated_at_local"),
        last_error.get("captured_at"),
        last_error.get("timestamp"),
        last_error.get("ts"),
    ]
    response = last_error.get("response")
    if isinstance(response, dict):
        candidates.extend([
            response.get("generated_at_local"),
            response.get("captured_at_local"),
            response.get("timestamp"),
        ])
    for value in candidates:
        parsed = parse_datetime(value)
        if parsed:
            return parsed
    return None


def classify_error(last_error: dict[str, Any]) -> tuple[str, str, str]:
    http_code = str(last_error.get("http_code") or "")
    response = last_error.get("response") if isinstance(last_error.get("response"), dict) else {}
    detail = response.get("detail") or response.get("error") or last_error.get("detail") or last_error.get("error") or last_error
    text = normalize_text(detail)

    if http_code == "429" or "429" in text or "too many requests" in text or "rate limit" in text or "request limit" in text:
        return "P3", "rate-limit failure after healthy heartbeat", "rate_limit"
    if http_code.startswith("5") or "http 5" in text or "tick_failed" in text:
        return "P2", "HTTP 5xx / tick failure after healthy heartbeat", "http_5xx_tick_failure"
    if "pipeline_error" in text:
        return "P2", "PIPELINE_ERROR after healthy heartbeat", "pipeline_error"
    if "github" in text and ("persist" in text or "push" in text or "commit" in text):
        return "P2", "GitHub state-persistence failure after healthy heartbeat", "github_persistence"
    return "P2", "new scheduler/backend failure after healthy heartbeat", "scheduler_backend_failure"


def evaluate_state(
    health_result: FetchResult,
    last_error_result: FetchResult,
    now_local: datetime,
) -> MonitorDecision:
    if not health_result.ok:
        reason = f"health.json missing/unreadable on {STATE_REF}: {health_result.error or 'unknown error'}"
        return MonitorDecision(
            notify=True,
            severity="P1",
            reason=reason,
            fingerprint=fingerprint("health_unreadable", reason),
            health_fields=health_field_snapshot(None),
            details={"health_error": health_result.error, "health_missing": health_result.missing},
        )

    health = health_result.data or {}
    fields = health_field_snapshot(health)
    generated = parse_datetime(health.get("generated_at_local"))
    if generated is None:
        reason = "health.json generated_at_local is missing or invalid"
        return MonitorDecision(
            notify=True,
            severity="P1",
            reason=reason,
            fingerprint=fingerprint("health_bad_timestamp", reason),
            health_fields=fields,
            details={"generated_at_local": health.get("generated_at_local")},
        )

    if active_window(now_local):
        age_minutes = (now_local - generated).total_seconds() / 60.0
        if age_minutes > STALE_MINUTES:
            reason = f"health.json heartbeat is stale: {age_minutes:.1f} minutes old during active window"
            return MonitorDecision(
                notify=True,
                severity="P1",
                reason=reason,
                fingerprint=fingerprint("health_stale", {"generated_at_local": health.get("generated_at_local"), "window": TZ_NAME}),
                health_fields=fields,
                details={"age_minutes": round(age_minutes, 1), "active_window": True},
            )

    version = parse_version(health.get("version"))
    if version < MIN_VERSION:
        reason = f"health version below 3.60.0: {health.get('version')}"
        return MonitorDecision(
            notify=True,
            severity="P2",
            reason=reason,
            fingerprint=fingerprint("health_version", health.get("version")),
            health_fields=fields,
            details={"min_version": ".".join(map(str, MIN_VERSION)), "parsed_version": version},
        )

    if str(health.get("status", "")).lower() != "ok":
        reason = f"health status is not ok: {health.get('status')}"
        return MonitorDecision(
            notify=True,
            severity="P1",
            reason=reason,
            fingerprint=fingerprint("health_status", health.get("status")),
            health_fields=fields,
            details={"status": health.get("status")},
        )

    if last_error_result.ok and isinstance(last_error_result.data, dict):
        last_error = last_error_result.data
        err_at = extract_error_timestamp(last_error)
        if err_at and err_at > generated:
            severity, reason, category = classify_error(last_error)
            return MonitorDecision(
                notify=True,
                severity=severity,
                reason=reason,
                fingerprint=fingerprint(category, {
                    "http_code": last_error.get("http_code"),
                    "response": last_error.get("response"),
                    "detail": last_error.get("detail"),
                    "error": last_error.get("error"),
                }),
                health_fields=fields,
                details={
                    "last_error_at_local": err_at.isoformat(),
                    "health_generated_at_local": generated.isoformat(),
                    "last_error_http_code": last_error.get("http_code"),
                    "last_error": last_error,
                },
            )

    return MonitorDecision(
        notify=False,
        severity="OK",
        reason="health current and ok; no newer persisted scheduler/backend failure",
        fingerprint="ok",
        health_fields=fields,
        details={
            "active_window": active_window(now_local),
            "health_generated_at_local": generated.isoformat(),
            "last_error_missing": last_error_result.missing,
            "database_persisted_is_informational": True,
            "latest_json_timestamp_is_not_a_health_signal": True,
        },
    )


def load_dedupe_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def apply_dedupe(decision: MonitorDecision, state: dict[str, Any], now_local: datetime, dedupe_hours: float) -> MonitorDecision:
    if not decision.notify:
        return decision
    last = state.get("last_alert") if isinstance(state.get("last_alert"), dict) else {}
    last_fp = last.get("fingerprint")
    last_notified = parse_datetime(last.get("notified_at_local"))
    last_severity = last.get("severity")
    if last_fp == decision.fingerprint and last_severity == decision.severity and last_notified:
        elapsed = now_local - last_notified
        if elapsed < timedelta(hours=dedupe_hours):
            return MonitorDecision(
                notify=False,
                severity=decision.severity,
                reason=f"duplicate alert suppressed for {elapsed.total_seconds() / 60.0:.1f} minutes: {decision.reason}",
                fingerprint=decision.fingerprint,
                health_fields=decision.health_fields,
                details={**decision.details, "suppressed_duplicate": True},
                suppressed=True,
            )
    return decision


def save_dedupe_state(path: Path, decision: MonitorDecision, now_local: datetime) -> None:
    payload: dict[str, Any] = {
        "updated_at_local": now_local.isoformat(),
        "last_check": {
            "severity": decision.severity,
            "notify": decision.notify,
            "suppressed": decision.suppressed,
            "fingerprint": decision.fingerprint,
            "reason": decision.reason,
        },
    }
    if decision.notify:
        payload["last_alert"] = {
            "notified_at_local": now_local.isoformat(),
            "severity": decision.severity,
            "fingerprint": decision.fingerprint,
            "reason": decision.reason,
        }
    elif decision.severity == "OK":
        payload["last_ok_local"] = now_local.isoformat()
    else:
        previous = load_dedupe_state(path)
        if "last_alert" in previous:
            payload["last_alert"] = previous["last_alert"]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def markdown(decision: MonitorDecision) -> str:
    lines = []
    if decision.notify:
        lines.append(f"⚠️ Soccer backend alert ({decision.severity})")
    elif decision.suppressed:
        lines.append(f"🔕 Soccer backend duplicate alert suppressed ({decision.severity})")
    else:
        lines.append("✅ Soccer backend healthy")
    lines.append("")
    lines.append(f"Reason: {decision.reason}")
    lines.append(f"Fingerprint: {decision.fingerprint}")
    lines.append("")
    lines.append("Health fields:")
    for key in HEALTH_FIELDS:
        lines.append(f"- {key}: {decision.health_fields.get(key)!r}")
    return "\n".join(lines)


def as_json(decision: MonitorDecision) -> str:
    payload = {
        "notify": decision.notify,
        "suppressed": decision.suppressed,
        "severity": decision.severity,
        "reason": decision.reason,
        "fingerprint": decision.fingerprint,
        "health_fields": decision.health_fields,
        "details": decision.details,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def main() -> int:
    args = parse_args()
    now_local = parse_datetime(args.now_local) if args.now_local else datetime.now(LOCAL_TZ)
    if now_local is None:
        raise SystemExit("Invalid --now-local value")

    health = fetch_json_file(args.repo, HEALTH_PATH, args.ref, args.token)
    last_error = fetch_json_file(args.repo, LAST_ERROR_PATH, args.ref, args.token)
    # latest.json may be useful for operators, but it is intentionally not part of health evaluation.
    _ = LATEST_PATH

    decision = evaluate_state(health, last_error, now_local)
    state_path = Path(args.state_file)
    if not args.no_dedupe:
        state = load_dedupe_state(state_path)
        decision = apply_dedupe(decision, state, now_local, args.dedupe_hours)
        save_dedupe_state(state_path, decision, now_local)

    print(as_json(decision) if args.json else markdown(decision))
    if decision.notify:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
