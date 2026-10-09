#!/usr/bin/env python3
"""Read-only Soccer Edge source/research freshness guard.

Never infers missing formations, rewrites historical rows or promotes a model.
Exit 2 for critical data/operations anomalies; healthy-but-underpowered is allowed.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


def dt(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def hours_since(now, when):
    if when is None:
        return None
    return round(max(0.0, (now - when).total_seconds() / 3600), 2)


def read_json(path, critical):
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(d, dict):
            raise ValueError("expected object")
        return d
    except (OSError, ValueError) as e:
        critical.append(f"REQUIRED_ARTIFACT_INVALID:{path.name}:{type(e).__name__}")
        return {}


def monitor(state: Path, *, now: datetime, research_commit_utc: str | None,
            history_latest_date: str | None):
    critical, warnings = [], []
    analysis = state / "soccer_edge_state" / "analysis"
    health = read_json(state / "soccer_edge_state" / "health.json", critical)
    ledger = read_json(analysis / "signal_ledger_summary.json", critical)
    corners = read_json(analysis / "corners_baseline.json", critical)
    fm4 = read_json(analysis / "formation_matchup_fm4_style_ablation_v1.json", critical)
    team = read_json(analysis / "team_corners_validation.json", critical)
    validation = read_json(analysis / "v4_022_corners_oos_validation.json", critical)

    live_age_h = hours_since(now, dt(health.get("generated_at_utc")))
    evidence_age_h = hours_since(now, dt(ledger.get("last_evidence_at_utc")))
    research_age_h = hours_since(now, dt(research_commit_utc))
    last_corner_eval = max(
        (dt(row.get("kickoff_local")) for row in (corners.get("evaluations") or [])
         if isinstance(row, dict) and dt(row.get("kickoff_local")) is not None),
        default=None,
    )
    corner_eval_age_h = hours_since(now, last_corner_eval)
    source_latest = dt(((fm4.get("style_profile") or {}).get("source_date_range") or {}).get("latest"))
    style_age_h = hours_since(now, source_latest)
    latest_history = dt(history_latest_date + "T00:00:00+00:00") if history_latest_date else None
    history_age_h = hours_since(now, latest_history)

    if live_age_h is None or live_age_h > 6:
        critical.append("LIVE_HEALTH_STALE_OR_MISSING_GT_6H")
    if evidence_age_h is None or evidence_age_h > 12:
        critical.append("CANONICAL_LEDGER_EVIDENCE_STALE_OR_MISSING_GT_12H")
    if health and health.get("database_persisted") is not True:
        critical.append("LIVE_DATABASE_NOT_PERSISTED")
    if research_age_h is None or research_age_h > 48:
        critical.append("CORNERS_RESEARCH_ARTIFACT_STALE_OR_UNVERIFIED_GT_48H")
    elif research_age_h > 30:
        warnings.append("CORNERS_RESEARCH_ARTIFACT_AGE_OVER_30H")

    # No automatic assertion that 44/100 must rise every day.
    # This detects a separate coverage failure while live canonical evidence advances.
    if evidence_age_h is not None and evidence_age_h <= 12:
        if corner_eval_age_h is None or corner_eval_age_h > 168:
            critical.append("CORNERS_VERIFIED_EVALUATION_COVERAGE_GAP_GT_7D")
        if style_age_h is None or style_age_h > 168:
            critical.append("FM4_VERIFIED_STYLE_SOURCE_GAP_GT_7D")
    if history_age_h is None or history_age_h > 48:
        critical.append("HISTORY_FILE_DATE_STALE_OR_UNKNOWN_GT_48H")

    n = corners.get("formation_adjusted_evaluations")
    oos = corners.get("walk_forward_evaluations")
    audit = corners.get("formation_eligibility_audit") or {}
    if n is None or oos is None or not isinstance(n, int) or not isinstance(oos, int):
        critical.append("CORNERS_OOS_COUNTERS_MISSING")
    elif not (0 <= n <= oos):
        critical.append("CORNERS_OOS_COUNTERS_INCONSISTENT")
    if n is not None and audit.get("formation_adjusted_evaluations") != n:
        critical.append("CORNERS_ELIGIBILITY_AUDIT_MISMATCH")
    if corners and corners.get("status") != "RESEARCH_ONLY_CORNERS_BASELINE":
        critical.append("CORNERS_STATUS_CONTRACT_CHANGED")
    if validation and validation.get("production_promotion_allowed") is not False:
        critical.append("PRODUCTION_PROMOTION_CONTRACT_VIOLATION")
    if not team.get("schema_version"):
        critical.append("TEAM_CORNERS_VALIDATION_SCHEMA_MISSING")

    report = {
        "schema_version": "1.0.0",
        "generated_at_utc": now.isoformat(),
        "status": "CRITICAL" if critical else ("WARNING" if warnings else "HEALTHY"),
        "read_only": True,
        "predictions_rewritten": False,
        "model_weights_changed": False,
        "thresholds_changed": False,
        "auto_promotion_allowed": False,
        "measurements": {
            "live_health_age_hours": live_age_h,
            "canonical_evidence_age_hours": evidence_age_h,
            "research_commit_age_hours": research_age_h,
            "last_corners_oos_kickoff_utc": last_corner_eval.isoformat() if last_corner_eval else None,
            "corners_oos_age_hours": corner_eval_age_h,
            "latest_fm4_style_source_utc": source_latest.isoformat() if source_latest else None,
            "fm4_source_age_hours": style_age_h,
            "latest_history_file_date": history_latest_date,
            "history_file_age_hours": history_age_h,
            "corners_oos_evaluations": oos,
            "formation_adjusted_evaluations": n,
            "formation_adjusted_minimum": 100,
            "formation_missing_reason_counts": audit.get("reason_counts"),
            "team_corners_status": team.get("status"),
            "v4_corners_status": validation.get("status"),
        },
        "critical": sorted(set(critical)),
        "warnings": sorted(set(warnings)),
        "interpretation": (
            "Coverage or pipeline anomaly; investigate source and workflow. "
            "Do not fabricate formation history or lower OOS/True CLV thresholds. "
            "Underpowered gate by itself is not a pipeline failure."
        ),
    }
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", required=True, type=Path)
    ap.add_argument("--research-commit-utc", default="")
    ap.add_argument("--history-latest-date", default="")
    ap.add_argument("--now-utc", default="")
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    now = dt(args.now_utc) if args.now_utc else datetime.now(timezone.utc)
    if now is None:
        ap.error("--now-utc must be an ISO timestamp")
    history_date = args.history_latest_date
    if history_date and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", history_date):
        ap.error("history date must be YYYY-MM-DD")
    report = monitor(args.state_dir, now=now,
                     research_commit_utc=args.research_commit_utc,
                     history_latest_date=history_date or None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "critical", "warnings", "measurements")},
                     indent=2))
    return 2 if report["critical"] else 0


if __name__ == "__main__":
    sys.exit(main())
