#!/usr/bin/env python3
"""Read-only multi-market maturity audit. Never promotes betting markets."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPORTS = {
    "FT_TOTALS": "v4_016_ft_totals_production_validation.json",
    "1X2": "v4_017_1x2_calibration_validation.json",
    "BTTS": "v4_018_btts_calibration_validation.json",
    "TEAM_TOTALS": "v4_019_team_totals_oos_validation.json",
    "1H": "v4_020_1h_oos_validation.json",
    "2H": "v4_021_2h_oos_validation.json",
    "CORNERS": "v4_022_corners_oos_validation.json",
    "CARDS": "phase14_cards_referee_validation.json",
    "PLAYER_PROPS": "phase15_player_props_validation.json",
}
EXPECTED_CANONICAL = {"1X2": "1X2", "FT_TOTALS": "FT_TOTALS", "BTTS": "BTTS"}


def parse_dt(x):
    try:
        if not x:
            return None
        return datetime.fromisoformat(str(x).replace("Z", "+00:00")).astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def age_h(now, x):
    dt = parse_dt(x)
    if dt is None:
        return None
    return round(max(0.0, (now-dt).total_seconds()/3600), 2)


def audit(state: Path, commits: dict, now: datetime):
    analysis = state / "soccer_edge_state" / "analysis"
    critical, warnings = [], []

    def source(filename):
        path = analysis / filename
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(d, dict):
                raise ValueError("non-object report")
            return d
        except (OSError, ValueError) as e:
            critical.append("INVALID_REQUIRED_REPORT:"+filename+":"+type(e).__name__)
            return {}

    def age(filename):
        return age_h(now, commits.get(filename))

    canonical = source("clv_v4_postgres_report.json")
    oos = source("oos_calibration_v4_report.json")
    props_close = source("player_props_clv_v4_report.json")
    props_oos = source("player_props_oos_v4_report.json")
    settlement = source("settlement_postgres_v4.json")
    market_performance = source("market_performance_summary.json")
    derivative = source("research_derivative_market_audit.json")

    counts = canonical.get("family_counts") or {}
    canonical_ages = age("clv_v4_postgres_report.json")
    if canonical_ages is None or canonical_ages > 30:
        critical.append("CANONICAL_CLV_REPORT_STALE_GT_30H")
    family_reports = {}

    for family, filename in REPORTS.items():
        d = source(filename)
        cls = d.get("true_clv") or {}
        reported_rows = cls.get("rows")
        report_age = age(filename)
        report_blockers = d.get("blockers") or []
        family_reports[family] = {
            "report": filename,
            "report_age_hours": report_age,
            "status": d.get("status") or "NOT_VERIFIED",
            "production_promotion_allowed": d.get("production_promotion_allowed"),
            "true_clv_rows": reported_rows,
            "true_clv_unique_fixtures": cls.get("unique_fixtures"),
            "blocker_count": len(report_blockers),
            "blockers": report_blockers[:16],
        }
        if d and d.get("production_promotion_allowed") is not False:
            critical.append("UNAUTHORIZED_PROMOTION_FLAG:"+family)
        if family in EXPECTED_CANONICAL:
            expected = counts.get(EXPECTED_CANONICAL[family])
            if isinstance(expected, int) and isinstance(reported_rows, int):
                if expected != reported_rows:
                    family_reports[family]["canonical_true_clv_rows"] = expected
                    if canonical_ages is not None and canonical_ages > 2:
                        critical.append(f"CLV_SOURCE_REPORT_COUNT_MISMATCH:{family}:{reported_rows}_VS_{expected}")
                    else:
                        warnings.append("CLV_FAMILY_REFRESH_PENDING:"+family)
            elif canonical and d:
                critical.append("CLV_FAMILY_COUNT_NOT_VERIFIABLE:"+family)
        # Unchanged scientific OOS need not generate a new git commit each day.
        # Old snapshots warrant inspection, not invented new outcomes.
        if report_age is None or report_age > 72:
            warnings.append("MARKET_REPORT_OLD_OR_UNVERIFIED:"+family)

    team = family_reports["TEAM_TOTALS"]
    expected_team = counts.get("HOME_TT", 0) + counts.get("AWAY_TT", 0)
    if isinstance(expected_team, int) and isinstance(team.get("true_clv_rows"), int):
        if expected_team != team["true_clv_rows"] and canonical_ages is not None and canonical_ages > 2:
            critical.append("TEAM_TOTALS_CLV_SOURCE_REPORT_COUNT_MISMATCH")
        team["canonical_true_clv_rows"] = expected_team

    # Research zero rows stays zero. Only lack of persistence is an operational fault.
    props_close_age = age("player_props_clv_v4_report.json")
    props_oos_age = age("player_props_oos_v4_report.json")
    if (props_oos_age is not None and props_oos_age <= 36
            and (props_close_age is None or props_close_age > 72)):
        critical.append("PLAYER_PROPS_CLV_DIAGNOSTIC_NOT_PERSISTED_GT_72H")
    if props_close and props_close.get("production_promotion_allowed") is not False:
        critical.append("PLAYER_PROPS_CLV_UNAUTHORIZED_PROMOTION")
    if props_oos and props_oos.get("production_promotion_allowed") is not False:
        critical.append("PLAYER_PROPS_OOS_UNAUTHORIZED_PROMOTION")

    actual_actionable = (settlement.get("source_diagnostics") or {}).get("actionable_classification_events")
    if actual_actionable == 0:
        settlement_state = "NO_NEW_ACTIONABLE_EVENTS_LEGITIMATE_STATIC_SETTLEMENT"
    elif isinstance(actual_actionable, int) and actual_actionable > 0:
        settlement_state = "ACTIONABLE_EVENTS_EXIST_RECONCILE_GRADING"
        if (age("market_performance_summary.json") or 9999) > 72:
            warnings.append("SETTLEMENT_SUMMARY_STALE_WITH_ACTIONABLE_EVENTS")
    else:
        settlement_state = "SETTLEMENT_ACTIONABLE_COVERAGE_NOT_VERIFIED"
        warnings.append("SETTLEMENT_POSTGRES_ACTIONABLE_COVERAGE_MISSING")

    deriv = {
        "candidate_rows": derivative.get("candidate_rows"),
        "classified_rows": derivative.get("classified_rows"),
        "unclassified_rows": derivative.get("unclassified_rows"),
        "unique_fixtures": derivative.get("unique_fixtures"),
        "provider_requests_added": derivative.get("provider_requests_added"),
    }
    if (deriv.get("classified_rows") is not None
            and deriv.get("unclassified_rows") is not None
            and deriv.get("candidate_rows") is not None
            and deriv["classified_rows"]+deriv["unclassified_rows"] != deriv["candidate_rows"]):
        critical.append("DERIVATIVE_MARKET_TAXONOMY_COUNT_INCONSISTENT")

    return {
        "schema_version": "1.0.0",
        "generated_at_utc": now.isoformat(),
        "status": "CRITICAL" if critical else ("WARNING" if warnings else "HEALTHY"),
        "read_only": True,
        "retroactive_predictions_rewritten": False,
        "strict_close_semantics_changed": False,
        "model_weights_changed": False,
        "production_promotion_allowed": False,
        "canonical_clv_age_hours": canonical_ages,
        "canonical_clv_family_counts": counts,
        "families": family_reports,
        "player_props": {
            "clv_report_age_hours": props_close_age,
            "oos_report_age_hours": props_oos_age,
            "signals_with_probabilities": props_close.get("signal_rows"),
            "strict_true_clv_rows": props_close.get("true_clv_rows"),
            "oos_unique_fixtures": props_oos.get("unique_fixtures"),
            "oos_player_game_rows": props_oos.get("player_game_rows"),
            "oos_model_weight": props_oos.get("decision_weight"),
        },
        "settlement": {
            "status": settlement_state,
            "new_pg_actionable_classifications": actual_actionable,
            "pg_settled_decisions": settlement.get("settled_decisions"),
            "commercial_settlement_decisions": market_performance.get("settlement_decisions"),
            "commercial_summary_age_hours": age("market_performance_summary.json"),
        },
        "canonical_oos": {
            "status": oos.get("status"),
            "current_model_rows": oos.get("current_model_rows"),
            "report_age_hours": age("oos_calibration_v4_report.json"),
            "historical_predictions_recomputed": (oos.get("anti_leakage") or {}).get("historical_predictions_recomputed"),
        },
        "derivative_market_capture": deriv,
        "critical": sorted(set(critical)),
        "warnings": sorted(set(warnings)),
        "interpretation": (
            "A passing operational check does not promote betting markets. "
            "No new strict close, weak calibration, sparse OOS or absent referee/XI "
            "history are scientific/data-availability holds, not automatically broken workflows."
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", type=Path, required=True)
    ap.add_argument("--commits", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--now-utc", default="")
    args = ap.parse_args()
    now = parse_dt(args.now_utc) if args.now_utc else datetime.now(timezone.utc)
    if now is None:
        ap.error("invalid --now-utc")
    commits = json.loads(args.commits.read_text(encoding="utf-8"))
    r = audit(args.state_dir, commits, now)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(r, sort_keys=True, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status":r["status"],"critical":r["critical"],"warnings":r["warnings"]},indent=2))
    return 2 if r["critical"] else 0


if __name__ == "__main__":
    sys.exit(main())
