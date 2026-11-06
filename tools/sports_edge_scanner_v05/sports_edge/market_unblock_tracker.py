"""Ordered, read-only Soccer market research audit for Bet365 exclusively.

Legacy canonical CLV is NOT assumed to be Bet365-specific without explicit
bookmaker_id=8, bookmaker=Bet365 and source verification.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from .market_registry import FAMILY_FILES

VERSION = "SOCCER_BET365_MARKET_UNBLOCK_V1.0"
PHASES = [
    ("1X2", "A_REVIEW"),
    ("TEAM_TOTALS", "A_REVIEW"),
    ("BTTS", "B_CLV"),
    ("FT_TOTALS", "B_CLV"),
    ("FIRST_HALF", "C_MODEL"),
    ("SECOND_HALF", "C_MODEL"),
    ("FT_CORNERS", "C_MODEL"),
    ("TEAM_CORNERS", "C_MODEL"),
    ("CARDS", "D_DATA"),
    ("PLAYER_PROPS", "D_DATA"),
    ("FOULS", "D_DATA"),
    ("FORMATION_STYLE", "D_DATA"),
]
ACTIONS = {
    "1X2": "Review multiclass walk-forward vs baseline; validate Bet365-only close and per-fixture ROI; no automatic promotion.",
    "TEAM_TOTALS": "Audit OOS by home/away, line and Over/Under; investigate negative aggregate hypothetical ROI; verify Bet365-only close.",
    "BTTS": "Test experimental OOS calibration across leagues and buckets; collect real Bet365 matched entry/close snapshots.",
    "FT_TOTALS": "Obtain Bet365 closes for exact FT lines, including currently unobserved Asian thresholds, then repeat validation.",
    "FIRST_HALF": "Train a dedicated 1H challenger that beats baseline on Brier and log-loss before any BET evaluation.",
    "SECOND_HALF": "Train 2H challenger with independently verified halftime state, beating baseline on Brier/log-loss/MAE.",
    "FT_CORNERS": "Increase formation-adjusted examples and league stability; separately store Bet365 exact corners closes.",
    "TEAM_CORNERS": "Validate league/venue segments and parent corner model; do not inherit FT corners calibration.",
    "CARDS": "Connect verified referee assignments and settlement rules; validate cards OOS and Bet365 close history.",
    "PLAYER_PROPS": "Build player/XI/minutes-aligned OOS ledger per prop family and exact Bet365 line history.",
    "FOULS": "Create independent count model and walk-forward ledger after confirming coverage and Bet365 market availability.",
    "FORMATION_STYLE": "Increase comparable prior XI, coach and style samples and validate ablation out-of-sample.",
}


def _read(source: Path) -> dict[str, Any]:
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError, UnicodeError):
        return {}


def _get(row: dict[str, Any], *keys: str) -> Any:
    value: Any = row
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _sample(report: dict[str, Any], family: str) -> tuple[int | None, str]:
    lookup = {
        "1X2": (("canonical_multiclass_oos", "evaluated_rows"), "multiclass_oos_rows"),
        "TEAM_TOTALS": (("oos_sample", "evaluated_fixtures"), "oos_fixtures"),
        "BTTS": (("canonical_oos_calibration", "rows"), "calibration_oos_rows"),
        "FT_TOTALS": (("sample", "model_settled"), "settled_NOT_independent_oos"),
        "FIRST_HALF": (("calibration", "n"), "oos_rows"),
        "SECOND_HALF": (("oos_sample", "walk_forward_evaluated"), "oos_rows"),
        "FT_CORNERS": (("ft_corners", "walk_forward_evaluations"), "oos_evaluations"),
        "TEAM_CORNERS": (("team_corners", "evaluated_fixtures"), "oos_fixtures"),
        "CARDS": (("yellow_cards", "oos_n"), "yellow_card_oos_rows"),
    }
    if family not in lookup:
        return None, "not_verified"
    keys, unit = lookup[family]
    n = _get(report, *keys)
    return (n if isinstance(n, int) and not isinstance(n, bool) and n >= 0 else None), unit


def _baseline(report: dict[str, Any], family: str) -> dict[str, Any]:
    if family == "1X2":
        base = _get(report, "canonical_multiclass_oos", "baseline") or {}
        candidate = _get(report, "canonical_multiclass_oos", "temperature_scaled") or {}
        brier_key, log_key = "multiclass_brier", "multiclass_log_loss"
    elif family == "FIRST_HALF":
        base = _get(report, "calibration", "baseline") or {}
        candidate = _get(report, "calibration", "challenger") or {}
        brier_key, log_key = "brier", "log_loss"
    elif family == "SECOND_HALF":
        base, candidate = report.get("baseline") or {}, report.get("challenger") or {}
        brier_key, log_key = "brier_o1_5", "log_loss_o1_5"
    else:
        return {"status": "NOT_VERIFIED"}
    if not isinstance(base, dict) or not isinstance(candidate, dict):
        return {"status": "NOT_VERIFIED"}
    b0, b1 = base.get(brier_key), candidate.get(brier_key)
    l0, l1 = base.get(log_key), candidate.get(log_key)
    if any(not isinstance(x, (int, float)) or isinstance(x, bool) for x in (b0, b1, l0, l1)):
        return {"status": "NOT_VERIFIED"}
    return {"status": "BOTH_IMPROVED" if b1 < b0 and l1 < l0 else "BASELINE_NOT_BEATEN",
            "baseline_brier": b0, "challenger_brier": b1,
            "baseline_log_loss": l0, "challenger_log_loss": l1,
            "brier_improvement": round(b0 - b1, 8),
            "log_loss_improvement": round(l0 - l1, 8)}


def _clv(report: dict[str, Any]) -> dict[str, Any]:
    legacy = report.get("true_clv")
    if not isinstance(legacy, dict):
        return {"historical_rows_unattributed": None, "bet365_verified_rows": None,
                "provenance": "NOT_VERIFIED"}
    n = legacy.get("rows")
    n = n if isinstance(n, int) and not isinstance(n, bool) and n >= 0 else None
    verified = (legacy.get("bookmaker_id") == 8
                and str(legacy.get("bookmaker") or "").lower() == "bet365"
                and legacy.get("source_verified") is True)
    return {"historical_rows_unattributed": n,
            "bet365_verified_rows": n if verified else None,
            "provenance": "VERIFIED_BET365_ID8" if verified else "NOT_VERIFIED",
            "historical_minimum_reported": legacy.get("minimum_rows")}



def _release_blockers(report: dict[str, Any], family: str, status: str,
                      clv: dict[str, Any]) -> list[str]:
    """Explicit, non-authorizing checklist: canonical research != approved BET.

    Provenance requires actual Bet365-ID8 matched-source evidence. Canonical
    CLV rows, even when numerous, are not Bet365-specific by themselves.
    Hypothetical ROI diagnostics are risk flags, never estimates of model EV.
    """
    blockers = []
    if not report:
        return ["CANONICAL_VALIDATION_REPORT_MISSING"]
    if report.get("production_promotion_allowed") is not True:
        blockers.append("CANONICAL_PRODUCTION_APPROVAL_NOT_GRANTED")
    if report.get("manual_review_required") is not False:
        blockers.append("MANUAL_APPROVAL_REQUIRED")
    if clv.get("provenance") != "VERIFIED_BET365_ID8":
        blockers.append("BET365_ID8_ENTRY_CLOSE_PROVENANCE_NOT_VERIFIED")
    if status not in ("CALIBRATION_REVIEW_ELIGIBLE", "OOS_REVIEW_ELIGIBLE"):
        blockers.append("CANONICAL_MARKET_STILL_RESEARCH_HOLD")
    for warning in report.get("warnings") or []:
        if isinstance(warning, str) and warning:
            blockers.append("CANONICAL_REVIEW_WARNING:" + warning)
    if family == "1X2":
        outcome = _get(report, "canonical_clv_outcome_context",
                       "fixture_equal_weight_roi",
                       "mean_hypothetical_roi_units_per_fixture")
        if isinstance(outcome, (int, float)) and not isinstance(outcome, bool) and outcome < 0:
            blockers.append("NEGATIVE_CANONICAL_HYPOTHETICAL_FIXTURE_ROI_REVIEW")
    if family == "TEAM_TOTALS":
        by_line = _get(report, "canonical_clv_outcome_context", "by_line")
        if isinstance(by_line, dict):
            for line, data in sorted(by_line.items()):
                if not isinstance(data, dict):
                    continue
                value = data.get("hypothetical_roi_per_priced_settled_observation")
                if isinstance(value, (int, float)) and not isinstance(value, bool) and value < 0:
                    blockers.append("NEGATIVE_CANONICAL_HYPOTHETICAL_ROI_LINE_" +
                                    str(line).replace(".", "_"))
    return sorted(set(blockers))

def build_market_unblock_report(source_dir: str | Path, *,
                                generated_at: str | None = None) -> dict[str, Any]:
    """Triage all twelve market families in declared order, zero external calls."""
    source = Path(source_dir)
    lookup = {family: filename for family, _, filename in FAMILY_FILES}
    reports = {fn: _read(source / fn) for fn in set(x for x in lookup.values() if x)}
    rows = []
    for index, (family, phase) in enumerate(PHASES, 1):
        filename = lookup[family]
        report = reports.get(filename) or {}
        status = str(report.get("status") or "NOT_VERIFIED")
        blockers = report.get("blockers")
        if not isinstance(blockers, list):
            blockers = _get(report, "health", "blockers")
        blockers = [str(x) for x in blockers] if isinstance(blockers, list) else []
        if not report:
            blockers.append("CANONICAL_MARKET_EVIDENCE_MISSING")
        n, unit = _sample(report, family)
        comp, clv = _baseline(report, family), _clv(report)
        if comp.get("status") == "BASELINE_NOT_BEATEN":
            stage = "MODEL_REBUILD_REQUIRED"
        elif status in ("CALIBRATION_REVIEW_ELIGIBLE", "OOS_REVIEW_ELIGIBLE") and not blockers:
            stage = "MANUAL_REVIEW_ELIGIBLE"
        elif report:
            stage = "RESEARCH_HOLD"
        else:
            stage = "NOT_VERIFIED"
        rows.append({
            "priority": index, "phase": phase, "market_family": family,
            "source_file": filename, "source_model_version": report.get("model_version"),
            "source_status": status, "source_available": bool(report),
            "report_timestamp": report.get("generated_at_utc") or report.get("generated_at"),
            "report_freshness": "NOT_VERIFIED",
            "sample_n": n, "sample_unit": unit,
            "baseline_comparison": comp, "clv": clv, "blockers": blockers,
            "research_decision": stage, "release_gate": "NOT_APPROVED",
            "release_blockers": _release_blockers(report, family, status, clv),
            "production_promotion_allowed": False,
            "bet_eligible": False, "next_action": ACTIONS[family],
        })
    return {
        "schema_version": VERSION,
        "generated_at_utc": generated_at or datetime.now(timezone.utc).isoformat(),
        "sport": "soccer", "bookmaker": "Bet365", "bookmaker_id": 8,
        "no_other_bookmaker_required": True,
        "historical_reports_are_not_live_confirmation": True,
        "bet365_only_clv_coverage_verified": False,
        "bet_eligible_families": 0, "families_audited": len(rows),
        "manual_review_candidates": [r["market_family"] for r in rows if r["research_decision"] == "MANUAL_REVIEW_ELIGIBLE"],
        "model_rebuild_candidates": [r["market_family"] for r in rows if r["research_decision"] == "MODEL_REBUILD_REQUIRED"],
        "families": rows,
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Research digest; no automatic promotion or BET."""
    lines = [
        "# Bet365-only Soccer market research audit",
        "",
        "Canonical CLV counts are historical and NOT Bet365-only unless provenance is explicit.",
        "Research eligibility is NOT production approval.",
        "",
        "| Order | Market | Canonical status | Sample N | CLV historical (book unverified) | CLV Bet365 verified | Stage |",
        "|---:|---|---|---:|---:|---|---|",
    ]
    for row in report["families"]:
        clv = row["clv"]
        historical = clv["historical_rows_unattributed"]
        bet365 = clv["bet365_verified_rows"]
        lines.append(
            f"| {row['priority']} | {row['market_family']} | {row['source_status']} "
            f"| {row['sample_n'] if row['sample_n'] is not None else 'N/V'} "
            f"| {historical if historical is not None else 'N/V'} "
            f"| {bet365 if bet365 is not None else 'N/V'} "
            f"| {row['research_decision']} |"
        )
    lines.extend(["", "## Release gate: NOT APPROVED (all markets)", "",
                  "Research-review readiness never enables BET. Historical canonical "
                  "CLV counts do not verify Bet365-ID8 entry/close provenance.", ""])
    for row in report["families"]:
        if row["market_family"] in ("1X2", "TEAM_TOTALS"):
            reasons = ", ".join(row["release_blockers"]) or "NOT_VERIFIED"
            lines.append(f"- **{row['market_family']}**: {reasons}")
    lines.extend(["", "## Next actions, in order", ""])
    for row in report["families"]:
        lines.append(f"{row['priority']}. **{row['market_family']}**: {row['next_action']}")
    lines.extend(["", "No BET or staking authorized by this audit.", ""])
    return "\n".join(lines)
