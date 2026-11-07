"""Read-only FE11 universal baseline/outcome count integrity.

Input is an authenticated, protected Render OIDC aggregate JSON receipt.
No external requests, no SQL writes and no betting/market actions here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

VERSION = "SOCCER_FE11_LIFECYCLE_INVENTORY_GUARD_v1.0"
# First independently verified immutable baseline from Nov 6 23:54 CDMX:
# 23 actual future approved-league fixtures in PostgreSQL, not test fixtures.
VERIFIED_REAL_BASELINE_FLOOR = 23
VERIFIED_REAL_HISTORY_FLOOR = 3
COUNT_KEYS = (
    "eligible_pregame_baselines",
    "matched_official_final_fixtures",
    "current_future_baselines",
    "official_period_complete_fixtures",
    "official_ft_corners_verified_fixtures",
    "official_period_and_corners_complete_fixtures",
    "joint_goals_corners_handicap_training_rows",
)


def verify(bridge: Any) -> dict[str, Any]:
    blockers: list[str] = []
    if not isinstance(bridge, dict):
        bridge = {}
        blockers.append("BRIDGE_NOT_VERIFIED")
    original = bridge.get("fe11_history_inventory")
    universe = bridge.get("fe11_universal_inventory")
    if not isinstance(original, dict):
        original = {}
        blockers.append("HISTORICAL_PROFILE_INVENTORY_NOT_VERIFIED")
    if not isinstance(universe, dict):
        universe = {}
        blockers.append("UNIVERSAL_LIFECYCLE_INVENTORY_NOT_VERIFIED")
    if original.get("status") != "VERIFIED_READ":
        blockers.append("HISTORICAL_PROFILE_DB_READ_FAILED")
    if universe.get("status") != "VERIFIED_READ":
        blockers.append("UNIVERSAL_DB_READ_FAILED")

    preexisting = original.get("research_rows_persisted")
    if type(preexisting) is not int or preexisting < VERIFIED_REAL_HISTORY_FLOOR:
        blockers.append("HISTORICAL_PROFILE_REGRESSION")
    if original.get("production_bets_authorized") is not False:
        blockers.append("HISTORICAL_BET_AUTH_NOT_DISABLED")
    if original.get("provider_requests_added") != 0:
        blockers.append("HISTORICAL_AUDIT_PROVIDER_CALLS_NOT_ZERO")

    counts: dict[str, int | None] = {}
    for key in COUNT_KEYS:
        item = universe.get(key)
        counts[key] = item if type(item) is int and item >= 0 else None
        if counts[key] is None:
            blockers.append("INVALID_AGGREGATE_" + key.upper())
    pre = counts["eligible_pregame_baselines"]
    final = counts["matched_official_final_fixtures"]
    future = counts["current_future_baselines"]
    periods = counts["official_period_complete_fixtures"]
    corners = counts["official_ft_corners_verified_fixtures"]
    complete = counts["official_period_and_corners_complete_fixtures"]
    if pre is not None and pre < VERIFIED_REAL_BASELINE_FLOOR:
        blockers.append("VERIFIED_PREMATCH_BASELINE_REGRESSION")
    for name, child, parent in (
        ("FUTURE_EXCEEDS_CAPTURED", future, pre),
        ("RESULTS_EXCEED_PREMATCH", final, pre),
        ("PERIODS_EXCEED_RESULTS", periods, final),
        ("CORNERS_EXCEED_RESULTS", corners, final),
        ("PERIOD_CORNERS_EXCEED_PERIODS", complete, periods),
        ("PERIOD_CORNERS_EXCEED_CORNERS", complete, corners),
    ):
        if child is not None and parent is not None and child > parent:
            blockers.append(name)
    if universe.get("production_bets_authorized") is not False:
        blockers.append("UNIVERSAL_BET_AUTH_NOT_DISABLED")
    if universe.get("provider_requests_added") != 0:
        blockers.append("UNIVERSAL_AUDIT_PROVIDER_CALLS_NOT_ZERO")
    linked = final if isinstance(final, int) else None
    future_count = future if isinstance(future, int) else None
    no_final = (
        max(0, pre - future_count - linked)
        if isinstance(pre, int) and future_count is not None
        and linked is not None else None
    )
    # NOT a settlement failure: postponed, suspended, still active and
    # source-unreported matches can all appear in nonfuture/no-final.
    return {
        "version": VERSION,
        "status": "VERIFIED_AGGREGATE_CONSISTENCY" if not blockers else "NOT_VERIFIED",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "historical_profiles_retained": preexisting if type(preexisting) is int else None,
        **counts,
        "nonfuture_without_matched_final_count": no_final,
        "nonfuture_warning": "NOT_A_VERIFIED_FAILURE_INCLUDES_LIVE_AND_POSTPONED",
        "closure_state": (
            "OFFICIAL_POSTGAME_LINKAGE_OBSERVED" if linked is not None and linked > 0
            else "NO_MATCHED_OFFICIAL_FINAL_AS_OF_AUDIT"
        ),
        "blockers": blockers,
        "provider_requests_added": 0,
        "database_writes_added": 0,
        "bet_eligible": False,
        "model_weights_changed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = json.loads(args.bridge.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        payload = {}
    result = verify(payload)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "VERIFIED_AGGREGATE_CONSISTENCY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
