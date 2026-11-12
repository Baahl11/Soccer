"""Read-only audit: configured scanner leagues vs an already-captured official slate.

This does NOT grant research/model/market eligibility, request provider data or
alter the fixture model. A source's two-date window is NOT a scanner one-date
fixture count and must never be presented as such.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any

VERSION = "SOCCER_SCANNER_SOURCE_SCOPE_RECONCILIATION_v1.0"


def _utc(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return result.astimezone(timezone.utc) if result.tzinfo is not None else None


def audit(receipt: Any, league_ids: Any, *, now: datetime) -> dict[str, Any]:
    base = {
        "version": VERSION,
        "status": "NOT_VERIFIED",
        "source_scope": "PROSPECTIVE_TWO_DATE_WINDOW_NOT_SCANNER_SINGLE_DAY",
        "total_source_eligible": None,
        "in_configured_leagues": None,
        "outside_configured_leagues": None,
        "uncovered_competitions": [],
        "model_eligibility_promoted": False,
        "market_inputs_used": False,
        "bet_eligible": False,
        "provider_requests_added": 0,
    }
    if not isinstance(receipt, dict) or not isinstance(league_ids, str):
        return {**base, "reason": "SOURCE_OR_SCOPE_NOT_VERIFIED"}
    parts = league_ids.split(",")
    if not 1 <= len(parts) <= 30 or any(not p.isdecimal() or not 0 < int(p) < 100000 for p in parts):
        return {**base, "reason": "INVALID_CONFIGURED_LEAGUES"}
    allowed = set(map(int, parts))
    generated = _utc(receipt.get("generated_at_utc"))
    if (
        receipt.get("status") != "CAPTURED_RESEARCH_ONLY" or generated is None
        or now.tzinfo is None or not -timedelta(minutes=5) <= now-generated <= timedelta(hours=6)
    ):
        return {**base, "reason": "OFFICIAL_SPORT_RECEIPT_MISSING_OR_STALE"}
    rows = receipt.get("league_coverage_report")
    if (
        not isinstance(rows, list) or receipt.get("league_coverage_report_truncated") is not False
        or type(receipt.get("eligible_fixture_ids")) is not int
        or type(receipt.get("league_coverage_report_total_leagues")) is not int
        or len(rows) != receipt["league_coverage_report_total_leagues"]
    ):
        return {**base, "reason": "INCOMPLETE_SOURCE_LEAGUE_BREAKDOWN"}
    seen: set[int] = set()
    counted: list[tuple[int, str, int]] = []
    for row in rows:
        if not isinstance(row, dict):
            return {**base, "reason": "INVALID_SOURCE_COMPETITION"}
        lid, n, name = row.get("competition_id"), row.get("eligible"), row.get("league")
        if (
            type(lid) is not int or lid <= 0 or lid in seen or type(n) is not int or n < 0
            or not isinstance(name, str) or not name.strip()
        ):
            return {**base, "reason": "INVALID_SOURCE_COMPETITION"}
        seen.add(lid)
        counted.append((lid, name[:90], n))
    total = sum(n for _, _, n in counted)
    if total != receipt["eligible_fixture_ids"]:
        return {**base, "reason": "SOURCE_SCOPE_COUNTS_NOT_RECONCILED"}
    matching = sum(n for lid, _, n in counted if lid in allowed)
    missing = sorted(
        ((lid, name, n) for lid, name, n in counted if lid not in allowed and n > 0),
        key=lambda row: (-row[2], row[0]),
    )
    return {
        **base,
        "status": "SCOPE_OVERLAP" if matching else "SCOPE_ZERO_OVERLAP",
        "reason": (
            "VERIFIED_RESEARCH_SCOPE_INTERSECTION"
            if matching else "NO_CURRENT_SOURCE_FIXTURES_IN_CONFIGURED_COMPETITIONS"
        ),
        "source_generated_at_utc": generated.isoformat(),
        "configured_league_count": len(allowed),
        "source_competitions": len(rows),
        "total_source_eligible": total,
        "in_configured_leagues": matching,
        "outside_configured_leagues": total-matching,
        "uncovered_competitions": [
            {"league_id": lid, "league": name, "eligible": n}
            for lid, name, n in missing[:8]
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--leagues", required=True)
    args = parser.parse_args()
    try:
        receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        receipt = {}
    result = audit(receipt, args.leagues, now=datetime.now(timezone.utc))
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    # Scope diagnostics are advisory; not-verified evidence never promotes
    # predictions or warrants bypassing the normal research validation gates.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
