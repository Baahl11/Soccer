"""Budgeted, result-only FT corners for eligible FE11 frozen pregame fixtures.

No odds, probabilities, market tickets, synthetic zeros or bet decisions.
A separate queue rechecks official finished fixture_statistics for matches
whose prior sport-first identity and final score were captured. A capped
attempt ledger prevents repeated same-fixture quota burn.
"""
from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Any

from mcp_gateway import automation_v2 as v2

VERSION = "SOCCER_FE11_POSTGAME_CORNERS_v1.0"
SOURCE = "API_FOOTBALL_FIXTURE_STATISTICS"
MAX_FIXTURES = max(0, min(10, int(os.getenv("SOCCER_FE11_POSTGAME_CORNERS_PER_TICK", "6"))))

PENDING_SQL = """
    WITH final AS (
        SELECT DISTINCT ON (fixture_id) fixture_id, final_status
        FROM soccer_fe11_official_outcome_snapshots
        ORDER BY fixture_id, observed_at DESC
    )
    SELECT b.fixture_id, b.home_team_id, b.away_team_id, b.kickoff
    FROM soccer_fe11_pregame_baselines b
    JOIN final o ON o.fixture_id = b.fixture_id AND o.final_status = 'FT'
    LEFT JOIN soccer_fe11_verified_corner_snapshots c ON c.fixture_id = b.fixture_id
    LEFT JOIN soccer_fe11_corner_capture_attempts a ON a.fixture_id = b.fixture_id
    WHERE c.fixture_id IS NULL
      AND b.kickoff <= NOW() AND b.kickoff > NOW() - interval '30 days'
      AND (a.attempt_count IS NULL OR a.attempt_count < 5)
      AND (a.last_attempt_at IS NULL OR a.last_attempt_at <= NOW() - interval '12 hours')
    ORDER BY b.kickoff ASC, b.fixture_id ASC
    LIMIT %s
"""


def _corner_value(team: dict) -> int | None:
    rows = team.get("statistics")
    if not isinstance(rows, list):
        return None
    candidates = [r.get("value") for r in rows
                  if isinstance(r, dict) and str(r.get("type") or "").strip().lower() == "corner kicks"]
    if len(candidates) != 1:
        return None
    v = candidates[0]
    if type(v) is int:
        result = v
    elif isinstance(v, str) and v.strip().isdigit():
        result = int(v.strip())
    else:
        return None
    return result if 0 <= result <= 60 else None


def parse_verified_ft_corners(raw: Any, home_id: int, away_id: int) -> dict | None:
    """Requires exactly two distinct participating team IDs and two real data points."""
    if (not isinstance(raw, list) or len(raw) != 2
            or type(home_id) is not int or type(away_id) is not int
            or home_id <= 0 or away_id <= 0 or home_id == away_id):
        return None
    found: dict[int, int] = {}
    for item in raw:
        if not isinstance(item, dict):
            return None
        team = item.get("team")
        team_id = team.get("id") if isinstance(team, dict) else None
        if type(team_id) is not int or team_id not in (home_id, away_id) or team_id in found:
            return None
        corners = _corner_value(item)
        if corners is None:
            return None
        found[team_id] = corners
    return {"home": found[home_id], "away": found[away_id]} if len(found) == 2 else None


def pending(limit: int) -> list[dict]:
    from mcp_gateway import persistence
    if limit <= 0 or not persistence.persistence_configured():
        return []
    persistence.ensure_schema()
    with persistence._connect() as conn:
        with conn.cursor() as cur:
            cur.execute(PENDING_SQL, (limit,))
            return [
                {"fixture_id": row[0], "home_team_id": row[1],
                 "away_team_id": row[2], "kickoff": row[3].isoformat()}
                for row in cur.fetchall()
            ]


async def collect(tick: dict[str, Any]) -> None:
    """Append compact queue audit and verified observations to current tick.

    Invoked after all existing sport-first and bookmaker decisions are complete.
    Provider budget can only tighten/reduce selection. Missing corners are
    NEVER treated as zero, and no historical prediction is changed.
    """
    tick["fe11_postgame_corner_observations"] = []
    tick["fe11_postgame_corner_attempts"] = []
    tick["fe11_postgame_corner_summary"] = {
        "version": VERSION, "status": "NOT_ATTEMPTED",
        "selected": 0, "captured": 0, "provider_requests_added": 0,
        "bet_eligible": False, "market_inputs_used": False,
    }
    budget_left = int(v2.MAX_API_CALLS_PER_TICK) - int(v2._API_CALLS_THIS_TICK)
    if MAX_FIXTURES <= 0:
        tick["fe11_postgame_corner_summary"]["status"] = "DISABLED"
        return
    if budget_left < 4:
        tick["fe11_postgame_corner_summary"]["status"] = "DEFERRED_TICK_BUDGET"
        return
    if v2._LAST_DAILY_REMAINING is None or v2._LAST_DAILY_REMAINING <= 125:
        tick["fe11_postgame_corner_summary"]["status"] = "DEFERRED_DAILY_QUOTA"
        return
    limit = min(MAX_FIXTURES, max(0, budget_left - 2))
    try:
        rows = pending(limit)
    except Exception:
        tick["fe11_postgame_corner_summary"]["status"] = "NOT_VERIFIED_DB_QUERY_FAILED"
        return

    tick["fe11_postgame_corner_summary"]["selected"] = len(rows)
    if not rows:
        tick["fe11_postgame_corner_summary"]["status"] = "NO_ELIGIBLE_PENDING_FT_CORNERS"
        return
    before = int(v2._API_CALLS_THIS_TICK)
    for fx in rows:
        if (int(v2._API_CALLS_THIS_TICK) >= int(v2.MAX_API_CALLS_PER_TICK) - 2
                or v2._LAST_DAILY_REMAINING is None or v2._LAST_DAILY_REMAINING <= 125):
            break
        fid = fx["fixture_id"]
        result = None
        status = "NOT_VERIFIED_PROVIDER_STATS"
        try:
            response = await v2._budgeted_api_get("fixtures/statistics", {"fixture": fid})
            result = parse_verified_ft_corners(
                response.get("response"), fx["home_team_id"], fx["away_team_id"],
            )
            if result is not None:
                status = "VERIFIED_FT_CORNERS"
        except Exception:
            status = "NOT_VERIFIED_PROVIDER_ERROR"
        observed = datetime.now(timezone.utc).isoformat()
        tick["fe11_postgame_corner_attempts"].append({
            "fixture_id": fid,
            "kickoff": fx["kickoff"],
            "observed_at_utc": observed,
            "status": status,
            "provider_requests_added": 1,
        })
        if result is not None:
            tick["fe11_postgame_corner_observations"].append({
                "fixture_id": fid, "kickoff": fx["kickoff"],
                "home_team_id": fx["home_team_id"],
                "away_team_id": fx["away_team_id"],
                "observed_at_utc": observed,
                "home_corners": result["home"],
                "away_corners": result["away"],
                "period": "FT_90",
                "source": SOURCE,
                "source_response_verified": True,
                "bet_eligible": False,
            })
    tick["fe11_postgame_corner_summary"].update({
        "status": "VERIFIED_CAPTURES" if tick["fe11_postgame_corner_observations"]
                  else "NOT_VERIFIED_NO_COMPLETE_PROVIDER_STATS",
        "captured": len(tick["fe11_postgame_corner_observations"]),
        "attempted": len(tick["fe11_postgame_corner_attempts"]),
        "provider_requests_added": max(0, int(v2._API_CALLS_THIS_TICK) - before),
    })
