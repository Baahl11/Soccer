"""FE11 all-eligible official-slate capture: no bookmaker or per-fixture API calls.

One provider fixtures/date response can freeze a baseline for EVERY valid
approved-league NS fixture. When it later finishes, the SAME official date
slate supplies score and 1H/2H labels; missing fields remain NULL. A score
is not proof of completed corners or a calibrated multi-leg market.

The runtime must persist ONLY rows assembled from a fresh official slate,
never reconstruct pregame baselines for completed fixtures.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from mcp_gateway import fe11_registry_next_day_v1 as registry

VERSION = "SOCCER_FE11_ALL_ELIGIBLE_LIFECYCLE_v1.0"
APPROVED_LEAGUES = registry.SCANNER_APPROVED_LEAGUES
CANONICAL_FIELDS = registry.PERSISTENCE_FIELDS
TERMINAL_STATUSES = frozenset({"FT", "AET", "PEN"})
MAX_ELIGIBLE_PER_TICK = 3000
SOURCE = "API_FOOTBALL_FIXTURES_DATE"
PREMATCH_SOURCE_STATUS = "NS"
NO_BET = False


def _utc(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc) if dt.tzinfo is not None else None
    except ValueError:
        return None


def _positive(value: Any) -> bool:
    return type(value) is int and 0 < value < 10**12


def _goal_pair(value: Any) -> dict[str, int] | None:
    if not isinstance(value, dict):
        return None
    home, away = value.get("home"), value.get("away")
    if type(home) is not int or type(away) is not int or not (0 <= home <= 50 and 0 <= away <= 50):
        return None
    return {"home": home, "away": away}


def _identity(row: Any) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    fid, league, home, away = (
        row.get("fixture_id"), row.get("league_id"),
        row.get("home_team_id"), row.get("away_team_id"),
    )
    kickoff = _utc(row.get("kickoff"))
    if (not _positive(fid) or not _positive(league)
            or not _positive(home) or not _positive(away) or home == away
            or league not in APPROVED_LEAGUES
            or type(row.get("season")) is not int
            or not 2000 <= row["season"] <= 2100
            or kickoff is None
            or not isinstance(row.get("home_team"), str)
            or not row["home_team"].strip()
            or not isinstance(row.get("away_team"), str)
            or not row["away_team"].strip()):
        return None
    obj = {key: row.get(key) for key in CANONICAL_FIELDS}
    obj["kickoff"] = kickoff.isoformat()
    return obj


def _outcome(row: dict[str, Any], identity: dict[str, Any]) -> dict[str, Any] | None:
    final = _goal_pair(row.get("goals"))
    if final is None:
        return None
    score = row.get("score") if isinstance(row.get("score"), dict) else {}
    halftime = _goal_pair(score.get("halftime"))
    regulation_final = _goal_pair(score.get("fulltime"))
    # The aggregate score can include extra time or penalties. Only derive
    # regulation 2H from separately reported 90-minute fulltime and HT.
    second_half = None
    if halftime and regulation_final:
        h = regulation_final["home"] - halftime["home"]
        a = regulation_final["away"] - halftime["away"]
        if h >= 0 and a >= 0:
            second_half = {"home": h, "away": a}
    return {
        "fixture_id": identity["fixture_id"],
        "kickoff": identity["kickoff"],
        "home_team_id": identity["home_team_id"],
        "away_team_id": identity["away_team_id"],
        "status": row["status"],
        "goals": final,
        "halftime": halftime,
        "regulation_fulltime": regulation_final,
        "second_half": second_half,
        "corners_ft": None,
        "corners_verified": False,
        "market_inputs_used": False,
        "bet_eligible": False,
        "status_source": SOURCE,
    }


def build_capture(
    fixture_rows: Any, *, now_utc: datetime,
    next_day_roster: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a complete, bounded all-eligible capture with source audit.

    next_day_roster must be revalidated at its independent authoritative
    boundary. It can only extend the pregame date coverage, not add results.
    Fail closed if duplicate fixture IDs conflict or scope is oversized.
    """
    if not isinstance(fixture_rows, list) or now_utc.tzinfo is None:
        raise ValueError("FE11_OFFICIAL_SLATE_OR_TIMESTAMP_INVALID")
    now = now_utc.astimezone(timezone.utc)
    official = list(fixture_rows)
    if next_day_roster is not None:
        event = registry.discovery_event(next_day_roster)
        if event is not None:
            official.extend(event["fixtures"])
    valid: dict[int, dict[str, Any]] = {}
    invalid_identities = 0
    eligible_source_count = 0
    duplicate_count = 0
    for row in official:
        if not isinstance(row, dict) or row.get("league_id") not in APPROVED_LEAGUES:
            continue
        eligible_source_count += 1
        identity = _identity(row)
        if identity is None:
            invalid_identities += 1
            continue
        fid = identity["fixture_id"]
        if fid in valid:
            previous = valid[fid]
            if (previous["identity"] != identity or previous["status"] != row.get("status")
                    or previous["goals"] != _goal_pair(row.get("goals"))):
                return _blocked(
                    now, "NOT_VERIFIED_SOURCE_IDENTITY_CONFLICT",
                    eligible_source_count, invalid_identities,
                )
            duplicate_count += 1
            continue
        valid[fid] = {
            "identity": identity, "status": row.get("status"),
            "goals": _goal_pair(row.get("goals")), "raw": row,
        }
        if len(valid) > MAX_ELIGIBLE_PER_TICK:
            return _blocked(
                now, "NOT_VERIFIED_SOURCE_TOO_LARGE",
                eligible_source_count, invalid_identities,
            )
    pregame: list[dict[str, Any]] = []
    finished: list[dict[str, Any]] = []
    discarded_status = 0
    final_score_missing = 0
    for value in valid.values():
        identity, status, raw = value["identity"], value["status"], value["raw"]
        kickoff = _utc(identity["kickoff"])
        if status == PREMATCH_SOURCE_STATUS and kickoff > now:
            pregame.append(identity)
        elif status in TERMINAL_STATUSES and kickoff <= now:
            outcome = _outcome(raw, identity)
            if outcome:
                finished.append(outcome)
            else:
                final_score_missing += 1
        else:
            discarded_status += 1
    pregame.sort(key=lambda x: (x["kickoff"], x["fixture_id"]))
    finished.sort(key=lambda x: (x["kickoff"], x["fixture_id"]))
    return {
        "version": VERSION,
        "status": "VERIFIED_OFFICIAL_ALL_ELIGIBLE_CAPTURE" if invalid_identities == 0
                  else "PARTIAL_IDENTITY_NOT_VERIFIED",
        "source": SOURCE,
        "source_observed_at_utc": now.isoformat(),
        "approved_league_ids": sorted(APPROVED_LEAGUES),
        "eligible_source_count": eligible_source_count,
        "unique_eligible_verified": len(valid),
        "pregame_count": len(pregame),
        "official_final_count": len(finished),
        "invalid_identities": invalid_identities,
        "duplicate_identical": duplicate_count,
        "excluded_not_pregame_or_final": discarded_status,
        "final_score_missing": final_score_missing,
        "pregame_fixtures": pregame,
        "official_final_fixtures": finished,
        "market_inputs_used": False,
        "bet_eligible": False,
        "probabilities_calibrated": False,
        "provider_requests_added": 0,
    }


def _blocked(now: datetime, status: str, eligible: int, invalid: int) -> dict[str, Any]:
    return {
        "version": VERSION, "status": status,
        "source": SOURCE, "source_observed_at_utc": now.isoformat(),
        "approved_league_ids": sorted(APPROVED_LEAGUES),
        "eligible_source_count": eligible, "unique_eligible_verified": 0,
        "pregame_count": 0, "official_final_count": 0,
        "invalid_identities": invalid, "pregame_fixtures": [],
        "official_final_fixtures": [], "market_inputs_used": False,
        "bet_eligible": False, "probabilities_calibrated": False,
        "provider_requests_added": 0,
    }


def safe_tick_capture(value: Any, *, now_utc: str) -> tuple[list[dict], list[dict]]:
    """Second validation at persistence boundary; no opaque caller-supplied rows."""
    if not isinstance(value, dict) or value.get("version") != VERSION:
        return [], []
    observed = _utc(value.get("source_observed_at_utc"))
    expected = _utc(now_utc)
    if (observed is None or expected is None or observed != expected
            or value.get("source") != SOURCE or value.get("market_inputs_used") is not False
            or value.get("bet_eligible") is not False
            or value.get("probabilities_calibrated") is not False
            or value.get("provider_requests_added") != 0
            or value.get("status") not in {"VERIFIED_OFFICIAL_ALL_ELIGIBLE_CAPTURE",
                                           "PARTIAL_IDENTITY_NOT_VERIFIED"}):
        return [], []
    pregame = value.get("pregame_fixtures")
    terminal = value.get("official_final_fixtures")
    if (not isinstance(pregame, list) or not isinstance(terminal, list)
            or type(value.get("pregame_count")) is not int or type(value.get("official_final_count")) is not int
            or len(pregame) != value["pregame_count"] or len(terminal) != value["official_final_count"]
            or len(pregame) + len(terminal) > MAX_ELIGIBLE_PER_TICK):
        return [], []
    seen_pre, seen_final = set(), set()
    for row in pregame:
        if (not isinstance(row, dict) or set(row) != set(CANONICAL_FIELDS)
                or _identity(row) != row or row.get("status") != "NS"
                or _utc(row["kickoff"]) <= observed
                or row["fixture_id"] in seen_pre):
            return [], []
        seen_pre.add(row["fixture_id"])
    for row in terminal:
        if (not isinstance(row, dict) or row.get("status") not in TERMINAL_STATUSES
                or not _positive(row.get("fixture_id"))
                or row.get("bet_eligible") is not False
                or row.get("market_inputs_used") is not False
                or row.get("status_source") != SOURCE
                or row.get("corners_verified") is not False
                or row.get("corners_ft") is not None
                or _utc(row.get("kickoff")) is None
                or _utc(row["kickoff"]) > observed
                or not _positive(row.get("home_team_id")) or not _positive(row.get("away_team_id"))
                or row["home_team_id"] == row["away_team_id"]
                or _goal_pair(row.get("goals")) != row.get("goals")
                or row.get("halftime") not in (None, _goal_pair(row.get("halftime")))
                or row.get("regulation_fulltime") not in (None, _goal_pair(row.get("regulation_fulltime")))
                or row.get("second_half") not in (None, _goal_pair(row.get("second_half")))
                or row["fixture_id"] in seen_final):
            return [], []
        seen_final.add(row["fixture_id"])
    return pregame, terminal
