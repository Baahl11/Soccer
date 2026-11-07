"""Strict FE11 next-day canonical fixture roster, independent of price/market.

Only the official API-Football fixtures/date payload, already obtained by the
authorized scheduler, may seed soccer_fixtures through the EXISTING
DAILY_DISCOVERY persistence mechanism. Never infer IDs from a model profile.

The scheduler decides if an *additional* date request fits its real quota.
No provider calls, database connections, or writes occur in this module.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

VERSION = "SOCCER_FE11_NEXT_DAY_REGISTRY_PRESEED_v0.1"
SCANNER_APPROVED_LEAGUES = frozenset({2, 3, 17, 18, 39, 61, 78, 135, 140, 262, 848})
LOCAL_TIMEZONE = ZoneInfo("America/Mexico_City")
MAX_VERIFIED_NEXT_DAY = 200
PERSISTENCE_FIELDS = (
    "fixture_id", "league_id", "league", "country", "season", "round",
    "kickoff", "status", "status_long", "home_team_id", "home_team",
    "away_team_id", "away_team", "venue", "city",
)


def _utc(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else None
    except ValueError:
        return None


def _identifier(value: Any) -> bool:
    return type(value) is int and 0 < value < 10**12


def select_verified_next_day(
    provider_fixture_rows: Any, *, now_utc: datetime,
    league_ids: frozenset[int] = SCANNER_APPROVED_LEAGUES,
    max_fixtures: int = MAX_VERIFIED_NEXT_DAY,
) -> dict[str, Any]:
    """Fail closed on malformed source, duplicate conflicts or oversized slate.

    Individual non-NS/cancelled/outside-next-local-date records are excluded.
    The resulting fixture dictionaries have only the canonical registry
    columns accepted by persistence._upsert_fixture; no betting fields.
    """
    if (not isinstance(now_utc, datetime) or now_utc.tzinfo is None
            or not isinstance(provider_fixture_rows, list)
            or not isinstance(league_ids, frozenset)
            or not league_ids
            or any(not _identifier(v) for v in league_ids)
            or type(max_fixtures) is not int or not 1 <= max_fixtures <= 300):
        raise ValueError("FE11_NEXT_DAY_SOURCE_OR_BOUNDS_INVALID")
    now = now_utc.astimezone(timezone.utc)
    next_local_date = now.astimezone(LOCAL_TIMEZONE).date() + timedelta(days=1)
    selected: dict[int, dict[str, Any]] = {}
    excluded = {
        "OUTSIDE_APPROVED_LEAGUES": 0, "NOT_NS_PREMATCH": 0,
        "NOT_NEXT_LOCAL_DATE": 0, "MALFORMED_IDENTITY": 0,
        "IDENTICAL_DUPLICATES": 0,
    }
    for raw in provider_fixture_rows:
        if not isinstance(raw, dict):
            excluded["MALFORMED_IDENTITY"] += 1
            continue
        league = raw.get("league_id")
        if type(league) is not int or league not in league_ids:
            excluded["OUTSIDE_APPROVED_LEAGUES"] += 1
            continue
        if raw.get("status") != "NS":
            excluded["NOT_NS_PREMATCH"] += 1
            continue
        kickoff = _utc(raw.get("kickoff"))
        if kickoff is None or kickoff.astimezone(LOCAL_TIMEZONE).date() != next_local_date:
            excluded["NOT_NEXT_LOCAL_DATE"] += 1
            continue
        fid = raw.get("fixture_id")
        home, away = raw.get("home_team_id"), raw.get("away_team_id")
        season = raw.get("season")
        if (not _identifier(fid) or not _identifier(home)
                or not _identifier(away) or home == away
                or type(season) is not int or not 2000 <= season <= 2100
                or kickoff <= now
                or not isinstance(raw.get("home_team"), str)
                or not raw["home_team"].strip()
                or not isinstance(raw.get("away_team"), str)
                or not raw["away_team"].strip()):
            excluded["MALFORMED_IDENTITY"] += 1
            continue
        safe = {key: raw.get(key) for key in PERSISTENCE_FIELDS}
        safe["kickoff"] = kickoff.isoformat()
        if fid in selected:
            if safe != selected[fid]:
                return {
                    "version": VERSION, "status": "NOT_VERIFIED_IDENTITY_CONFLICT",
                    "date": next_local_date.isoformat(), "fixtures": [],
                    "count": 0, "source": "API_FOOTBALL_FIXTURES_DATE",
                    "bet_eligible": False, "market_inputs_used": False,
                    "blockers": ["DUPLICATE_FIXTURE_CONFLICT"],
                }
            excluded["IDENTICAL_DUPLICATES"] += 1
            continue
        selected[fid] = safe
        if len(selected) > max_fixtures:
            return {
                "version": VERSION, "status": "NOT_VERIFIED_TOO_MANY_FIXTURES",
                "date": next_local_date.isoformat(), "fixtures": [],
                "count": 0, "source": "API_FOOTBALL_FIXTURES_DATE",
                "bet_eligible": False, "market_inputs_used": False,
                "blockers": ["PRESEED_BOUNDED_CAP_EXCEEDED"],
            }
    fixtures = sorted(selected.values(), key=lambda f: (f["kickoff"], f["fixture_id"]))
    return {
        "version": VERSION, "status": "VERIFIED_PROVIDER_ROSTER" if fixtures else "NO_VERIFIED_NEXT_DAY_FIXTURES",
        "date": next_local_date.isoformat(), "fixtures": fixtures,
        "count": len(fixtures), "excluded": excluded,
        "source": "API_FOOTBALL_FIXTURES_DATE",
        "source_observed_at_utc": now.isoformat(),
        "market_inputs_used": False, "bet_eligible": False,
        "classification": "RESEARCH_ONLY",
        "blockers": [],
    }


def discovery_event(roster: dict[str, Any]) -> dict[str, Any] | None:
    """Recognized by existing persistence.persist_tick, no new DB writer."""
    if (not isinstance(roster, dict)
            or roster.get("version") != VERSION
            or roster.get("status") != "VERIFIED_PROVIDER_ROSTER"
            or roster.get("market_inputs_used") is not False
            or roster.get("bet_eligible") is not False
            or type(roster.get("count")) is not int
            or not isinstance(roster.get("fixtures"), list)
            or len(roster["fixtures"]) != roster["count"]
            or not 0 < roster["count"] <= MAX_VERIFIED_NEXT_DAY
            or roster.get("source") != "API_FOOTBALL_FIXTURES_DATE"
            or _utc(roster.get("source_observed_at_utc")) is None
            or not isinstance(roster.get("date"), str)):
        return None
    # Protect the persistence boundary even if a caller mutates the
    # previously validated roster. No price/model extras can be upserted.
    for fx in roster["fixtures"]:
        if (not isinstance(fx, dict) or set(fx) != set(PERSISTENCE_FIELDS)
                or fx.get("status") != "NS"
                or fx.get("league_id") not in SCANNER_APPROVED_LEAGUES
                or not _identifier(fx.get("fixture_id"))
                or not _identifier(fx.get("home_team_id"))
                or not _identifier(fx.get("away_team_id"))
                or fx["home_team_id"] == fx["away_team_id"]):
            return None
        kickoff = _utc(fx.get("kickoff"))
        if (kickoff is None
                or kickoff.astimezone(LOCAL_TIMEZONE).date().isoformat() != roster["date"]
                or kickoff <= _utc(roster["source_observed_at_utc"])):
            return None
    return {
        "event_type": "DAILY_DISCOVERY",
        "stage": "FE11_NEXT_DAY_PRESEED",
        "date": roster["date"],
        "timezone": "America/Mexico_City",
        "upcoming_count": roster["count"],
        "fixtures": roster["fixtures"],
        "source": "API_FOOTBALL_FIXTURES_DATE",
        "source_observed_at_utc": roster["source_observed_at_utc"],
        "classification": "RESEARCH_ONLY",
        "sport_first": True,
        "market_data_included": False,
        "bet_eligible": False,
        "calibrated_joint_probability": None,
        "notes": [
            "Official identity-only fixture registry coverage; neither model forecasts nor bookmaker inputs.",
            "Does not authorize FE11 historical import or change BET/market/availability rules.",
        ],
    }
