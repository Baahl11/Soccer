from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from typing import Any

from mcp_gateway import automation as base
from mcp_gateway import automation_v3 as v3
from mcp_gateway import automation_v18 as v18

# Presentation / coverage metadata only. Sporting weights and market decision
# logic remain exactly the v18 production path.
MODEL_VERSION = v18.MODEL_VERSION
AUTOMATION_VERSION = "2.8.0"

_ORIGINAL_DAILY_DISCOVERY = v3._daily_discovery_event
_REGISTRY_THIS_TICK: list[dict[str, Any]] = []
_REGISTRY_ERRORS_THIS_TICK: list[str] = []

_IDENTITY_TTL = timedelta(days=30)


def _identity_key(league_id: Any, season: Any) -> str:
    return f"{league_id}:{season}"


def _clean(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _remember_or_restore_identity(fx: dict[str, Any], now: datetime) -> None:
    league_id = fx.get("league_id")
    season = fx.get("season")
    if league_id is None or season is None:
        return
    key = _identity_key(league_id, season)
    league = _clean(fx.get("league"))
    country = _clean(fx.get("country"))

    if league or country:
        base._cache_set(
            "league_identity",
            key,
            {
                "league_id": league_id,
                "season": season,
                "league": league,
                "country": country,
            },
            now,
        )
        return

    cached = base._cache_get("league_identity", key, _IDENTITY_TTL, now)
    if not isinstance(cached, dict):
        return
    if not _clean(fx.get("league")) and _clean(cached.get("league")):
        fx["league"] = cached.get("league")
    if not _clean(fx.get("country")) and _clean(cached.get("country")):
        fx["country"] = cached.get("country")


def _coverage_row(
    sample: dict[str, Any],
    coverage: dict[str, Any],
    fixture_count: int,
    eligible_fixture_count: int,
) -> dict[str, Any]:
    tier = str(coverage.get("data_tier") or "D")
    known = bool(coverage.get("known"))
    return {
        "league_id": sample.get("league_id"),
        "season": sample.get("season"),
        "country": _clean(sample.get("country")) or "NOT VERIFIED",
        "competition": _clean(sample.get("league")) or f"League #{sample.get('league_id')}",
        "fixtures_in_scanned_slate": fixture_count,
        "pregame_eligible_fixtures": eligible_fixture_count,
        "provider_coverage_known": known,
        "coverage": {
            "events": bool(coverage.get("events")) if known else None,
            "lineups": bool(coverage.get("lineups")) if known else None,
            "fixture_statistics": bool(coverage.get("statistics_fixtures")) if known else None,
            "player_statistics": bool(coverage.get("statistics_players")) if known else None,
            "injuries": bool(coverage.get("injuries")) if known else None,
            "odds": bool(coverage.get("odds")) if known else None,
        },
        "data_tier": tier,
        "sporting_screen_scope": "ELIGIBLE" if tier in {"A", "B", "C"} else "RESEARCH_OR_PASS",
        "bet_default_scope": "A_B_ONLY" if tier in {"A", "B"} else "NOT_DEFAULT_BET_ELIGIBLE",
        "galaxy_activation_required": False,
        "galaxy_role": "OPTIONAL_BACKBONE_OR_SHADOW; NOT_A_SLATE_ELIGIBILITY_GATE",
    }


async def _capture_registry_discovery(
    fixtures: list[dict[str, Any]], now_utc: datetime, local_now: datetime
) -> dict[str, Any] | None:
    """Capture complete-slate league coverage without per-league provider calls.

    v3._coverage_fast uses the existing cached bulk /leagues map (one map for all
    competitions), so this registry never performs an API-Football request per
    league. Fixture identity is restored from local cache when Galaxy's compact
    slate only carries league_id/season.
    """
    global _REGISTRY_THIS_TICK, _REGISTRY_ERRORS_THIS_TICK

    for fx in fixtures:
        if isinstance(fx, dict):
            _remember_or_restore_identity(fx, now_utc)

    grouped: dict[str, dict[str, Any]] = {}
    for fx in fixtures:
        if not isinstance(fx, dict):
            continue
        league_id = fx.get("league_id")
        season = fx.get("season")
        if league_id is None or season is None:
            continue
        key = _identity_key(league_id, season)
        row = grouped.setdefault(
            key,
            {
                "sample": fx,
                "fixtures": 0,
                "eligible": 0,
            },
        )
        row["fixtures"] += 1
        try:
            kickoff = base._dt(str(fx.get("kickoff")))
            is_future = kickoff >= now_utc
        except Exception:
            is_future = False
        blocked_status = fx.get("status") in base.CANCELLED_STATUSES | base.POSTPONED_STATUSES
        if is_future and not blocked_status:
            row["eligible"] += 1

    registry: list[dict[str, Any]] = []
    errors: list[str] = []
    for key, group in grouped.items():
        sample = group["sample"]
        try:
            coverage = await v3._coverage_fast(
                int(sample["league_id"]), int(sample["season"]), now_utc
            )
        except Exception as exc:
            coverage = {"known": False, "data_tier": "D"}
            errors.append(f"{key}:{str(exc)[:120]}")
        registry.append(
            _coverage_row(
                sample,
                coverage,
                int(group["fixtures"]),
                int(group["eligible"]),
            )
        )

    registry.sort(
        key=lambda row: (
            {"A": 0, "B": 1, "C": 2, "D": 3}.get(str(row.get("data_tier")), 9),
            str(row.get("country") or ""),
            str(row.get("competition") or ""),
        )
    )
    _REGISTRY_THIS_TICK = registry
    _REGISTRY_ERRORS_THIS_TICK = errors[:20]

    return await _ORIGINAL_DAILY_DISCOVERY(fixtures, now_utc, local_now)


def _registry_lookup() -> dict[str, dict[str, Any]]:
    return {
        _identity_key(row.get("league_id"), row.get("season")): row
        for row in _REGISTRY_THIS_TICK
    }


def _score(shortlist: dict[str, Any], key: str) -> float | None:
    value = shortlist.get(key)
    try:
        return round(float(value), 1) if value is not None else None
    except (TypeError, ValueError):
        return None


def _reason(event: dict[str, Any]) -> str | None:
    market_decision = event.get("market_decision")
    if isinstance(market_decision, dict):
        for key in ("reason", "status_reason", "block_reason"):
            value = _clean(market_decision.get(key))
            if value:
                return value
    shortlist = event.get("sporting_shortlist")
    if isinstance(shortlist, dict):
        value = _clean(shortlist.get("reason"))
        if value:
            return value
    notes = event.get("notes")
    if isinstance(notes, list) and notes:
        return _clean(notes[0])
    return _clean(event.get("error"))


def _presentation_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    registry = _registry_lookup()
    rows: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict) or event.get("event_type") == "DAILY_DISCOVERY":
            continue
        fx = event.get("fixture")
        if not isinstance(fx, dict):
            continue
        reg = registry.get(_identity_key(fx.get("league_id"), fx.get("season"))) or {}
        shortlist = event.get("sporting_shortlist")
        if not isinstance(shortlist, dict):
            shortlist = {}
        best = event.get("best_market")
        if not isinstance(best, dict):
            best = {}
        rows.append(
            {
                "fixture_id": fx.get("fixture_id"),
                "country": _clean(fx.get("country")) or reg.get("country") or "NOT VERIFIED",
                "competition": _clean(fx.get("league")) or reg.get("competition") or f"League #{fx.get('league_id')}",
                "data_tier": (event.get("coverage") or {}).get("data_tier") if isinstance(event.get("coverage"), dict) else reg.get("data_tier"),
                "kickoff": fx.get("kickoff"),
                "stage": event.get("stage"),
                "home": fx.get("home_team"),
                "away": fx.get("away_team"),
                "tracks": shortlist.get("tracks") or [],
                "side_score": _score(shortlist, "side_edge_score"),
                "goals_score": _score(shortlist, "goal_environment_score"),
                "two_way_score": _score(shortlist, "two_way_scoring_score"),
                "availability_confidence": event.get("availability_confidence"),
                "market_source": event.get("market_source"),
                "market": best.get("market"),
                "selection": best.get("selection"),
                "line": best.get("line"),
                "price": best.get("decimal_price"),
                "bookmaker": best.get("bookmaker"),
                "classification": event.get("classification") or "WATCH",
                "tier": event.get("tier"),
                "stake_units": event.get("stake_units", 0.0),
                "reason": _reason(event),
            }
        )

    order = {"BET": 0, "LEAN": 1, "WATCH": 2, "PASS": 3, "CLOSE": 4, "POSTGAME": 5}
    rows.sort(
        key=lambda row: (
            order.get(str(row.get("classification")), 9),
            str(row.get("country") or ""),
            str(row.get("competition") or ""),
            str(row.get("kickoff") or ""),
        )
    )
    return rows


async def run_tick() -> dict[str, Any]:
    global _REGISTRY_THIS_TICK, _REGISTRY_ERRORS_THIS_TICK
    _REGISTRY_THIS_TICK = []
    _REGISTRY_ERRORS_THIS_TICK = []

    previous_discovery = v3._daily_discovery_event
    v3._daily_discovery_event = _capture_registry_discovery
    try:
        payload = await v18.run_tick()
    finally:
        v3._daily_discovery_event = previous_discovery

    tiers = Counter(str(row.get("data_tier") or "D") for row in _REGISTRY_THIS_TICK)
    eligible = sum(
        int(row.get("pregame_eligible_fixtures") or 0)
        for row in _REGISTRY_THIS_TICK
        if row.get("data_tier") in {"A", "B", "C"}
    )
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    payload["league_coverage_registry"] = {
        "schema_version": "1.0.0",
        "generated_at_local": payload.get("generated_at_local"),
        "scope": "ALL_COMPETITIONS_PRESENT_IN_SCANNED_PROVIDER_SLATE; NOT_LIMITED_TO_GALAXY_ACTIVE_LEAGUES",
        "provider_calls_per_league": 0,
        "galaxy_activation_is_eligibility_gate": False,
        "competition_count": len(_REGISTRY_THIS_TICK),
        "data_tier_counts": dict(tiers),
        "pregame_data_eligible_fixture_count": eligible,
        "errors": list(_REGISTRY_ERRORS_THIS_TICK),
        "competitions": list(_REGISTRY_THIS_TICK),
    }
    payload["match_table_rows"] = _presentation_rows(payload.get("events") or [])
    payload["presentation_contract"] = {
        "default_format": "MARKDOWN_TABLES",
        "grouping": ["classification", "country", "competition"],
        "detailed_sections": ["BET", "LEAN", "WATCH"],
        "pass_display": "SUMMARY_BY_COMPETITION_UNLESS_FULL_SLATE_REQUESTED",
        "columns": [
            "kickoff",
            "match",
            "data_tier",
            "sporting_scores",
            "availability_confidence",
            "market",
            "price",
            "classification",
            "reason",
        ],
        "notes": [
            "Show analyzed matches in tables, not stacked prose.",
            "Separate BET, LEAN, WATCH and PASS-summary sections.",
            "Within each section divide by country/competition when useful.",
            "Never hide NOT VERIFIED availability or market information.",
        ],
    }
    payload["coverage_registry_policy"] = (
        "GLOBAL_SLATE_FIRST; DATA_QUALITY_NOT_LEAGUE_WHITELIST; GALAXY_ACTIVE_LEAGUES_DO_NOT_DEFINE_ELIGIBILITY"
    )
    return payload
