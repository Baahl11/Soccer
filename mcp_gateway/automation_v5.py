from __future__ import annotations

import os
from collections import Counter
from datetime import datetime, timedelta, timezone as dt_timezone
from typing import Any

import httpx

from mcp_gateway import automation as base
from mcp_gateway import automation_v2 as v2
from mcp_gateway import automation_v3 as v3
from mcp_gateway import automation_v4 as v4
from mcp_gateway.soccer_model import build_raw_projection, public_raw_projection

MODEL_VERSION = "SOCCER EDGE ENGINE v1.0"
AUTOMATION_VERSION = "1.5.2"

MAX_DEEP_DIVE_FIXTURES_PER_TICK = int(os.getenv("SOCCER_EDGE_MAX_DEEP_DIVE_FIXTURES_PER_TICK", "8"))
MAX_UPCOMING_MARKET_CAPTURE_FIXTURES = max(
    20,
    int(os.getenv("SOCCER_EDGE_MAX_UPCOMING_MARKET_CAPTURE_FIXTURES", "80")),
)
SHORTLIST_TTL = timedelta(hours=14)

PLAYER_PROPS_XI_RESEARCH_STAGES = {"T-20", "T-10"}
MAX_PLAYER_PROPS_XI_RESEARCH_FIXTURES_PER_TICK = max(
    1,
    int(os.getenv("SOCCER_PLAYER_PROPS_XI_RESEARCH_MAX_FIXTURES_PER_TICK", "4")),
)
_PLAYER_PROPS_XI_RESEARCH_ATTEMPTS = 0
_PLAYER_PROPS_XI_RESEARCH_CAPTURED = 0

STAGE_PRIORITY = {
    "T-40": 0,
    "T-20": 1,
    "T-10": 2,
    "CLOSE": 3,
    "T-60": 4,
    "T-90": 5,
    "T-30": 6,
    "POSTGAME": 7,
}
STAGE_TARGET = {"T-90": 90, "T-60": 60, "T-40": 40, "T-30": 30, "T-20": 20, "T-10": 10, "CLOSE": 0}

# These rules do not create sporting edge. They only break ties when scarce
# API budget forces us to decide which already-data-eligible fixture is refreshed first.
PRIORITY_COMPETITIONS = {
    ("world", "uefa champions league"),
    ("world", "uefa europa league"),
    ("world", "uefa conference league"),
    ("world", "copa libertadores"),
    ("world", "copa sudamericana"),
    ("england", "premier league"),
    ("spain", "la liga"),
    ("germany", "bundesliga"),
    ("italy", "serie a"),
    ("france", "ligue 1"),
    ("mexico", "liga mx"),
    ("usa", "major league soccer"),
    ("brazil", "serie a"),
    ("argentina", "liga profesional argentina"),
    ("portugal", "primeira liga"),
    ("netherlands", "eredivisie"),
}


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _competition_priority(fx: dict[str, Any]) -> int:
    key = ((fx.get("country") or "").strip().lower(), (fx.get("league") or "").strip().lower())
    return 0 if key in PRIORITY_COMPETITIONS else 1


def _coverage_strength(coverage: dict[str, Any]) -> int:
    return sum(
        1
        for key in ("lineups", "statistics_fixtures", "statistics_players", "injuries", "odds")
        if coverage.get(key)
    )


def _screen_shortlist(raw: dict[str, Any]) -> dict[str, Any]:
    if raw.get("status") != "MODELED_LIMITED":
        return {
            "shortlisted": False,
            "rank": 0.0,
            "tracks": [],
            "reason": raw.get("reason") or "RAW_PROJECTION_NOT_AVAILABLE",
        }

    scores = raw.get("screen_scores") or {}
    side = _num(scores.get("side_edge_score"))
    goals = _num(scores.get("goal_environment_score"))
    two_way = _num(scores.get("two_way_scoring_score"))

    tracks: list[str] = []
    ranks: list[float] = []

    # Master v1.0: 60+ can enter the side shortlist; 70+ is a meaningful signal.
    if side is not None and side >= 60:
        tracks.append("SIDE")
        ranks.append(side)

    # Master v1.0: 70+ is a meaningful high-goal signal; <50 is a meaningful
    # suppression signal. Convert suppression to a ranking score only for queueing.
    if goals is not None and goals >= 70:
        tracks.append("GOALS_OVER")
        ranks.append(goals)
    elif goals is not None and goals < 50:
        tracks.append("GOALS_UNDER")
        ranks.append(100.0 - goals)

    # Master v1.0: 65+ is a viable two-way scoring path.
    if two_way is not None and two_way >= 65:
        tracks.append("TWO_WAY")
        ranks.append(two_way)

    return {
        "shortlisted": bool(tracks),
        "rank": round(max(ranks) if ranks else 0.0, 1),
        "tracks": tracks,
        "side_edge_score": side,
        "goal_environment_score": goals,
        "two_way_scoring_score": two_way,
        "reason": "SPORTING_SCREEN_PASS" if tracks else "SPORTING_SCREEN_BELOW_SHORTLIST",
    }


def _shortlist_get(fixture_id: int, now: datetime) -> dict[str, Any] | None:
    value = base._cache_get("sport_shortlist", str(fixture_id), SHORTLIST_TTL, now)
    return value if isinstance(value, dict) else None


def _shortlist_set(fixture_id: int, value: dict[str, Any], now: datetime) -> None:
    base._cache_set("sport_shortlist", str(fixture_id), value, now)


def _stage_already_processed(fixture_id: int, stage: str, now: datetime) -> bool:
    conn = base._cache_conn()
    row = conn.execute(
        "SELECT updated_at FROM processed_stages WHERE fixture_id=? AND stage=?",
        (fixture_id, stage),
    ).fetchone()
    return bool(row and now.timestamp() - float(row[0]) < timedelta(hours=6).total_seconds())


def _mark_stage_processed(fixture_id: int, stage: str, now: datetime) -> None:
    conn = base._cache_conn()
    conn.execute(
        """
        INSERT INTO processed_stages(fixture_id, stage, updated_at)
        VALUES(?,?,?)
        ON CONFLICT(fixture_id, stage) DO UPDATE SET updated_at=excluded.updated_at
        """,
        (fixture_id, stage, now.timestamp()),
    )
    conn.commit()


async def _team_stats_12h(team_id: int, league_id: int, season: int, now: datetime) -> dict[str, Any]:
    key = f"{team_id}:{league_id}:{season}"
    cached = base._cache_get("team_stats", key, timedelta(hours=12), now)
    if isinstance(cached, dict):
        return cached
    payload = await base._api_get(
        "teams/statistics", {"team": team_id, "league": league_id, "season": season}
    )
    compact = base._compact_team_stats(payload)
    base._cache_set("team_stats", key, compact, now)
    return compact


async def _recent_12h(team_id: int, now: datetime) -> list[dict[str, Any]]:
    key = str(team_id)
    cached = base._cache_get("recent", key, timedelta(hours=12), now)
    if isinstance(cached, list):
        return cached
    payload = await base._api_get(
        "fixtures", {"team": team_id, "last": 8, "timezone": base.TIMEZONE_NAME}
    )
    matches = [base._compact_fixture(x) for x in payload.get("response", [])]
    base._cache_set("recent", key, matches, now)
    return matches


async def _injuries_2h(fixture_id: int, now: datetime) -> list[dict[str, Any]]:
    key = str(fixture_id)
    cached = base._cache_get("injuries", key, timedelta(hours=2), now)
    if isinstance(cached, list):
        return cached
    compact = base._compact_injuries(await base._api_get("injuries", {"fixture": fixture_id}))
    base._cache_set("injuries", key, compact, now)
    return compact


async def _lineup_cached(fixture_id: int, now: datetime) -> dict[str, Any]:
    key = str(fixture_id)
    confirmed = base._cache_get("lineup_confirmed", key, timedelta(hours=12), now)
    if isinstance(confirmed, dict):
        return confirmed
    pending = base._cache_get("lineup_pending", key, timedelta(minutes=5), now)
    if isinstance(pending, dict):
        return pending

    compact = base._compact_lineups(
        await base._api_get("fixtures/lineups", {"fixture": fixture_id})
    )
    if compact.get("both_xi_confirmed") and compact.get("both_goalkeepers_confirmed"):
        base._cache_set("lineup_confirmed", key, compact, now)
    else:
        base._cache_set("lineup_pending", key, compact, now)
    return compact


def _realign_cached_player_props(
    compact: dict[str, Any],
    lineup: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(compact, dict):
        return compact
    rows = []
    starters = base._confirmed_starters(lineup)
    for row in compact.get("research_cards_props_markets") or []:
        if not isinstance(row, dict) or row.get("research_family") != "PLAYER_PROPS":
            rows.append(row)
            continue
        subfamily = row.get("research_subfamily")
        values = [
            base._align_research_player_value(
                value,
                confirmed_starters=starters,
                research_subfamily=subfamily,
            )
            if isinstance(value, dict) else value
            for value in (row.get("values") or [])
        ]
        rows.append({
            **row,
            "values": values,
            "confirmed_xi_at_quote": bool(starters),
            "xi_aligned_value_rows": sum(
                1 for value in values
                if isinstance(value, dict)
                and value.get("xi_alignment_status") == "MATCHED_CONFIRMED_XI"
            ),
        })
    return {
        **compact,
        "research_cards_props_markets": rows,
    }


async def _odds_7m(
    fixture_id: int,
    now: datetime,
    lineup: dict[str, Any] | None = None,
) -> dict[str, Any]:
    key = str(fixture_id)
    cached = base._cache_get("odds_compact", key, timedelta(minutes=7), now)
    if isinstance(cached, dict):
        aligned = _realign_cached_player_props(cached, lineup)
        return {
            **aligned,
            "source": "LOCAL_ODDS_CACHE",
            "resolution_status": "PRICE_CACHE_HIT_LOCAL",
        }
    compact = base._compact_odds(
        await base._api_get("odds", {"fixture": fixture_id, "page": 1}),
        lineup=lineup,
    )
    # Persist the provider payload without the transient source marker so a
    # later cache read cannot masquerade as a fresh provider observation.
    base._cache_set("odds_compact", key, compact, now)
    return {
        **compact,
        "source": "API_FOOTBALL_ODDS_V3",
        "resolution_status": "PRICE_API_RESOLVED",
    }


async def _try_player_props_xi_research(
    event: dict[str, Any],
    fx: dict[str, Any],
    stage: str,
    coverage: dict[str, Any],
    now: datetime,
) -> bool:
    global _PLAYER_PROPS_XI_RESEARCH_ATTEMPTS
    global _PLAYER_PROPS_XI_RESEARCH_CAPTURED

    if stage not in PLAYER_PROPS_XI_RESEARCH_STAGES:
        return False
    if not coverage.get("lineups") or not coverage.get("odds"):
        return False
    if _PLAYER_PROPS_XI_RESEARCH_ATTEMPTS >= MAX_PLAYER_PROPS_XI_RESEARCH_FIXTURES_PER_TICK:
        return False

    _PLAYER_PROPS_XI_RESEARCH_ATTEMPTS += 1
    lineup = await _lineup_cached(fx["fixture_id"], now)
    event["lineups"] = lineup
    event["availability_confidence"] = (
        0.90
        if lineup.get("both_xi_confirmed") and lineup.get("both_goalkeepers_confirmed")
        else 0.75
    )

    if lineup.get("both_xi_confirmed") is not True:
        event["research_player_props_xi_capture"] = {
            "status": "XI_NOT_CONFIRMED",
            "stage": stage,
            "research_only": True,
            "decision_weight": 0.0,
            "production_promotion_allowed": False,
        }
        return False

    market = await _odds_7m(
        fx["fixture_id"],
        now,
        lineup=lineup,
    )
    event["market"] = market
    event["market_use"] = "PLAYER_PROPS_XI_RESEARCH_ONLY"
    event["classification"] = "RESEARCH_ONLY"
    event["bet_eligible"] = False
    event["stake_units"] = 0.0
    event["decision_weight"] = 0.0

    prop_rows = [
        row for row in (market.get("research_cards_props_markets") or [])
        if isinstance(row, dict)
        and row.get("research_family") == "PLAYER_PROPS"
    ]
    xi_aligned_values = sum(
        1
        for row in prop_rows
        for value in (row.get("values") or [])
        if isinstance(value, dict)
        and value.get("xi_alignment_status") == "MATCHED_CONFIRMED_XI"
    )
    event["research_player_props_xi_capture"] = {
        "status": "XI_CONFIRMED_RESEARCH_CAPTURED",
        "stage": stage,
        "market_source": market.get("source"),
        "market_resolution_status": market.get("resolution_status"),
        "player_prop_market_rows": len(prop_rows),
        "xi_aligned_value_rows": xi_aligned_values,
        "research_only": True,
        "decision_weight": 0.0,
        "production_promotion_allowed": False,
        "primary_market_decision_created": False,
    }
    event["notes"].append(
        "Confirmed-XI Player Props research capture retained despite sporting shortlist miss; no production decision created."
    )
    _PLAYER_PROPS_XI_RESEARCH_CAPTURED += 1
    return True


async def _cheap_sport_bundle(fx: dict[str, Any], now: datetime) -> dict[str, Any]:
    # First-stage SPORT FIRST screen: season home/away splits only. Recent-form
    # requests are reserved for candidates that survive the screen.
    home_stats = await _team_stats_12h(fx["home_team_id"], fx["league_id"], fx["season"], now)
    away_stats = await _team_stats_12h(fx["away_team_id"], fx["league_id"], fx["season"], now)
    return {
        "sport_data": "AVAILABLE",
        "home_stats": home_stats,
        "away_stats": away_stats,
        "home_recent": [],
        "away_recent": [],
        "screen_layer": "CHEAP_SEASON_SPLIT",
    }


async def _refine_sport_bundle(fx: dict[str, Any], cheap: dict[str, Any], now: datetime) -> dict[str, Any]:
    home_recent = await _recent_12h(fx["home_team_id"], now)
    away_recent = await _recent_12h(fx["away_team_id"], now)
    return {
        "sport_data": "AVAILABLE",
        "home_stats": cheap.get("home_stats") or {},
        "away_stats": cheap.get("away_stats") or {},
        "home_recent": home_recent,
        "away_recent": away_recent,
        "screen_layer": "REFINED_WITH_RECENT",
    }


def _best_decision_fields(best: dict[str, Any]) -> dict[str, Any]:
    return {
        "family": best.get("family"),
        "market": best.get("market"),
        "selection": best.get("selection"),
        "line": best.get("line"),
        "decimal_price": best.get("decimal_price"),
        "bookmaker": best.get("bookmaker"),
        "p_breakeven": best.get("p_breakeven"),
        "p_market_fair": best.get("p_market_fair"),
        "p_raw": best.get("p_raw"),
        "p_shrunk": best.get("p_shrunk"),
        "prob_edge_pp": best.get("prob_edge_pp"),
        "estimated_ev": best.get("estimated_ev"),
        "shrink_weight": best.get("shrink_weight"),
        "tier": best.get("tier"),
        "classification": best.get("classification"),
        "stake_units": best.get("stake_units"),
    }


async def _priority_event(
    fx: dict[str, Any],
    stage: str,
    coverage: dict[str, Any],
    now: datetime,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "event_type": "SOCCER_REFRESH",
        "stage": stage,
        "fixture": fx,
        "coverage": coverage,
        "model_version": MODEL_VERSION,
        "classification": "WATCH",
        "tier": None,
        "stake_units": 0.0,
        "bet_eligible": False,
        "availability_confidence": None,
        "notes": [],
    }

    prior_shortlist = _shortlist_get(fx["fixture_id"], now)

    if stage == "POSTGAME":
        if coverage.get("statistics_fixtures"):
            stats = await base._api_get("fixtures/statistics", {"fixture": fx["fixture_id"]})
            event["match_stats"] = stats.get("response", [])
        event["result"] = {"goals": fx.get("goals"), "score": fx.get("score"), "status": fx.get("status")}
        event["classification"] = "POSTGAME"
        event["sporting_shortlist"] = prior_shortlist
        return event

    if stage == "CLOSE":
        if not prior_shortlist or not prior_shortlist.get("shortlisted"):
            event["classification"] = "PASS"
            event["notes"].append("Closing market skipped: fixture never entered a sporting shortlist.")
            event["market_skipped_by_sport_screen"] = True
            return event
        event["sporting_shortlist"] = prior_shortlist
        if coverage.get("odds"):
            event["market"] = await _odds_7m(fx["fixture_id"], now)
            event["market_use"] = "CLOSING_SNAPSHOT_FOR_SHORTLISTED_FIXTURE"
        else:
            event["market"] = "NOT VERIFIED"
        event["classification"] = "CLOSE"
        return event

    raw_internal: dict[str, Any] | None = None
    refined_bundle: dict[str, Any] | None = None
    shortlisted = bool(prior_shortlist and prior_shortlist.get("shortlisted"))

    if stage in v2.SPORTING_STAGES:
        cheap = await _cheap_sport_bundle(fx, now)
        raw_cheap = build_raw_projection(fx, cheap, None)
        cheap_screen = _screen_shortlist(raw_cheap)
        event["sporting_screen_initial"] = cheap_screen

        if not cheap_screen["shortlisted"]:
            _shortlist_set(fx["fixture_id"], cheap_screen, now)
            event["raw_projection"] = public_raw_projection(raw_cheap)
            event["sporting_shortlist"] = cheap_screen
            if await _try_player_props_xi_research(event, fx, stage, coverage, now):
                event["market_skipped_by_sport_screen"] = False
                return event
            event["classification"] = "PASS"
            event["market_skipped_by_sport_screen"] = stage in v2.MARKET_STAGES
            event["notes"].append(
                "SPORT FIRST screen below shortlist threshold; recent-form and production market requests deferred."
            )
            return event

        refined_bundle = await _refine_sport_bundle(fx, cheap, now)
        raw_internal = build_raw_projection(fx, refined_bundle, None)
        refined_screen = _screen_shortlist(raw_internal)
        event["sporting_screen_refined"] = refined_screen
        _shortlist_set(fx["fixture_id"], refined_screen, now)
        event["sporting_shortlist"] = refined_screen
        shortlisted = bool(refined_screen.get("shortlisted"))

        if not shortlisted:
            event["raw_projection"] = public_raw_projection(raw_internal)
            if await _try_player_props_xi_research(event, fx, stage, coverage, now):
                event["market_skipped_by_sport_screen"] = False
                return event
            event["classification"] = "PASS"
            event["market_skipped_by_sport_screen"] = stage in v2.MARKET_STAGES
            event["notes"].append(
                "Refined sporting screen fell below shortlist threshold; no production market request made."
            )
            return event
    else:
        event["sporting_shortlist"] = prior_shortlist

    if not shortlisted:
        if await _try_player_props_xi_research(event, fx, stage, coverage, now):
            return event
        event["classification"] = "PASS"
        event["notes"].append("No active sporting shortlist for this refresh.")
        return event

    # Expensive availability calls happen only after the sporting shortlist exists.
    if stage in v2.INJURY_STAGES and coverage.get("injuries"):
        event["injuries"] = await _injuries_2h(fx["fixture_id"], now)
    elif stage in v2.INJURY_STAGES:
        event["injuries"] = "NOT VERIFIED"

    if stage in v2.LINEUP_STAGES and coverage.get("lineups"):
        lineup = await _lineup_cached(fx["fixture_id"], now)
        event["lineups"] = lineup
        if lineup["both_xi_confirmed"] and lineup["both_goalkeepers_confirmed"]:
            event["availability_confidence"] = 0.90
        else:
            event["availability_confidence"] = 0.70 if stage in {"T-60", "T-40"} else 0.60
            event["notes"].append("Material lineup/goalkeeper information remains NOT VERIFIED.")
    elif stage in v2.LINEUP_STAGES:
        event["lineups"] = "NOT VERIFIED"
        event["availability_confidence"] = 0.60

    # Rebuild raw projection after availability confidence is known. This does not
    # use market data and preserves SPORT FIRST.
    if raw_internal is not None:
        if event.get("availability_confidence") is not None and refined_bundle is not None:
            raw_internal = build_raw_projection(
                fx, refined_bundle, event.get("availability_confidence")
            )
        event["raw_projection"] = public_raw_projection(raw_internal)
        if raw_internal.get("status") == "MODELED_LIMITED":
            event["notes"].append(
                "Automated raw projection is LIMITED: xG/npxG/PPDA/field tilt remain NOT VERIFIED; goals are not substituted as xG."
            )

    # Detailed market is requested only for sporting-shortlisted candidates.
    if stage in v2.MARKET_STAGES and coverage.get("odds"):
        event["market"] = await _odds_7m(
            fx["fixture_id"],
            now,
            lineup=event.get("lineups") if isinstance(event.get("lineups"), dict) else None,
        )
        event["market_use"] = "MARKET_COMPARISON_AFTER_SPORTING_SHORTLIST"
    elif stage in v2.MARKET_STAGES:
        event["market"] = "NOT VERIFIED"

    if stage in {"T-40", "T-20", "T-10"}:
        if raw_internal is None:
            event["market_decision"] = {
                "status": "WATCH",
                "reason": "RAW_PROJECTION_NOT_AVAILABLE",
                "decisions": [],
            }
        else:
            market_decision = v4._safe_evaluate_market(
                raw_internal,
                event.get("market"),
                coverage,
                event.get("availability_confidence"),
                stage,
                event.get("lineups"),
            )
            event["market_decision"] = market_decision
            best = market_decision.get("best_decision") or {}
            event["classification"] = market_decision.get("status") or "WATCH"
            event["tier"] = best.get("tier")
            event["stake_units"] = best.get("stake_units", 0.0)
            event["bet_eligible"] = event["classification"] == "BET"
            if best:
                event["best_market"] = _best_decision_fields(best)

    if stage == "T-20":
        lineup = event.get("lineups")
        if (
            not isinstance(lineup, dict)
            or not lineup.get("both_xi_confirmed")
            or not lineup.get("both_goalkeepers_confirmed")
        ):
            event["bet_eligible"] = False
            event["classification"] = "WATCH"
            event["notes"].append("T-20 lineup/GK gate failed: BET eligibility blocked.")

    return event


async def run_tick() -> dict[str, Any]:
    global _PLAYER_PROPS_XI_RESEARCH_ATTEMPTS
    global _PLAYER_PROPS_XI_RESEARCH_CAPTURED

    v2._API_CALLS_THIS_TICK = 0
    v2._LAST_DAILY_REMAINING = None
    _PLAYER_PROPS_XI_RESEARCH_ATTEMPTS = 0
    _PLAYER_PROPS_XI_RESEARCH_CAPTURED = 0

    # v1.4 protections remain active.
    base._api_get = v4._paced_api_get
    v2.evaluate_market = v4._safe_evaluate_market

    now_utc = datetime.now(dt_timezone.utc)
    local_now = now_utc.astimezone(base.TIMEZONE)
    base._prune_cache(now_utc)

    dates = [local_now.date()]
    if local_now.hour >= 22:
        dates.append((local_now + timedelta(days=1)).date())

    fixtures: list[dict[str, Any]] = []
    quota: dict[str, Any] = {}

    original_coverage = base._coverage
    base._coverage = v3._coverage_fast

    base._HTTP_CLIENT = httpx.AsyncClient(
        timeout=base.TIMEOUT,
        limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
    )
    try:
        for d in dates:
            payload = await base._api_get(
                "fixtures", {"date": d.isoformat(), "timezone": base.TIMEZONE_NAME}
            )
            quota = payload.get("quota", quota)
            for row in payload.get("response", []):
                fx = base._compact_fixture(row)
                if (
                    fx.get("fixture_id")
                    and fx.get("kickoff")
                    and fx.get("league_id")
                    and fx.get("season")
                ):
                    fixtures.append(fx)

        upcoming_market_capture_fixtures = sorted(
            [
                fx
                for fx in fixtures
                if base._dt(fx["kickoff"]) > now_utc
                and fx.get("status") not in base.CANCELLED_STATUSES | base.POSTPONED_STATUSES
            ],
            key=lambda fx: base._dt(fx["kickoff"]),
        )[:MAX_UPCOMING_MARKET_CAPTURE_FIXTURES]

        events: list[dict[str, Any]] = []
        discovery = await v3._daily_discovery_event(fixtures, now_utc, local_now)
        if discovery is not None:
            events.append(discovery)

        due: list[dict[str, Any]] = []
        low_data_counts: Counter[str] = Counter()

        for fx in fixtures:
            kickoff = base._dt(fx["kickoff"])
            minutes_to = (kickoff - now_utc).total_seconds() / 60.0
            stage = base._stage_for(minutes_to, fx.get("status") or "")
            if not stage:
                continue

            if stage == "POSTGAME":
                minutes_since = -minutes_to
                if minutes_since < 95 or minutes_since > 240:
                    continue
                prior = _shortlist_get(fx["fixture_id"], now_utc)
                if not prior or not prior.get("shortlisted"):
                    continue

            coverage = await v3._coverage_fast(fx["league_id"], fx["season"], now_utc)
            tier = coverage.get("data_tier") or "D"
            prior = _shortlist_get(fx["fixture_id"], now_utc)
            prior_rank = _num((prior or {}).get("rank")) or 0.0
            target = STAGE_TARGET.get(stage)
            proximity = abs(minutes_to - target) if target is not None else 0.0

            due.append(
                {
                    "fx": fx,
                    "stage": stage,
                    "coverage": coverage,
                    "tier": tier,
                    "priority": (
                        0 if tier in {"A", "B"} else 1,
                        STAGE_PRIORITY.get(stage, 99),
                        0 if prior and prior.get("shortlisted") else 1,
                        0 if tier == "A" else 1 if tier == "B" else 2 if tier == "C" else 3,
                        _competition_priority(fx),
                        -_coverage_strength(coverage),
                        -prior_rank,
                        proximity,
                        kickoff,
                    ),
                }
            )

        due.sort(key=lambda item: item["priority"])

        deep_dive_processed = 0
        deferred_due_to_priority = 0
        deferred_due_to_budget = 0
        market_requests_avoided_by_screen = 0
        shortlist_events = 0

        for item in due:
            fx = item["fx"]
            stage = item["stage"]
            coverage = item["coverage"]
            tier = item["tier"]

            if _stage_already_processed(fx["fixture_id"], stage, now_utc):
                continue

            if tier not in {"A", "B"}:
                low_data_counts[f"{tier}:{stage}"] += 1
                _mark_stage_processed(fx["fixture_id"], stage, now_utc)
                continue

            if deep_dive_processed >= MAX_DEEP_DIVE_FIXTURES_PER_TICK:
                deferred_due_to_priority += 1
                continue

            try:
                event = await _priority_event(fx, stage, coverage, now_utc)
            except v2.TickBudgetExceeded as exc:
                deferred_due_to_budget += 1
                events.append(
                    {
                        "event_type": "QUOTA_GUARD",
                        "stage": stage,
                        "fixture": fx,
                        "model_version": MODEL_VERSION,
                        "classification": "WATCH",
                        "error": str(exc),
                    }
                )
                break
            except Exception as exc:
                # Do not mark the stage processed. A later tick/stage may retry.
                events.append(
                    {
                        "event_type": "PIPELINE_ERROR",
                        "stage": stage,
                        "fixture": fx,
                        "model_version": MODEL_VERSION,
                        "classification": "WATCH",
                        "error": str(exc)[:500],
                    }
                )
                continue

            deep_dive_processed += 1
            _mark_stage_processed(fx["fixture_id"], stage, now_utc)
            if event.get("market_skipped_by_sport_screen"):
                market_requests_avoided_by_screen += 1
            if (event.get("sporting_shortlist") or {}).get("shortlisted"):
                shortlist_events += 1
            events.append(event)

        if low_data_counts:
            events.append(
                {
                    "event_type": "LOW_DATA_SCREEN_SUMMARY",
                    "stage": "SLATE_SCREEN",
                    "classification": "PASS",
                    "model_version": MODEL_VERSION,
                    "screened_out_count": sum(low_data_counts.values()),
                    "by_tier_stage": dict(low_data_counts),
                    "notes": [
                        "Data Tier C/D fixtures remain counted in the slate but do not consume deep-dive API budget.",
                        "Detailed market/lineup refresh is reserved for Data Tier A/B sporting candidates.",
                    ],
                }
            )

        actionable = [
            e for e in events if e.get("stage") in {"T-40", "T-20", "T-10", "CLOSE"}
        ]
        bets = [e for e in events if e.get("classification") == "BET"]

        return {
            "service": "soccer-edge-automation",
            "version": AUTOMATION_VERSION,
            "model_version": MODEL_VERSION,
            "generated_at_utc": now_utc.isoformat(),
            "generated_at_local": local_now.isoformat(),
            "timezone": base.TIMEZONE_NAME,
            "fixture_scan_count": len(fixtures),
            "upcoming_market_capture_fixture_count": len(upcoming_market_capture_fixtures),
            "upcoming_market_capture_fixtures": upcoming_market_capture_fixtures,
            "due_fixture_count": len(due),
            "event_count": len(events),
            "actionable_refresh_count": len(actionable),
            "bet_candidate_count": len(bets),
            "api_calls_this_tick": v2._API_CALLS_THIS_TICK,
            "max_api_calls_per_tick": v2.MAX_API_CALLS_PER_TICK,
            "last_daily_remaining": v2._LAST_DAILY_REMAINING,
            "deferred_due_to_budget": deferred_due_to_budget,
            "deferred_due_to_priority": deferred_due_to_priority,
            "deep_dive_processed_count": deep_dive_processed,
            "shortlist_event_count": shortlist_events,
            "screened_out_low_data_count": sum(low_data_counts.values()),
            "market_requests_avoided_by_sport_screen": market_requests_avoided_by_screen,
            "player_props_xi_research_attempts": _PLAYER_PROPS_XI_RESEARCH_ATTEMPTS,
            "player_props_xi_research_captured": _PLAYER_PROPS_XI_RESEARCH_CAPTURED,
            "player_props_xi_research_max_per_tick": MAX_PLAYER_PROPS_XI_RESEARCH_FIXTURES_PER_TICK,
            "max_deep_dive_fixtures_per_tick": MAX_DEEP_DIVE_FIXTURES_PER_TICK,
            "priority_queue": "DATA_TIER_THEN_STAGE_THEN_PRIOR_SHORTLIST_THEN_COMPETITION_THEN_COVERAGE",
            "request_pacing_seconds": v4.MIN_REQUEST_INTERVAL_SECONDS,
            "rate_limit_max_retries": v4.RATE_LIMIT_MAX_RETRIES,
            "first_half_market_model": "BLOCKED_PENDING_EXPLICIT_1H_MODEL",
            "quota": quota,
            "events": events,
            "database_persistence": "OPTIONAL_NOT_REQUIRED_FOR_SCHEDULER",
        }
    finally:
        base._coverage = original_coverage
        if base._HTTP_CLIENT is not None:
            await base._HTTP_CLIENT.aclose()
            base._HTTP_CLIENT = None
        if base._CACHE_CONN is not None:
            base._CACHE_CONN.commit()
