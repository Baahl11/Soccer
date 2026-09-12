from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone as dt_timezone
from typing import Any

import httpx

from mcp_gateway import automation as base
from mcp_gateway.soccer_model import build_raw_projection, evaluate_market, public_raw_projection

SPORTING_STAGES = {"T-90", "T-60", "T-40", "T-20", "T-10"}
MARKET_STAGES = {"T-40", "T-20", "T-10", "CLOSE"}
LINEUP_STAGES = {"T-60", "T-40", "T-30", "T-20", "T-10"}
INJURY_STAGES = {"T-90", "T-60", "T-40", "T-20"}
MAX_API_CALLS_PER_TICK = int(os.getenv("SOCCER_EDGE_MAX_API_CALLS_PER_TICK", "35"))
_API_CALLS_THIS_TICK = 0
_LAST_DAILY_REMAINING: int | None = None
_ORIGINAL_API_GET = base._api_get


class TickBudgetExceeded(RuntimeError):
    pass


async def _budgeted_api_get(endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
    global _API_CALLS_THIS_TICK, _LAST_DAILY_REMAINING
    if _API_CALLS_THIS_TICK >= MAX_API_CALLS_PER_TICK:
        raise TickBudgetExceeded(
            f"Per-tick API budget reached ({MAX_API_CALLS_PER_TICK}); lower-priority work deferred."
        )
    if _LAST_DAILY_REMAINING is not None and _LAST_DAILY_REMAINING <= 50:
        raise TickBudgetExceeded("Daily API reserve guard reached; lower-priority work deferred.")
    _API_CALLS_THIS_TICK += 1
    payload = await _ORIGINAL_API_GET(endpoint, params)
    remaining = (payload.get("quota") or {}).get("daily_remaining")
    try:
        if remaining is not None:
            _LAST_DAILY_REMAINING = int(remaining)
    except (TypeError, ValueError):
        pass
    return payload


# All API calls made by the base helper functions in this short-lived worker are
# routed through the same hard budget. At 35 calls/tick and 10-minute cadence,
# the theoretical scheduler maximum is 5,040/day, leaving substantial room from
# a 7,500/day Pro quota for interactive calls, failures and unusual slates.
base._api_get = _budgeted_api_get


async def _event_for_fixture(fx: dict[str, Any], stage: str, now: datetime) -> dict[str, Any]:
    coverage = await base._coverage(fx["league_id"], fx["season"], now)
    event: dict[str, Any] = {
        "event_type": "SOCCER_REFRESH",
        "stage": stage,
        "fixture": fx,
        "coverage": coverage,
        "model_version": "SOCCER EDGE ENGINE v1.0",
        "classification": "WATCH",
        "tier": None,
        "stake_units": 0.0,
        "bet_eligible": False,
        "availability_confidence": None,
        "notes": [],
    }

    # Automated BET requires Data Tier A/B. Do not spend scarce API quota doing
    # team/lineup/odds deep dives for C/D fixtures that cannot pass that gate.
    if coverage.get("data_tier") not in {"A", "B"}:
        event["classification"] = "PASS" if coverage.get("data_tier") == "D" else "WATCH"
        event["notes"].append("Automated deep dive skipped: Data Tier A/B required for automated BET eligibility.")
        return event

    # SPORT FIRST. Sporting inputs and raw projection are created before market retrieval.
    sporting: dict[str, Any] | None = None
    if stage in SPORTING_STAGES:
        sporting = await base._sport_bundle(fx, coverage, now)
        event["sporting"] = sporting

    if stage in INJURY_STAGES and coverage.get("injuries"):
        event["injuries"] = base._compact_injuries(await base._api_get("injuries", {"fixture": fx["fixture_id"]}))
    elif stage in INJURY_STAGES:
        event["injuries"] = "NOT VERIFIED"

    if stage in LINEUP_STAGES and coverage.get("lineups"):
        lineup = base._compact_lineups(await base._api_get("fixtures/lineups", {"fixture": fx["fixture_id"]}))
        event["lineups"] = lineup
        if lineup["both_xi_confirmed"] and lineup["both_goalkeepers_confirmed"]:
            event["availability_confidence"] = 0.90
        else:
            event["availability_confidence"] = 0.70 if stage in {"T-60", "T-40"} else 0.60
            event["notes"].append("Material lineup/goalkeeper information remains NOT VERIFIED.")
    elif stage in LINEUP_STAGES:
        event["lineups"] = "NOT VERIFIED"
        event["availability_confidence"] = 0.60

    raw_internal: dict[str, Any] | None = None
    if sporting and sporting.get("sport_data") == "AVAILABLE":
        raw_internal = build_raw_projection(fx, sporting, event.get("availability_confidence"))
        event["raw_projection"] = public_raw_projection(raw_internal)
        if raw_internal.get("status") == "MODELED_LIMITED":
            event["notes"].append(
                "Automated raw projection is LIMITED: xG/npxG/PPDA/field tilt remain NOT VERIFIED; goals are not substituted as xG."
            )
        else:
            event["notes"].append("Raw projection could not be generated from verified sporting inputs.")

    # Market retrieval occurs only after the raw sporting projection has been built.
    if stage in MARKET_STAGES and coverage.get("odds"):
        event["market"] = base._compact_odds(await base._api_get("odds", {"fixture": fx["fixture_id"], "page": 1}))
        event["market_use"] = "MARKET_COMPARISON_AFTER_RAW_PROJECTION"
    elif stage in MARKET_STAGES:
        event["market"] = "NOT VERIFIED"

    if stage in {"T-40", "T-20", "T-10"}:
        if raw_internal is None:
            event["market_decision"] = {"status": "WATCH", "reason": "RAW_PROJECTION_NOT_AVAILABLE", "decisions": []}
        else:
            market_decision = evaluate_market(
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
                event["best_market"] = {
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

    if stage == "T-20":
        lineup = event.get("lineups")
        if not isinstance(lineup, dict) or not lineup.get("both_xi_confirmed") or not lineup.get("both_goalkeepers_confirmed"):
            event["bet_eligible"] = False
            event["classification"] = "WATCH"
            event["notes"].append("T-20 lineup/GK gate failed: BET eligibility blocked.")

    if stage == "POSTGAME":
        if coverage.get("statistics_fixtures"):
            stats = await base._api_get("fixtures/statistics", {"fixture": fx["fixture_id"]})
            event["match_stats"] = stats.get("response", [])
        event["result"] = {"goals": fx.get("goals"), "score": fx.get("score"), "status": fx.get("status")}
        event["classification"] = "POSTGAME"

    if stage == "CLOSE":
        event["classification"] = "CLOSE"

    return event


async def run_tick() -> dict[str, Any]:
    global _API_CALLS_THIS_TICK, _LAST_DAILY_REMAINING
    _API_CALLS_THIS_TICK = 0
    _LAST_DAILY_REMAINING = None

    now_utc = datetime.now(dt_timezone.utc)
    local_now = now_utc.astimezone(base.TIMEZONE)
    base._prune_cache(now_utc)

    dates = [local_now.date()]
    if local_now.hour >= 22:
        dates.append((local_now + timedelta(days=1)).date())

    fixtures: list[dict[str, Any]] = []
    quota: dict[str, Any] = {}

    base._HTTP_CLIENT = httpx.AsyncClient(
        timeout=base.TIMEOUT,
        limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
    )
    try:
        for d in dates:
            payload = await base._api_get("fixtures", {"date": d.isoformat(), "timezone": base.TIMEZONE_NAME})
            quota = payload.get("quota", quota)
            for row in payload.get("response", []):
                fx = base._compact_fixture(row)
                if fx.get("fixture_id") and fx.get("kickoff"):
                    fixtures.append(fx)

        events: list[dict[str, Any]] = []

        if local_now.hour == 6 and local_now.minute < 15:
            upcoming = []
            for fx in fixtures:
                kickoff = base._dt(fx["kickoff"])
                if kickoff >= now_utc and fx.get("status") not in base.CANCELLED_STATUSES | base.POSTPONED_STATUSES:
                    upcoming.append(fx)
            events.append({
                "event_type": "DAILY_DISCOVERY",
                "stage": "MORNING",
                "date": local_now.date().isoformat(),
                "timezone": base.TIMEZONE_NAME,
                "upcoming_count": len(upcoming),
                "fixtures": upcoming,
                "classification": "PRE-FINAL",
                "sport_first": True,
                "market_data_included": False,
                "model_version": "SOCCER EDGE ENGINE v1.0",
                "notes": [
                    "Complete configured slate discovered. Resource-aware sporting/model screens run automatically as each fixture enters its pregame window."
                ],
            })

        due: list[tuple[int, datetime, dict[str, Any], str]] = []
        priority = {"T-40": 0, "T-20": 1, "T-10": 2, "CLOSE": 3, "T-60": 4, "T-90": 5, "T-30": 6, "POSTGAME": 7}
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
            due.append((priority.get(stage, 99), kickoff, fx, stage))

        due.sort(key=lambda item: (item[0], item[1]))
        deferred_due_to_budget = 0
        for _, _, fx, stage in due:
            if not base._dedupe_stage(fx["fixture_id"], stage, now_utc):
                continue
            try:
                events.append(await _event_for_fixture(fx, stage, now_utc))
            except TickBudgetExceeded as exc:
                deferred_due_to_budget += 1
                events.append({
                    "event_type": "QUOTA_GUARD",
                    "stage": stage,
                    "fixture": fx,
                    "model_version": "SOCCER EDGE ENGINE v1.0",
                    "classification": "WATCH",
                    "error": str(exc),
                })
                break
            except Exception as exc:
                events.append({
                    "event_type": "PIPELINE_ERROR",
                    "stage": stage,
                    "fixture": fx,
                    "model_version": "SOCCER EDGE ENGINE v1.0",
                    "classification": "WATCH",
                    "error": str(exc)[:500],
                })

        actionable = [e for e in events if e.get("stage") in {"T-40", "T-20", "T-10", "CLOSE"}]
        bets = [e for e in events if e.get("classification") == "BET"]
        return {
            "service": "soccer-edge-automation",
            "version": "1.2.1",
            "model_version": "SOCCER EDGE ENGINE v1.0",
            "generated_at_utc": now_utc.isoformat(),
            "generated_at_local": local_now.isoformat(),
            "timezone": base.TIMEZONE_NAME,
            "fixture_scan_count": len(fixtures),
            "due_fixture_count": len(due),
            "event_count": len(events),
            "actionable_refresh_count": len(actionable),
            "bet_candidate_count": len(bets),
            "api_calls_this_tick": _API_CALLS_THIS_TICK,
            "max_api_calls_per_tick": MAX_API_CALLS_PER_TICK,
            "last_daily_remaining": _LAST_DAILY_REMAINING,
            "deferred_due_to_budget": deferred_due_to_budget,
            "events": events,
            "quota": quota,
            "database_persistence": "OPTIONAL_NOT_REQUIRED_FOR_SCHEDULER",
        }
    finally:
        if base._HTTP_CLIENT is not None:
            await base._HTTP_CLIENT.aclose()
            base._HTTP_CLIENT = None
        if base._CACHE_CONN is not None:
            base._CACHE_CONN.commit()
