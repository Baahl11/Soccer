"""Prospective SPORT ONLY pilot using the existing canonical v5 sporting path.

Deliberately does NOT call automation_v129.run_tick, price_resolver, odds,
market_evaluate, CLV, settlements, or any betting decision. Research only.
Run manually off Render; no persistent DB credentials or cron.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import re

MAX_PROVIDER_REQUESTS = 40
MIN_DAILY_REMAINING = 1000
MAX_CANDIDATES = 20
WINDOW_MINUTES = 180
KICKOFF_CUTOFF = timedelta(minutes=5)
OFFICIAL_HOSTS = {"v3.football.api-sports.io", "api-football-v1.p.rapidapi.com"}
ALLOWED_ENDPOINTS = {
    "/fixtures", "/leagues", "/teams/statistics", "/injuries", "/fixtures/lineups",
}
SENSITIVE_PHASES = {"odds", "predictions", "price", "market", "clv", "bet"}


class SportOnlyBudgetStop(RuntimeError):
    """Fail closed before any prohibited or excess network request."""


def allowed_provider_request(method: str, host: str, path: str) -> bool:
    return method.upper() == "GET" and host in OFFICIAL_HOSTS and path in ALLOWED_ENDPOINTS


def _utc(raw: str) -> datetime:
    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("TIMESTAMP_WITHOUT_TIMEZONE")
    return dt.astimezone(timezone.utc)


def _valid_raw(raw: dict) -> bool:
    keys = ("raw_home_win_prob", "raw_draw_prob", "raw_away_win_prob",
            "raw_btts_yes_prob", "raw_over_2_5_prob")
    vals = [raw.get(k) for k in keys]
    return (raw.get("status") == "MODELED_LIMITED"
            and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in vals)
            and abs(sum(vals[:3]) - 1) <= 0.002)


def _frozen_record(fx: dict, raw: dict, *, started: datetime, finished: datetime,
                   runtime_sha: str, shortlist: dict, availability: dict,
                   stage: str, feature_reads: dict, sporting_inputs: dict) -> dict | None:
    try:
        kickoff = _utc(fx["kickoff"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (started <= finished <= kickoff - KICKOFF_CUTOFF) or not _valid_raw(raw):
        return None
    if not (type(fx.get("fixture_id")) is int and fx["fixture_id"] > 0):
        return None
    return {
        "schema_version": "1.0.0",
        "status": "CANONICAL_V5_SPORT_ONLY_RESEARCH_LIMITED",
        "fixture_id": fx["fixture_id"],
        "kickoff_utc": kickoff.isoformat(),
        "sport_analysis_started_at_utc": started.isoformat(),
        "sport_analysis_finished_at_utc": finished.isoformat(),
        "feature_observation": feature_reads,
        "sporting_inputs": sporting_inputs,
        "sporting_inputs_sha256": hashlib.sha256(
            json.dumps(sporting_inputs, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "league_id": fx.get("league_id"),
        "home_team_id": fx.get("home_team_id"),
        "away_team_id": fx.get("away_team_id"),
        "source_runtime_commit": runtime_sha,
        "model_version": raw.get("model_version"),
        "projection_model": raw.get("projection_model"),
        "raw_sport_projection": raw,
        "sport_shortlist": shortlist,
        "source_stage": stage,
        "availability": availability,
        "model_depth": "SEASON_SPLITS_PLUS_RECENT_FORM; XI_WHEN_AVAILABLE; ADVANCED_XG_NOT_VERIFIED",
        "full_deep_dive_verified": False,
        "availability_verified_for_bet": False,
        "market_used_to_generate_raw_probability": False,
        "market_price_verified": False,
        "canonical_clv_eligible": False,
        "bet_eligible": False,
        "production_promotion_allowed": False,
        "retroactive_forecast_created": False,
    }


async def _sport_fixture(fx, coverage, stage, now, runtime_sha, v5, model):
    """Reuse actual canonical v5 screening/refinement, but NEVER its market event."""
    started = datetime.now(timezone.utc)
    cheap = await v5._cheap_sport_bundle(fx, now)
    initial = model.build_raw_projection(fx, cheap, None)
    initial_shortlist = v5._screen_shortlist(initial)
    refined = cheap
    shortlist = initial_shortlist
    if initial_shortlist.get("shortlisted"):
        refined = await v5._refine_sport_bundle(fx, cheap, now)
        # Only finished history strictly before the start of this model run.
        # No future/post-kickoff data from malformed provider/cache fixtures.
        refined = dict(refined)
        for side in ("home_recent", "away_recent"):
            clean = []
            for match in (refined.get(side) or []):
                try:
                    if (_utc(match["kickoff"]) < started
                            and match.get("status") in {"FT", "AET", "PEN"}):
                        clean.append(match)
                except (KeyError, TypeError, ValueError):
                    continue
            refined[side] = clean
        interim = model.build_raw_projection(fx, refined, None)
        shortlist = v5._screen_shortlist(interim)
    availability = {
        "lineup_state": "NOT VERIFIED", "goalkeepers": "NOT VERIFIED",
        "injuries": "NOT VERIFIED", "availability_confidence": 0.50,
    }
    if shortlist.get("shortlisted"):
        if stage in {"T-90", "T-60", "T-40", "T-20"} and coverage.get("injuries"):
            # Record whether a provider response exists, not a claim of health.
            injuries = await v5._injuries_2h(fx["fixture_id"], now)
            availability["injuries"] = "PROVIDER_RESPONSE" if isinstance(injuries, list) else "NOT VERIFIED"
        if stage in {"T-60", "T-40", "T-20", "T-10"} and coverage.get("lineups"):
            lineup = await v5._lineup_cached(fx["fixture_id"], now)
            confirmed = bool(lineup.get("both_xi_confirmed") and
                             lineup.get("both_goalkeepers_confirmed"))
            availability["lineup_state"] = "CONFIRMED_API" if confirmed else "NOT VERIFIED"
            availability["goalkeepers"] = "CONFIRMED_API" if confirmed else "NOT VERIFIED"
            availability["availability_confidence"] = 0.90 if confirmed else 0.60
    raw = model.public_raw_projection(model.build_raw_projection(
        fx, refined, availability["availability_confidence"]
    ))
    finished = datetime.now(timezone.utc)
    return _frozen_record(
        fx, raw, started=started, finished=finished, runtime_sha=runtime_sha,
        shortlist=shortlist, availability=availability, stage=stage,
        sporting_inputs={
            "home_stats": refined.get("home_stats"),
            "away_stats": refined.get("away_stats"),
            "home_recent": refined.get("home_recent"),
            "away_recent": refined.get("away_recent"),
        },
        feature_reads={
            "team_statistics": "FETCHED_OR_CACHE_AT_RUN_TIME",
            "recent_fixtures": "FETCHED_OR_CACHE_AT_RUN_TIME" if initial_shortlist.get("shortlisted") else "NOT_REQUESTED",
            "verified_source_updated_at": "NOT VERIFIED",
            "observation_not_after_utc": finished.isoformat(),
        },
    ), ("MODELED_LIMITED" if _valid_raw(raw) else "SPORT_MODEL_NOT_SUPPORTED")


async def run(runtime_sha: str, output: Path) -> dict:
    if not re.fullmatch("[0-9a-f]{40}", runtime_sha):
        raise RuntimeError("INVALID_PINNED_RUNTIME_SHA")
    if not os.getenv("API_FOOTBALL_KEY"):
        raise RuntimeError("API_FOOTBALL_KEY_MISSING")
    if os.getenv("DATABASE_URL"):
        raise RuntimeError("SPORT_ONLY_MUST_NOT_HAVE_DATABASE_URL")
    if os.getenv("SOCCER_EDGE_MAX_API_CALLS_PER_TICK") != str(MAX_PROVIDER_REQUESTS):
        raise RuntimeError("UNVERIFIED_GLOBAL_HTTP_BUDGET")

    import httpx
    from mcp_gateway import automation as base
    from mcp_gateway import automation_v2 as v2
    from mcp_gateway import automation_v3 as v3
    from mcp_gateway import automation_v5 as v5
    from mcp_gateway import soccer_model as model

    # Independent disposable process and cache. Protect the transport itself
    # against a helper that bypasses base._api_get.
    traffic = {"outgoing": 0, "denied": 0, "quota_headers": 0,
               "last_daily_remaining": None, "by_path": {}}
    original_send = httpx.AsyncClient.send
    original_get = base._api_get

    async def limited_send(client, request, *args, **kwargs):
        if not allowed_provider_request(request.method, request.url.host, request.url.path):
            traffic["denied"] += 1
            raise SportOnlyBudgetStop("SPORT_ONLY_NON_SPORT_NETWORK_BLOCKED")
        if traffic["outgoing"] >= MAX_PROVIDER_REQUESTS:
            traffic["denied"] += 1
            raise SportOnlyBudgetStop("SPORT_ONLY_GLOBAL_PROVIDER_CAP_REACHED")
        remaining = traffic["last_daily_remaining"]
        if remaining is not None and remaining < MIN_DAILY_REMAINING:
            raise SportOnlyBudgetStop("SPORT_ONLY_DAILY_QUOTA_RESERVE_REACHED")
        traffic["outgoing"] += 1
        path = request.url.path
        traffic["by_path"][path] = traffic["by_path"].get(path, 0) + 1
        response = await original_send(client, request, *args, **kwargs)
        rem = response.headers.get("x-ratelimit-requests-remaining")
        if rem is not None and rem.isdigit():
            traffic["quota_headers"] += 1
            traffic["last_daily_remaining"] = int(rem)
            if int(rem) < MIN_DAILY_REMAINING:
                raise SportOnlyBudgetStop("SPORT_ONLY_DAILY_QUOTA_RESERVE_REACHED")
        else:
            raise SportOnlyBudgetStop("SPORT_ONLY_DAILY_QUOTA_HEADER_NOT_VERIFIED")
        return response

    async def only_sport(endpoint, params):
        if "/" + endpoint.lstrip("/") not in ALLOWED_ENDPOINTS:
            raise SportOnlyBudgetStop("SPORT_ONLY_FORBIDDEN_ENDPOINT")
        return await v2._ORIGINAL_API_GET(endpoint, params)

    httpx.AsyncClient.send = limited_send
    base._api_get = only_sport
    base._HTTP_CLIENT = httpx.AsyncClient(
        timeout=base.TIMEOUT,
        limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
    )
    rows, dropped = [], {}
    scanned, due_ab, attempts = 0, 0, 0
    now = datetime.now(timezone.utc)
    try:
        # Full current-day slate is counted. Candidate admission is bounded and
        # does not depend on bookmaker quotes or their availability.
        local = now.astimezone(base.TIMEZONE)
        dates = [local.date()]
        if local.hour >= 21:
            dates.append((local + timedelta(days=1)).date())
        fixtures = []
        seen = set()
        for date in dates:
            slate = await only_sport("fixtures", {
                "date": date.isoformat(), "timezone": base.TIMEZONE_NAME,
            })
            for item in slate.get("response", []):
                fx = base._compact_fixture(item)
                fid = fx.get("fixture_id")
                if type(fid) is int and fid > 0 and fid not in seen:
                    fixtures.append(fx)
                    seen.add(fid)
        scanned = len(fixtures)
        candidates = []
        for fx in fixtures:
            try:
                kick = _utc(fx["kickoff"])
                if not (now + KICKOFF_CUTOFF < kick <= now + timedelta(minutes=WINDOW_MINUTES)):
                    continue
                if fx.get("status") in base.CANCELLED_STATUSES | base.POSTPONED_STATUSES | base.FINISHED_STATUSES:
                    continue
                coverage = await v3._coverage_fast(fx["league_id"], fx["season"], now)
                if coverage.get("data_tier") not in ("A", "B"):
                    continue
                due_ab += 1
                minutes = (kick - now).total_seconds() / 60
                stage = base._stage_for(minutes, fx.get("status") or "") or "EARLY_RESEARCH"
                candidates.append((kick, fx, coverage, stage))
            except (KeyError, TypeError, ValueError):
                dropped["INVALID_FIXTURE_OR_KICKOFF"] = dropped.get("INVALID_FIXTURE_OR_KICKOFF", 0) + 1
        candidates.sort(key=lambda it: it[0])
        for _, fx, coverage, stage in candidates[:MAX_CANDIDATES]:
            if traffic["outgoing"] >= MAX_PROVIDER_REQUESTS - 6:
                dropped["NOT_ANALYZED_RESOURCE_CAP"] = (
                    dropped.get("NOT_ANALYZED_RESOURCE_CAP", 0) + 1
                )
                break
            attempts += 1
            try:
                row, reason = await _sport_fixture(fx, coverage, stage, now, runtime_sha, v5, model)
            except SportOnlyBudgetStop:
                dropped["NOT_ANALYZED_RESOURCE_CAP"] = dropped.get("NOT_ANALYZED_RESOURCE_CAP", 0) + 1
                break
            except Exception as exc:
                dropped["SPORT_PROCESSING_ERROR_" + type(exc).__name__] = (
                    dropped.get("SPORT_PROCESSING_ERROR_" + type(exc).__name__, 0) + 1
                )
                continue
            if row is None:
                dropped[reason] = dropped.get(reason, 0) + 1
            else:
                rows.append(row)
        if len(candidates) > MAX_CANDIDATES:
            dropped["NOT_ANALYZED_CANDIDATE_CAP"] = len(candidates) - MAX_CANDIDATES
    finally:
        base._api_get = original_get
        await base._HTTP_CLIENT.aclose()
        base._HTTP_CLIENT = None
        httpx.AsyncClient.send = original_send
        if base._CACHE_CONN is not None:
            base._CACHE_CONN.commit()
            base._CACHE_CONN.close()
            base._CACHE_CONN = None

    # Global safety proof. Do not write any observations unless the complete
    # run is clean and the official provider quota header was independently read.
    if (traffic["denied"] or traffic["outgoing"] > MAX_PROVIDER_REQUESTS
            or traffic["quota_headers"] == 0
            or traffic["last_daily_remaining"] is None
            or traffic["last_daily_remaining"] < MIN_DAILY_REMAINING):
        raise RuntimeError("SPORT_ONLY_PROVIDER_PROOF_FAILED_NO_OUTPUT_SAVED")
    report = {
        "schema_version": "1.0.0", "status": "SPORT_ONLY_RESEARCH_CANARY",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_sha": runtime_sha, "fixture_scan_count": scanned,
        "tier_ab_due_in_180m": due_ab, "candidate_attempts": attempts,
        "real_frozen_raw_count": len(rows), "full_deep_dives_verified": 0,
        "drop_reasons": dropped, "actual_provider_calls": traffic["outgoing"],
        "provider_calls_by_endpoint": traffic["by_path"],
        "transport_budget_denied": traffic["denied"],
        "quota_headers_verified": traffic["quota_headers"],
        "last_daily_remaining": traffic["last_daily_remaining"],
        "model_stages": "CANONICAL_V5_CHEAP_SCREEN_REFINED_FORM_AVAILABILITY_RAW",
        "markets_queried": 0, "price_resolver_invocations": 0, "clv_rows_added": 0,
        "db_writes": 0, "bet_eligible": False, "production_promotion_allowed": False,
        "raw_research_snapshots": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report


def self_test():
    assert allowed_provider_request("GET", "v3.football.api-sports.io", "/teams/statistics")
    assert not allowed_provider_request("GET", "v3.football.api-sports.io", "/odds")
    assert not allowed_provider_request("GET", "evil.example", "/fixtures")
    assert not allowed_provider_request("POST", "v3.football.api-sports.io", "/fixtures")
    now = datetime(2026, 11, 10, 17, tzinfo=timezone.utc)
    fx = {"fixture_id": 1, "kickoff": (now + timedelta(hours=1)).isoformat()}
    raw = {
        "status": "MODELED_LIMITED", "raw_home_win_prob": .45,
        "raw_draw_prob": .25, "raw_away_win_prob": .30,
        "raw_btts_yes_prob": .5, "raw_over_2_5_prob": .6,
    }
    kw = dict(started=now, finished=now + timedelta(minutes=1),
              runtime_sha="a"*40, shortlist={"shortlisted": True},
              availability={"availability_confidence": .5},
              stage="T-60", feature_reads={}, sporting_inputs={})
    row = _frozen_record(fx, raw, **kw)
    assert row and not row["bet_eligible"] and row["full_deep_dive_verified"] is False
    assert row["market_used_to_generate_raw_probability"] is False
    assert _frozen_record(fx, raw, **{**kw, "finished": now + timedelta(hours=1)}) is None
    assert _frozen_record(fx, {**raw, "raw_home_win_prob": float("nan")}, **kw) is None
    assert _frozen_record({**fx, "kickoff": (now-timedelta(seconds=1)).isoformat()}, raw, **kw) is None

    class FakeV5:
        @staticmethod
        async def _cheap_sport_bundle(fx, now):
            return {"sport_data": "AVAILABLE"}
        @staticmethod
        async def _refine_sport_bundle(fx, cheap, now):
            return {"sport_data": "AVAILABLE", "refined": True}
        @staticmethod
        def _screen_shortlist(raw):
            return {"shortlisted": True}
        @staticmethod
        async def _injuries_2h(fid, now):
            return []
        @staticmethod
        async def _lineup_cached(fid, now):
            return {"both_xi_confirmed": False, "both_goalkeepers_confirmed": False}

    class FakeModel:
        @staticmethod
        def build_raw_projection(fx, sporting, confidence):
            return raw
        @staticmethod
        def public_raw_projection(raw):
            return raw

    future_fx = {**fx, "kickoff": (datetime.now(timezone.utc)
                 + timedelta(hours=1)).isoformat()}
    row, reason = asyncio.run(_sport_fixture(
        future_fx, {"injuries": True, "lineups": True}, "T-40",
        datetime.now(timezone.utc), "a"*40, FakeV5, FakeModel))
    assert reason == "MODELED_LIMITED" and row is not None
    assert row["availability"]["lineup_state"] == "NOT VERIFIED"
    assert row["availability_verified_for_bet"] is False
    assert row["raw_sport_projection"] == raw
    assert len(row["sporting_inputs_sha256"]) == 64
    assert row["sporting_inputs"]["home_recent"] == []
    print("SPORT_ONLY_OFFLINE_CONTRACT_PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--runtime-sha", default="")
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    if a.output is None:
        ap.error("--output required for live run")
    report = asyncio.run(run(a.runtime_sha, a.output))
    print(json.dumps({k: v for k, v in report.items() if k != "raw_research_snapshots"},
                     sort_keys=True))


if __name__ == "__main__":
    main()
