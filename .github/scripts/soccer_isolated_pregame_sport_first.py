"""Bounded prospective SPORT-FIRST raw-goals projection outside Render.

Reuse the *exact* soccer_edge-mcp-v1/mcp_gateway/soccer_model.py mathematics.
No odds, markets, bet promotion, synthetic rows, postgame reconstruction, or
historical re-dating. Results are immutable research snapshots in GitHub state;
Postgres/OOS ingestion requires a separate authenticated as-of bridge.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any
import urllib.parse
import urllib.request

VERSION = "ISOLATED_SOCCER_SPORT_FIRST_PREGAME_v1.1.0"
BASE_URL = "https://v3.football.api-sports.io"
TIMEZONE = ZoneInfo("America/Mexico_City")
ALLOWED_ENDPOINTS = frozenset({"fixtures", "teams/statistics"})
MAX_PROVIDER_REQUESTS = 38  # 2 date slates + 2 verified season splits for at most 18 fixtures
MAX_FIXTURES = 18
MIN_VERIFIED_SPLIT_SAMPLE = 5
MIN_DAILY_REMAINING = 1000
MIN_KICKOFF_LEAD = timedelta(minutes=45)
MAX_KICKOFF_LOOKAHEAD = timedelta(hours=32)
MAX_SOURCE_SIZE = 1_000_000
OUTPUT_LIMIT = 5000


def _dt(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else None
    except ValueError:
        return None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
        return number if 0 <= number < 1000 and number == number else None
    except (TypeError, ValueError):
        return None


def _integer(value: Any) -> int | None:
    return value if type(value) is int and value > 0 else None


def _fixture(row: dict[str, Any], now: datetime) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    game = row.get("fixture") or {}
    league = row.get("league") or {}
    teams = row.get("teams") or {}
    home = teams.get("home") or {}
    away = teams.get("away") or {}
    kickoff = _dt(game.get("date"))
    fid = _integer(game.get("id"))
    lid = _integer(league.get("id"))
    season = league.get("season")
    home_id = _integer(home.get("id"))
    away_id = _integer(away.get("id"))
    status = (game.get("status") or {}).get("short")
    if (fid is None or lid is None or type(season) is not int or
            season < 2000 or home_id is None or away_id is None or
            home_id == away_id or kickoff is None or status != "NS"):
        return None
    if not MIN_KICKOFF_LEAD <= kickoff-now <= MAX_KICKOFF_LOOKAHEAD:
        return None
    return {
        "fixture_id": fid, "kickoff": kickoff.isoformat(),
        "league_id": lid, "league": str(league.get("name") or ""),
        "country": str(league.get("country") or ""), "season": season,
        "home_team_id": home_id, "home_team": str(home.get("name") or ""),
        "away_team_id": away_id, "away_team": str(away.get("name") or ""),
        "source_status": status,
    }


class OfficialProvider:
    def __init__(self, key: str):
        if not key:
            raise RuntimeError("API_FOOTBALL_KEY_NOT_CONFIGURED")
        self._key = key
        self.calls = 0
        self.remaining: int | None = None

    def get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        if endpoint not in ALLOWED_ENDPOINTS:
            raise RuntimeError("NON_SPORT_PROVIDER_ENDPOINT_REJECTED")
        if self.calls >= MAX_PROVIDER_REQUESTS:
            raise RuntimeError("SPORT_PROVIDER_CALL_CAP_EXCEEDED")
        if self.remaining is not None and self.remaining <= MIN_DAILY_REMAINING:
            raise RuntimeError("PROVIDER_DAILY_QUOTA_RESERVE")
        query = urllib.parse.urlencode(params)
        req = urllib.request.Request(
            f"{BASE_URL}/{endpoint}?{query}",
            headers={"x-apisports-key": self._key, "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            raw = response.read(MAX_SOURCE_SIZE + 1)
            quota_raw = response.headers.get("x-ratelimit-requests-remaining")
        self.calls += 1
        if len(raw) > MAX_SOURCE_SIZE:
            raise RuntimeError("OFFICIAL_SPORT_RESPONSE_TOO_LARGE")
        try:
            remaining = int(quota_raw)
        except (ValueError, TypeError):
            raise RuntimeError("PROVIDER_DAILY_QUOTA_NOT_VERIFIED") from None
        if remaining < MIN_DAILY_REMAINING:
            raise RuntimeError("PROVIDER_DAILY_QUOTA_BELOW_RESERVE")
        self.remaining = remaining if self.remaining is None else min(remaining, self.remaining)
        payload = json.loads(raw)
        if not isinstance(payload, dict) or not isinstance(payload.get("response"), (list, dict)):
            raise RuntimeError("UNVERIFIED_OFFICIAL_SPORT_RESPONSE")
        if payload.get("errors"):
            raise RuntimeError("API_FOOTBALL_RETURNED_ERRORS")
        return payload


def _valid_team_stats(payload: dict[str, Any], fx: dict[str, Any], team_key: str) -> dict[str, Any] | None:
    response = payload.get("response")
    if not isinstance(response, dict):
        return None
    team = response.get("team") or {}
    league = response.get("league") or {}
    if team.get("id") != fx[f"{team_key}_team_id"]:
        return None
    if league.get("id") != fx["league_id"] or league.get("season") != fx["season"]:
        return None
    goals = response.get("goals") or {}
    fixtures = response.get("fixtures") or {}
    which = "home" if team_key == "home" else "away"
    played = (fixtures.get("played") or {}).get(which)
    if type(played) is not int or played < 3:
        return None
    for section in ("for", "against"):
        rate = ((goals.get(section) or {}).get("average") or {}).get(which)
        if _number(rate) is None:
            return None
    # Exact canonical shape accepted by soccer_model.build_raw_projection
    return {"goals": {"for": goals.get("for"), "against": goals.get("against")},
            "fixtures": fixtures, "form": response.get("form")}


def _load_canonical_model(model_path: Path):
    if not model_path.is_file():
        raise RuntimeError("CANONICAL_SOCCER_MODEL_NOT_AVAILABLE")
    spec = importlib.util.spec_from_file_location("soccer_model_isolated_runtime", model_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("CANONICAL_SOCCER_MODEL_IMPORT_FAILED")
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    if not hasattr(model, "build_raw_projection") or not hasattr(model, "public_raw_projection"):
        raise RuntimeError("CANONICAL_SOCCER_MODEL_CONTRACT_MISSING")
    return model


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.stat().st_size > 20_000_000:
        raise RuntimeError("SPORT_RESEARCH_STATE_TOO_LARGE")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or _integer(value.get("fixture_id")) is None:
            raise RuntimeError("EXISTING_SPORT_STATE_INTEGRITY_FAILED")
        rows.append(value)
    if len(rows) > 5000:
        raise RuntimeError("SPORT_RESEARCH_REGISTRY_UNBOUNDED")
    return rows


def _build_one(fx: dict[str, Any], provider, model, model_sha: str,
               clock) -> dict[str, Any]:
    kickoff = _dt(fx["kickoff"])
    if kickoff is None or kickoff <= clock() + MIN_KICKOFF_LEAD:
        raise RuntimeError("KICKOFF_LEAD_NOT_VERIFIED")
    stats = {}
    for team_key in ("home", "away"):
        source = provider.get("teams/statistics", {
            "team": fx[f"{team_key}_team_id"],
            "league": fx["league_id"],
            "season": fx["season"],
        })
        stats[team_key] = _valid_team_stats(source, fx, team_key)
        if stats[team_key] is None:
            return {
                "fixture_id": fx["fixture_id"], "status": "NOT_MODELED",
                "reason": f"{team_key.upper()}_VERIFIED_SEASON_SPLITS_MISSING",
            }
    observed_at = clock()
    if observed_at >= kickoff or observed_at < kickoff-MAX_KICKOFF_LOOKAHEAD:
        raise RuntimeError("MODEL_NOT_ACTUALLY_CAPTURED_PREGAME")
    # No lineups were fetched: conservative uncertainty, no BET eligibility.
    raw = model.build_raw_projection(fx, {
        "home_stats": stats["home"], "away_stats": stats["away"],
        "home_recent": [], "away_recent": [],
    }, availability_confidence=0.50)
    if raw.get("status") != "MODELED_LIMITED":
        return {"fixture_id": fx["fixture_id"], "status": "NOT_MODELED",
                "reason": raw.get("reason") or "CANONICAL_SPORT_MODEL_DID_NOT_PROJECT"}
    public = model.public_raw_projection(raw)
    sample = public.get("sample") if isinstance(public.get("sample"), dict) else {}
    minimum = sample.get("minimum_split_sample")
    if type(minimum) is not int or minimum < MIN_VERIFIED_SPLIT_SAMPLE:
        return {
            "fixture_id": fx["fixture_id"], "status": "NOT_MODELED",
            "reason": "INSUFFICIENT_VERIFIED_HOME_AWAY_SPLIT_SAMPLE",
        }
    if not all(_number(public.get(key)) is not None for key in (
            "raw_home_goal_rate", "raw_away_goal_rate",
            "raw_home_win_prob", "raw_draw_prob", "raw_away_win_prob",
            "raw_over_2_5_prob", "raw_btts_yes_prob")):
        raise RuntimeError("CANONICAL_SPORT_PROJECTION_NOT_NUMERIC")
    if abs(sum(public[k] for k in ("raw_home_win_prob", "raw_draw_prob",
                                  "raw_away_win_prob")) - 1.0) > 0.001:
        raise RuntimeError("UNVERIFIED_SPORT_PROBABILITY_MASS")
    stamp = observed_at.isoformat()
    return {
        "schema_version": "1.0.0", "status": "FROZEN_RESEARCH_PREMATCH_SPORT_PROJECTION",
        "fixture_id": fx["fixture_id"], "kickoff": fx["kickoff"],
        "captured_at_utc": stamp, "generated_at_utc": stamp,
        "source": "LIVE_API_FOOTBALL_OFFICIAL_PREGAME_SPORT_INPUTS",
        "provider_observed_before_kickoff": True,
        "fixture": fx,
        "sporting_source": {
            "team_season_splits": stats,
            "recent_form": "NOT_COLLECTED_IN_LOW_COST_ISOLATED_PILOT",
            "injuries": "NOT VERIFIED", "lineups": "NOT VERIFIED",
            "availability_confidence": 0.50,
        },
        "model_version": str(public.get("model_version") or model.MODEL_VERSION),
        "projection_model": str(public.get("projection_model") or model.PROJECTION_MODEL),
        "canonical_runtime_model_sha": model_sha,
        "model_file_sha256": hashlib.sha256(Path(model.__file__).read_bytes()).hexdigest(),
        "raw_sport_projection": public,
        "raw_projection_before_market": True,
        "provider_odds_requests_added": 0,
        "historical_predictions_recomputed": False,
        "oos_validated": False, "clv_validated": False,
        "bet_eligible": False, "decision_weight": 0.0,
        "production_promotion_allowed": False,
        "model_inputs_missing": ["CONFIRMED_XI", "INJURIES", "RECENT_FORM",
                                 "ADVANCED_XG_METRICS"],
        "limitations": "CANONICAL_GOAL_RATE_POISSON_LIMITED: season-split baseline only, not an actionable sports edge",
    }


def _select_diverse_upcoming(choices: list[dict[str, Any]], limit: int, now: datetime) -> list[dict[str, Any]]:
    """SPORT schedule first: upcoming <=6h, then other games, diverse leagues.

    This only selects a hard-capped research sample. No odds or tier eligibility.
    """
    chosen: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for bucket in (
        [fx for fx in choices if (_dt(fx["kickoff"]) - now) <= timedelta(hours=6)],
        [fx for fx in choices if (_dt(fx["kickoff"]) - now) > timedelta(hours=6)],
    ):
        seen_leagues: set[int] = set()
        for fx in bucket:
            league_id = fx["league_id"]
            if league_id in seen_leagues or fx["fixture_id"] in seen_ids:
                continue
            chosen.append(fx)
            seen_leagues.add(league_id)
            seen_ids.add(fx["fixture_id"])
            if len(chosen) >= limit:
                return chosen
        for fx in bucket:
            if fx["fixture_id"] in seen_ids:
                continue
            chosen.append(fx)
            seen_ids.add(fx["fixture_id"])
            if len(chosen) >= limit:
                return chosen
    return chosen


def capture(provider, model, model_sha: str, existing: list[dict[str, Any]], *,
            clock, max_fixtures: int = MAX_FIXTURES):
    if not 1 <= max_fixtures <= MAX_FIXTURES:
        raise ValueError("UNBOUNDED_SPORT_FIXTURE_CAP")
    now = clock()
    if now.tzinfo is None:
        raise ValueError("NOW_MUST_BE_AWARE")
    local_today = now.astimezone(TIMEZONE).date()
    dates = [local_today, local_today + timedelta(days=1)]
    if len(dates) > 2:
        raise RuntimeError("UNEXPECTED_SLATE_DAY_COUNT")
    candidates = []
    for d in dates:
        response = provider.get("fixtures", {"date": d.isoformat(), "timezone": "America/Mexico_City"})
        rows = response.get("response") or []
        if not isinstance(rows, list) or len(rows) > OUTPUT_LIMIT:
            raise RuntimeError("SLATE_UNBOUNDED")
        candidates.extend(f for row in rows if (f := _fixture(row, now)) is not None)
    already = {r["fixture_id"] for r in existing}
    unique = {}
    for fx in candidates:
        unique.setdefault(fx["fixture_id"], fx)
    choices = sorted(
        (f for f in unique.values() if f["fixture_id"] not in already),
        key=lambda f: (f["kickoff"], f["fixture_id"]),
    )
    frozen, excluded = [], []
    for fx in _select_diverse_upcoming(choices, max_fixtures, now):
        result = _build_one(fx, provider, model, model_sha, clock)
        if result.get("status") == "FROZEN_RESEARCH_PREMATCH_SPORT_PROJECTION":
            frozen.append(result)
        else:
            excluded.append(result)
    if provider.calls > MAX_PROVIDER_REQUESTS:
        raise RuntimeError("PROVIDER_CALL_CAP_EXCEEDED")
    return {
        "status": "CAPTURED_RESEARCH_ONLY" if frozen else "NO_NEW_VALID_PREGAME_MODELS",
        "generated_at_utc": clock().isoformat(),
        "candidate_fixtures": len(choices),
        "selection_policy": "UPCOMING_WITHIN_6H_THEN_LATER;DISTINCT_LEAGUES_FIRST;NO_BET_TIER_ASSUMED",
        "attempted_fixtures": len(frozen)+len(excluded),
        "new_frozen_pregame_models": len(frozen),
        "excluded": excluded, "frozen": frozen,
        "provider_requests_added": provider.calls,
        "daily_remaining_reported": provider.remaining,
        "postgres_write_queries": 0,
        "provider_odds_requests_added": 0,
        "retroactive_pregame_signals_created": False,
        "production_promotion_allowed": False,
    }


def self_test(model_path: Path):
    model = _load_canonical_model(model_path)
    now = datetime(2026, 11, 9, 18, tzinfo=timezone.utc)
    fixture = {
        "fixture": {"id": 101, "date": (now + timedelta(hours=4)).isoformat(),
                    "status": {"short": "NS"}},
        "league": {"id": 55, "season": 2026, "name": "TEST_ONLY"},
        "teams": {"home": {"id": 5}, "away": {"id": 6}},
    }
    f = _fixture(fixture, now)
    assert f and f["fixture_id"] == 101
    assert _fixture({**fixture, "fixture":{**fixture["fixture"],"date":(now-timedelta(minutes=1)).isoformat()}}, now) is None
    assert _fixture({**fixture, "fixture":{**fixture["fixture"],"status":{"short":"FT"}}}, now) is None
    class Offline:
        calls = 0
        remaining = 7000
        def get(self, endpoint, params):
            assert endpoint in ALLOWED_ENDPOINTS
            self.calls += 1
            if endpoint == "fixtures":
                return {"response": [fixture]}
            team_id = params["team"]
            return {"response": {
                "team": {"id": team_id}, "league": {"id":55,"season":2026},
                "fixtures":{"played":{"home":6,"away":6,"total":12}},
                "goals":{
                    "for":{"average":{"home":"1.70","away":"1.20"}},
                    "against":{"average":{"home":"0.90","away":"1.30"}},
                },
            }}
    provider = Offline()
    audit = capture(provider, model, "fixed-test-repo-sha", [], clock=lambda:now)
    assert audit["status"] == "CAPTURED_RESEARCH_ONLY"
    assert audit["new_frozen_pregame_models"] == 1
    assert audit["provider_requests_added"] == 4
    record = audit["frozen"][0]
    assert record["captured_at_utc"] < record["kickoff"]
    assert record["raw_projection_before_market"] is True
    assert record["raw_sport_projection"]["status"] == "MODELED_LIMITED"
    assert record["bet_eligible"] is False
    assert record["provider_odds_requests_added"] == 0
    deduped = capture(Offline(),model,"fixed-test-repo-sha",[record],clock=lambda:now)
    assert deduped["new_frozen_pregame_models"] == 0
    bad = {**fixture,"teams":{"home":{"id":5},"away":{"id":5}}}
    assert _fixture(bad,now) is None
    sample = [
        {**f, "fixture_id":i,"league_id":55 if i < 109 else 56,
         "kickoff":(now+timedelta(hours=(2 if i<109 else 3))).isoformat()}
        for i in range(101,111)
    ]
    fair = _select_diverse_upcoming(sample, 3, now)
    assert len(fair)==3 and len({x["league_id"] for x in fair})==2
    assert fair[0]["fixture_id"]==101
    invalid = {"response":{"team":{"id":0},"league":{"id":55,"season":2026},
               "fixtures":{"played":{"home":6}}}}
    assert _valid_team_stats(invalid,f,"home") is None
    print("ISOLATED_SPORT_FIRST_OFFLINE_PASS: NO_MARKET, REAL_CANONICAL_MODEL, PREGAME, DEDUP, BOUNDED")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model-file",type=Path,required=True)
    parser.add_argument("--runtime-model-sha",required=True)
    parser.add_argument("--registry",type=Path)
    parser.add_argument("--receipt",type=Path)
    args = parser.parse_args()
    if not args.runtime_model_sha or len(args.runtime_model_sha) > 128:
        parser.error("model SHA must be nonempty and bounded")
    if args.self_test:
        self_test(args.model_file)
        return
    if args.registry is None or args.receipt is None:
        parser.error("--registry and --receipt required")
    model = _load_canonical_model(args.model_file)
    prior = _read_rows(args.registry)
    provider = OfficialProvider(os.environ.get("API_FOOTBALL_KEY", ""))
    result = capture(provider, model, args.runtime_model_sha, prior, clock=lambda:datetime.now(timezone.utc))
    if result["frozen"]:
        args.registry.parent.mkdir(parents=True,exist_ok=True)
        with args.registry.open("a",encoding="utf-8") as file:
            for row in result["frozen"]:
                file.write(json.dumps(row,sort_keys=True,ensure_ascii=False)+"\n")
    receipt = {k:v for k,v in result.items() if k!="frozen"}
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(receipt,sort_keys=True,ensure_ascii=False))


if __name__ == "__main__":
    main()
