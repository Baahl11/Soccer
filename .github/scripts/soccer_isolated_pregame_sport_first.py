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
import tempfile
from pathlib import Path
from typing import Any
import urllib.parse
import urllib.request

VERSION = "ISOLATED_SOCCER_SPORT_FIRST_PREGAME_v1.4.0"
BASE_URL = "https://v3.football.api-sports.io"
TIMEZONE = ZoneInfo("America/Mexico_City")
ALLOWED_ENDPOINTS = frozenset({"fixtures", "teams/statistics"})
MAX_PROVIDER_REQUESTS = 38  # 2 date slates + 2 verified season splits for at most 18 fixtures
MAX_FIXTURES = 18
MIN_VERIFIED_SPLIT_SAMPLE = 5
MIN_DAILY_REMAINING = 1500  # 20% of 7,500 API-Football Pro assumption
MIN_KICKOFF_LEAD = timedelta(minutes=45)
MAX_KICKOFF_LOOKAHEAD = timedelta(hours=32)
MAX_SOURCE_SIZE = 1_000_000
OUTPUT_LIMIT = 5000
# Private, in-process memoization and read-only reuse of original frozen
# season-split snapshots. NEVER create a new public provider-data cache file.
TEAM_SPLIT_CACHE_TTL = timedelta(hours=12)
MAX_TEAM_SPLIT_CACHE_ENTRIES = 3000
NEGATIVE_COOLDOWN = timedelta(hours=8)
NEGATIVE_HISTORY_RETENTION = timedelta(hours=36)
NEGATIVE_CACHE_MAX_ENTRIES = 2000
NEGATIVE_REASONS = frozenset({
    "HOME_VERIFIED_SEASON_SPLITS_MISSING",
    "AWAY_VERIFIED_SEASON_SPLITS_MISSING",
    "INSUFFICIENT_VERIFIED_HOME_AWAY_SPLIT_SAMPLE",
})


def _read_negative_cache(path: Path) -> dict[int, dict[str, str]]:
    """Read small source-negative metadata, not stats or model projections."""
    if not path.exists():
        return {}
    if path.stat().st_size > MAX_SOURCE_SIZE:
        raise RuntimeError("SPORT_NEGATIVE_CACHE_UNBOUNDED")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != "1.0":
        raise RuntimeError("SPORT_NEGATIVE_CACHE_SCHEMA_INVALID")
    entries = data.get("entries")
    if not isinstance(entries, dict) or len(entries) > NEGATIVE_CACHE_MAX_ENTRIES:
        raise RuntimeError("SPORT_NEGATIVE_CACHE_ENTRIES_INVALID")
    result: dict[int, dict[str, str]] = {}
    for fid, record in entries.items():
        if not isinstance(fid, str) or not fid.isdecimal() or not isinstance(record, dict):
            continue
        numeric = int(fid)
        if not 0 < numeric < 10**12:
            continue
        retry = _dt(record.get("retry_after_utc"))
        if record.get("reason") not in NEGATIVE_REASONS or retry is None:
            continue
        result[numeric] = {
            "reason": record["reason"],
            "retry_after_utc": retry.isoformat(),
        }
    return result


def _write_negative_cache(
    path: Path, current: dict[int, dict[str, str]],
    excluded: list[dict[str, Any]], frozen: list[dict[str, Any]],
    *, now: datetime,
) -> None:
    """Expiry-bounded and idempotent; no source payload or personal data stored."""
    entries: dict[int, dict[str, str]] = {
        fid: record for fid, record in current.items()
        if _dt(record.get("retry_after_utc")) and
           _dt(record["retry_after_utc"]) > now - NEGATIVE_HISTORY_RETENTION
    }
    for row in excluded:
        fid = row.get("fixture_id")
        reason = row.get("reason")
        if type(fid) is int and fid > 0 and reason in NEGATIVE_REASONS:
            entries[fid] = {
                "reason": reason,
                "retry_after_utc": (now + NEGATIVE_COOLDOWN).isoformat(),
            }
    for row in frozen:
        fid = row.get("fixture_id")
        if type(fid) is int:
            entries.pop(fid, None)
    if len(entries) > NEGATIVE_CACHE_MAX_ENTRIES:
        sorted_entries = sorted(
            entries.items(), key=lambda item: item[1]["retry_after_utc"],
            reverse=True,
        )
        entries = dict(sorted_entries[:NEGATIVE_CACHE_MAX_ENTRIES])
    doc = {
        "schema_version": "1.0",
        "generated_at_utc": now.isoformat(),
        "source": "SPORT_INPUT_NEGATIVE_REASON_ONLY",
        "model_predictions_recomputed": False,
        "provider_requests_added": 0,
        "entries": {str(key): value for key, value in sorted(entries.items())},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, sort_keys=True, indent=2)+"\n",encoding="utf-8")



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


class TeamSeasonSplitCache:
    """One sport-input key per (team, competition, season, home/away venue).

    Only reuse verified official team stats captured BEFORE this fixture's
    kickoff and within 12h of now. Seed from original pregame research rows
    already loaded into memory; do not write/duplicate provider data to a
    publicly downloadable GitHub cache. An observed source is not a new model.
    """
    def __init__(self, originals: list[dict[str, Any]] | None = None):
        self.entries: dict[tuple[int, int, int, str], dict[str, Any]] = {}
        self.hits = 0
        self.from_prior_snapshot = 0
        self.from_current_capture = 0
        self.misses = 0
        self.puts = 0
        for row in (originals or [])[-15000:]:
            if not isinstance(row, dict):
                continue
            fx = row.get("fixture") or {}
            if (row.get("status") != "FROZEN_RESEARCH_PREMATCH_SPORT_PROJECTION"
                    or row.get("provider_observed_before_kickoff") is not True
                    or row.get("raw_projection_before_market") is not True
                    or row.get("provider_odds_requests_added") != 0
                    or not isinstance(fx, dict)):
                continue
            at = _dt(row.get("captured_at_utc"))
            original_kick = _dt(fx.get("kickoff"))
            if at is None or original_kick is None or at >= original_kick:
                continue
            source = row.get("sporting_source") or {}
            splits = source.get("team_season_splits") or {}
            if not isinstance(splits, dict):
                continue
            for side in ("home", "away"):
                compact = splits.get(side)
                if not isinstance(compact, dict):
                    continue
                key = self.key(fx, side)
                if key is None or _verify_cached_split(compact, fx, side) is None:
                    continue
                previous = self.entries.get(key)
                if previous is None or at > previous["observed"]:
                    self.entries[key] = {
                        "observed": at, "source": "ORIGINAL_FROZEN_PREGAME",
                        "stats": compact,
                    }
        if len(self.entries) > MAX_TEAM_SPLIT_CACHE_ENTRIES:
            self.entries = dict(sorted(
                self.entries.items(), key=lambda x: x[1]["observed"],
                reverse=True,
            )[:MAX_TEAM_SPLIT_CACHE_ENTRIES])

    @staticmethod
    def key(fx: dict[str, Any], side: str) -> tuple[int, int, int, str] | None:
        vals = (
            fx.get(f"{side}_team_id"), fx.get("league_id"), fx.get("season"),
        )
        if side not in ("home", "away") or any(type(x) is not int or x <= 0 for x in vals):
            return None
        return (*vals, side)

    def get(self, fx: dict[str, Any], side: str, at: datetime) -> dict[str, Any] | None:
        key = self.key(fx, side)
        entry = self.entries.get(key) if key else None
        kickoff = _dt(fx.get("kickoff"))
        if (entry is None or kickoff is None or at.tzinfo is None
                or at < entry["observed"]
                or at - entry["observed"] >= TEAM_SPLIT_CACHE_TTL
                or entry["observed"] >= kickoff):
            self.misses += 1
            return None
        stats = _verify_cached_split(entry["stats"], fx, side)
        if stats is None:
            self.misses += 1
            return None
        self.hits += 1
        if entry["source"] == "ORIGINAL_FROZEN_PREGAME":
            self.from_prior_snapshot += 1
        else:
            self.from_current_capture += 1
        return stats

    def put(self, fx: dict[str, Any], side: str, stats: dict[str, Any], at: datetime) -> None:
        key = self.key(fx, side)
        kickoff = _dt(fx.get("kickoff"))
        if (key is None or kickoff is None or at.tzinfo is None or at >= kickoff
                or _verify_cached_split(stats, fx, side) is None):
            return
        if len(self.entries) >= MAX_TEAM_SPLIT_CACHE_ENTRIES and key not in self.entries:
            return
        old = self.entries.get(key)
        if old is None or old["observed"] <= at:
            self.entries[key] = {"observed": at, "source": "SAME_CAPTURE", "stats": stats}
            self.puts += 1


def _verify_cached_split(stats: Any, fx: dict[str, Any], side: str) -> dict[str, Any] | None:
    """Recheck original venue/season model requirements on every cache read."""
    if not isinstance(stats, dict):
        return None
    payload = {
        "response": {
            "team": {"id": fx.get(f"{side}_team_id")},
            "league": {"id": fx.get("league_id"), "season": fx.get("season")},
            "goals": stats.get("goals"),
            "fixtures": stats.get("fixtures"),
            "form": stats.get("form"),
        }
    }
    return _valid_team_stats(payload, fx, side)


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
               clock, split_cache: TeamSeasonSplitCache | None = None) -> dict[str, Any]:
    kickoff = _dt(fx["kickoff"])
    if kickoff is None or kickoff <= clock() + MIN_KICKOFF_LEAD:
        raise RuntimeError("KICKOFF_LEAD_NOT_VERIFIED")
    stats = {}
    for team_key in ("home", "away"):
        cached = split_cache.get(fx, team_key, clock()) if split_cache else None
        if cached is not None:
            stats[team_key] = cached
        else:
            source = provider.get("teams/statistics", {
                "team": fx[f"{team_key}_team_id"],
                "league": fx["league_id"],
                "season": fx["season"],
            })
            stats[team_key] = _valid_team_stats(source, fx, team_key)
            if stats[team_key] is not None and split_cache is not None:
                split_cache.put(fx, team_key, stats[team_key], clock())
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
            clock, max_fixtures: int = MAX_FIXTURES,
            negative_cache: dict[int, dict[str, str]] | None = None):
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
    cached_negative = negative_cache if isinstance(negative_cache, dict) else {}
    deferred = {
        fid for fid, record in cached_negative.items()
        if type(fid) is int and isinstance(record, dict)
        and record.get("reason") in NEGATIVE_REASONS
        and (retry := _dt(record.get("retry_after_utc"))) is not None and retry > now
    }
    unique = {}
    for fx in candidates:
        unique.setdefault(fx["fixture_id"], fx)
    choices = sorted(
        (f for f in unique.values() if f["fixture_id"] not in already
         and f["fixture_id"] not in deferred),
        # Screen never-attempted fixtures before expired hard failures except
        # when a retried fixture kicks off within 90 minutes. Preserve old
        # negative cache metadata briefly solely for fair selection.
        key=lambda f: (
            0 if (_dt(f["kickoff"]) - now) <= timedelta(minutes=90) else
            (1 if f["fixture_id"] not in cached_negative else 2),
            f["kickoff"], f["fixture_id"],
        ),
    )
    # Shared season splits across fixtures: in-memory only, fed by immutable
    # original captures and fresh observations. No public duplicate of source
    # data; no extra API source or changed calibrated model weights.
    split_cache = TeamSeasonSplitCache(existing)
    frozen, excluded = [], []
    selected = _select_diverse_upcoming(choices, max_fixtures, now)
    for fx in selected:
        result = _build_one(fx, provider, model, model_sha, clock, split_cache)
        if result.get("status") == "FROZEN_RESEARCH_PREMATCH_SPORT_PROJECTION":
            frozen.append(result)
        else:
            excluded.append(result)
    if provider.calls > MAX_PROVIDER_REQUESTS:
        raise RuntimeError("PROVIDER_CALL_CAP_EXCEEDED")
    by_league: dict[str, dict[str, Any]] = {}
    for fx in unique.values():
        key = str(fx["league_id"])
        item = by_league.setdefault(key, {
            "league": fx["league"], "competition_id": fx["league_id"],
            "eligible": 0, "attempted": 0, "modeled_limited": 0,
            "missing_or_insufficient": 0, "already_frozen": 0,
            "deferred_due_to_missing_source": 0, "unattempted": 0,
            "failure_reasons": {},
        })
        item["eligible"] += 1
        if fx["fixture_id"] in already:
            item["already_frozen"] += 1
        elif fx["fixture_id"] in deferred:
            item["deferred_due_to_missing_source"] += 1
        else:
            item["unattempted"] += 1
    fixture_lookup = {x["fixture_id"]: x for x in unique.values()}
    for row in frozen:
        fx = fixture_lookup.get(row["fixture_id"])
        if fx:
            item = by_league[str(fx["league_id"])]
            item["attempted"] += 1
            item["modeled_limited"] += 1
            item["unattempted"] -= 1
    for row in excluded:
        fx = fixture_lookup.get(row["fixture_id"])
        if fx:
            item = by_league[str(fx["league_id"])]
            item["attempted"] += 1
            item["missing_or_insufficient"] += 1
            item["unattempted"] -= 1
            reason = str(row.get("reason") or "NOT_VERIFIED")[:100]
            item["failure_reasons"][reason] = item["failure_reasons"].get(reason, 0) + 1
    report = sorted(by_league.values(), key=lambda x: (-x["eligible"], x["competition_id"]))
    return {
        "status": "CAPTURED_RESEARCH_ONLY" if frozen else "NO_NEW_VALID_PREGAME_MODELS",
        "generated_at_utc": clock().isoformat(),
        "candidate_fixtures": len(choices),
        "negative_cache_deferred_fixtures": sum(
            f["fixture_id"] in deferred and f["fixture_id"] not in already
            for f in unique.values()
        ),
        "negative_cache_policy": "SOURCE_MISSING_OR_SAMPLE_SMALL_8H_RETRY_NO_MODEL",
        "selection_policy": "NEAR_KICKOFF_THEN_UNTRIED_THEN_EXPIRED_RETRIES;DIVERSE_LEAGUES;NO_BETS",
        "previously_negative_eligible": sum(
            f["fixture_id"] in cached_negative for f in choices
        ),
        "attempted_fixtures": len(frozen)+len(excluded),
        "new_frozen_pregame_models": len(frozen),
        "eligible_fixture_ids": len(unique),
        "league_coverage_report": report[:250],
        "league_coverage_report_total_leagues": len(report),
        "league_coverage_report_truncated": len(report) > 250,
        "league_coverage_report_scope": "THIS_CAPTURE_TWO_DATE_WINDOW_ONLY",
        "season_split_cache_hits": split_cache.hits,
        "season_split_cache_hits_from_original_frozen": split_cache.from_prior_snapshot,
        "season_split_cache_hits_same_capture": split_cache.from_current_capture,
        "season_split_cache_misses": split_cache.misses,
        "season_split_cache_new_entries": split_cache.puts,
        "season_split_cache_new_public_cache_files_created": 0,
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
    assert deduped["league_coverage_report"][0]["already_frozen"] == 1
    # New eligible game, same home team+league+season. The verified original
    # pregame snapshot can provide that team's current-season split without
    # another provider call. Away opponent is fetched from the official source.
    second_fixture = {
        **fixture,
        "fixture": {**fixture["fixture"], "id": 102,
                    "date": (now+timedelta(hours=5)).isoformat()},
        "teams": {"home": {"id": 5}, "away": {"id": 7}},
    }
    class ReusingProvider(Offline):
        def get(self, endpoint, params):
            if endpoint == "fixtures":
                self.calls += 1
                return {"response": [fixture, second_fixture]}
            return super().get(endpoint, params)
    reused = capture(
        ReusingProvider(), model, "fixed-test-repo-sha", [record], clock=lambda: now
    )
    assert reused["new_frozen_pregame_models"] == 1
    assert reused["provider_requests_added"] == 3  # 2 dates + 1 new team
    assert reused["season_split_cache_hits"] == 1
    assert reused["season_split_cache_hits_from_original_frozen"] == 1
    assert reused["season_split_cache_misses"] == 1
    assert reused["season_split_cache_new_public_cache_files_created"] == 0
    assert reused["league_coverage_report"][0]["eligible"] == 2
    assert reused["league_coverage_report"][0]["attempted"] == 1
    assert reused["league_coverage_report"][0]["modeled_limited"] == 1
    assert reused["league_coverage_report"][0]["already_frozen"] == 1
    # A cached source from another competition, venue, or after kickoff is
    # never used as a current pre-match model input.
    wrong_league = {**f, "league_id": 56}
    data = TeamSeasonSplitCache([record])
    assert data.get(wrong_league, "home", now) is None
    assert data.get(f, "away", now+timedelta(hours=13)) is None
    assert data.get(f, "home", now-timedelta(minutes=1)) is None
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
    with tempfile.TemporaryDirectory() as folder:
        neg_path = Path(folder) / "negative.json"
        excluded = [{
            "fixture_id": f["fixture_id"],
            "status": "NOT_MODELED",
            "reason": "HOME_VERIFIED_SEASON_SPLITS_MISSING",
        }]
        _write_negative_cache(neg_path, {}, excluded, [], now=now)
        negative = _read_negative_cache(neg_path)
        assert f["fixture_id"] in negative
        assert negative[f["fixture_id"]]["reason"] in NEGATIVE_REASONS
        quiet_provider = Offline()
        suppressed = capture(
            quiet_provider, model, "fixed-test-repo-sha", [], clock=lambda: now,
            negative_cache=negative,
        )
        assert suppressed["new_frozen_pregame_models"] == 0
        assert suppressed["attempted_fixtures"] == 0
        assert suppressed["negative_cache_deferred_fixtures"] == 1
        assert suppressed["provider_requests_added"] == 2  # fixtures only
        _write_negative_cache(
            neg_path, negative, [], [],
            now=now + NEGATIVE_COOLDOWN + timedelta(minutes=1),
        )
        kept = _read_negative_cache(neg_path)
        assert f["fixture_id"] in kept  # expiry releases a retry, retains fairness history
        # When an expired hard failure and a new fixture compete in one
        # match slot, analyze the never-tried one first, not the stale retry.
        new_fixture = {
            **fixture,
            "fixture": {**fixture["fixture"], "id": 102,
                        "date": (now+timedelta(hours=5)).isoformat()},
        }
        old_fixture = {
            **fixture,
            "fixture": {**fixture["fixture"], "id": 101,
                        "date": (now+timedelta(hours=4)).isoformat()},
        }
        class PairProvider(Offline):
            def get(self, endpoint, params):
                if endpoint == "fixtures":
                    self.calls += 1
                    return {"response": [old_fixture, new_fixture]}
                return super().get(endpoint, params)
        expired = {101: {
            "reason": "HOME_VERIFIED_SEASON_SPLITS_MISSING",
            "retry_after_utc": (now-timedelta(minutes=1)).isoformat(),
        }}
        fair_provider = PairProvider()
        priority_run = capture(
            fair_provider, model, "fixed-test-repo-sha", [],
            clock=lambda: now, max_fixtures=1, negative_cache=expired,
        )
        assert priority_run["attempted_fixtures"] == 1
        assert priority_run["frozen"][0]["fixture_id"] == 102
        assert priority_run["provider_requests_added"] == 4
        assert priority_run["previously_negative_eligible"] == 1
        _write_negative_cache(
            neg_path, kept, [], [],
            now=now + NEGATIVE_COOLDOWN + NEGATIVE_HISTORY_RETENTION + timedelta(minutes=1),
        )
        assert _read_negative_cache(neg_path) == {}
    print("ISOLATED_SPORT_FIRST_OFFLINE_PASS: NO_MARKET, REAL_CANONICAL_MODEL, PREGAME, DEDUP, BOUNDED, NEGATIVE_CACHE, FAIR_UNTRIED_PRIORITY")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model-file",type=Path,required=True)
    parser.add_argument("--runtime-model-sha",required=True)
    parser.add_argument("--registry",type=Path)
    parser.add_argument("--receipt",type=Path)
    parser.add_argument("--negative-cache",type=Path)
    args = parser.parse_args()
    if not args.runtime_model_sha or len(args.runtime_model_sha) > 128:
        parser.error("model SHA must be nonempty and bounded")
    if args.self_test:
        self_test(args.model_file)
        return
    if args.registry is None or args.receipt is None or args.negative_cache is None:
        parser.error("--registry, --receipt and --negative-cache required")
    model = _load_canonical_model(args.model_file)
    prior = _read_rows(args.registry)
    negatives = _read_negative_cache(args.negative_cache)
    provider = OfficialProvider(os.environ.get("API_FOOTBALL_KEY", ""))
    result = capture(provider, model, args.runtime_model_sha, prior, clock=lambda:datetime.now(timezone.utc), negative_cache=negatives)
    if result["frozen"]:
        args.registry.parent.mkdir(parents=True,exist_ok=True)
        with args.registry.open("a",encoding="utf-8") as file:
            for row in result["frozen"]:
                file.write(json.dumps(row,sort_keys=True,ensure_ascii=False)+"\n")
    _write_negative_cache(
        args.negative_cache, negatives, result["excluded"], result["frozen"],
        now=_dt(result["generated_at_utc"]) or datetime.now(timezone.utc),
    )
    receipt = {k:v for k,v in result.items() if k!="frozen"}
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(receipt,sort_keys=True,ensure_ascii=False))


if __name__ == "__main__":
    main()
