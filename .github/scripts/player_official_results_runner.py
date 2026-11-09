"""Bounded official postgame player collector for GitHub runner only.

Research-only source refresh. It neither touches Render/Postgres nor reconstructs
pregame projections, odds, CLV or historical out-of-sample predictions.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

API_BASE = "https://v3.football.api-sports.io"
MIN_DAILY_REMAINING = 1000
MAX_FIXTURES_PER_RUN = 5
STATUS = "RESEARCH_PLAYER_TREND_REGISTRY_BACKFILL_STORE"


def as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        n = int(value)
        return n if n >= 0 and str(value).strip() == str(n) else None
    except (TypeError, ValueError):
        return None


def as_num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        n = float(value)
        return n if n >= 0 and n != float("inf") and n == n else None
    except (TypeError, ValueError):
        return None


def remaining_from_headers(headers: Any) -> int | None:
    for name in ("x-ratelimit-requests-remaining", "X-RateLimit-Requests-Remaining"):
        val = headers.get(name)
        if val is not None:
            return as_int(val)
    return None


def provider_get(endpoint: str, params: dict[str, Any], key: str) -> tuple[dict, int | None]:
    if endpoint not in {"fixtures", "fixtures/players"}:
        raise ValueError("ENDPOINT_NOT_ALLOWLISTED")
    url = API_BASE + "/" + endpoint + "?" + urlencode(params)
    request = Request(url, headers={"x-apisports-key": key, "Accept": "application/json"})
    with urlopen(request, timeout=25) as response:
        payload = json.loads(response.read(8_000_001))
        remaining = remaining_from_headers(response.headers)
    if not isinstance(payload, dict) or payload.get("errors"):
        raise ValueError("INVALID_PROVIDER_RESPONSE")
    return payload, remaining


def candidate_fixtures(payload: dict[str, Any], now: datetime) -> list[dict[str, Any]]:
    found: dict[int, dict[str, Any]] = {}
    for row in payload.get("response") or []:
        if not isinstance(row, dict):
            continue
        fixture = row.get("fixture") or {}
        status = fixture.get("status") or {}
        teams = row.get("teams") or {}
        goals = row.get("goals") or {}
        fid = as_int(fixture.get("id"))
        home_id = as_int((teams.get("home") or {}).get("id"))
        away_id = as_int((teams.get("away") or {}).get("id"))
        home_goals = as_int(goals.get("home"))
        away_goals = as_int(goals.get("away"))
        if (status.get("short") != "FT" or not fid or not home_id or not away_id
                or home_id == away_id or home_goals is None or away_goals is None):
            continue  # AET/PEN/PST/NS and missing scores are not 90m finalized.
        try:
            kickoff = datetime.fromisoformat(str(fixture.get("date")).replace("Z", "+00:00"))
            if kickoff.tzinfo is None or kickoff >= now:
                continue
        except ValueError:
            continue
        found[fid] = {
            "fixture_id": fid,
            "kickoff": kickoff.astimezone(timezone.utc).isoformat(),
            "home_team_id": home_id,
            "away_team_id": away_id,
        }
    return sorted(found.values(), key=lambda x: (x["kickoff"], x["fixture_id"]))


def compact_players(payload: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]] | None:
    response = payload.get("response")
    if not isinstance(response, list) or len(response) != 2:
        return None
    teams = []
    seen_players: set[int] = set()
    for group in response:
        if not isinstance(group, dict) or not isinstance(group.get("players"), list):
            return None
        meta = group.get("team") or {}
        tid = as_int(meta.get("id"))
        if not tid or tid not in {candidate["home_team_id"], candidate["away_team_id"]}:
            return None
        rows = []
        for player_row in group["players"]:
            if not isinstance(player_row, dict):
                continue
            p = player_row.get("player") or {}
            pid = as_int(p.get("id"))
            stats = player_row.get("statistics")
            if not pid or pid in seen_players or not isinstance(stats, list) or not stats:
                continue
            s = stats[0]
            if not isinstance(s, dict) or not isinstance(s.get("games"), dict):
                continue
            games = s["games"]
            minutes = as_num(games.get("minutes"))
            if minutes is None or minutes > 130:
                continue
            shots = s.get("shots") or {}
            goals = s.get("goals") or {}
            cards = s.get("cards") or {}
            passes = s.get("passes") or {}
            tackles = s.get("tackles") or {}
            duels = s.get("duels") or {}
            seen_players.add(pid)
            rows.append({
                "player_id": pid,
                "name": p.get("name"),
                "position": games.get("position"),
                "minutes": minutes,
                "substitute": games.get("substitute"),
                "captain": games.get("captain"),
                "rating": as_num(games.get("rating")),
                "shots": as_num(shots.get("total")),
                "shots_on_target": as_num(shots.get("on")),
                "goals": as_num(goals.get("total")),
                "assists": as_num(goals.get("assists")),
                "saves": as_num(goals.get("saves")),
                "goals_conceded": as_num(goals.get("conceded")),
                "yellow_cards": as_num(cards.get("yellow")),
                "red_cards": as_num(cards.get("red")),
                "passes": as_num(passes.get("total")),
                "key_passes": as_num(passes.get("key")),
                "tackles": as_num(tackles.get("total")),
                "duels": as_num(duels.get("total")),
                "duels_won": as_num(duels.get("won")),
            })
        teams.append({"team_id": tid, "team": meta.get("name"), "players": rows})
    if ({t["team_id"] for t in teams} != {candidate["home_team_id"], candidate["away_team_id"]}
            or sum(len(t["players"]) for t in teams) < 10):
        return None
    return teams


def load_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"captures": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("status") != STATUS or not isinstance(data.get("captures"), list):
        raise ValueError("UNRECOGNIZED_EXISTING_REGISTRY")
    return data


def collect(
    fixture_date: str,
    existing: dict[str, Any],
    get: Callable[[str, dict[str, Any]], tuple[dict, int | None]],
    *,
    now: datetime,
    max_fixtures: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    known = {}
    for capture in existing.get("captures") or []:
        if isinstance(capture, dict):
            fid = as_int(capture.get("fixture_id"))
            if fid:
                known[fid] = capture
    response, remaining = get("fixtures", {"date": fixture_date, "timezone": "America/Mexico_City"})
    candidates = [f for f in candidate_fixtures(response, now) if f["fixture_id"] not in known]
    attempted = captured = rejected = 0
    errors = []
    # Unknown quota must never turn into unrestricted requests.
    if remaining is not None and remaining > MIN_DAILY_REMAINING:
        for fixture in candidates[:max_fixtures]:
            if remaining is None or remaining <= MIN_DAILY_REMAINING:
                break
            attempted += 1
            try:
                stats, remaining = get("fixtures/players", {"fixture": fixture["fixture_id"]})
                teams = compact_players(stats, fixture)
                if teams is None:
                    rejected += 1
                    continue
                known[fixture["fixture_id"]] = {
                    "fixture_id": fixture["fixture_id"],
                    "kickoff": fixture["kickoff"],
                    "capture_phase": "POSTGAME_RUNNER_OFFICIAL_API",
                    "captured_at_utc": now.isoformat(),
                    "source": "API_FOOTBALL_V3_FIXTURES_PLAYERS",
                    "provider_fixture_final_status": "FT",
                    "teams": teams,
                }
                captured += 1
            except Exception:
                errors.append("PROVIDER_ERROR_REDACTED")
                break  # fail closed, do not retry and burn further quota
    else:
        errors.append("QUOTA_UNKNOWN_OR_BELOW_RESERVE")
    captures = sorted(known.values(), key=lambda c: (str(c.get("kickoff") or ""), int(c["fixture_id"])))[-500:]
    updated = {
        "schema_version": "1.0.0",
        "status": STATUS,
        "generated_at_utc": now.isoformat(),
        "capture_count": len(captures),
        "captures": captures,
        "future_registry_use_only": True,
        "retroactive_pregame_signal_created": False,
        "eligible_for_historical_oos_reconstruction": False,
        "production_promotion_allowed": False,
    }
    report = {
        "status": "RESEARCH_ONLY_POSTGAME_CAPTURE",
        "date": fixture_date,
        "finalized_missing_from_registry": len(candidates),
        "attempted": attempted,
        "captured": captured,
        "rejected_provider_payloads": rejected,
        "provider_requests_added": attempted + 1,
        "daily_remaining_reported": remaining,
        "errors": errors,
        "retroactive_predictions_created": False,
        "oos_validation_complete": False,
        "production_promotion_allowed": False,
    }
    return updated, report


def self_test() -> None:
    now = datetime(2026, 11, 8, 12, 0, tzinfo=timezone.utc)
    def fixture(status: str = "FT", fid: int = 123) -> dict:
        return {"fixture": {"id": fid, "date": "2026-11-07T18:00:00+00:00", "status": {"short": status}},
                "teams": {"home": {"id": 11}, "away": {"id": 22}},
                "goals": {"home": 2, "away": 0}}
    def stats(team_id: int, start: int) -> dict:
        return {"team": {"id": team_id, "name": "Verified team"},
                "players": [{"player": {"id": start+i, "name": "Test"},
                             "statistics": [{"games": {"minutes": 90, "position": "F"},
                                             "shots": {"total": None, "on": 1}}]} for i in range(6)]}
    source = {"fixtures": {"response": [fixture(), fixture("AET", 456)]},
              "fixtures/players": {"response": [stats(11, 100), stats(22, 200)]}}
    def get(endpoint: str, params: dict) -> tuple[dict, int]:
        return source[endpoint], 7000
    output, report = collect("2026-11-07", {"captures": []}, get, now=now, max_fixtures=5)
    assert report["captured"] == 1 and report["provider_requests_added"] == 2
    assert output["captures"][0]["teams"][0]["players"][0]["shots"] is None
    assert output["production_promotion_allowed"] is False
    output2, report2 = collect("2026-11-07", output, get, now=now, max_fixtures=5)
    assert report2["captured"] == 0 and report2["provider_requests_added"] == 1
    _, report3 = collect("2026-11-07", {"captures": []}, lambda e,p:(source[e],None), now=now, max_fixtures=5)
    assert report3["attempted"] == 0 and report3["provider_requests_added"] == 1
    source["fixtures/players"] = {"response": [stats(11, 100), stats(999, 200)]}
    _, report4 = collect("2026-11-07", {"captures": []}, get, now=now, max_fixtures=5)
    assert report4["captured"] == 0 and report4["rejected_provider_payloads"] == 1
    print("OFFLINE_TESTS_PASS: FT_ONLY, OFFICIAL_IDS, MISSING_NOT_ZERO, DEDUP, QUOTA_FAIL_CLOSED, RESEARCH_ONLY")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="state/soccer_edge_state/analysis/player_trend_registry_backfill.json")
    parser.add_argument("--date", default=None)
    parser.add_argument("--max-fixtures", type=int, default=3)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not 1 <= args.max_fixtures <= MAX_FIXTURES_PER_RUN:
        parser.error("max-fixtures out of bounds")
    now = datetime.now(timezone.utc)
    local = now.astimezone(ZoneInfo("America/Mexico_City"))
    fixture_date = args.date or (local.date() - timedelta(days=1)).isoformat()
    date = datetime.strptime(fixture_date, "%Y-%m-%d").date()
    if date >= local.date() or date < local.date() - timedelta(days=7):
        parser.error("date must be one of the previous 7 local calendar dates")
    key = os.getenv("API_FOOTBALL_KEY", "")
    if not key:
        print(json.dumps({"status": "SKIPPED_MISSING_GITHUB_ACTIONS_SECRET", "provider_requests_added": 0,
                          "production_promotion_allowed": False}))
        return
    path = Path(args.registry)
    existing = load_registry(path)
    updated, report = collect(
        fixture_date, existing, lambda e, p: provider_get(e, p, key),
        now=now, max_fixtures=args.max_fixtures,
    )
    # Never rewrite historical predictions, market prices, results or OOS labels.
    if report["captured"]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
            json.dump(updated, tmp, ensure_ascii=False, indent=2)
            tmp.write("\n")
            tmp_name = tmp.name
        os.replace(tmp_name, path)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
