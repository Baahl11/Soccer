"""Strict as-of OOS join: live canonical PostgreSQL forecasts to official FT outcomes.

Offline/testable. Never reconstructs a prediction from an outcome, reads prices,
or alters any historical tracking row. Uses the sport-specific OOS scorer from
soccer-edge-mcp-v1 for all Brier/logloss/RMSE metrics.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

SHOTS_STATUS = "LIVE_RESEARCH_SHOTS_PROPS"
SOT_STATUS = "LIVE_RESEARCH_SOT_PROPS"
PLAYER_STATUS = {"SHOTS": "LIVE_RESEARCH_SHOTS_DISTRIBUTION",
                 "SOT": "LIVE_RESEARCH_SOT_DISTRIBUTION"}
MARKETS = {"SHOTS": ("player_shots_intelligence", "shots", "expected_shots"),
           "SOT": ("player_sot_intelligence", "shots_on_target", "expected_sot")}
MAX_SNAPSHOTS = 25
MODEL_VERSION = "SOCCER_PLAYER_PROPS_OOS_V4_1.1.1"


def as_utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return result.astimezone(timezone.utc) if result.tzinfo else None
    except ValueError:
        return None


def num(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        n = float(value)
        return n if math.isfinite(n) and n >= 0 else None
    except (ValueError, TypeError):
        return None


def posid(value: Any) -> int | None:
    return value if type(value) is int and 0 < value < 10**13 else None


def fixture_ids(registry: dict) -> list[int]:
    captures = registry.get("captures") or []
    eligible = [x for x in captures
                if isinstance(x, dict)
                and x.get("capture_phase") == "POSTGAME_RUNNER_OFFICIAL_API"
                and x.get("source") == "API_FOOTBALL_V3_FIXTURES_PLAYERS"
                and x.get("provider_fixture_final_status") == "FT"
                and posid(x.get("fixture_id"))
                and as_utc(x.get("kickoff"))
                and as_utc(x.get("captured_at_utc"))
                and as_utc(x["captured_at_utc"]) > as_utc(x["kickoff"])]
    eligible.sort(key=lambda x: (as_utc(x["kickoff"]), x["fixture_id"]), reverse=True)
    out: list[int] = []
    for c in eligible:
        if c["fixture_id"] not in out:
            out.append(c["fixture_id"])
        if len(out) == MAX_SNAPSHOTS:
            break
    return out


def grade_export(export: dict, registry: dict, existing_keys: set[tuple[int, str, int]]):
    """Return zero or more validated *existing* model forecasts joined to truth."""
    from mcp_gateway import player_props_oos_postgres_v4 as scorer

    if export.get("status") != "VERIFIED_FROZEN_PLAYER_PREGAME_READONLY":
        raise ValueError("FROZEN_SOURCE_NOT_VERIFIED")
    ids = fixture_ids(registry)
    if not ids or int(export.get("queried_fixtures", -1)) != len(ids):
        raise ValueError("FROZEN_SOURCE_QUERY_COUNT_MISMATCH")
    allowed = set(ids)
    official = {x["fixture_id"]: x for x in registry.get("captures") or []
                if isinstance(x, dict) and x.get("fixture_id") in allowed}
    seen = set()
    added: list[dict] = []
    counters = Counter()
    for snapshot in export.get("rows") or []:
        if not isinstance(snapshot, dict):
            raise ValueError("FROZEN_SOURCE_ROW_NOT_OBJECT")
        fid = posid(snapshot.get("fixture_id"))
        if fid not in allowed or fid in seen or len(seen) >= MAX_SNAPSHOTS:
            raise ValueError("FROZEN_SNAPSHOT_IDENTITY_NOT_ALLOWED")
        seen.add(fid)
        cap = official[fid]
        frozen = as_utc(snapshot.get("generated_at"))
        kickoff = as_utc(snapshot.get("kickoff"))
        cap_kickoff = as_utc(cap.get("kickoff"))
        observed = as_utc(cap.get("captured_at_utc"))
        if not (frozen and kickoff and cap_kickoff and observed
                and frozen < kickoff == cap_kickoff < observed):
            counters["NOT_IDENTICAL_FROZEN_KICKOFF_OR_CAPTURE_TIMELINE"] += 1
            continue
        home = posid(snapshot.get("home_team_id"))
        away = posid(snapshot.get("away_team_id"))
        if home is None or away is None or home == away:
            counters["UNVERIFIED_FIXTURE_TEAMS"] += 1
            continue
        team_groups = cap.get("teams") or []
        if (len(team_groups) != 2 or
                {posid(t.get("team_id")) for t in team_groups if isinstance(t, dict)} != {home, away}):
            counters["OFFICIAL_TEAM_IDS_MISMATCH"] += 1
            continue
        actuals: dict[int, tuple[int, dict]] = {}
        for team in team_groups:
            for player in team.get("players") or []:
                if not isinstance(player, dict):
                    continue
                pid = posid(player.get("player_id"))
                if pid and pid not in actuals and num(player.get("minutes")) is not None:
                    actuals[pid] = (team["team_id"], player)

        payload = snapshot.get("event_payload")
        if not isinstance(payload, dict):
            counters["MISSING_FROZEN_MODEL_PAYLOAD"] += 1
            continue
        filtered_payload = {"fixture": {"fixture_id": fid, "kickoff": snapshot["kickoff"]}}
        for family, (key, actual_key, expected_key) in MARKETS.items():
            intel = payload.get(key)
            if not isinstance(intel, dict) or intel.get("status") != (SHOTS_STATUS if family == "SHOTS" else SOT_STATUS):
                counters[f"{family}_MODEL_NOT_FROZEN"] += 1
                continue
            if intel.get("fixture_id") != fid:
                counters[f"{family}_MODEL_FIXTURE_MISMATCH"] += 1
                continue
            verified_players = []
            for p in intel.get("players") or []:
                if not isinstance(p, dict) or p.get("status") != PLAYER_STATUS[family] or p.get("confirmed_starter") is not True:
                    continue
                pid, tid = posid(p.get("player_id")), posid(p.get("team_id"))
                actual = actuals.get(pid) if pid else None
                if not actual or actual[0] != tid or tid not in (home, away):
                    counters[f"{family}_PLAYER_ID_OR_TEAM_NOT_VERIFIED"] += 1
                    continue
                if num(actual[1].get(actual_key)) is None:
                    counters[f"{family}_OFFICIAL_STAT_MISSING"] += 1
                    continue
                reg_time = as_utc(p.get("registry_generated_at_utc"))
                if not reg_time or reg_time > frozen:
                    counters[f"{family}_PLAYER_REGISTRY_NOT_ASOF"] += 1
                    continue
                if num(p.get(expected_key)) is None:
                    counters[f"{family}_RAW_PROJECTION_NOT_VERIFIED"] += 1
                    continue
                lines = []
                for ln in p.get("lines") or []:
                    if not isinstance(ln, dict):
                        continue
                    line, prob, under = num(ln.get("line")), num(ln.get("p_over")), num(ln.get("p_under"))
                    if (line not in (0.5,1.5,2.5,3.5,4.5,5.5)
                            or prob is None or under is None
                            or prob > 1 or under > 1 or abs(prob+under-1) > 1e-4):
                        continue
                    lines.append({"line": line, "p_over": prob, "p_under": under})
                if not lines:
                    counters[f"{family}_NO_VALID_LINE_DISTRIBUTION"] += 1
                    continue
                safe = copy.copy(p)
                safe["lines"] = lines
                verified_players.append(safe)
            filtered_payload[key] = {**intel, "players": verified_players}

        pregame = {**snapshot, "event_payload": filtered_payload}
        postgame = {
            "fixture_id": fid,
            "generated_at": cap["captured_at_utc"],
            "event_payload": {
                "postgame_player_stats": {"teams": team_groups}
            },
        }
        candidates = scorer.build_oos_rows([pregame], [postgame])
        for r in candidates:
            pid = posid(r.get("player_id"))
            family = r.get("market_family")
            if not pid or family not in MARKETS:
                continue
            key = (fid, family, pid)
            if key in existing_keys:
                counters["ALREADY_GRADED"] += 1
                continue
            if not r.get("binary_rows"):
                counters["NO_PROBABILITY_ROWS"] += 1
                continue
            added.append({
                **r,
                "frozen_source": "POSTGRES_CANONICAL_SOCCER_REFRESH_EVENTS",
                "frozen_source_event_id": snapshot.get("source_event_id"),
                "official_result_source": cap.get("source"),
                "official_captured_at_utc": cap["captured_at_utc"],
                "asof_and_official_team_id_verified": True,
                "actionable": False,
                "decision_weight": 0.0,
                "production_promotion_allowed": False,
            })
            existing_keys.add(key)
            counters["VERIFIED_OOS_ROWS_ADDED"] += 1
    return added, counters, len(seen)


def run(export_path: Path, registry_path: Path, tracking_path: Path,
        report_path: Path, audit_path: Path) -> dict:
    from mcp_gateway import player_props_oos_postgres_v4 as scorer
    export = json.loads(export_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("status") != "RESEARCH_PLAYER_TREND_REGISTRY_BACKFILL_STORE":
        raise ValueError("OFFICIAL_REGISTRY_NOT_VERIFIED")
    old = [json.loads(line) for line in tracking_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    old_keys = {(x.get("fixture_id"), x.get("market_family"), x.get("player_id"))
                for x in old if isinstance(x, dict)}
    additions, excluded, found = grade_export(export, registry, old_keys)
    current_time = datetime.now(timezone.utc).isoformat()
    if additions:
        with tracking_path.open("a", encoding="utf-8") as writer:
            for r in additions:
                writer.write(json.dumps(r, ensure_ascii=False, sort_keys=True, allow_nan=False)+"\n")
        merged = old + additions
        summary = scorer.summarize_oos(merged)
        report.update(summary)
        report["generated_at_utc"] = current_time
        report["status"] = "COLLECTING_PLAYER_PROP_OOS"
        report["production_promotion_allowed"] = False
        report["decision_weight"] = 0.0
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    else:
        summary = scorer.summarize_oos(old)

    audit = {
        "schema_version": "1.0.0",
        "status": "OOS_ROWS_ADDED_FROM_TRUE_FROZEN_PREGAME" if additions else "ZERO_ELIGIBLE_FROZEN_PLAYER_MODELS",
        "generated_at_utc": current_time,
        "source": "OIDC_RENDER_READ_ONLY_POSTGRES_PLUS_VERIFIED_API_FOOTBALL_FT",
        "official_captures_total": registry.get("capture_count"),
        "official_captures_with_strict_provenance": len(fixture_ids(registry)),
        "frozen_fixtures_found": found,
        "existing_oos_player_rows": len(old),
        "new_oos_player_rows": len(additions),
        "updated_oos_player_rows": len(old) + len(additions),
        "new_rows_by_market": dict(Counter(x["market_family"] for x in additions)),
        "excluded_reasons": dict(excluded),
        "shots_player_game_rows": summary["families"]["SHOTS"]["player_game_rows"],
        "sot_player_game_rows": summary["families"]["SOT"]["player_game_rows"],
        "shots_minimum_review": 500,
        "sot_minimum_review": 500,
        "provider_requests_added": 0,
        "postgres_write_queries": 0,
        "retroactive_pregame_predictions_created": False,
        "production_promotion_allowed": False,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    return audit


def self_test():
    class FakeScorer:
        @staticmethod
        def build_oos_rows(pre, post):
            return [{
                "schema_version": "1.0.0", "fixture_id": 123, "stage": "T-20",
                "prediction_timestamp": pre[0]["generated_at"],
                "market_family": "SHOTS", "player_id": 51, "player_name": "Offline A",
                "team_id": 11, "position": "F", "expected_count": 1.5,
                "actual_count": 2, "expected_minutes": 90, "actual_minutes": 90,
                "count_error": 0.5, "minutes_error": 0,
                "binary_rows": [{"fixture_id": 123, "player_id": 51,
                                 "market_family": "SHOTS", "line": 1.5,
                                 "probability": 0.51, "outcome": 1}],
                "decision_weight": 0.0, "production_promotion_allowed": False,
            }] if pre[0]["event_payload"].get("player_shots_intelligence", {}).get("players") else []

    import types
    old_pkg = sys.modules.get("mcp_gateway")
    saved = sys.modules.get("mcp_gateway.player_props_oos_postgres_v4")
    fake_pkg = types.ModuleType("mcp_gateway")
    fake_pkg.__path__ = []
    fake_pkg.player_props_oos_postgres_v4 = FakeScorer
    sys.modules["mcp_gateway"] = fake_pkg
    sys.modules["mcp_gateway.player_props_oos_postgres_v4"] = FakeScorer
    try:
        kick = datetime(2026,11,8,6,0,tzinfo=timezone.utc)
        snap = (kick-timedelta(minutes=20)).isoformat()
        cap = {"fixture_id":123, "kickoff":kick.isoformat(),
               "capture_phase":"POSTGAME_RUNNER_OFFICIAL_API",
               "source":"API_FOOTBALL_V3_FIXTURES_PLAYERS",
               "provider_fixture_final_status":"FT",
               "captured_at_utc":(kick+timedelta(days=1)).isoformat(),
               "teams":[{"team_id":11, "players":[{"player_id":51,"minutes":90,"shots":2,"shots_on_target":1}]},
                        {"team_id":22, "players":[{"player_id":52,"minutes":90,"shots":0,"shots_on_target":0}]}]}
        reg = {"captures":[cap]}
        intel = {"status":SHOTS_STATUS,"fixture_id":123,"players":[{
            "player_id":51,"team_id":11,"status":PLAYER_STATUS["SHOTS"],
            "confirmed_starter":True,
            "registry_generated_at_utc":(kick-timedelta(days=1)).isoformat(),
            "expected_shots":1.5,"lines":[{"line":1.5,"p_over":0.51,"p_under":0.49}]
        }]}
        one = {"fixture_id":123,"home_team_id":11,"away_team_id":22,
               "source_event_id":987,"generated_at":snap,"kickoff":kick.isoformat(),
               "stage":"T-20","event_payload":{"player_shots_intelligence":intel}}
        exp={"status":"VERIFIED_FROZEN_PLAYER_PREGAME_READONLY","queried_fixtures":1,"rows":[one]}
        out,reasons,found=grade_export(exp,reg,set())
        assert found==1 and len(out)==1 and out[0]["frozen_source_event_id"]==987
        dup,r2,_=grade_export(exp,reg,{(123,"SHOTS",51)})
        assert not dup and r2["ALREADY_GRADED"]==1
        unsafe=copy.deepcopy(exp)
        unsafe["rows"][0]["event_payload"]["player_shots_intelligence"]["players"][0]["registry_generated_at_utc"]=(kick+timedelta(days=1)).isoformat()
        assert not grade_export(unsafe,reg,set())[0]
        badkick=copy.deepcopy(exp)
        badkick["rows"][0]["kickoff"]=(kick+timedelta(minutes=1)).isoformat()
        assert not grade_export(badkick,reg,set())[0]
        badteam=copy.deepcopy(exp)
        badteam["rows"][0]["away_team_id"]=99
        assert not grade_export(badteam,reg,set())[0]
        missing=copy.deepcopy(reg)
        missing["captures"][0]["teams"][0]["players"][0]["shots"]=None
        assert not grade_export(exp,missing,set())[0]
        print("OFFLINE_OOS_JOIN_TESTS_PASS: ASOF, TEAMS, NO_PHANTOM_ZERO, DEDUP, NO_BET")
    finally:
        if saved is None: del sys.modules["mcp_gateway.player_props_oos_postgres_v4"]
        else: sys.modules["mcp_gateway.player_props_oos_postgres_v4"] = saved
        if old_pkg is None: del sys.modules["mcp_gateway"]
        else: sys.modules["mcp_gateway"] = old_pkg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--export-file")
    p.add_argument("--registry")
    p.add_argument("--tracking")
    p.add_argument("--report")
    p.add_argument("--audit")
    args = p.parse_args()
    if args.self_test:
        self_test()
        return
    if not all([args.export_file, args.registry, args.tracking, args.report, args.audit]):
        p.error("export-file, registry, tracking, report and audit are required")
    output = run(Path(args.export_file), Path(args.registry),
                 Path(args.tracking), Path(args.report), Path(args.audit))
    print(json.dumps(output, sort_keys=True, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
