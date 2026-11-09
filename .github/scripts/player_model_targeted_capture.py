"""Capture at most one official FT fixture selected from genuine frozen model IDs.

Research-only: the inventory is NOT result evidence. Independent provider
fixture FT + official player identities/stats must pass the existing bounded
collector before any OOS join. No Render backfill and no retrospective models.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo


def _dt(raw):
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc) if dt.tzinfo else None
    except (ValueError, TypeError):
        return None


def select_target(inventory: dict, registry: dict, *, now: datetime):
    if inventory.get("status") != "FROZEN_PLAYER_MODEL_FIXTURE_INVENTORY_RESEARCH_ONLY":
        raise ValueError("UNVERIFIED_FROZEN_MODEL_INVENTORY")
    if type(inventory.get("candidate_fixtures")) is not int:
        raise ValueError("INVALID_INVENTORY_COUNT")
    candidates = inventory.get("candidates")
    if (not isinstance(candidates, list)
            or len(candidates) != inventory["candidate_fixtures"]
            or len(candidates) > 20):
        raise ValueError("UNBOUNDED_INVENTORY")
    if registry.get("status") != "RESEARCH_PLAYER_TREND_REGISTRY_BACKFILL_STORE":
        raise ValueError("UNVERIFIED_OFFICIAL_OUTCOME_REGISTRY")
    if now.tzinfo is None:
        raise ValueError("NOW_MUST_BE_TIMEZONE_AWARE")
    local_today = now.astimezone(ZoneInfo("America/Mexico_City")).date()
    known = {x["fixture_id"] for x in registry.get("captures") or []
             if isinstance(x, dict) and type(x.get("fixture_id")) is int}
    for item in candidates:
        if not isinstance(item, dict):
            raise ValueError("INVALID_INVENTORY_ROW")
        fid = item.get("fixture_id")
        if type(fid) is not int or fid <= 0:
            raise ValueError("UNVERIFIED_FIXTURE_ID")
        if item.get("official_final_result_verified") is not False or item.get("oos_eligible") is not False:
            raise ValueError("INVENTORY_MUST_NOT_ASSERT_FT_OR_OOS")
        if fid in known:
            continue
        kickoff = _dt(item.get("kickoff"))
        if kickoff is None or kickoff > now - timedelta(hours=4):
            continue
        matchday = kickoff.astimezone(ZoneInfo("America/Mexico_City")).date()
        if local_today - timedelta(days=7) <= matchday < local_today:
            return fid, matchday.isoformat()
    return None


def capture_one(inventory_file: Path, registry_file: Path, *, now: datetime):
    inventory = json.loads(inventory_file.read_text(encoding="utf-8"))
    registry = json.loads(registry_file.read_text(encoding="utf-8"))
    target = select_target(inventory, registry, now=now)
    if target is None:
        return {
            "status": "NO_UNCAPTURED_FROZEN_MODEL_FIXTURE_IN_SAFE_WINDOW",
            "provider_requests_added": 0,
            "captured": 0,
            "production_promotion_allowed": False,
        }
    if not os.getenv("API_FOOTBALL_KEY"):
        raise RuntimeError("API_FOOTBALL_SECRET_NOT_CONFIGURED")
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("player_official_results_runner.py")),
        "--registry", str(registry_file),
        "--fixture-id", str(target[0]),
        "--date", target[1],
        "--max-fixtures", "1",
    ]
    res = subprocess.run(cmd, text=True, capture_output=True, timeout=45, check=False)
    if res.returncode:
        raise RuntimeError("BOUNDED_PROVIDER_COLLECTOR_FAILED")
    try:
        report = json.loads(res.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        raise RuntimeError("BOUNDED_PROVIDER_REPORT_NOT_VERIFIED") from None
    remaining = report.get("daily_remaining_reported")
    if (report.get("status") != "RESEARCH_ONLY_POSTGAME_CAPTURE"
            or report.get("errors")
            or report.get("captured") not in (0, 1)
            or report.get("attempted") not in (0, 1)
            or report.get("provider_requests_added") not in (1, 2)
            or type(remaining) is not int or remaining < 1000
            or report.get("production_promotion_allowed") is not False
            or report.get("retroactive_predictions_created") is not False):
        raise RuntimeError("OFFICIAL_TARGETED_CAPTURE_NOT_VERIFIED")
    return {
        "status": "OFFICIAL_FT_CAPTURE_VERIFIED" if report["captured"] else "PROVIDER_FT_NOT_YET_VERIFIED",
        "fixture_id": target[0],
        "captured": report["captured"],
        "provider_requests_added": report["provider_requests_added"],
        "daily_remaining_reported": remaining,
        "retroactive_predictions_created": False,
        "production_promotion_allowed": False,
    }


def self_test():
    utc = timezone.utc
    now = datetime(2026, 11, 9, 20, 0, tzinfo=utc)
    fixture = {
        "fixture_id": 123,
        "kickoff": "2026-11-07T06:00:00+00:00",
        "official_final_result_verified": False,
        "oos_eligible": False,
    }
    inventory = {"status": "FROZEN_PLAYER_MODEL_FIXTURE_INVENTORY_RESEARCH_ONLY",
                 "candidate_fixtures": 1, "candidates": [fixture]}
    registry = {"status": "RESEARCH_PLAYER_TREND_REGISTRY_BACKFILL_STORE", "captures": []}
    assert select_target(inventory, registry, now=now) == (123, "2026-11-07")
    registry["captures"] = [{"fixture_id": 123}]
    assert select_target(inventory, registry, now=now) is None
    registry["captures"] = []
    assert select_target({**inventory, "candidate_fixtures": 0, "candidates": []}, registry, now=now) is None
    try:
        select_target({**inventory, "candidate_fixtures": 2}, registry, now=now)
        raise AssertionError("Expected fail closed on unbounded count mismatch")
    except ValueError:
        pass
    try:
        select_target({**inventory,"candidates":[{**fixture,"official_final_result_verified":True}]},registry,now=now)
        raise AssertionError("Expected fail closed on fake FT")
    except ValueError:
        pass
    old = {**fixture, "kickoff": "2026-10-15T06:00:00+00:00"}
    assert select_target({**inventory, "candidates": [old]}, registry, now=now) is None
    future = {**fixture, "kickoff": "2026-11-09T23:00:00+00:00"}
    assert select_target({**inventory, "candidates": [future]}, registry, now=now) is None
    print("OFFLINE_TARGETED_CAPTURE_PASS: TIME, DEDUP, PROVENANCE, SINGLE_FIXTURE, FT_REQUIRED")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", required=False)
    parser.add_argument("--registry", required=False)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.inventory or not args.registry:
        parser.error("inventory and registry are required")
    out = capture_one(Path(args.inventory), Path(args.registry), now=datetime.now(timezone.utc))
    print(json.dumps(out, sort_keys=True))


if __name__ == "__main__":
    main()
