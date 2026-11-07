"""Isolated, quota-bounded FE11 complete approved-league fixture lifecycle.

This worker never imports the canonical v129 sporting+price model. It runs
in its OWN short-lived, cgroup-guarded subprocess behind GitHub OIDC and
the shared DB-heavy admission lock, at a different cadence from /internal/tick.
All state comes from official API-Football fixtures/date and, when budget
permits, verified FT-only fixture/statistics per previously frozen fixture.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import json
import os
import sys
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from mcp_gateway import automation as provider
from mcp_gateway import fe11_universal_lifecycle_v1 as universal
from mcp_gateway import fe11_universal_persistence_v1 as writer
from mcp_gateway import fe11_corners_persistence_v1 as corner_writer
from mcp_gateway import persistence

VERSION = "FE11_STANDALONE_OFFICIAL_SLATE_COLLECTOR_v1.0"
LOCAL_TZ = ZoneInfo("America/Mexico_City")
MAX_TOTAL_PROVIDER_CALLS = 12
MAX_DATE_PROVIDER_CALLS = 2
MIN_DAILY_REMAINING_FOR_OPTIONAL = 150
MAX_PROVIDER_FIXTURES_PER_DATE = 5000


def _remaining(payload: dict[str, Any]) -> int | None:
    try:
        value = (payload.get("quota") or {}).get("daily_remaining")
        if value is None:
            return None
        n = int(value)
        return n if n >= 0 else None
    except (TypeError, ValueError):
        return None


def requested_dates(now: datetime) -> list[str]:
    """Today always; yesterday for morning settlement, tomorrow for afternoon.
    Both optional periods use at most ONE additional provider date request.
    """
    if not isinstance(now, datetime) or now.tzinfo is None:
        raise ValueError("FE11_COLLECTION_TIME_NOT_VERIFIED")
    local = now.astimezone(LOCAL_TZ)
    today = local.date()
    other = today - timedelta(days=1) if local.hour < 12 else today + timedelta(days=1)
    return [today.isoformat(), other.isoformat()]


def _approved_from_date(
    response: dict[str, Any], date: str,
) -> tuple[list[dict], int]:
    """Filter after complete, nonpaginated official-date response verification."""
    if not isinstance(response, dict) or not isinstance(response.get("response"), list):
        raise ValueError("OFFICIAL_FIXTURES_RESPONSE_NOT_VERIFIED")
    raw = response["response"]
    if len(raw) > MAX_PROVIDER_FIXTURES_PER_DATE:
        raise ValueError("OFFICIAL_FIXTURES_DATE_SOURCE_UNBOUNDED")
    if response.get("results") is not None and (
        type(response["results"]) is not int or response["results"] != len(raw)
    ):
        raise ValueError("OFFICIAL_FIXTURES_DATE_INCOMPLETE")
    paging = response.get("paging") or {}
    if isinstance(paging, dict) and int(paging.get("total") or 1) > 1:
        raise ValueError("OFFICIAL_FIXTURES_PAGINATION_UNRESOLVED")
    result = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        lid = (item.get("league") or {}).get("id")
        if lid not in universal.APPROVED_LEAGUES:
            continue
        fx = provider._compact_fixture(item)
        kickoff = universal._utc(fx.get("kickoff"))
        if kickoff is None or kickoff.astimezone(LOCAL_TZ).date().isoformat() != date:
            raise ValueError("OFFICIAL_LOCAL_DATE_FIXTURE_MISMATCH")
        result.append(fx)
    return result, len(raw)


async def run_once(*, now_utc: datetime | None = None) -> dict[str, Any]:
    if not persistence.persistence_configured():
        raise RuntimeError("FE11_CANONICAL_DATABASE_NOT_CONFIGURED")
    persistence.ensure_schema()
    if not os.getenv("API_FOOTBALL_KEY", "").strip():
        raise RuntimeError("FE11_OFFICIAL_PROVIDER_KEY_NOT_CONFIGURED")
    now = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
    dates = requested_dates(now)
    requests = 0
    quota = None
    total_seen = 0
    compact: list[dict] = []
    date_audit = []
    official_get = provider._api_get  # capture original before other imports
    async with httpx.AsyncClient(
        timeout=provider.TIMEOUT,
        limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
    ) as client:
        provider._HTTP_CLIENT = client
        try:
            for index, date in enumerate(dates):
                if index > 0 and (
                    quota is None or quota <= MIN_DAILY_REMAINING_FOR_OPTIONAL
                    or requests >= MAX_DATE_PROVIDER_CALLS
                ):
                    date_audit.append({"date": date, "status": "DEFERRED_PROVIDER_RESERVE"})
                    continue
                requests += 1
                fetched = await official_get(
                    "fixtures", {"date": date, "timezone": "America/Mexico_City"}
                )
                values, original_count = _approved_from_date(fetched, date)
                total_seen += original_count
                compact.extend(values)
                quota = _remaining(fetched)
                date_audit.append({
                    "date": date, "status": "OFFICIAL_DATE_COMPLETE",
                    "provider_fixtures_seen": original_count,
                    "approved_league_fixtures": len(values),
                })
            observed = datetime.now(timezone.utc)
            capture = universal.build_capture(compact, now_utc=observed)
            if capture["status"] not in {
                "VERIFIED_OFFICIAL_ALL_ELIGIBLE_CAPTURE", "PARTIAL_IDENTITY_NOT_VERIFIED",
            }:
                raise ValueError("FE11_SLATE_CONFLICT_OR_UNBOUNDED_SOURCE")
            tick = {
                "generated_at_utc": capture["source_observed_at_utc"],
                "fe11_universal_lifecycle": capture,
            }
            with persistence._connect() as conn:
                with conn.cursor() as cur:
                    writer.persist(cur, tick, persistence._upsert_fixture)

            # A separate incremental FT90 corner queue follows score persistence.
            # It does not block the universal basic fixture/score freeze when
            # quota is missing. The original deep-dive budget is unchanged.
            corner_attempts: list[dict] = []
            corner_observations: list[dict] = []
            corner_status = "DEFERRED_DAILY_QUOTA"
            if quota is not None and quota > MIN_DAILY_REMAINING_FOR_OPTIONAL:
                from mcp_gateway import fe11_postgame_corners_v1 as corner_model
                cap = min(corner_model.MAX_FIXTURES, MAX_TOTAL_PROVIDER_CALLS - requests)
                if cap > 0:
                    pending = corner_model.pending(cap)
                    corner_status = "NO_PENDING_VERIFIED_FT" if not pending else "ATTEMPTED"
                    for fixture in pending:
                        if requests >= MAX_TOTAL_PROVIDER_CALLS or quota is None or quota <= MIN_DAILY_REMAINING_FOR_OPTIONAL:
                            break
                        requests += 1
                        try:
                            stats = await official_get(
                                "fixtures/statistics", {"fixture": int(fixture["fixture_id"])}
                            )
                            quota = _remaining(stats)
                            observed_corners = corner_model.parse_verified_ft_corners(
                                stats.get("response"),
                                int(fixture["home_team_id"]),
                                int(fixture["away_team_id"]),
                            )
                            state = "VERIFIED_FT_CORNERS" if observed_corners is not None else "NOT_VERIFIED_PROVIDER_STATS"
                        except Exception:
                            observed_corners = None
                            state = "NOT_VERIFIED_PROVIDER_ERROR"
                        captured = datetime.now(timezone.utc).isoformat()
                        corner_attempts.append({
                            "fixture_id": fixture["fixture_id"],
                            "kickoff": fixture["kickoff"],
                            "observed_at_utc": captured,
                            "status": state,
                            "provider_requests_added": 1,
                        })
                        if observed_corners is not None:
                            corner_observations.append({
                                "fixture_id": fixture["fixture_id"],
                                "kickoff": fixture["kickoff"],
                                "home_team_id": fixture["home_team_id"],
                                "away_team_id": fixture["away_team_id"],
                                "observed_at_utc": captured,
                                "home_corners": observed_corners["home"],
                                "away_corners": observed_corners["away"],
                                "period": "FT_90",
                                "source": corner_model.SOURCE,
                                "source_response_verified": True,
                                "bet_eligible": False,
                            })
                    if corner_attempts:
                        corner_status = "VERIFIED_CAPTURES" if corner_observations else "NOT_VERIFIED_MISSING_CORNERS"
            if corner_attempts:
                tick["fe11_postgame_corner_summary"] = {
                    "version": "SOCCER_FE11_POSTGAME_CORNERS_v1.0",
                    "status": corner_status,
                    "captured": len(corner_observations),
                    "selected": len(corner_attempts),
                    "bet_eligible": False,
                    "market_inputs_used": False,
                }
                tick["fe11_postgame_corner_attempts"] = corner_attempts
                tick["fe11_postgame_corner_observations"] = corner_observations
                with persistence._connect() as conn:
                    with conn.cursor() as cur:
                        corner_writer.persist(cur, tick)
        finally:
            provider._HTTP_CLIENT = None

    with persistence._connect() as conn:
        with conn.cursor() as cur:
            inventory = writer.inventory(cur)
    return {
        "status": "VERIFIED_ALL_ELIGIBLE_LIFECYCLE" if capture["status"] == "VERIFIED_OFFICIAL_ALL_ELIGIBLE_CAPTURE"
                  else "PARTIAL_OFFICIAL_IDENTITY_NOT_VERIFIED",
        "version": VERSION,
        "source": universal.SOURCE,
        "source_observed_at_utc": capture["source_observed_at_utc"],
        "source_date_audit": date_audit,
        "provider_fixtures_seen": total_seen,
        "approved_eligible_source_rows": capture["eligible_source_count"],
        "pregame_source_accepted": capture["pregame_count"],
        "official_terminal_source_accepted": capture["official_final_count"],
        "source_invalid_identities": capture["invalid_identities"],
        "official_corner_attempts": len(corner_attempts),
        "official_corner_verified": len(corner_observations),
        "corner_status": corner_status,
        "provider_requests_added": requests,
        "last_daily_remaining": quota,
        "database_persisted": True,
        "inventory": inventory,
        "bet_eligible": False,
        "probabilities_calibrated": False,
        "market_inputs_used": False,
    }


def main() -> int:
    try:
        result = asyncio.run(run_once())
        print(json.dumps(result, ensure_ascii=False, separators=(",", ":")), flush=True)
        return 0
    except Exception as exc:
        print("FE11_UNIVERSAL_WORKER_FAILED " + type(exc).__name__,
              file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
