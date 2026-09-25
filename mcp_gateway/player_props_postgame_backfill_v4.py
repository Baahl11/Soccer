from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import time
from typing import Any

from mcp_gateway import automation_v2 as v2
from mcp_gateway import persistence as persistence_base
from mcp_gateway import player_props_phase15_coverage_audit as coverage_audit
from mcp_gateway import player_trends

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_PLAYER_PROPS_POSTGAME_BACKFILL_V4_1.0.0"
MAX_FIXTURES_PER_RUN = 5
MIN_DAILY_REMAINING = 50
MIN_REQUEST_INTERVAL_SECONDS = 0.8


def select_candidate_fixture_ids(audit: dict[str, Any], *, max_fixtures: int = MAX_FIXTURES_PER_RUN) -> list[int]:
    selected: list[int] = []
    seen: set[int] = set()
    for row in audit.get("fixtures") or []:
        if not isinstance(row, dict) or row.get("fixture_id") is None:
            continue
        families = row.get("families") if isinstance(row.get("families"), dict) else {}
        is_candidate = any(
            isinstance(value, dict)
            and value.get("oos_recoverability") == "PROVIDER_BACKFILL_CANDIDATE"
            for value in families.values()
        )
        if not is_candidate:
            continue
        fid = int(row["fixture_id"])
        if fid in seen:
            continue
        selected.append(fid)
        seen.add(fid)
        if len(selected) >= max_fixtures:
            break
    return selected


def make_backfill_event(
    fixture_id: int,
    compact: dict[str, Any],
    *,
    provider_daily_remaining: int | None,
) -> dict[str, Any]:
    stats = dict(compact)
    stats["fixture_id"] = int(fixture_id)
    stats["capture_phase"] = "POSTGAME_BACKFILL"
    stats["finalized_fixture_required"] = True
    stats["goalkeeper_fields_retained"] = ["saves", "goals_conceded"]
    stats["player_card_fields_retained"] = ["yellow_cards", "red_cards"]
    return {
        "event_type": "RESEARCH_BACKFILL",
        "stage": "POSTGAME_BACKFILL",
        "fixture": {"fixture_id": int(fixture_id)},
        "classification": "PASS",
        "bet_eligible": False,
        "actionable": False,
        "decision_weight": 0.0,
        "postgame_player_stats": stats,
        "backfill": {
            "reason": "PHASE15_OOS_OUTCOME_RECOVERY",
            "pregame_signal_required": True,
            "finalized_result_required": True,
            "retroactive_pregame_signal_created": False,
            "retroactive_market_created": False,
            "provider_daily_remaining_after_call": provider_daily_remaining,
        },
    }


def _has_players(compact: dict[str, Any]) -> bool:
    return any(
        isinstance(team, dict) and bool(team.get("players"))
        for team in (compact.get("teams") or [])
    )


def _already_materialized(conn, fixture_id: int) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1
            FROM soccer_refresh_events
            WHERE fixture_id = %s
              AND stage IN ('POSTGAME','POSTGAME_BACKFILL')
              AND payload ? 'postgame_player_stats'
            LIMIT 1
            """,
            (int(fixture_id),),
        )
        return cur.fetchone() is not None


def _persist_backfill(conn, event: dict[str, Any], *, generated_at: datetime) -> None:
    fixture_id = int((event.get("fixture") or {}).get("fixture_id"))
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO soccer_refresh_events (
                fixture_id, stage, event_type, classification, availability_confidence,
                bet_eligible, data_tier, generated_at, payload
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
            """,
            (
                fixture_id,
                "POSTGAME_BACKFILL",
                "RESEARCH_BACKFILL",
                "PASS",
                None,
                False,
                None,
                generated_at,
                json.dumps(event),
            ),
        )


async def run_backfill(*, lookback_days: int = 180, max_fixtures: int = MAX_FIXTURES_PER_RUN) -> dict[str, Any]:
    lookback_days = max(1, min(int(lookback_days), 730))
    max_fixtures = max(1, min(int(max_fixtures), MAX_FIXTURES_PER_RUN))
    if not persistence_base.persistence_configured():
        return {
            "schema_version": SCHEMA_VERSION,
            "model_version": MODEL_VERSION,
            "status": "POSTGRES_NOT_CONFIGURED",
            "provider_requests_added": 0,
            "production_promotion_allowed": False,
        }

    persistence_base.ensure_schema()
    with persistence_base._connect() as conn:
        pregame, postgame, finalized = coverage_audit._load_rows(
            conn,
            lookback_days=lookback_days,
            max_rows=50000,
        )
        audit = coverage_audit.build_audit(pregame, postgame, finalized)
        candidates = select_candidate_fixture_ids(audit, max_fixtures=max_fixtures)

        attempted = captured = already_materialized = unavailable = 0
        details: list[dict[str, Any]] = []
        daily_remaining = v2._LAST_DAILY_REMAINING
        last_request_started = 0.0

        for fixture_id in candidates:
            if _already_materialized(conn, fixture_id):
                already_materialized += 1
                details.append({"fixture_id": fixture_id, "status": "ALREADY_MATERIALIZED"})
                continue
            if fixture_id not in finalized:
                details.append({"fixture_id": fixture_id, "status": "SKIPPED_NOT_FINALIZED"})
                continue
            if daily_remaining is not None and daily_remaining <= MIN_DAILY_REMAINING:
                details.append({"fixture_id": fixture_id, "status": "SKIPPED_DAILY_RESERVE_GUARD"})
                break

            wait_for = MIN_REQUEST_INTERVAL_SECONDS - (time.monotonic() - last_request_started)
            if wait_for > 0:
                await asyncio.sleep(wait_for)

            attempted += 1
            last_request_started = time.monotonic()
            try:
                raw = await v2._ORIGINAL_API_GET("fixtures/players", {"fixture": int(fixture_id)})
                remaining = (raw.get("quota") or {}).get("daily_remaining")
                try:
                    if remaining is not None:
                        daily_remaining = int(remaining)
                        v2._LAST_DAILY_REMAINING = daily_remaining
                except (TypeError, ValueError):
                    pass

                compact = player_trends._compact(raw)
                if not _has_players(compact):
                    unavailable += 1
                    details.append({
                        "fixture_id": fixture_id,
                        "status": "PROVIDER_RETURNED_NO_PLAYER_ROWS",
                        "daily_remaining": daily_remaining,
                    })
                    continue

                event = make_backfill_event(
                    fixture_id,
                    compact,
                    provider_daily_remaining=daily_remaining,
                )
                _persist_backfill(conn, event, generated_at=datetime.now(timezone.utc))
                captured += 1
                details.append({
                    "fixture_id": fixture_id,
                    "status": "CAPTURED",
                    "player_rows": sum(
                        len(team.get("players") or [])
                        for team in compact.get("teams") or []
                        if isinstance(team, dict)
                    ),
                    "daily_remaining": daily_remaining,
                })
            except Exception as exc:
                unavailable += 1
                details.append({
                    "fixture_id": fixture_id,
                    "status": "PROVIDER_ERROR",
                    "error": str(exc)[:180],
                    "daily_remaining": daily_remaining,
                })

    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": "PLAYER_PROP_POSTGAME_BACKFILL_COMPLETE",
        "candidate_fixtures": candidates,
        "candidate_fixture_count": len(candidates),
        "attempted": attempted,
        "captured": captured,
        "already_materialized": already_materialized,
        "unavailable": unavailable,
        "details": details,
        "provider_requests_added": attempted,
        "max_provider_requests_per_run": MAX_FIXTURES_PER_RUN,
        "daily_remaining_after_run": daily_remaining,
        "decision_weight": 0.0,
        "production_promotion_allowed": False,
        "policy": (
            "ONLY FINALIZED PLAYER OUTCOMES MAY BE BACKFILLED; AN ORIGINAL PREGAME MODEL SIGNAL "
            "MUST ALREADY EXIST; NEVER CREATE RETROACTIVE XI, MODEL PROBABILITIES, MARKET PRICES, "
            "OR CLOSING LINES; MAX FIVE PROVIDER REQUESTS PER RUN; STOP AT DAILY RESERVE GUARD."
        ),
    }
