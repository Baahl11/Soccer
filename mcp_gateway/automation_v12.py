from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone as dt_timezone
from typing import Any

from mcp_gateway import automation as base
from mcp_gateway import automation_v6 as v6
from mcp_gateway import automation_v10 as v10
from mcp_gateway import automation_v11 as v11
from mcp_gateway.galaxyparlay_client import configured as galaxy_configured

MODEL_VERSION = "SOCCER EDGE ENGINE v1.1"
AUTOMATION_VERSION = "2.1.0"

_ORIGINAL_ADAPTIVE_API_GET = v6._adaptive_paced_api_get

SLATE_NEAR_WINDOW = timedelta(
    minutes=int(os.getenv("GALAXYPARLAY_SLATE_NEAR_WINDOW_MINUTES", "180"))
)
SLATE_NEAR_MAX_AGE = timedelta(
    minutes=int(os.getenv("GALAXYPARLAY_SLATE_NEAR_MAX_AGE_MINUTES", "30"))
)
SLATE_FAR_MAX_AGE = timedelta(
    hours=int(os.getenv("GALAXYPARLAY_SLATE_FAR_MAX_AGE_HOURS", "12"))
)
RECONCILIATION_WINDOW_MINUTES = int(
    os.getenv("SOCCER_EDGE_SLATE_RECONCILIATION_WINDOW_MINUTES", "10")
)

_SLATE_METRICS: dict[str, int] = {}


def _reset_slate_metrics() -> None:
    _SLATE_METRICS.clear()
    _SLATE_METRICS.update(
        galaxy_slate_reads=0,
        galaxy_slate_hits=0,
        galaxy_slate_fallbacks=0,
        galaxy_slate_stale_rejections=0,
        api_slate_calls_avoided=0,
        api_slate_reconciliation_calls=0,
        duplicate_requests_detected=0,
    )


def _bump(key: str, amount: int = 1) -> None:
    _SLATE_METRICS[key] = int(_SLATE_METRICS.get(key, 0)) + amount


def _dt(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        result = value
    else:
        text = str(value).strip().replace("Z", "+00:00")
        try:
            result = datetime.fromisoformat(text)
        except ValueError:
            return None
    if result.tzinfo is None:
        result = result.replace(tzinfo=dt_timezone.utc)
    return result.astimezone(dt_timezone.utc)


def _status_long(short: str | None) -> str | None:
    return {
        "NS": "Not Started",
        "TBD": "Time To Be Defined",
        "1H": "First Half",
        "HT": "Halftime",
        "2H": "Second Half",
        "ET": "Extra Time",
        "P": "Penalty In Progress",
        "FT": "Match Finished",
        "AET": "Match Finished After Extra Time",
        "PEN": "Match Finished After Penalties",
        "BT": "Break Time",
        "SUSP": "Match Suspended",
        "INT": "Match Interrupted",
        "PST": "Match Postponed",
        "CANC": "Match Cancelled",
        "ABD": "Match Abandoned",
        "AWD": "Technical Loss",
        "WO": "WalkOver",
        "LIVE": "In Progress",
    }.get(str(short or "").upper())


def _galaxy_fixture_to_api_row(row: dict[str, Any]) -> dict[str, Any] | None:
    fixture = row.get("fixture") or {}
    if not isinstance(fixture, dict):
        return None
    required = (
        fixture.get("fixture_id"),
        fixture.get("kickoff_time"),
        fixture.get("home_team_id"),
        fixture.get("away_team_id"),
        fixture.get("league_id"),
        fixture.get("season"),
    )
    if any(value in {None, ""} for value in required):
        return None

    kickoff_dt = _dt(fixture.get("kickoff_time"))
    status = str(fixture.get("status") or "")
    home_score = fixture.get("home_score")
    away_score = fixture.get("away_score")

    return {
        "fixture": {
            "id": fixture.get("fixture_id"),
            "date": fixture.get("kickoff_time"),
            "timestamp": int(kickoff_dt.timestamp()) if kickoff_dt else None,
            "timezone": base.TIMEZONE_NAME,
            "status": {
                "short": status,
                "long": _status_long(status),
                "elapsed": None,
            },
            "venue": {
                "name": fixture.get("venue"),
                "city": fixture.get("city"),
            },
        },
        "league": {
            "id": fixture.get("league_id"),
            "name": None,
            "country": None,
            "season": fixture.get("season"),
            "round": fixture.get("round"),
        },
        "teams": {
            "home": {
                "id": fixture.get("home_team_id"),
                "name": fixture.get("home_team_name"),
            },
            "away": {
                "id": fixture.get("away_team_id"),
                "name": fixture.get("away_team_name"),
            },
        },
        "goals": {"home": home_score, "away": away_score},
        "score": {
            "halftime": {
                "home": fixture.get("halftime_home"),
                "away": fixture.get("halftime_away"),
            },
            "fulltime": {"home": home_score, "away": away_score},
            "extratime": {"home": None, "away": None},
            "penalty": {"home": None, "away": None},
        },
    }


def _row_materially_stale(row: dict[str, Any], now: datetime) -> bool:
    fixture = row.get("fixture") or {}
    if not isinstance(fixture, dict):
        return True
    kickoff = _dt(fixture.get("kickoff_time"))
    updated = _dt(fixture.get("updated_at"))
    if kickoff is None:
        return True
    if updated is None:
        return abs(kickoff - now) <= SLATE_NEAR_WINDOW

    max_age = (
        SLATE_NEAR_MAX_AGE
        if abs(kickoff - now) <= SLATE_NEAR_WINDOW
        else SLATE_FAR_MAX_AGE
    )
    age = now - updated
    return age > max_age or age < timedelta(minutes=-5)


def _hourly_reconciliation_due() -> bool:
    local_now = datetime.now(base.TIMEZONE)
    window = max(1, min(30, RECONCILIATION_WINDOW_MINUTES))
    return local_now.minute < window


async def _galaxy_slate_payload(match_date: str, now: datetime) -> dict[str, Any] | None:
    client = v10._GALAXY_CLIENT
    if client is None or not galaxy_configured():
        return None

    _bump("galaxy_slate_reads")
    base_url = os.getenv("GALAXYPARLAY_BASE_URL", "").strip().rstrip("/")
    if not base_url:
        return None

    try:
        response = await client.get(
            f"{base_url}/api/sports-edge/v1/slate",
            params={
                "match_date": match_date,
                "limit": 500,
                "core_only": "true",
                "include_features": "false",
            },
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return None

    if not isinstance(payload, dict) or payload.get("external_api_calls") not in {0, "0"}:
        return None
    rows = payload.get("fixtures") or []
    if not isinstance(rows, list) or not rows:
        return None

    if any(
        _row_materially_stale(row, now)
        for row in rows
        if isinstance(row, dict)
    ):
        _bump("galaxy_slate_stale_rejections")
        return None

    response_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        converted = _galaxy_fixture_to_api_row(row)
        if converted is not None:
            response_rows.append(converted)
    if not response_rows:
        return None

    _bump("galaxy_slate_hits")
    _bump("api_slate_calls_avoided")
    return {
        "source": "GalaxyParlay persisted data",
        "retrieved_at_utc": now.isoformat(),
        "endpoint": "/api/sports-edge/v1/slate",
        "parameters": {"match_date": match_date},
        "results": len(response_rows),
        "response": response_rows,
        "quota": {},
        "galaxy_contract_version": payload.get("contract_version"),
        "external_api_calls": 0,
    }


async def _galaxy_first_slate_api_get(
    endpoint: str, params: dict[str, Any]
) -> dict[str, Any]:
    is_daily_slate = (
        endpoint.strip("/") == "fixtures"
        and bool(params.get("date"))
        and not params.get("id")
        and not params.get("team")
        and not params.get("last")
    )
    if not is_daily_slate:
        return await _ORIGINAL_ADAPTIVE_API_GET(endpoint, params)

    if _hourly_reconciliation_due():
        _bump("api_slate_reconciliation_calls")
        return await _ORIGINAL_ADAPTIVE_API_GET(endpoint, params)

    now = datetime.now(dt_timezone.utc)
    galaxy_payload = await _galaxy_slate_payload(str(params.get("date")), now)
    if galaxy_payload is not None:
        return galaxy_payload

    _bump("galaxy_slate_fallbacks")
    return await _ORIGINAL_ADAPTIVE_API_GET(endpoint, params)


async def run_tick() -> dict[str, Any]:
    _reset_slate_metrics()
    previous_adaptive = v6._adaptive_paced_api_get
    v6._adaptive_paced_api_get = _galaxy_first_slate_api_get
    try:
        payload = await v11.run_tick()
    finally:
        v6._adaptive_paced_api_get = previous_adaptive

    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    metrics = dict(payload.get("galaxy_first_metrics") or {})
    metrics.update(_SLATE_METRICS)
    metrics["provider_requests_avoided"] = sum(
        int(metrics.get(key, 0) or 0)
        for key in (
            "api_team_stats_calls_avoided",
            "api_recent_calls_avoided",
            "api_odds_calls_avoided",
            "api_slate_calls_avoided",
        )
    )
    payload["galaxy_first_metrics"] = metrics
    policy = dict(payload.get("duplicate_request_policy") or {})
    policy["slate"] = (
        "GALAXY_FIRST_EXCEPT_ONE_HOURLY_AUTHORITATIVE_API_RECONCILIATION; "
        "API_FOOTBALL_FALLBACK_ONLY_IF_GALAXY_MISSING_OR_MATERIALLY_STALE"
    )
    payload["duplicate_request_policy"] = policy
    payload["slate_source_policy"] = {
        "normal_ticks": "GALAXYPARLAY_PERSISTED_FIRST",
        "authoritative_reconciliation": "API_FOOTBALL_ONCE_PER_HOUR",
        "near_kickoff_max_age_minutes": int(SLATE_NEAR_MAX_AGE.total_seconds() // 60),
        "far_fixture_max_age_hours": int(SLATE_FAR_MAX_AGE.total_seconds() // 3600),
        "duplicate_requests_detected_target": 0,
    }
    return payload
