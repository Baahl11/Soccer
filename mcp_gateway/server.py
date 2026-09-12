import os
from datetime import date as Date, datetime, timezone as dt_timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

API_BASE_URL = os.getenv("API_FOOTBALL_BASE_URL", "https://v3.football.api-sports.io").rstrip("/")
DEFAULT_TIMEZONE = os.getenv("SOCCER_TIMEZONE", "America/Mexico_City")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("API_FOOTBALL_TIMEOUT", "20"))

TRANSPORT_SECURITY = TransportSecuritySettings(
    enable_dns_rebinding_protection=True,
    allowed_hosts=[
        "soccer-edge-api.onrender.com",
        "soccer-edge-api.onrender.com:*",
        "localhost:*",
        "127.0.0.1:*",
    ],
    allowed_origins=[
        "https://chatgpt.com",
        "https://chat.openai.com",
        "https://platform.openai.com",
    ],
)

mcp = FastMCP(
    "Soccer Edge API",
    instructions=(
        "Read-only structured soccer data gateway for SPORTS EDGE ENGINE. "
        "API-Football data is factual input, not a betting recommendation. "
        "Missing data must remain unverified; never infer injuries, lineups, odds, or statistics."
    ),
    stateless_http=True,
    json_response=True,
    transport_security=TRANSPORT_SECURITY,
)


class APIFootballError(RuntimeError):
    pass


def _api_key() -> str:
    key = os.getenv("API_FOOTBALL_KEY", "").strip()
    if not key:
        raise APIFootballError(
            "API_FOOTBALL_KEY is not configured on the server. "
            "Set it as a Render environment variable."
        )
    return key


def _positive_int(value: int, field: str) -> int:
    if value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _valid_season(season: int) -> int:
    if season < 1900 or season > 2100:
        raise ValueError("season must be a four-digit year")
    return season


def _clean_params(params: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in params.items() if value is not None and value != ""}


async def _get(endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    headers = {
        "x-apisports-key": _api_key(),
        "Accept": "application/json",
        "User-Agent": "sports-edge-engine-soccer/1.0",
    }
    url = f"{API_BASE_URL}/{endpoint.lstrip('/')}"
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        try:
            response = await client.get(url, headers=headers, params=_clean_params(params or {}))
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise APIFootballError(f"API-Football timeout on /{endpoint}") from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 429:
                raise APIFootballError("API-Football rate limit reached (HTTP 429)") from exc
            raise APIFootballError(f"API-Football HTTP error {status} on /{endpoint}") from exc
        except httpx.HTTPError as exc:
            raise APIFootballError(f"API-Football transport error on /{endpoint}") from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise APIFootballError(f"API-Football returned invalid JSON on /{endpoint}") from exc

    if not isinstance(payload, dict):
        raise APIFootballError(f"Unexpected API-Football response shape on /{endpoint}")

    api_errors = payload.get("errors")
    if api_errors:
        raise APIFootballError(f"API-Football reported errors on /{endpoint}: {api_errors}")

    return {
        "source": "API-Football v3",
        "retrieved_at_utc": datetime.now(dt_timezone.utc).isoformat(),
        "endpoint": f"/{endpoint.lstrip('/')}",
        "parameters": _clean_params(params or {}),
        "results": payload.get("results"),
        "paging": payload.get("paging"),
        "response": payload.get("response", []),
        "quota": {
            "daily_limit": response.headers.get("x-ratelimit-requests-limit"),
            "daily_remaining": response.headers.get("x-ratelimit-requests-remaining"),
            "per_minute_limit": response.headers.get("X-RateLimit-Limit"),
            "per_minute_remaining": response.headers.get("X-RateLimit-Remaining"),
        },
    }


@mcp.tool()
async def get_today_fixtures(
    match_date: str | None = None,
    timezone: str = DEFAULT_TIMEZONE,
) -> dict[str, Any]:
    """Get soccer fixtures for a calendar date in an IANA timezone.
    Defaults to the current date in America/Mexico_City.
    Use this as the structured slate source; verify configured competition eligibility separately.
    """
    try:
        zone = ZoneInfo(timezone)
    except ZoneInfoNotFoundError as exc:
        raise ValueError("timezone must be a valid IANA timezone, e.g. America/Mexico_City") from exc
    if match_date is None:
        match_date = datetime.now(zone).date().isoformat()
    try:
        Date.fromisoformat(match_date)
    except ValueError as exc:
        raise ValueError("match_date must be YYYY-MM-DD") from exc
    return await _get("fixtures", {"date": match_date, "timezone": timezone})


@mcp.tool()
async def get_fixture(fixture_id: int) -> dict[str, Any]:
    """Get one fixture by API-Football fixture_id, including current status and core metadata."""
    return await _get("fixtures", {"id": _positive_int(fixture_id, "fixture_id")})


@mcp.tool()
async def get_league_coverage(league_id: int, season: int) -> dict[str, Any]:
    """Get league-season metadata and coverage flags before calling optional downstream endpoints."""
    return await _get(
        "leagues",
        {
            "id": _positive_int(league_id, "league_id"),
            "season": _valid_season(season),
        },
    )


@mcp.tool()
async def get_team_stats(team_id: int, league_id: int, season: int) -> dict[str, Any]:
    """Get API-Football team season statistics for one team in one league-season."""
    return await _get(
        "teams/statistics",
        {
            "team": _positive_int(team_id, "team_id"),
            "league": _positive_int(league_id, "league_id"),
            "season": _valid_season(season),
        },
    )


@mcp.tool()
async def get_injuries(fixture_id: int) -> dict[str, Any]:
    """Get provider-reported injuries/suspensions for a fixture.
    Empty results mean no provider records were returned, not proof that every player is available.
    """
    return await _get("injuries", {"fixture": _positive_int(fixture_id, "fixture_id")})


@mcp.tool()
async def get_lineups(fixture_id: int) -> dict[str, Any]:
    """Get provider-reported lineups, formations, coaches and substitutes for a fixture.
    Treat incomplete/empty responses as NOT VERIFIED until a current reliable source confirms them.
    """
    return await _get("fixtures/lineups", {"fixture": _positive_int(fixture_id, "fixture_id")})


@mcp.tool()
async def get_odds(
    fixture_id: int,
    bookmaker_id: int | None = None,
    page: int = 1,
) -> dict[str, Any]:
    """Get current API-Football prematch odds for a fixture.
    Market prices must be timestamped by the calling system; do not treat stale snapshots as current.
    """
    if page < 1:
        raise ValueError("page must be >= 1")
    params: dict[str, Any] = {
        "fixture": _positive_int(fixture_id, "fixture_id"),
        "page": page,
    }
    if bookmaker_id is not None:
        params["bookmaker"] = _positive_int(bookmaker_id, "bookmaker_id")
    return await _get("odds", params)


@mcp.tool()
async def get_match_stats(fixture_id: int) -> dict[str, Any]:
    """Get provider-supported team match statistics for a fixture."""
    return await _get("fixtures/statistics", {"fixture": _positive_int(fixture_id, "fixture_id")})


@mcp.tool()
async def get_player_match_stats(fixture_id: int) -> dict[str, Any]:
    """Get provider-supported player match statistics for a fixture."""
    return await _get("fixtures/players", {"fixture": _positive_int(fixture_id, "fixture_id")})


@mcp.tool()
async def get_head_to_head(
    team_a_id: int,
    team_b_id: int,
    last: int = 10,
) -> dict[str, Any]:
    """Get recent head-to-head fixtures as supporting context only; never use raw H2H as primary evidence."""
    if last < 1 or last > 100:
        raise ValueError("last must be between 1 and 100")
    h2h = f"{_positive_int(team_a_id, 'team_a_id')}-{_positive_int(team_b_id, 'team_b_id')}"
    return await _get("fixtures/headtohead", {"h2h": h2h, "last": last})


@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request) -> Response:
    return JSONResponse(
        {
            "status": "ok",
            "service": "soccer-edge-api",
            "version": "1.0.2",
            "api_key_configured": bool(os.getenv("API_FOOTBALL_KEY", "").strip()),
        }
    )


app = mcp.streamable_http_app()
