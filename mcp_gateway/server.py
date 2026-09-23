import asyncio
import os
import sys
from datetime import date as Date, datetime, timezone as dt_timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from typing import Any

import httpx
import jwt
from jwt import PyJWKClient
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from mcp_gateway.persistence import persistence_configured
from mcp_gateway import training_dataset_v4

API_BASE_URL = os.getenv("API_FOOTBALL_BASE_URL", "https://v3.football.api-sports.io").rstrip("/")
DEFAULT_TIMEZONE = os.getenv("SOCCER_TIMEZONE", "America/Mexico_City")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("API_FOOTBALL_TIMEOUT", "20"))
GITHUB_OIDC_AUDIENCE = os.getenv("GITHUB_OIDC_AUDIENCE", "soccer-edge-render")
GITHUB_REPOSITORY = os.getenv("GITHUB_REPOSITORY_ALLOWED", "Baahl11/Soccer")
GITHUB_WORKFLOW_PATH = ".github/workflows/soccer-edge-scheduler.yml"
GITHUB_ISSUER = "https://token.actions.githubusercontent.com"
GITHUB_JWKS_URL = "https://token.actions.githubusercontent.com/.well-known/jwks"
_JWK_CLIENT = PyJWKClient(GITHUB_JWKS_URL, cache_keys=True)

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


def _compact_fixture(item: dict[str, Any]) -> dict[str, Any]:
    fixture = item.get("fixture") or {}
    league = item.get("league") or {}
    teams = item.get("teams") or {}
    home = teams.get("home") or {}
    away = teams.get("away") or {}
    status = fixture.get("status") or {}
    venue = fixture.get("venue") or {}
    return {
        "fixture_id": fixture.get("id"),
        "kickoff": fixture.get("date"),
        "timestamp": fixture.get("timestamp"),
        "timezone": fixture.get("timezone"),
        "status": status.get("short"),
        "status_long": status.get("long"),
        "elapsed": status.get("elapsed"),
        "league_id": league.get("id"),
        "league": league.get("name"),
        "country": league.get("country"),
        "season": league.get("season"),
        "round": league.get("round"),
        "home_team_id": home.get("id"),
        "home_team": home.get("name"),
        "away_team_id": away.get("id"),
        "away_team": away.get("name"),
        "venue": venue.get("name"),
        "city": venue.get("city"),
    }


def _github_oidc_claims(request: Request) -> dict[str, Any]:
    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer "):
        raise PermissionError("Missing bearer token")
    token = auth[7:].strip()
    signing_key = _JWK_CLIENT.get_signing_key_from_jwt(token)
    claims = jwt.decode(
        token,
        signing_key.key,
        algorithms=["RS256"],
        audience=GITHUB_OIDC_AUDIENCE,
        issuer=GITHUB_ISSUER,
        options={"require": ["exp", "iat", "iss", "aud", "sub"]},
    )
    if claims.get("repository") != GITHUB_REPOSITORY:
        raise PermissionError("Repository not allowed")
    workflow_ref = claims.get("workflow_ref", "")
    expected_prefix = f"{GITHUB_REPOSITORY}/{GITHUB_WORKFLOW_PATH}@"
    if not workflow_ref.startswith(expected_prefix):
        raise PermissionError("Workflow not allowed")
    if claims.get("ref") != "refs/heads/main":
        raise PermissionError("Only main branch scheduler is allowed")
    return claims


@mcp.tool()
async def get_today_fixtures(
    match_date: str | None = None,
    timezone: str = DEFAULT_TIMEZONE,
) -> dict[str, Any]:
    """Get a compact soccer slate for a calendar date in an IANA timezone."""
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

    raw = await _get("fixtures", {"date": match_date, "timezone": timezone})
    items = raw.get("response") or []
    compact = [_compact_fixture(item) for item in items[:100]]
    return {
        "source": raw.get("source"),
        "retrieved_at_utc": raw.get("retrieved_at_utc"),
        "match_date": match_date,
        "timezone": timezone,
        "total_results": len(items),
        "returned_results": len(compact),
        "truncated": len(items) > len(compact),
        "fixtures": compact,
        "quota": raw.get("quota"),
    }


@mcp.tool()
async def get_fixture(fixture_id: int) -> dict[str, Any]:
    """Get one fixture by API-Football fixture_id, including current status and core metadata."""
    return await _get("fixtures", {"id": _positive_int(fixture_id, "fixture_id")})


@mcp.tool()
async def get_league_coverage(league_id: int, season: int) -> dict[str, Any]:
    """Get league-season metadata and coverage flags before calling optional downstream endpoints."""
    return await _get("leagues", {"id": _positive_int(league_id, "league_id"), "season": _valid_season(season)})


@mcp.tool()
async def get_team_stats(team_id: int, league_id: int, season: int) -> dict[str, Any]:
    """Get API-Football team season statistics for one team in one league-season."""
    return await _get("teams/statistics", {
        "team": _positive_int(team_id, "team_id"),
        "league": _positive_int(league_id, "league_id"),
        "season": _valid_season(season),
    })


@mcp.tool()
async def get_injuries(fixture_id: int) -> dict[str, Any]:
    """Get provider-reported injuries/suspensions for a fixture. Empty results are not proof that all players are available."""
    return await _get("injuries", {"fixture": _positive_int(fixture_id, "fixture_id")})


@mcp.tool()
async def get_lineups(fixture_id: int) -> dict[str, Any]:
    """Get provider-reported lineups, formations, coaches and substitutes for a fixture."""
    return await _get("fixtures/lineups", {"fixture": _positive_int(fixture_id, "fixture_id")})


@mcp.tool()
async def get_odds(fixture_id: int, bookmaker_id: int | None = None, page: int = 1) -> dict[str, Any]:
    """Get current API-Football prematch odds for a fixture."""
    if page < 1:
        raise ValueError("page must be >= 1")
    params: dict[str, Any] = {"fixture": _positive_int(fixture_id, "fixture_id"), "page": page}
    if bookmaker_id is not None:
        params["bookmaker"] = _positive_int(bookmaker_id, "bookmaker_id")
    return await _get("odds", params)


@mcp.tool()
async def get_prematch_bet_types(
    bet_id: int | None = None,
    search: str | None = None,
) -> dict[str, Any]:
    """Get the API-Football /odds/bets prematch market catalog for audit/reference."""
    params: dict[str, Any] = {}
    if bet_id is not None:
        params["id"] = _positive_int(bet_id, "bet_id")
    if search is not None:
        value = str(search).strip()
        if value:
            params["search"] = value
    return await _get("odds/bets", params)


@mcp.tool()
async def get_prematch_bookmakers(
    bookmaker_id: int | None = None,
    search: str | None = None,
) -> dict[str, Any]:
    """Get the API-Football /odds/bookmakers catalog for verified bookmaker identity."""
    params: dict[str, Any] = {}
    if bookmaker_id is not None:
        params["id"] = _positive_int(bookmaker_id, "bookmaker_id")
    if search is not None:
        value = str(search).strip()
        if value:
            params["search"] = value
    return await _get("odds/bookmakers", params)


@mcp.tool()
async def get_match_stats(fixture_id: int) -> dict[str, Any]:
    """Get provider-supported team match statistics for a fixture."""
    return await _get("fixtures/statistics", {"fixture": _positive_int(fixture_id, "fixture_id")})


@mcp.tool()
async def get_player_match_stats(fixture_id: int) -> dict[str, Any]:
    """Get provider-supported player match statistics for a fixture."""
    return await _get("fixtures/players", {"fixture": _positive_int(fixture_id, "fixture_id")})


@mcp.tool()
async def get_head_to_head(team_a_id: int, team_b_id: int, last: int = 10) -> dict[str, Any]:
    """Get recent head-to-head fixtures as supporting context only."""
    if last < 1 or last > 100:
        raise ValueError("last must be between 1 and 100")
    h2h = f"{_positive_int(team_a_id, 'team_a_id')}-{_positive_int(team_b_id, 'team_b_id')}"
    return await _get("fixtures/headtohead", {"h2h": h2h, "last": last})


@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request) -> Response:
    return JSONResponse({
        "status": "ok",
        "service": "soccer-edge-api",
        "version": "1.4.0",
        "api_key_configured": bool(os.getenv("API_FOOTBALL_KEY", "").strip()),
        "scheduler_endpoint": True,
        "scheduler_isolated_worker": True,
        "database_persistence_configured": persistence_configured(),
    })


@mcp.custom_route("/internal/tick", methods=["POST"])
async def internal_tick(request: Request) -> Response:
    try:
        _github_oidc_claims(request)
    except Exception as exc:
        return JSONResponse({"error": "unauthorized", "detail": str(exc)[:200]}, status_code=401)

    env = os.environ.copy()
    env.setdefault("MALLOC_ARENA_MAX", "2")
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "mcp_gateway.tick_worker",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stderr_lines: list[str] = []

        async def _pump_worker_stderr() -> None:
            assert proc.stderr is not None
            while True:
                raw = await proc.stderr.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").rstrip()
                stderr_lines.append(line)
                if len(stderr_lines) > 200:
                    del stderr_lines[:-200]

        assert proc.stdout is not None
        stderr_task = asyncio.create_task(_pump_worker_stderr())
        stdout_task = asyncio.create_task(proc.stdout.read())
        try:
            await asyncio.wait_for(proc.wait(), timeout=420)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            await stderr_task
            stdout_task.cancel()
            return JSONResponse({"error": "tick_timeout"}, status_code=504)

        await stderr_task
        stdout = await stdout_task
        stderr_text = "\n".join(stderr_lines)
        if proc.returncode != 0:
            detail = stderr_text[-1000:]
            return JSONResponse({"error": "tick_failed", "detail": detail}, status_code=500)
        if not stdout:
            return JSONResponse({"error": "tick_failed", "detail": "worker returned empty output"}, status_code=500)

        # Return worker JSON bytes directly. Avoid parsing and re-serializing the
        # potentially large payload in the long-lived MCP process.
        return Response(content=stdout, media_type="application/json", status_code=200)
    except Exception as exc:
        return JSONResponse({"error": "tick_failed", "detail": str(exc)[:500]}, status_code=500)


@mcp.custom_route("/internal/training-dataset/build", methods=["POST"])
async def internal_training_dataset_build(request: Request) -> Response:
    try:
        _github_oidc_claims(request)
    except Exception as exc:
        return JSONResponse({"error": "unauthorized", "detail": str(exc)[:200]}, status_code=401)

    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}

    cutoff = body.get("cutoff")
    backfill_limit = body.get("backfill_limit", 5000)
    try:
        backfill_limit = max(1, min(int(backfill_limit), 20000))
    except (TypeError, ValueError):
        return JSONResponse({"error": "invalid_backfill_limit"}, status_code=400)

    try:
        result = await asyncio.to_thread(
            training_dataset_v4.build_and_persist,
            cutoff=str(cutoff) if cutoff else None,
            backfill_limit=backfill_limit,
        )
        return JSONResponse(result)
    except Exception as exc:
        return JSONResponse(
            {"error": "training_dataset_build_failed", "detail": str(exc)[:500]},
            status_code=500,
        )


app = mcp.streamable_http_app()
