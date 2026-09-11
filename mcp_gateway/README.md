# Soccer Edge API MCP Gateway

Read-only MCP gateway for SPORTS EDGE ENGINE. It exposes selected API-Football v3 endpoints as structured tools while keeping `API_FOOTBALL_KEY` server-side.

## Render

Build command:

```bash
pip install -r mcp_gateway/requirements.txt
```

Start command:

```bash
uvicorn mcp_gateway.server:app --host 0.0.0.0 --port $PORT
```

Required secret:

```text
API_FOOTBALL_KEY=<set only in Render Environment>
```

Non-secret defaults:

```text
API_FOOTBALL_BASE_URL=https://v3.football.api-sports.io
API_FOOTBALL_TIMEOUT=20
SOCCER_TIMEZONE=America/Mexico_City
```

## Endpoints

- `/health` — health check; exposes only whether the key is configured, never the key itself.
- `/mcp` — MCP Streamable HTTP endpoint.

## Tools

- `get_today_fixtures`
- `get_fixture`
- `get_league_coverage`
- `get_team_stats`
- `get_injuries`
- `get_lineups`
- `get_odds`
- `get_match_stats`
- `get_player_match_stats`
- `get_head_to_head`

The gateway does not produce betting recommendations. Missing data remains unverified. API-Football predictions are intentionally not exposed as a raw-model input.
