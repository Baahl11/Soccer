from __future__ import annotations

from pathlib import Path

from starlette.responses import FileResponse, JSONResponse
from starlette.staticfiles import StaticFiles

from mcp_gateway import supabase_auth_v4


DIST_DIR = Path(__file__).resolve().parents[1] / "web_v3" / "dist"
ASSETS_DIR = DIST_DIR / "assets"


async def app_page(request):
    return FileResponse(
        DIST_DIR / "index.html",
        media_type="text/html",
        headers={"Cache-Control": "no-store"},
    )


async def auth_config(request):
    auth = supabase_auth_v4.public_auth_config()
    return JSONResponse(
        {
            "supabase_url": auth.get("project_url"),
            "publishable_key": auth.get("publishable_key"),
            "auth_configured": bool(auth.get("configured")),
            "api_base": "/app/api/v2",
        },
        headers={"Cache-Control": "no-store"},
    )


assets = StaticFiles(directory=str(ASSETS_DIR), check_dir=False)
