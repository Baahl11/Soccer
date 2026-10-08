from __future__ import annotations

from pathlib import Path

from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles


DIST_DIR = Path(__file__).resolve().parents[1] / "web_v3" / "dist"
ASSETS_DIR = DIST_DIR / "assets"


async def app_page(request):
    return FileResponse(
        DIST_DIR / "index.html",
        media_type="text/html",
        headers={"Cache-Control": "no-store"},
    )


assets = StaticFiles(directory=str(ASSETS_DIR), check_dir=False)
