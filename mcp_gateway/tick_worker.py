import asyncio
import json
import logging
import sys

import httpx

from mcp_gateway import automation_v6, automation_v56
from mcp_gateway.persistence_v2 import persist_tick

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

SHORTLIST_STATE_URL = (
    "https://raw.githubusercontent.com/Baahl11/Soccer/"
    "soccer-edge-state/soccer_edge_state/shortlist_state.json"
)


def _read_seed() -> dict:
    try:
        if not sys.stdin.isatty():
            raw = sys.stdin.buffer.read()
            if raw:
                payload = json.loads(raw.decode("utf-8"))
                if isinstance(payload, dict):
                    seed = payload.get("shortlist_state")
                    if isinstance(seed, dict):
                        return seed
    except Exception:
        pass
    try:
        response = httpx.get(SHORTLIST_STATE_URL, timeout=5.0, follow_redirects=True)
        if response.status_code == 200:
            seed = response.json()
            return seed if isinstance(seed, dict) else {}
    except Exception:
        pass
    return {}


async def _main() -> int:
    try:
        imported = automation_v6.import_shortlist_state(_read_seed())
        payload = await automation_v56.run_tick()
        payload["shortlist_seed_imported"] = imported
        try:
            db_payload = dict(payload)
            db_payload.pop("shortlist_state", None)
            payload["database_persisted"] = persist_tick(db_payload)
        except Exception as db_exc:
            payload["database_persisted"] = False
            payload["database_error"] = str(db_exc)[:300]
        sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        sys.stdout.flush()
        return 0
    except Exception as exc:
        sys.stderr.write(f"tick_failed: {str(exc)[:500]}\n")
        sys.stderr.flush()
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
