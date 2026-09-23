import asyncio
import json
import logging
import os
import resource
import sys

import httpx

from mcp_gateway import automation_v6, automation_v92
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


def _rss_mb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)


def _probe(label: str) -> None:
    if os.getenv("SOCCER_EDGE_MEMORY_PROBE", "1") == "1":
        sys.stderr.write(f"WORKER_MEM {label} peak_rss_mb={_rss_mb()}\\n")
        sys.stderr.flush()


async def _main() -> int:
    try:
        imported = automation_v6.import_shortlist_state(_read_seed())
        _probe("before_run_tick")
        payload = await automation_v92.run_tick()
        _probe("after_run_tick")
        payload["shortlist_seed_imported"] = imported
        try:
            # Avoid retaining a second top-level payload mapping on the 512 MB
            # Render instance. shortlist_state is durable scheduler handoff data,
            # not relational tick history, so remove it only while persisting.
            shortlist_state = payload.pop("shortlist_state", None)
            try:
                _probe("before_persist_tick")
                payload["database_persisted"] = persist_tick(payload)
                _probe("after_persist_tick")
            finally:
                if shortlist_state is not None:
                    payload["shortlist_state"] = shortlist_state
        except Exception as db_exc:
            payload["database_persisted"] = False
            payload["database_error"] = str(db_exc)[:300]
        _probe("before_json_dumps")
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        _probe("after_json_dumps")
        sys.stdout.write(encoded)
        sys.stdout.flush()
        _probe("after_stdout_flush")
        return 0
    except Exception as exc:
        sys.stderr.write(f"tick_failed: {str(exc)[:500]}\n")
        sys.stderr.flush()
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
