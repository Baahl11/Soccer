import asyncio
import json
import logging
import sys

import httpx

from mcp_gateway import automation_v6, automation_v7, automation_v123, product_views_v4
from mcp_gateway.persistence_v2 import persist_tick

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

SHORTLIST_STATE_URL = (
    "https://raw.githubusercontent.com/Baahl11/Soccer/"
    "soccer-edge-state/soccer_edge_state/shortlist_state.json"
)
FAIRNESS_STATE_URL = (
    "https://raw.githubusercontent.com/Baahl11/Soccer/"
    "soccer-edge-state/soccer_edge_state/scheduler_fairness_state.json"
)


def _remote_state(url: str) -> dict:
    try:
        response = httpx.get(url, timeout=5.0, follow_redirects=True)
        if response.status_code == 200:
            seed = response.json()
            return seed if isinstance(seed, dict) else {}
    except Exception:
        pass
    return {}


def _read_seeds() -> tuple[dict, dict]:
    shortlist_seed: dict = {}
    fairness_seed: dict = {}
    try:
        if not sys.stdin.isatty():
            raw = sys.stdin.buffer.read()
            if raw:
                payload = json.loads(raw.decode("utf-8"))
                if isinstance(payload, dict):
                    candidate = payload.get("shortlist_state")
                    if isinstance(candidate, dict):
                        shortlist_seed = candidate
                    candidate = payload.get("scheduler_fairness_state")
                    if isinstance(candidate, dict):
                        fairness_seed = candidate
    except Exception:
        pass

    if not shortlist_seed:
        shortlist_seed = _remote_state(SHORTLIST_STATE_URL)
    if not fairness_seed:
        fairness_seed = _remote_state(FAIRNESS_STATE_URL)
    return shortlist_seed, fairness_seed


def _normalize_refresh_event_model_lineage(payload: dict) -> dict:
    """Make persisted refresh-event lineage match the final runtime tick lineage.

    Older refresh-event constructors still carry a legacy v1.0 literal. The
    top-level tick model_version is the canonical runtime lineage used by the
    OOS ledger, so normalize only metadata immediately before persistence.
    This does not alter projections, decisions, thresholds, gates, or stages.
    """
    runtime_model_version = payload.get("model_version")
    normalized = 0
    mismatched = 0
    if runtime_model_version:
        for event in payload.get("events") or []:
            if not isinstance(event, dict) or event.get("event_type") != "SOCCER_REFRESH":
                continue
            previous = event.get("model_version")
            if previous != runtime_model_version:
                mismatched += 1
                event["model_version"] = runtime_model_version
                normalized += 1

    result = {
        "status": "NORMALIZED_TO_RUNTIME_TICK" if normalized else "ALREADY_ALIGNED",
        "runtime_model_version": runtime_model_version,
        "mismatched_refresh_events": mismatched,
        "normalized_refresh_events": normalized,
        "prediction_logic_changed": False,
        "thresholds_changed": False,
        "gates_changed": False,
    }
    payload["model_lineage_normalization"] = result
    return result


async def _main() -> int:
    try:
        shortlist_seed, fairness_seed = _read_seeds()
        imported = automation_v6.import_shortlist_state(shortlist_seed)
        fairness_imported = automation_v7.fair_scheduler.import_state(fairness_seed)
        payload = await automation_v123.run_tick()
        payload.setdefault("status", "ok")
        payload["shortlist_seed_imported"] = imported
        payload["fair_scheduler_seed_imported"] = fairness_imported
        _normalize_refresh_event_model_lineage(payload)
        try:
            # Avoid retaining a second top-level payload mapping on the 512 MB
            # Render instance. shortlist_state is durable scheduler handoff data,
            # not relational tick history, so remove it only while persisting.
            shortlist_state = payload.pop("shortlist_state", None)
            scheduler_fairness_state = payload.pop("scheduler_fairness_state", None)
            try:
                payload["database_persisted"] = persist_tick(payload)
            finally:
                if shortlist_state is not None:
                    payload["shortlist_state"] = shortlist_state
                if scheduler_fairness_state is not None:
                    payload["scheduler_fairness_state"] = scheduler_fairness_state
        except Exception as db_exc:
            payload["database_persisted"] = False
            payload["database_error"] = str(db_exc)[:300]

        # Phase24 views were initially assembled upstream before the final
        # price-resolver/CLV annotations and before persistence completed.
        # Rebuild once here from the final payload so the returned/state-branch
        # snapshot reflects final API caps, maturation counters and DB health.
        try:
            product = product_views_v4.build_views(payload, limit=product_views_v4.MAX_ROWS_PER_VIEW)
            payload["dashboard_views"] = product["views"]
            phase24 = payload.get("phase24_final_product_experience")
            if isinstance(phase24, dict):
                phase24["model_version"] = product_views_v4.MODEL_VERSION
                phase24["row_limit_per_view"] = product["row_limit_per_view"]
                phase24["dashboard_views_refreshed_after_persistence"] = True
        except Exception as dashboard_exc:
            payload["dashboard_refresh_error"] = str(dashboard_exc)[:300]

        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        sys.stdout.write(encoded)
        sys.stdout.flush()
        return 0
    except Exception as exc:
        sys.stderr.write(f"tick_failed: {str(exc)[:500]}\n")
        sys.stderr.flush()
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))