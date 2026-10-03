import asyncio
import json
import logging
import sys
import time

import httpx

from mcp_gateway import automation_v6, automation_v7, automation_v129, product_views_v4
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


def _emit_timing(stage: str, **fields: object) -> None:
    record = {"stage": stage, **fields}
    sys.stderr.write(
        "TICK_TIMING "
        + json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=str)
        + "\n"
    )
    sys.stderr.flush()


def _elapsed_ms(started: float) -> int:
    return int(round((time.monotonic() - started) * 1000.0))


async def _main() -> int:
    total_started = time.monotonic()
    timings: dict[str, int] = {}
    restorers: list[tuple[object, str, object]] = []

    def install_async_timing(module: object, attr: str, stage: str) -> None:
        original = getattr(module, attr)
        restorers.append((module, attr, original))
        calls_key = f"{stage}_calls"
        elapsed_key = f"{stage}_ms"

        async def wrapped(*args, **kwargs):
            call_index = int(timings.get(calls_key, 0)) + 1
            timings[calls_key] = call_index
            started = time.monotonic()
            _emit_timing(f"{stage}_start", call=call_index)
            try:
                return await original(*args, **kwargs)
            finally:
                elapsed = _elapsed_ms(started)
                timings[elapsed_key] = int(timings.get(elapsed_key, 0)) + elapsed
                _emit_timing(f"{stage}_done", call=call_index, elapsed_ms=elapsed)

        setattr(module, attr, wrapped)

    def install_sync_timing(module: object, attr: str, stage: str) -> None:
        original = getattr(module, attr)
        restorers.append((module, attr, original))
        calls_key = f"{stage}_calls"
        elapsed_key = f"{stage}_ms"

        def wrapped(*args, **kwargs):
            call_index = int(timings.get(calls_key, 0)) + 1
            timings[calls_key] = call_index
            started = time.monotonic()
            _emit_timing(f"{stage}_start", call=call_index)
            try:
                return original(*args, **kwargs)
            finally:
                elapsed = _elapsed_ms(started)
                timings[elapsed_key] = int(timings.get(elapsed_key, 0)) + elapsed
                _emit_timing(f"{stage}_done", call=call_index, elapsed_ms=elapsed)

        setattr(module, attr, wrapped)

    try:
        stage_started = time.monotonic()
        _emit_timing("seed_state_start")
        shortlist_seed, fairness_seed = _read_seeds()
        timings["seed_state_ms"] = _elapsed_ms(stage_started)
        _emit_timing("seed_state_done", elapsed_ms=timings["seed_state_ms"])

        stage_started = time.monotonic()
        imported = automation_v6.import_shortlist_state(shortlist_seed)
        fairness_imported = automation_v7.fair_scheduler.import_state(fairness_seed)
        timings["seed_import_ms"] = _elapsed_ms(stage_started)
        _emit_timing("seed_import_done", elapsed_ms=timings["seed_import_ms"])

        # Diagnostic-only nested timing. Every wrapper delegates to the exact
        # existing function and is restored immediately after automation_v129.
        # No provider budget, selection logic, thresholds, gates or persistence
        # semantics are changed by these probes.
        v128 = automation_v129.v128
        v127 = v128.v127
        v126 = v127.v126
        v125 = v126.v125
        v124 = v125.v124
        v123 = v124.v123
        v121 = v123.v121
        v120 = v121.v120

        install_async_timing(v128, "run_tick", "v128_run_tick")
        install_async_timing(v127, "run_tick", "v127_run_tick")
        install_async_timing(v126, "run_tick", "v126_run_tick")
        install_async_timing(v125, "run_tick", "v125_run_tick")
        install_async_timing(v124, "run_tick", "v124_run_tick")
        install_async_timing(v123, "run_tick", "v123_run_tick")
        install_async_timing(v121, "run_tick", "v121_run_tick")
        install_async_timing(v120, "run_tick", "v120_run_tick")
        install_async_timing(v126.v2, "_ORIGINAL_API_GET", "provider_network_request")
        install_async_timing(v123.price_resolver_v4, "resolve_payload", "price_resolver_v4_resolve_payload")
        install_sync_timing(
            v128.market_residual_challenger_v4,
            "build_report",
            "market_residual_build_report",
        )
        install_sync_timing(
            automation_v129.team_totals_close_provenance_history_v4,
            "load_report",
            "team_totals_history_audit",
        )
        install_sync_timing(
            automation_v129.team_totals_close_provenance_v4,
            "build_report",
            "team_totals_current_tick_audit",
        )

        stage_started = time.monotonic()
        _emit_timing("automation_v129_start")
        try:
            payload = await automation_v129.run_tick()
        finally:
            timings["automation_v129_total_ms"] = _elapsed_ms(stage_started)
            _emit_timing(
                "automation_v129_done",
                elapsed_ms=timings["automation_v129_total_ms"],
            )
            for module, attr, original in reversed(restorers):
                setattr(module, attr, original)
            restorers.clear()

        payload.setdefault("status", "ok")
        payload["shortlist_seed_imported"] = imported
        payload["fair_scheduler_seed_imported"] = fairness_imported
        payload["tick_stage_timings_ms"] = timings

        stage_started = time.monotonic()
        _normalize_refresh_event_model_lineage(payload)
        timings["model_lineage_normalization_ms"] = _elapsed_ms(stage_started)
        _emit_timing(
            "model_lineage_normalization_done",
            elapsed_ms=timings["model_lineage_normalization_ms"],
        )

        stage_started = time.monotonic()
        _emit_timing("persistence_start")
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
        timings["persistence_ms"] = _elapsed_ms(stage_started)
        _emit_timing("persistence_done", elapsed_ms=timings["persistence_ms"])

        # Phase24 views were initially assembled upstream before the final
        # price-resolver/CLV annotations and before persistence completed.
        # Rebuild once here from the final payload so the returned/state-branch
        # snapshot reflects final API caps, maturation counters and DB health.
        stage_started = time.monotonic()
        _emit_timing("dashboard_refresh_start")
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
        timings["dashboard_refresh_ms"] = _elapsed_ms(stage_started)
        _emit_timing("dashboard_refresh_done", elapsed_ms=timings["dashboard_refresh_ms"])

        timings["total_before_json_ms"] = _elapsed_ms(total_started)
        _emit_timing("json_encode_start", elapsed_ms=timings["total_before_json_ms"])
        stage_started = time.monotonic()
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        timings["json_encode_ms"] = _elapsed_ms(stage_started)
        timings["total_worker_ms"] = _elapsed_ms(total_started)
        _emit_timing(
            "json_encode_done",
            elapsed_ms=timings["json_encode_ms"],
            total_worker_ms=timings["total_worker_ms"],
            output_bytes=len(encoded.encode("utf-8")),
        )
        sys.stdout.write(encoded)
        sys.stdout.flush()
        return 0
    except Exception as exc:
        for module, attr, original in reversed(restorers):
            try:
                setattr(module, attr, original)
            except Exception:
                pass
        _emit_timing("tick_worker_exception", elapsed_ms=_elapsed_ms(total_started), error=str(exc)[:200])
        sys.stderr.write(f"tick_failed: {str(exc)[:500]}\n")
        sys.stderr.flush()
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
