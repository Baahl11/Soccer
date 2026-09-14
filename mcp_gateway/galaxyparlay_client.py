from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx

DEFAULT_TIMEOUT_SECONDS = 6.0
MAX_SHADOW_FIXTURES_PER_TICK = 8


def configured() -> bool:
    return bool(os.getenv("GALAXYPARLAY_BASE_URL", "").strip()) and os.getenv(
        "GALAXYPARLAY_SHADOW_ENABLED", "false"
    ).strip().lower() in {"1", "true", "yes", "on"}


def _base_url() -> str:
    return os.getenv("GALAXYPARLAY_BASE_URL", "").strip().rstrip("/")


def _compact_prediction(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "market_key": row.get("market_key"),
        "prediction": row.get("prediction"),
        "prediction_json": row.get("prediction_json"),
        "probability": row.get("probability"),
        "confidence_score": row.get("confidence_score"),
        "quality_grade": row.get("quality_grade"),
        "model_version": row.get("model_version"),
        "calibration_status": row.get("calibration_status"),
        "predicted_at": row.get("predicted_at"),
        "publishable": row.get("publishable"),
        "publication_state": row.get("publication_state"),
        "publication_reason": row.get("publication_reason"),
    }


async def get_fixture_shadow(
    client: httpx.AsyncClient, fixture_id: int
) -> dict[str, Any]:
    url = f"{_base_url()}/api/sports-edge/v1/fixture/{fixture_id}"
    try:
        response = await client.get(
            url,
            params={"core_only": "true", "include_features": "false"},
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("GalaxyParlay returned a non-object payload")
        if payload.get("external_api_calls") not in {0, "0"}:
            raise ValueError("GalaxyParlay shadow endpoint did not assert external_api_calls=0")
        return {
            "status": "AVAILABLE",
            "source": payload.get("source") or "GalaxyParlay persisted data",
            "contract_version": payload.get("contract_version"),
            "data_cutoff": payload.get("data_cutoff"),
            "fixture_id": fixture_id,
            "predictions": [
                _compact_prediction(row)
                for row in (payload.get("predictions") or [])
                if isinstance(row, dict)
            ],
            "quality_scores": payload.get("quality_scores") or [],
            "odds": payload.get("odds") or [],
            "external_api_calls": 0,
            "decision_influence": "NONE_SHADOW_ONLY",
        }
    except httpx.HTTPStatusError as exc:
        return {
            "status": "UNAVAILABLE",
            "fixture_id": fixture_id,
            "error": f"HTTP {exc.response.status_code}",
            "decision_influence": "NONE_SHADOW_ONLY",
        }
    except Exception as exc:
        return {
            "status": "UNAVAILABLE",
            "fixture_id": fixture_id,
            "error": str(exc)[:240],
            "decision_influence": "NONE_SHADOW_ONLY",
        }


async def enrich_events(events: list[dict[str, Any]]) -> dict[str, int]:
    if not configured():
        return {"requested": 0, "available": 0, "errors": 0}

    candidates: list[tuple[int, dict[str, Any]]] = []
    seen: set[int] = set()
    for event in events:
        if event.get("stage") not in {"T-40", "T-30", "T-20", "T-10", "CLOSE"}:
            continue
        fixture = event.get("fixture") or {}
        fixture_id = fixture.get("fixture_id")
        shortlist = event.get("sporting_shortlist") or {}
        if not fixture_id or not shortlist.get("shortlisted"):
            continue
        fixture_id = int(fixture_id)
        if fixture_id in seen:
            continue
        seen.add(fixture_id)
        candidates.append((fixture_id, event))
        if len(candidates) >= MAX_SHADOW_FIXTURES_PER_TICK:
            break

    if not candidates:
        return {"requested": 0, "available": 0, "errors": 0}

    timeout = float(os.getenv("GALAXYPARLAY_TIMEOUT", str(DEFAULT_TIMEOUT_SECONDS)))
    limits = httpx.Limits(max_connections=4, max_keepalive_connections=2)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        results = await asyncio.gather(
            *(get_fixture_shadow(client, fixture_id) for fixture_id, _ in candidates)
        )

    available = 0
    errors = 0
    for (_, event), shadow in zip(candidates, results):
        event["galaxy_shadow"] = shadow
        if shadow.get("status") == "AVAILABLE":
            available += 1
        else:
            errors += 1

    return {"requested": len(candidates), "available": available, "errors": errors}
