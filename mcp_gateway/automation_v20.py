from __future__ import annotations

import os
from collections import Counter
from typing import Any

from mcp_gateway import automation_v19 as v19
from mcp_gateway.presentation_tables import validate_presentation_payload

MODEL_VERSION = v19.MODEL_VERSION
AUTOMATION_VERSION = "2.8.1"

MAX_REGISTRY_ROWS = max(25, int(os.getenv("SOCCER_EDGE_MAX_REGISTRY_ROWS", "300")))
MAX_MATCH_TABLE_ROWS = max(25, int(os.getenv("SOCCER_EDGE_MAX_MATCH_TABLE_ROWS", "200")))


def _bounded_registry(payload: dict[str, Any]) -> None:
    registry = payload.get("league_coverage_registry")
    if not isinstance(registry, dict):
        return

    competitions = registry.get("competitions")
    if not isinstance(competitions, list):
        competitions = []

    total = len(competitions)
    attached = competitions[:MAX_REGISTRY_ROWS]
    registry["competition_count"] = total
    registry["competition_rows_attached"] = len(attached)
    registry["competitions_truncated"] = total > len(attached)
    registry["max_competition_rows"] = MAX_REGISTRY_ROWS
    registry["competitions"] = attached


def _bounded_match_rows(payload: dict[str, Any]) -> None:
    rows = payload.get("match_table_rows")
    if not isinstance(rows, list):
        rows = []

    total = len(rows)
    class_counts = Counter(str(row.get("classification") or "WATCH") for row in rows if isinstance(row, dict))
    attached = rows[:MAX_MATCH_TABLE_ROWS]
    payload["match_table_row_count"] = total
    payload["match_table_rows_attached"] = len(attached)
    payload["match_table_rows_truncated"] = total > len(attached)
    payload["match_table_max_rows"] = MAX_MATCH_TABLE_ROWS
    payload["match_table_classification_counts"] = dict(class_counts)
    payload["match_table_rows"] = attached


async def run_tick() -> dict[str, Any]:
    payload = await v19.run_tick()

    _bounded_registry(payload)
    _bounded_match_rows(payload)

    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    payload["presentation_state_policy"] = (
        "BOUNDED_PERSISTED_PRESENTATION_ONLY; FULL SPORTING/MARKET DECISIONS UNCHANGED; "
        "TRUNCATION_IS_EXPLICIT_AND_NEVER_CHANGES_CLASSIFICATION"
    )
    payload["presentation_validation"] = validate_presentation_payload(payload)
    return payload
