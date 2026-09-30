from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v127 as v127
from mcp_gateway import market_residual_challenger_v4

MODEL_VERSION = v127.MODEL_VERSION
AUTOMATION_VERSION = "4.37.0-market-residual-challenger"


async def run_tick() -> dict[str, Any]:
    payload = await v127.run_tick()
    rows = payload.get("match_table_rows")
    if not isinstance(rows, list):
        rows = []

    report = market_residual_challenger_v4.build_report(rows)
    payload["v211_market_residual_challenger"] = report
    payload["v211_checkpoint"] = (
        "MARKET RESIDUAL CHALLENGER ACTIVE IN RESEARCH ONLY: calibrated model probability is "
        "compared with the same captured de-vig market probability. The report has zero decision "
        "weight, adds no provider requests, and cannot promote or create a production BET."
    )
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
