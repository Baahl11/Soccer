from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v127 as v127
from mcp_gateway import market_residual_challenger_v4

MODEL_VERSION = v127.MODEL_VERSION
AUTOMATION_VERSION = "4.37.1-market-residual-challenger"


async def run_tick() -> dict[str, Any]:
    payload = await v127.run_tick()
    rows = payload.get("match_table_rows")
    if not isinstance(rows, list):
        rows = []

    # Runtime remains self-contained and zero-call. Historical settlement/RPS
    # labels and strict-close CLV are joined only by the offline V211 workflow;
    # the live report still exposes residual/reliability from same-tick rows.
    report = market_residual_challenger_v4.build_report(rows)
    payload["v211_market_residual_challenger"] = report
    payload["v211_checkpoint"] = (
        "MARKET RESIDUAL CHALLENGER ACTIVE IN RESEARCH ONLY: calibrated model probability is "
        "compared with the same captured de-vig market probability. Offline V211 evaluation adds "
        "dedicated-settlement Brier/log-loss/RPS and existing strict-close CLV without changing "
        "strict-close semantics. The challenger has zero decision weight, adds no provider requests, "
        "and cannot promote or create a production BET."
    )
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
