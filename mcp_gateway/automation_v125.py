from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v124 as v124
from mcp_gateway import btts_paid_odds_intelligence
from mcp_gateway import price_resolver_v4

MODEL_VERSION = v124.MODEL_VERSION
AUTOMATION_VERSION = "4.34.0-btts-paid-entry"


async def run_tick() -> dict[str, Any]:
    # Observe the existing paid /odds resolver path only. The wrapped function
    # remains the same fetch path and retains the exact provider-call budget.
    original_fetch, captured = btts_paid_odds_intelligence.install_fetch_observer(
        price_resolver_v4
    )
    try:
        payload = await v124.run_tick()
    finally:
        btts_paid_odds_intelligence.restore_fetch_observer(
            price_resolver_v4,
            original_fetch,
        )

    capture = btts_paid_odds_intelligence.attach(payload, captured)
    payload["v207_btts_paid_odds_entry_capture"] = dict(capture)
    payload["v207_checkpoint"] = (
        "BTTS TRUE-CLV ENTRY CAPTURE: reuse already-paid fresh /odds responses for fixtures that "
        "already have an explicit TWO_WAY/BTTS research signal, require a real Yes/No price pair, "
        "de-vig the entry price, and persist one research-only priced BTTS row. No provider calls are "
        "added and a later true close still requires a strictly later pre-kickoff provider_update."
    )
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
