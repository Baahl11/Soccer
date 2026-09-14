from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v8
from mcp_gateway.galaxyparlay_client import configured as galaxy_configured
from mcp_gateway.galaxyparlay_client import enrich_events

AUTOMATION_VERSION = "1.9.0"
MODEL_VERSION = "SOCCER EDGE ENGINE v1.0"


async def run_tick() -> dict[str, Any]:
    payload = await automation_v8.run_tick()
    events = payload.get("events") or []
    metrics = await enrich_events(events) if isinstance(events, list) else {
        "requested": 0,
        "available": 0,
        "errors": 0,
    }

    payload["version"] = AUTOMATION_VERSION
    payload["galaxyparlay_shadow_enabled"] = galaxy_configured()
    payload["galaxyparlay_shadow_requested"] = metrics["requested"]
    payload["galaxyparlay_shadow_available"] = metrics["available"]
    payload["galaxyparlay_shadow_errors"] = metrics["errors"]
    payload["galaxyparlay_decision_influence"] = "NONE_SHADOW_ONLY"
    payload["integration_policy"] = (
        "GalaxyParlay persisted predictions/quality/odds are recorded for cross-model validation only. "
        "They cannot promote or downgrade Soccer Edge classifications in v1.9."
    )
    return payload
