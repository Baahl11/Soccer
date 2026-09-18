from __future__ import annotations
from typing import Any
from mcp_gateway import automation_v77 as v77
from mcp_gateway import weather_intelligence
MODEL_VERSION=v77.MODEL_VERSION
AUTOMATION_VERSION="3.54.0"
async def run_tick()->dict[str,Any]:
    payload=await v77.run_tick(); metrics=weather_intelligence.attach(payload)
    payload["weather_intelligence"]={"schema_version":weather_intelligence.SCHEMA_VERSION,**metrics,"decision_weight":0.0,"canonical_model_weights_changed":False,"canonical_bet_logic_changed":False,"production_status":"EXTERNAL_SOURCE_BLOCKED"}
    payload["v354_provider_requests_added"]=0; payload["v354_model_weights_changed"]=False; payload["v354_canonical_bet_logic_changed"]=False
    payload["v354_weather_checkpoint"]="LIVE_WEATHER_GUARD_ATTACHED; APPROVED_FORECAST_SOURCE_REQUIRED"
    payload["version"]=AUTOMATION_VERSION; payload["model_version"]=MODEL_VERSION; return payload
