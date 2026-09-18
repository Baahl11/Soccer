from __future__ import annotations
from typing import Any
from mcp_gateway import automation_v76 as v76
from mcp_gateway import travel_intelligence
MODEL_VERSION=v76.MODEL_VERSION
AUTOMATION_VERSION="3.53.0"
async def run_tick()->dict[str,Any]:
    payload=await v76.run_tick(); metrics=travel_intelligence.attach(payload)
    payload["travel_intelligence"]={"schema_version":travel_intelligence.SCHEMA_VERSION,**metrics,"decision_weight":0.0,"canonical_model_weights_changed":False,"canonical_bet_logic_changed":False,"production_status":"SOURCE_GUARD_NOT_ACTIONABLE"}
    payload["v353_provider_requests_added"]=0; payload["v353_model_weights_changed"]=False; payload["v353_canonical_bet_logic_changed"]=False
    payload["v353_travel_checkpoint"]="DESTINATION_METADATA_LIVE; DISTANCE_TIMEZONE_ALTITUDE_BLOCKED_PENDING_VERIFIED_COORDINATES"
    payload["version"]=AUTOMATION_VERSION; payload["model_version"]=MODEL_VERSION; return payload
