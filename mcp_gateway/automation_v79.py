from __future__ import annotations
from typing import Any
from mcp_gateway import automation_v78 as v78
from mcp_gateway import competition_context_intelligence
MODEL_VERSION=v78.MODEL_VERSION
AUTOMATION_VERSION="3.55.0"
async def run_tick()->dict[str,Any]:
    payload=await v78.run_tick(); metrics=competition_context_intelligence.attach(payload)
    payload["competition_context_intelligence"]={"schema_version":competition_context_intelligence.SCHEMA_VERSION,**metrics,"decision_weight":0.0,"canonical_model_weights_changed":False,"canonical_bet_logic_changed":False,"production_status":"STRUCTURED_CONTEXT_ONLY"}
    payload["v355_provider_requests_added"]=0; payload["v355_model_weights_changed"]=False; payload["v355_canonical_bet_logic_changed"]=False
    payload["v355_competition_context_checkpoint"]="ROUND_STAGE_CONTEXT_LIVE; TABLE_AGGREGATE_MOTIVATION_BLOCKED_UNTIL_VERIFIED"
    payload["version"]=AUTOMATION_VERSION; payload["model_version"]=MODEL_VERSION; return payload
