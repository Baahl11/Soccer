from __future__ import annotations
from typing import Any

from mcp_gateway import automation_v3 as v3
from mcp_gateway import automation_v5 as v5
from mcp_gateway import automation_v7 as v7
from mcp_gateway import automation_v26 as v26

MODEL_VERSION = v26.MODEL_VERSION
AUTOMATION_VERSION = "3.3.0"

_ORIGINAL_COVERAGE_FAST = v3._coverage_fast
_ORIGINAL_QUEUE_PRIORITY = v7._queue_priority
_ORIGINAL_PRIORITY_EVENT = v5._priority_event

async def _sporting_eligible_coverage(league_id:int, season:int, now):
    coverage = await _ORIGINAL_COVERAGE_FAST(league_id, season, now)
    if not isinstance(coverage, dict) or coverage.get("data_tier") != "C":
        return coverage
    # v7 historically hard-blocks anything outside A/B before the sporting screen,
    # while the global registry explicitly defines C as sporting-screen eligible.
    # Present C as scheduler-eligible only, retaining the real tier in metadata.
    out = dict(coverage)
    out["_soccer_edge_original_data_tier"] = "C"
    out["_soccer_edge_sporting_only"] = True
    out["data_tier"] = "B"
    return out

def _tier_c_after_ab_priority(fx, stage, coverage, tier, prior, prior_rank, proximity, kickoff):
    if isinstance(coverage, dict) and coverage.get("_soccer_edge_original_data_tier") == "C":
        restored = dict(coverage); restored["data_tier"] = "C"
        return _ORIGINAL_QUEUE_PRIORITY(fx, stage, restored, "C", prior, prior_rank, proximity, kickoff)
    return _ORIGINAL_QUEUE_PRIORITY(fx, stage, coverage, tier, prior, prior_rank, proximity, kickoff)

async def _restore_real_tier_for_event(fx:dict[str,Any], stage:str, coverage:dict[str,Any], now):
    if isinstance(coverage, dict) and coverage.get("_soccer_edge_original_data_tier") == "C":
        coverage = dict(coverage)
        coverage["data_tier"] = "C"
        coverage.pop("_soccer_edge_original_data_tier", None)
        coverage.pop("_soccer_edge_sporting_only", None)
    # Real Tier C reaches the SPORT FIRST model, but existing market evaluator
    # still sees C and therefore cannot promote it to BET/LEAN under A/B gates.
    return await _ORIGINAL_PRIORITY_EVENT(fx, stage, coverage, now)

async def run_tick()->dict[str,Any]:
    prev_cov=v3._coverage_fast; prev_queue=v7._queue_priority; prev_event=v5._priority_event
    v3._coverage_fast=_sporting_eligible_coverage
    v7._queue_priority=_tier_c_after_ab_priority
    v5._priority_event=_restore_real_tier_for_event
    try:
        payload=await v26.run_tick()
    finally:
        v3._coverage_fast=prev_cov; v7._queue_priority=prev_queue; v5._priority_event=prev_event
    payload["version"]=AUTOMATION_VERSION
    payload["model_version"]=MODEL_VERSION
    payload["tier_c_sporting_policy"]=(
        "TIER_C_CAN_ENTER_SPORT_FIRST_SCREEN_AFTER_A_B_PRIORITY; REAL_TIER_C_IS_RESTORED_BEFORE_EVENT_EVALUATION; "
        "TIER_C_CANNOT_SUPPORT_BET_OR_LEAN; TIER_D_REMAINS_LOW_DATA_PASS; PROVIDER_HARD_CAP_UNCHANGED"
    )
    payload["tier_c_actionable_scope"]="RESEARCH_WATCH_PASS_ONLY"
    return payload
