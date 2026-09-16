from __future__ import annotations

from mcp_gateway import automation_v6 as v6
from mcp_gateway import automation_v33 as v33
from mcp_gateway import automation_v34 as v34
from mcp_gateway.galaxy_builder_v4 import build as build_galaxy_builder_v4

MODEL_VERSION = v33.MODEL_VERSION
AUTOMATION_VERSION = "3.11.0"
REGISTRY_CACHE_KEY_V2 = "_galaxy_candidate_registry_v2_market_backed"


async def run_tick() -> dict:
    payload = await v33.run_tick()

    builder = build_galaxy_builder_v4(payload)
    payload["galaxy_builder"] = builder
    current_tick_count = int(builder.get("candidate_count") or 0)
    payload["galaxy_builder_candidate_count_this_tick"] = current_tick_count
    payload["galaxy_builder_same_game_count_this_tick"] = int(builder.get("same_game_candidate_count") or 0)
    payload["galaxy_builder_multi_match_count_this_tick"] = int(builder.get("multi_match_candidate_count") or 0)
    payload["galaxy_builder_derivative_model_contract"] = builder.get("derivative_model_contract")
    payload["galaxy_builder_policy"] = builder.get("policy")
    payload["galaxy_builder_actionable_scope"] = (
        "V0.4 MARKET-BACKED RESEARCH; SGP LEGS MUST EXIST IN VERIFIED CURRENT MARKET; "
        "SAME-BOOK COMPONENT REFERENCE MUST REACH +110 TARGET FLOOR; LOGICALLY REDUNDANT LEGS BLOCKED; "
        "ONE PRIMARY SGP PER FIXTURE; EXACT CORRELATION-ADJUSTED SPORTSBOOK SGP QUOTE STILL REQUIRED FOR BET"
    )

    # Bump the rolling-registry namespace so v0.3 synthetic/unquoted candidates
    # cannot survive into v0.4 merely because their fixture is not due this tick.
    v34.REGISTRY_CACHE_KEY = REGISTRY_CACHE_KEY_V2
    payload["galaxy_builder"] = v34._rolling_builder(payload)
    payload["galaxy_builder_active_candidate_count"] = int(payload["galaxy_builder"].get("candidate_count") or 0)
    payload["galaxy_builder_rolling_registry_policy"] = (
        "V2 MARKET-BACKED REGISTRY; ACTIVE CANDIDATES SURVIVE NON-DUE TICKS; "
        "OLD V0.3 REGISTRY ISOLATED; REEVALUATION REPLACES PRIOR FIXTURE CANDIDATE; KICKOFF EXPIRES"
    )

    payload["shortlist_state"] = v6.export_shortlist_state()
    payload["shortlist_state_count"] = len(payload["shortlist_state"])
    payload["v311_provider_requests_added"] = 0
    payload["v311_model_weights_changed"] = False
    payload["v311_canonical_bet_logic_changed"] = False
    payload["v311_synthetic_unquoted_sgp_lines_allowed"] = False
    payload["v311_one_primary_sgp_per_fixture"] = True
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
