from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v121 as v121
from mcp_gateway import automation_v92 as v92
from mcp_gateway import automation_v112 as v112
from mcp_gateway import price_resolver_v4
from mcp_gateway import team_totals_intelligence

MODEL_VERSION = v121.MODEL_VERSION
AUTOMATION_VERSION = "4.31.2-team-totals-leftover-odds"


def _annotate_checkpoint(payload: dict[str, Any]) -> None:
    resolution = payload.get("price_resolution_v4") if isinstance(payload.get("price_resolution_v4"), dict) else {}
    payload["price_resolution_checkpoint"] = {
        "schema_version": "1.0.0",
        "status": resolution.get("status") or "NOT_RUN",
        "model_version": price_resolver_v4.MODEL_VERSION,
        "candidate_rows": int(resolution.get("candidate_rows") or 0),
        "unique_candidate_fixtures": int(resolution.get("unique_candidate_fixtures") or 0),
        "api_calls_added": int(resolution.get("api_calls_added") or 0),
        "resolution_counts": dict(resolution.get("resolution_counts") or {}),
        "simulated_odds_allowed": False,
        "calibrated_probability_fabricated": False,
        "phase16_recomputed_after_price_resolution": True,
        "team_totals_recomputed_after_price_resolution": True,
        "team_totals_post_resolution": dict(payload.get("team_totals_post_resolution") or {}),
        "canonical_bet_logic_changed": False,
        "model_weights_changed": False,
        "production_promotion_allowed": False,
        "research_spillover_candidate_fixtures": resolution.get("research_spillover_candidate_fixtures", 0),
        "research_spillover_cache_hits": resolution.get("research_spillover_cache_hits", 0),
        "research_spillover_api_calls_added": resolution.get("research_spillover_api_calls_added", 0),
        "research_spillover_fixtures_fetched": resolution.get("research_spillover_fixtures_fetched", 0),
        "research_spillover_market_rows_fetched": resolution.get("research_spillover_market_rows_fetched", 0),
        "note": (
            "Price resolver uses real API-Football /odds fixture quotes or fresh Postgres market snapshots. "
            "Resolved rows are re-evaluated by Team Totals research intelligence, execution-status separation "
            "and Phase16. Team Totals is cache-first and may use only provider budget left after every primary "
            "price target; those spillover calls remain research-only and cannot pre-empt FT Totals/BTTS/1X2. "
            "No calibrated probability is fabricated and no BET/tier/stake/model threshold is changed."
        ),
    }


async def run_tick() -> dict[str, Any]:
    payload = await v121.run_tick()

    await price_resolver_v4.resolve_payload(payload)

    # Team Totals is built earlier in the automation chain, before the price
    # resolver may attach real fixture /odds markets. Rebuild this research-only
    # derivative after price enrichment. The resolver may use only budget left
    # after all primary price targets, so derivative collection never pre-empts
    # FT Totals / BTTS / 1X2 price resolution.
    payload["team_totals_post_resolution"] = team_totals_intelligence.attach(payload)

    # Price enrichment changes exact market/selection/price/fair probability on
    # research visibility rows. Recompute readiness and Phase16 against that
    # enriched point-in-time view. Both functions remain research-only.
    v92._annotate_decision_separation(payload)
    v112._annotate_phase16(payload)

    _annotate_checkpoint(payload)
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
