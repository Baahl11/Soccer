from __future__ import annotations
from typing import Any

SCHEMA_VERSION = "1.0.0"
SUPPORTED_SCORE_MATRIX_FAMILIES = ["FT_GOALS", "BTTS", "DOUBLE_CHANCE"]
UNSUPPORTED_JOINT_FAMILIES = ["CORNERS", "TEAM_CORNERS", "CARDS", "TEAM_CARDS", "PLAYER_PROPS", "GK_SAVES"]

def attach(payload: dict[str, Any]) -> dict[str, Any]:
    builder=payload.get("galaxy_builder") if isinstance(payload.get("galaxy_builder"),dict) else {}
    candidates=[x for x in builder.get("same_game_candidates") or [] if isinstance(x,dict)]
    direct=[x for x in candidates if x.get("joint_probability_method")=="DIRECT_SCORE_MATRIX_CORRELATION_AWARE"]
    bad=[x for x in candidates if x.get("joint_probability_method") not in {None,"DIRECT_SCORE_MATRIX_CORRELATION_AWARE"}]
    families=sorted({str(leg.get("family")) for row in direct for leg in (row.get("legs") or []) if isinstance(leg,dict)})
    exact_quotes=sum(1 for x in direct if x.get("exact_parlay_quote") not in {None,"NOT_VERIFIED"})
    report={
        "schema_version":SCHEMA_VERSION,
        "status":"LIVE_RESEARCH_PARTIAL_JOINT_MODEL",
        "joint_probability_method":"DIRECT_SCORE_MATRIX_CORRELATION_AWARE",
        "supported_score_matrix_families":SUPPORTED_SCORE_MATRIX_FAMILIES,
        "families_seen_this_tick":families,
        "unsupported_joint_families":UNSUPPORTED_JOINT_FAMILIES,
        "same_game_candidates":len(candidates),
        "direct_score_matrix_candidates":len(direct),
        "unexpected_joint_method_candidates":len(bad),
        "exact_sgp_quotes_verified":exact_quotes,
        "marginal_multiplication_allowed_same_game":False,
        "actionable":False,
        "decision_weight":0.0,
        "production_blockers":[
            "UNDERLYING_RESEARCH_ONLY_LEGS_REMAIN_BLOCKED_WHERE_APPLICABLE",
            "EXACT_SPORTSBOOK_SGP_QUOTE_REQUIRED_FOR_BET",
            "NON_SCORE-MATRIX_FAMILIES_REQUIRE_SEPARATE_JOINT_MODEL",
        ],
        "policy":"SAME-GAME SCORE-DEPENDENT LEGS USE DIRECT SCORE-MATRIX INTERSECTION; NEVER MULTIPLY SAME-GAME MARGINALS.",
    }
    payload["galaxy_joint_intelligence"]=report
    return report
