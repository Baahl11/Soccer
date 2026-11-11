"""Immutable frozen SPORT vs independently supplied official FT grading.

No observed market price needed. Probability calibration and OOS sample
sizes are *family-isolated*. Never infer missing corners/cards/player shots
from absent records, nor promote BET based on a performance sample alone.
"""
from __future__ import annotations
from datetime import datetime, timezone
from collections import defaultdict
from typing import Any
import hashlib
import json
import math
from .soccer_family_research_v1 import MARKETS

VERSION = "SOCCER_OOS_FAMILY_VALIDATION_v1.0"
RESEARCH_MIN_FIXTURES_PER_FAMILY = 100  # Provisional research QA, NOT a BET unlock.
HALF_MARKETS = {"1H_1X2":"half_1", "1H_TOTALS":"half_1",
                "2H_1X2":"half_2", "2H_TOTALS":"half_2"}
COUNT_MARKETS = {"FT_CORNERS":"corners", "TEAM_CORNERS":"corners",
                 "FT_CARDS":"cards", "PLAYER_SHOTS":"player_shots"}


def _dt(s):
    if not isinstance(s, str):return None
    try:r=datetime.fromisoformat(s.replace("Z","+00:00"))
    except ValueError:return None
    return r.astimezone(timezone.utc) if r.tzinfo else None


def _integer(value):
    return type(value) is int and 0 <= value <= 100


def _count_score(scores, k):
    val=scores.get(k) if isinstance(scores,dict) else None
    return val if _integer(val) else None


def _outcome_goals(family, selection, score, line, entity):
    if family in {"1X2", "1H_1X2", "2H_1X2"}:
        if selection not in {"home","draw","away"}:return None
        return int((score["home"]>score["away"] if selection=="home" else
                    score["home"]==score["away"] if selection=="draw" else
                    score["home"]<score["away"]))
    if family=="BTTS":
        if selection not in {"yes","no"}:return None
        yes=score["home"]>0 and score["away"]>0
        return int(yes if selection=="yes" else not yes)
    if not (isinstance(selection,str) and
            selection in {f"over_{line:g}",f"under_{line:g}"}):
        return None
    if family in {"TEAM_TOTALS","TEAM_CORNERS"}:
        if entity not in {"home","away"}:return None
        count=score[entity]
    else:
        count=score["home"]+score["away"]
    return int(count>line if selection.startswith("over_") else count<line)


def _verified_score(row, family, player_id):
    s=row.get("official_settlement")
    if not isinstance(s, dict) or s.get("status")!="FT":
        return None
    if s.get("source_type")!="OFFICIAL_LEAGUE" or s.get("independent_auditor_verified") is not True:
        return None
    if not isinstance(s.get("official_source_url"),str) or not s["official_source_url"].startswith("https://"):
        return None
    if _dt(s.get("observed_at_utc")) is None:
        return None
    if family=="PLAYER_SHOTS":
        players=s.get("player_shots")
        if type(player_id) is not int or not isinstance(players,dict):
            return None
        v=players.get(str(player_id))
        return {"home":v,"away":0} if _integer(v) else None
    if family in HALF_MARKETS:
        node=s.get(HALF_MARKETS[family])
    elif family in {"FT_CORNERS","TEAM_CORNERS"}:
        node=s.get("corners")
    elif family=="FT_CARDS":
        node=s.get("cards")
    else:
        node=s.get("ft_goals")
    if not isinstance(node,dict) or not all(_integer(node.get(t)) for t in ("home","away")):
        return None
    return {"home":node["home"],"away":node["away"]}


def grade_frozen_record(
    row: dict[str,Any], *,
    trusted_official_ingestion_ids: frozenset[str] = frozenset(),
) -> dict[str,Any]:
    """One frozen selection; do not trust input's own official flag.

    An independent trusted ingestion registry MUST supply the allowlist.
    Default empty proves nothing; no supplied internet URL certifies the feed.
    """
    family=row.get("family")
    if family not in MARKETS:return {"status":"NOT_GRADED","reason":"UNSUPPORTED_FAMILY"}
    frozen=_dt(row.get("frozen_at_utc"))
    kickoff=_dt(row.get("kickoff_utc"))
    if not frozen or not kickoff or frozen >= kickoff:
        return {"status":"NOT_GRADED","reason":"FROZEN_ASOF_NOT_VERIFIED"}
    s=row.get("official_settlement",{})
    observed=_dt(s.get("observed_at_utc")) if isinstance(s,dict) else None
    if not observed or observed<=kickoff:
        return {"status":"NOT_GRADED","reason":"OFFICIAL_FT_POSTKICKOFF_TIME_NOT_VERIFIED"}
    ingestion_id=s.get("trusted_ingestion_id") if isinstance(s,dict) else None
    if (not isinstance(ingestion_id,str) or not ingestion_id
            or ingestion_id not in trusted_official_ingestion_ids):
        return {"status":"NOT_GRADED","reason":"INDEPENDENT_OFFICIAL_SETTLEMENT_INGEST_NOT_TRUSTED"}
    if row.get("model_version") in (None,""):
        return {"status":"NOT_GRADED","reason":"FROZEN_MODEL_VERSION_MISSING"}
    fixture=row.get("fixture_id")
    if type(fixture) is not int or fixture<=0:
        return {"status":"NOT_GRADED","reason":"FIXTURE_ID_NOT_VERIFIED"}
    prob=row.get("p_raw")
    if type(prob) not in (int,float) or not math.isfinite(prob) or not 0<prob<1:
        return {"status":"NOT_GRADED","reason":"FROZEN_PROBABILITY_INVALID"}
    selection=row.get("selection")
    line=row.get("line")
    if family in {"FT_TOTALS","TEAM_TOTALS","1H_TOTALS","2H_TOTALS","FT_CORNERS",
                  "TEAM_CORNERS","FT_CARDS","PLAYER_SHOTS"}:
        if type(line) not in (float,int) or line<0 or line>30 or abs(line*2-round(line*2))>1e-8 or round(line*2)%2!=1:
            return {"status":"NOT_GRADED","reason":"BINARY_LINE_NOT_VERIFIED"}
    elif line is not None:
        return {"status":"NOT_GRADED","reason":"NO_LINE_EXPECTED"}
    entity=row.get("entity")
    if family=="PLAYER_SHOTS":
        entity="home"
    result=_verified_score(row,family,row.get("player_id"))
    if result is None:
        return {"status":"NOT_GRADED","reason":"INDEPENDENT_OFFICIAL_SETTLEMENT_NOT_VERIFIED"}
    outcome=_outcome_goals(family,selection,result,line,entity)
    if outcome is None:
        return {"status":"NOT_GRADED","reason":"SELECTION_OR_ENTITY_NOT_VERIFIED"}
    identity={
        "fixture_id":fixture, "family":family,
        "selection":selection,"line":line,"entity":entity,
        "model_version":row["model_version"],"frozen_at_utc":frozen.isoformat(),
    }
    fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    return {"status":"OFFICIAL_FT_GRADED_RESEARCH_ONLY",
            "fixture_id":fixture,"family":family,"outcome":outcome,
            "p_raw":float(prob),"brier":round((float(prob)-outcome)**2,8),
            "model_version":row["model_version"],
            "frozen_fingerprint":fingerprint,"bet_eligible":False}


def family_oos_scorecards(
    records: list[dict[str,Any]], *,
    trusted_official_ingestion_ids: frozenset[str] = frozenset(),
) -> dict[str,Any]:
    """One probability per unique instrument, trusted ledger IDs required."""
    by_family=defaultdict(list)
    not_graded=defaultdict(int)
    seen=set()
    for row in records:
        g=grade_frozen_record(
            row, trusted_official_ingestion_ids=trusted_official_ingestion_ids)
        if g["status"]!="OFFICIAL_FT_GRADED_RESEARCH_ONLY":
            not_graded[g["reason"]]+=1
            continue
        token=g["frozen_fingerprint"]
        if token in seen:
            not_graded["DUPLICATE_FROZEN_INSTRUMENT"]+=1
            continue
        seen.add(token)
        by_family[g["family"]].append(g)
    families={}
    for family in MARKETS:
        grades=by_family[family]
        count=len(grades)
        unique_fixtures=len({x["fixture_id"] for x in grades})
        families[family]={
            "graded_instruments":count,
            "official_ft_unique_fixtures":unique_fixtures,
            "mean_brier":round(sum(x["brier"] for x in grades)/count,6) if count else None,
            "mean_predicted":round(sum(x["p_raw"] for x in grades)/count,6) if count else None,
            "mean_actual":round(sum(x["outcome"] for x in grades)/count,6) if count else None,
            "research_sample_gate":"MIN_100_UNIQUE_FT_REACHED" if unique_fixtures>=RESEARCH_MIN_FIXTURES_PER_FAMILY else "OOS_SAMPLE_INSUFFICIENT",
            "research_min_unique_fixtures":RESEARCH_MIN_FIXTURES_PER_FAMILY,
            "commercial_bet_gate_promoted":False,
            "pricing_source_verified":False,
            "true_clv_verified":False,
        }
    return {"version":VERSION,"status":"RESEARCH_OOS_REPORT",
            "official_result_verification":"INDEPENDENT_SOURCE_ATTESTATION_REQUIRED",
            "families":families,"not_graded_reasons":dict(not_graded),
            "bet_eligible":False,"commercial_promotion_allowed":False}
