"""SPORTS EDGE SOCCER observable commercial go/no-go, no wager generation.

Separates canonical CLV and approved production bets from isolated OOS baselines
and provider-proxy research quote visits. Zero evidence is reported as zero.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime,timezone,timedelta
import json
import math
from pathlib import Path

VERSION="SOCCER_COMMERCIAL_READINESS_AUDIT_v1.0.0"
SOURCE="API_FOOTBALL_ODDS_V3"
FAMILIES=("1X2","BTTS","FT_TOTALS","TEAM_TOTALS","1H","2H",
          "FT_CORNERS","TEAM_CORNERS","CARDS","PLAYER_PROPS")


def _dt(v):
    if not isinstance(v,str):
        return None
    try:
        t=datetime.fromisoformat(v.replace("Z","+00:00"))
        return t.astimezone(timezone.utc) if t.tzinfo else None
    except ValueError:
        return None


def _read_json(path):
    r=json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(r,dict):
        raise ValueError("INVALID_AUDIT_SOURCE")
    return r


def _jsonl(path,limit=5000):
    target=Path(path)
    rows=[json.loads(s) for s in target.read_text(encoding="utf-8").splitlines() if s.strip()] if target.is_file() else []
    if len(rows)>limit or any(not isinstance(x,dict) for x in rows):
        raise ValueError("UNBOUNDED_OR_INVALID_JSONL_SOURCE")
    return rows


def _all_quotes(path):
    folder=Path(path)
    files=sorted(folder.glob("*.jsonl")) if folder.is_dir() else []
    if len(files)>400:
        raise ValueError("PRICE_RESEARCH_PARTITIONS_UNBOUNDED")
    rows=[]
    for f in files:
        if f.stat().st_size>1_500_000:
            raise ValueError("PRICE_PARTITION_UNBOUNDED")
        rows.extend(_jsonl(f,1500))
    if len(rows)>30_000:
        raise ValueError("PRICE_VISITS_UNBOUNDED")
    return rows


def audit(frozen,official,import_receipt,quotes,clv,promotion,player,latest,now):
    if now.tzinfo is None:
        raise ValueError("AUDIT_CLOCK_NOT_UTC")
    if import_receipt.get("status")!="RESEARCH_ONLY_FROZEN_PG_IMPORT":
        raise ValueError("SOURCE_IMPORT_PROVENANCE_NOT_VERIFIED")
    if clv.get("status")!="ACTIVE_TRUE_CLV_SAMPLE":
        raise ValueError("CANONICAL_CLV_NOT_VERIFIED")
    if promotion.get("status")!="PROMOTION_REVIEW_FRAMEWORK_ACTIVE":
        raise ValueError("PROMOTION_FRAMEWORK_NOT_VERIFIED")
    freeze_ids=set()
    freeze_by_id={}
    for r in frozen:
        if r.get("status")!="FROZEN_RESEARCH_PREMATCH_SPORT_PROJECTION":
            raise ValueError("INVALID_FROZEN_RESEARCH_ROW")
        fid=r.get("fixture_id")
        if type(fid) is not int or fid in freeze_ids or not (_dt(r.get("captured_at_utc")) < _dt(r.get("kickoff"))):
            raise ValueError("FROZEN_ASOF_IDENTITY_INVALID")
        freeze_ids.add(fid)
        freeze_by_id[fid]=r
    imports=import_receipt.get("imported_keys") or []
    imported_ids={k.split(":",1)[0] for k in imports if isinstance(k,str)}
    verified_grade_ids=set()
    for g in official:
        fid=g.get("fixture_id")
        if fid not in freeze_by_id or fid in verified_grade_ids:
            raise ValueError("OFFICIAL_OOS_SOURCE_NOT_VERIFIED")
        if g.get("status")!="VERIFIED_FROZEN_PREGAME_SPORT_ONLY_OOS" or g.get("bet_eligible") is not False:
            raise ValueError("OOS_PROMOTION_OR_STATUS_NOT_VERIFIED")
        f=freeze_by_id[fid]
        if g.get("model_frozen_at_utc") != f["captured_at_utc"]:
            raise ValueError("RETROSPECTIVE_OOS_MODEL_MISMATCH")
        if not (_dt(g.get("official_result_verified_at_utc")) >= _dt(f["kickoff"])):
            raise ValueError("OFFICIAL_RESULT_BEFORE_KICKOFF")
        verified_grade_ids.add(fid)
    price_visits,exact_markets=0,0
    exact_but_stale_updates=0
    exact_with_post_model_updates=0
    no_exact_market_visits=0
    pairs=[]
    grouped=defaultdict(list)
    for q in quotes:
        if q.get("source")!=SOURCE:
            raise ValueError("NON_OFFICIAL_RESEARCH_PRICE_SOURCE")
        fid=q.get("fixture_id")
        original=freeze_by_id.get(fid)
        if not original:
            raise ValueError("PRICE_FIXTURE_WITHOUT_FROZEN_SPORT")
        recorded=_dt(q.get("captured_at_utc"))
        kick=_dt(original["kickoff"])
        frozen_at=_dt(original["captured_at_utc"])
        if (q.get("frozen_at_utc")!=original["captured_at_utc"]
                or not recorded or not (frozen_at<recorded<kick)
                or q.get("bet_eligible") is not False
                or q.get("production_promotion_allowed") is not False):
            raise ValueError("PRICE_NOT_ACTUALLY_ASOF_OR_RESEARCH_ONLY")
        price_visits+=1
        instr=q.get("instruments") or []
        if not isinstance(instr,list):
            raise ValueError("PRICE_INSTRUMENTS_INVALID")
        updated=_dt(q.get("provider_update_utc"))
        if instr:
            exact_markets+=1
            if updated is not None and updated>frozen_at:
                exact_with_post_model_updates+=1
            else:
                exact_but_stale_updates+=1
        else:
            no_exact_market_visits+=1
        for x in instr:
            if x.get("bookmaker_id")!=8 or x.get("family") not in ("1X2","BTTS","FT_TOTALS"):
                raise ValueError("UNSUPPORTED_INSTRUMENT")
            when=_dt(x.get("provider_update_utc"))
            price=x.get("decimal_price")
            if (not updated or when!=updated or when>recorded+timedelta(minutes=5)
                    or type(price) not in (int,float)
                    or not math.isfinite(price) or price<=1):
                raise ValueError("PRICE_EVIDENCE_NOT_VERIFIED")
            key=(fid,8,x.get("market_id"),x.get("family"),x.get("selection"),x.get("line"))
            grouped[key].append((recorded,updated,float(price)))
    priced_fixtures=set()
    for key,visits in grouped.items():
        visits.sort()
        fid=key[0]
        frozen_at=_dt(freeze_by_id[fid]["captured_at_utc"])
        kick=_dt(freeze_by_id[fid]["kickoff"])
        priced_fixtures.add(fid)
        first=visits[0]
        for next_visit in visits[1:]:
            if (first[0]<next_visit[0]<kick
                    and frozen_at<first[1]<next_visit[1]):
                pairs.append({
                    "fixture_id":fid,"market_family":key[3],
                    "market_id":key[2],"selection":key[4],"line":key[5],
                    "second_within_40min_of_kickoff":kick-next_visit[0]<=timedelta(minutes=40),
                    "official_FT_OOS_also_available":fid in verified_grade_ids,
                    "source":"RESEARCH_ONLY_PROVIDER_PROXY_PRICE_MOVEMENT",
                })
                break
    family_counts=clv.get("family_counts") or {}
    reviews=promotion.get("market_family_reviews") or []
    summaries={}
    for row in reviews:
        fam=row.get("market_family")
        if fam not in FAMILIES:
            continue
        summaries[fam]={
            "promotion_stage":row.get("recommended_state"),
            "canonical_true_clv_rows":row.get("true_clv_rows"),
            "settled_bets":row.get("settled"),
            "unique_clv_fixtures":row.get("unique_fixtures"),
            "gate_blockers":row.get("blockers",[])[:16],
            "production_bet_authorized":False,
        }
    if set(summaries)!=set(FAMILIES):
        raise ValueError("MISSING_CANONICAL_MARKET_PROMOTION_FAMILIES")
    # The isolated Poisson research model has a verified regulation FT result
    # for these three markets, but NOT the same status as canonical OOS.
    isolated_oos={x:len(verified_grade_ids) for x in ("1X2","BTTS","FT_TOTALS")}
    for fam,rows in isolated_oos.items():
        summaries[fam]["isolated_limited_model_FT_OOS_research_fixtures"]=rows
        summaries[fam]["research_exact_instrument_provider_pairs"]=sum(
            p["market_family"]==fam for p in pairs)
    latest_time=_dt(latest.get("generated_at_utc"))
    stale=latest_time is None or latest_time>now or now-latest_time>timedelta(hours=48)
    blocker=[
        "NO_VERIFIED_DIRECT_EXECUTABLE_BOOKMAKER_BETSLIP_QUOTE",
        "CANONICAL_MARKET_GATE_NOT_PROMOTED",
        "INSUFFICIENT_SPORT_MODEL_AVAILABILITY_AND_DEEP_DIVE_COVERAGE",
        "MARKET_INDEPENDENT_OOS_AND_SETTLEMENT_GATES_NOT_MET",
        "LICENSING_GEO_COMPLIANCE_PAYMENT_AND_PRODUCT_UX_NOT_VERIFIED",
    ]
    if not pairs:
        blocker.append("NO_NEW_TWO_VISIT_PROVIDER_UPDATE_QUOTE_PAIRS")
    if exact_but_stale_updates:
        blocker.append("EXACT_QUOTES_EXIST_BUT_LAST_PROVIDER_UPDATE_PRECEDES_SPORT_MODEL")
    if stale:
        blocker.append("LATEST_CANONICAL_SCHEDULER_STALE")
    if not verified_grade_ids:
        blocker.append("NO_ISOLATED_FT_OOS_YET")
    if player.get("production_promotion_allowed") is not False:
        raise ValueError("PLAYER_PROPS_SAFETY_GATE_NOT_VERIFIED")
    report={
        "schema_version":"1.0.0","model_version":VERSION,
        "generated_at_utc":now.isoformat(),
        "status":"RESEARCH_VALIDATED_COMMERCIAL_PRODUCTION_BLOCKED",
        "verified_frozen_sport_research_models":len(freeze_ids),
        "original_commit_imported_model_ids_count":len(imported_ids),
        "official_FT_research_OOS_fixture_count":len(verified_grade_ids),
        "research_OOS_1X2_BTTs_FT25_fixture_counts":isolated_oos,
        "actual_provider_proxy_price_visit_count":price_visits,
        "visits_with_exact_complete_markets":exact_markets,
        "visits_with_exact_market_but_stale_provider_update":exact_but_stale_updates,
        "visits_with_provider_update_after_original_frozen_model":exact_with_post_model_updates,
        "visits_without_complete_exact_market":no_exact_market_visits,
        "research_proxy_price_fixtures":len(priced_fixtures),
        "research_quote_instrument_pairs_with_advanced_provider_timestamp":len(pairs),
        "research_pairs_observed_within_40min_kickoff":sum(p["second_within_40min_of_kickoff"] for p in pairs),
        "same_fixture_FT_and_research_price_pairs":sum(p["official_FT_OOS_also_available"] for p in pairs),
        "canonical_true_clv_total":clv.get("comparable_true_clv_rows"),
        "canonical_clv_family_counts":family_counts,
        "canonical_promotion_counts":promotion.get("recommended_state_counts"),
        "market_family_gates":summaries,
        "sgp_joint_bookmaker_ticket_priced":"NOT VERIFIED",
        "plus110_live_executable_price":"NOT VERIFIED",
        "research_engine_operational_subsystem":"SPORT_FIRST_FROZEN_PG_AND_FT_OOS_PROVEN",
        "private_research_preview_eligibility":"REVIEW_REQUIRED_PRODUCT_SECURITY_AND_LEGAL",
        "paid_advisory_product_eligibility":"NO_GO",
        "paid_betting_alerts_eligibility":"NO_GO",
        "marketing_claims_or_advertising_clearance":"NOT VERIFIED",
        "legal_regulatory_data_license_approval":"NOT VERIFIED",
        "subscription_billing_security_and_age_controls":"NOT VERIFIED",
        "blockers":blocker,
        "canonical_clv_rows_created_by_this_audit":0,
        "retroactive_predictions_created":False,
        "production_promotion_allowed":False,
        "bet_eligible":False,
        "interpretation":"As-of Poisson research baseline and exact proxy odds visits do not inherit canonical BET, CLV or market calibration gates.",
    }
    return report


def self_test():
    now=datetime(2026,11,10,18,tzinfo=timezone.utc)
    kick=now-timedelta(hours=3)
    frozen=[{"fixture_id":1,"status":"FROZEN_RESEARCH_PREMATCH_SPORT_PROJECTION",
             "kickoff":kick.isoformat(),
             "captured_at_utc":(kick-timedelta(hours=2)).isoformat()}]
    official=[{"fixture_id":1,"status":"VERIFIED_FROZEN_PREGAME_SPORT_ONLY_OOS",
               "model_frozen_at_utc":frozen[0]["captured_at_utc"],
               "official_result_verified_at_utc":now.isoformat(),
               "bet_eligible":False}]
    gate={"status":"PROMOTION_REVIEW_FRAMEWORK_ACTIVE",
          "market_family_reviews":[{"market_family":f,"recommended_state":"RESEARCH","true_clv_rows":0,"settled":0,"unique_fixtures":0,"blockers":["NOT_READY"]} for f in FAMILIES],
          "recommended_state_counts":{"RESEARCH":10}}
    clv={"status":"ACTIVE_TRUE_CLV_SAMPLE","family_counts":{"1X2":53},
         "comparable_true_clv_rows":789}
    report=audit(frozen,official,
                 {"status":"RESEARCH_ONLY_FROZEN_PG_IMPORT","imported_keys":[f"1:{frozen[0]['captured_at_utc']}"]},
                 [],clv,gate,{"production_promotion_allowed":False},
                 {"generated_at_utc":now.isoformat()},now)
    assert report["official_FT_research_OOS_fixture_count"]==1
    assert report["canonical_true_clv_total"]==789
    assert report["research_quote_instrument_pairs_with_advanced_provider_timestamp"]==0
    assert report["paid_advisory_product_eligibility"]=="NO_GO"
    assert report["canonical_clv_rows_created_by_this_audit"]==0
    assert report["production_promotion_allowed"] is False
    print("SOCCER_COMMERCIAL_READINESS_OFFLINE_PASS: REAL_FROZEN_FT_NO_FAKE_ODDS_NO_AUTO_LAUNCH")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    for k in ("frozen","oos","import-receipt","quote-dir","clv","promotion","player","latest","output"):
        p.add_argument("--"+k,default=None)
    args=p.parse_args()
    if args.self_test:
        self_test()
        return
    if not all([args.frozen,args.oos,args.import_receipt,args.quote_dir,
                args.clv,args.promotion,args.player,args.latest,args.output]):
        p.error("all inputs and output are required")
    report=audit(_jsonl(args.frozen),_jsonl(args.oos),_read_json(args.import_receipt),
                 _all_quotes(args.quote_dir),_read_json(args.clv),
                 _read_json(args.promotion),_read_json(args.player),
                 _read_json(args.latest),datetime.now(timezone.utc))
    target=Path(args.output)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="market_family_gates"},ensure_ascii=False))


if __name__=="__main__":
    main()
