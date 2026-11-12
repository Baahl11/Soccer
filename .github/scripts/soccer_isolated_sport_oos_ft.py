"""Official FT-only OOS grades for genuinely frozen isolated sport projections.

No replay, no market/odds, no prediction recomputation. Matches fixture, teams,
kickoff and source timestamps; valid regulation FT only. Never promotes BET.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from soccer_isolated_pregame_sport_first import (
    OfficialProvider, _dt, _integer, _number, _read_rows,
)

VERSION = "ISOLATED_SOCCER_SPORT_FIRST_OFFICIAL_FT_OOS_v1.0.1"
MAX_FIXTURES_PER_RUN = 4
WAIT_AFTER_KICKOFF = timedelta(hours=3)
MAX_POSTGAME_AGE = timedelta(days=14)
MAX_UNRESOLVED_ATTEMPT_HISTORY = 5000


def _read_unresolved_attempt_history(path: Path, now: datetime) -> dict[int, datetime]:
    """Small retry metadata, not recomputed probabilities or result data."""
    if not path.exists():
        return {}
    if path.stat().st_size > 350_000:
        raise RuntimeError("FT_OOS_RECEIPT_UNBOUNDED")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise RuntimeError("FT_OOS_RECEIPT_INVALID")
    stored = receipt.get("unresolved_attempted_at_utc")
    if stored is None:
        legacy = receipt.get("unresolved_fixture_ids") or []
        at = _dt(receipt.get("generated_at_utc"))
        if not isinstance(legacy, list) or len(legacy)>MAX_UNRESOLVED_ATTEMPT_HISTORY:
            raise RuntimeError("FT_OOS_LEGACY_ATTEMPTS_INVALID")
        stored = {str(fid):at.isoformat() for fid in legacy
                  if _integer(fid) is not None and at is not None}
    if not isinstance(stored, dict) or len(stored)>MAX_UNRESOLVED_ATTEMPT_HISTORY:
        raise RuntimeError("FT_OOS_ATTEMPTS_UNBOUNDED")
    attempts = {}
    for raw_id, raw_time in stored.items():
        if not isinstance(raw_id,str) or not raw_id.isdecimal():
            raise RuntimeError("FT_OOS_ATTEMPT_ID_INVALID")
        fid=int(raw_id)
        stamp=_dt(raw_time)
        if not 0<fid<10**12 or stamp is None:
            raise RuntimeError("FT_OOS_ATTEMPT_TIMESTAMP_INVALID")
        if timedelta(0)<=now-stamp<=MAX_POSTGAME_AGE:
            attempts[fid]=stamp
    return attempts


def valid_pregame(f: dict[str, Any]) -> bool:
    if (not isinstance(f, dict) or
            f.get("status") != "FROZEN_RESEARCH_PREMATCH_SPORT_PROJECTION"
            or f.get("source") != "LIVE_API_FOOTBALL_OFFICIAL_PREGAME_SPORT_INPUTS"
            or f.get("production_promotion_allowed") is not False
            or f.get("bet_eligible") is not False
            or f.get("raw_projection_before_market") is not True):
        return False
    fx = f.get("fixture")
    if not isinstance(fx, dict) or fx.get("fixture_id") != f.get("fixture_id"):
        return False
    if not all(_integer(fx.get(k)) for k in
               ("home_team_id", "away_team_id", "fixture_id", "league_id")):
        return False
    kick, observed = _dt(f.get("kickoff")), _dt(f.get("captured_at_utc"))
    if not kick or not observed or fx.get("kickoff") != f.get("kickoff"):
        return False
    if not observed < kick - timedelta(minutes=30):
        return False
    raw = f.get("raw_sport_projection") or {}
    if raw.get("status") != "MODELED_LIMITED" or not isinstance(raw, dict):
        return False
    probs = ("raw_home_win_prob", "raw_draw_prob", "raw_away_win_prob",
             "raw_btts_yes_prob", "raw_over_2_5_prob")
    if any(_number(raw.get(k)) is None or not 0 <= float(raw[k]) <= 1 for k in probs):
        return False
    return abs(sum(float(raw[k]) for k in probs[:3])-1.0) < 0.001


def grade_if_official_ft(f: dict[str, Any], response: dict[str, Any],
                         observed_at: datetime) -> dict[str, Any] | None:
    if not valid_pregame(f):
        raise ValueError("FROZEN_PREGAME_MODEL_NOT_VERIFIED")
    rows = response.get("response")
    if not isinstance(rows,list) or len(rows)!=1 or not isinstance(rows[0],dict):
        return None
    row = rows[0]
    fixture = row.get("fixture") or {}
    teams = row.get("teams") or {}
    goals = row.get("goals") or {}
    saved = f["fixture"]
    if fixture.get("id") != f["fixture_id"]:
        return None
    if (teams.get("home") or {}).get("id") != saved["home_team_id"]:
        return None
    if (teams.get("away") or {}).get("id") != saved["away_team_id"]:
        return None
    if (fixture.get("status") or {}).get("short") != "FT":
        return None
    actual_kick = _dt(fixture.get("date"))
    original_kick = _dt(f["kickoff"])
    if not actual_kick or abs((actual_kick-original_kick).total_seconds())>600:
        return None
    if observed_at < original_kick + WAIT_AFTER_KICKOFF:
        return None
    home = goals.get("home")
    away = goals.get("away")
    if type(home) is not int or type(away) is not int or not (0 <= home <= 25 and 0 <= away <= 25):
        return None
    raw = f["raw_sport_projection"]
    winner = "HOME" if home>away else "AWAY" if away>home else "DRAW"
    actual = [int(winner == x) for x in ("HOME","DRAW","AWAY")]
    probabilities = [float(raw[k]) for k in ("raw_home_win_prob","raw_draw_prob","raw_away_win_prob")]
    multi_brier = sum((p-y)**2 for p,y in zip(probabilities,actual))
    actual_btts = int(home>0 and away>0)
    p_btts = float(raw["raw_btts_yes_prob"])
    actual_over25 = int(home+away>=3)
    p_over25 = float(raw["raw_over_2_5_prob"])
    likelihood = max(1e-12, probabilities[actual.index(1)])
    return {
        "schema_version":"1.0.0",
        "status":"VERIFIED_FROZEN_PREGAME_SPORT_ONLY_OOS",
        "fixture_id":f["fixture_id"],
        "kickoff":f["kickoff"],
        "model_frozen_at_utc":f["captured_at_utc"],
        "official_result_verified_at_utc":observed_at.isoformat(),
        "source":"API_FOOTBALL_V3_FIXTURES_OFFICIAL_FT",
        "result":{"home_goals":home,"away_goals":away,"status":"FT",
                  "home_team_id":saved["home_team_id"],
                  "away_team_id":saved["away_team_id"]},
        "model_version":f["model_version"],
        "projection_model":f["projection_model"],
        "canonical_runtime_model_sha":f["canonical_runtime_model_sha"],
        "evaluation":{
            "1X2":{"predicted":[round(p,6) for p in probabilities],"actual":winner,
                   "brier_multiclass":round(multi_brier,8),
                   "log_loss":round(-math.log(likelihood),8)},
            "BTTS":{"p_yes":round(p_btts,6),"actual_yes":bool(actual_btts),
                    "brier":round((p_btts-actual_btts)**2,8)},
            "FT_TOTALS_OVER_2_5":{"p_over":round(p_over25,6),"actual_over":bool(actual_over25),
                                  "brier":round((p_over25-actual_over25)**2,8)},
        },
        "market_price_verified":False,
        "clv_eligible":False,
        "profitability_estimated":False,
        "model_recomputed_after_result":False,
        "production_promotion_allowed":False,
        "bet_eligible":False,
    }


def evaluate(frozen: list[dict[str,Any]], existing: list[dict[str,Any]],
             provider, *, now: datetime,
             prior_unresolved_attempts: dict[int,datetime] | None = None) -> dict[str,Any]:
    if now.tzinfo is None:
        raise ValueError("UTC_TIME_REQUIRED")
    known = {x["fixture_id"] for x in existing}
    previous = prior_unresolved_attempts or {}
    if len(previous)>MAX_UNRESOLVED_ATTEMPT_HISTORY:
        raise RuntimeError("FT_OOS_ATTEMPT_HISTORY_UNBOUNDED")
    candidates = []
    invalid=0
    for f in frozen:
        if not valid_pregame(f):
            invalid+=1
            continue
        kick = _dt(f["kickoff"])
        if (f["fixture_id"] not in known
                and WAIT_AFTER_KICKOFF <= now-kick <= MAX_POSTGAME_AGE):
            candidates.append(f)
    # Unchecked originals first; retried originals in oldest-attempt order.
    # Non-FT matches cannot monopolize the four daily check slots.
    candidates.sort(key=lambda f:(
        f["fixture_id"] in previous,
        previous.get(f["fixture_id"],datetime.min.replace(tzinfo=timezone.utc)),
        f["kickoff"],f["fixture_id"],
    ))
    candidate_ids={f["fixture_id"] for f in candidates}
    attempts={
        fid:stamp for fid,stamp in previous.items()
        if fid in candidate_ids and fid not in known
        and timedelta(0)<=now-stamp<=MAX_POSTGAME_AGE
    }
    new=[]; unresolved=[]
    for f in candidates[:MAX_FIXTURES_PER_RUN]:
        response=provider.get("fixtures",{"id":f["fixture_id"]})
        graded=grade_if_official_ft(f,response,now)
        if graded is None:
            unresolved.append(f["fixture_id"])
            attempts[f["fixture_id"]]=now
        else:
            new.append(graded)
            attempts.pop(f["fixture_id"],None)
    if provider.calls > MAX_FIXTURES_PER_RUN:
        raise RuntimeError("OFFICIAL_FT_RESULT_CALL_CAP_EXCEEDED")
    return {
        "status":"VERIFIED_OFFICIAL_FT_OOS_CAPTURE" if new else "NO_NEW_VERIFIED_FT_OOS",
        "generated_at_utc":now.isoformat(),
        "previous_grades":len(existing),
        "frozen_records":len(frozen),
        "invalid_frozen_records_excluded":invalid,
        "candidate_finished_window":len(candidates),
        "official_ft_checked":min(len(candidates),MAX_FIXTURES_PER_RUN),
        "new_oos_fixture_rows":len(new),
        "new":new,
        "unresolved_fixture_ids":unresolved,
        "unresolved_attempted_at_utc":{
            str(fid):stamp.isoformat() for fid,stamp in sorted(attempts.items())
        },
        "unresolved_attempt_history_size":len(attempts),
        "attempt_selection_policy":"NEVER_CHECKED_THEN_LEAST_RECENT_UNRESOLVED_FT_ONLY",
        "provider_requests_added":provider.calls,
        "daily_remaining_reported":provider.remaining,
        "market_prices_used":False,
        "retroactive_predictions_created":False,
        "production_promotion_allowed":False,
        "bet_eligible":False,
    }


def self_test():
    now=datetime(2026,11,10,18,tzinfo=timezone.utc)
    kick=now-timedelta(hours=6)
    f={
        "status":"FROZEN_RESEARCH_PREMATCH_SPORT_PROJECTION",
        "source":"LIVE_API_FOOTBALL_OFFICIAL_PREGAME_SPORT_INPUTS",
        "fixture_id":10, "kickoff":kick.isoformat(),
        "captured_at_utc":(kick-timedelta(hours=2)).isoformat(),
        "fixture":{"fixture_id":10,"kickoff":kick.isoformat(),
                   "home_team_id":11,"away_team_id":12,"league_id":5},
        "raw_projection_before_market":True,
        "production_promotion_allowed":False,"bet_eligible":False,
        "model_version":"SOCCER EDGE ENGINE v1.0",
        "projection_model":"GOAL_RATE_POISSON_LIMITED_v0.1",
        "canonical_runtime_model_sha":"fixedmodelsha",
        "raw_sport_projection":{
            "status":"MODELED_LIMITED","raw_home_win_prob":0.4,
            "raw_draw_prob":0.3,"raw_away_win_prob":0.3,
            "raw_btts_yes_prob":0.55,"raw_over_2_5_prob":0.6},
    }
    game={"fixture":{"id":10,"date":kick.isoformat(),"status":{"short":"FT"}},
          "teams":{"home":{"id":11},"away":{"id":12}},
          "goals":{"home":2,"away":1}}
    class Offline:
        def __init__(self,response):
            self.calls=0;self.remaining=7000;self.response=response
        def get(self,endpoint,params):
            assert endpoint=="fixtures" and params=={"id":10}
            self.calls+=1
            return {"response":[self.response]}
    assert valid_pregame(f)
    first=evaluate([f],[],Offline(game),now=now)
    assert first["new_oos_fixture_rows"]==1 and first["provider_requests_added"]==1
    g=first["new"][0]
    assert g["evaluation"]["1X2"]["actual"]=="HOME"
    assert g["evaluation"]["FT_TOTALS_OVER_2_5"]["actual_over"] is True
    assert g["evaluation"]["BTTS"]["actual_yes"] is True
    assert g["production_promotion_allowed"] is False and g["clv_eligible"] is False
    assert evaluate([f],[g],Offline(game),now=now)["provider_requests_added"]==0
    not_yet=Offline(game)
    assert evaluate([f],[],not_yet,now=kick+timedelta(hours=1))["provider_requests_added"]==0
    assert grade_if_official_ft(f,{ "response":[{**game,"fixture":{**game["fixture"],"status":{"short":"NS"}}}]},now) is None
    assert grade_if_official_ft(f,{ "response":[{**game,"teams":{"home":{"id":99},"away":{"id":12}}}]},now) is None
    assert grade_if_official_ft(f,{ "response":[{**game,"goals":{"home":None,"away":1}}]},now) is None
    assert not valid_pregame({**f,"captured_at_utc":(kick+timedelta(minutes=1)).isoformat()})
    # Twelve frozen fixtures, all lacking FT: every ID checked in 3 runs.
    pool=[{**f,"fixture_id":i,"fixture":{**f["fixture"],"fixture_id":i}}
          for i in range(20,32)]
    class NeverFinished:
        def __init__(self):
            self.calls=0
            self.remaining=7000
        def get(self,endpoint,params):
            assert endpoint=="fixtures" and "id" in params
            self.calls+=1
            return {"response":[]}
    history={}
    checked=[]
    for minutes in (0,20,40):
        step=evaluate(pool,[],NeverFinished(),now=now+timedelta(minutes=minutes),
                      prior_unresolved_attempts=history)
        checked.extend(step["unresolved_fixture_ids"])
        assert step["provider_requests_added"]==4
        assert step["unresolved_attempt_history_size"]==len(checked)
        history={int(fid):_dt(t) for fid,t in step["unresolved_attempted_at_utc"].items()}
    assert checked==list(range(20,32))
    again=evaluate(pool,[],NeverFinished(),now=now+timedelta(hours=1),
                   prior_unresolved_attempts=history)
    assert again["unresolved_fixture_ids"]==[20,21,22,23]
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/"legacy.json"
        path.write_text(json.dumps({"generated_at_utc":now.isoformat(),
                                   "unresolved_fixture_ids":[20,21,22,23]}),encoding="utf-8")
        migrated=_read_unresolved_attempt_history(path,now)
        assert sorted(migrated)==[20,21,22,23]
        next_run=evaluate(pool,[],NeverFinished(),now=now+timedelta(minutes=1),
                          prior_unresolved_attempts=migrated)
        assert next_run["unresolved_fixture_ids"]==[24,25,26,27]
    print("ISOLATED_SPORT_FT_OOS_OFFLINE_PASS: FREEZE,FIXTURE,OFFICIAL_FT,NO_ZERO,NO_PRICE,NO_REPLAY,FAIR_RETRY_ROTATION")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--frozen",type=Path)
    p.add_argument("--oos",type=Path)
    p.add_argument("--receipt",type=Path)
    args=p.parse_args()
    if args.self_test:
        self_test()
        return
    if not all([args.frozen,args.oos,args.receipt]):
        p.error("frozen,oos,receipt required")
    frozen=_read_rows(args.frozen)
    existing=_read_rows(args.oos)
    now=datetime.now(timezone.utc)
    previous=_read_unresolved_attempt_history(args.receipt,now)
    provider=OfficialProvider(os.environ.get("API_FOOTBALL_KEY",""))
    result=evaluate(frozen,existing,provider,now=now,
                    prior_unresolved_attempts=previous)
    if result["new"]:
        args.oos.parent.mkdir(parents=True,exist_ok=True)
        with args.oos.open("a",encoding="utf-8") as out:
            for row in result["new"]:
                out.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n")
    receipt={k:v for k,v in result.items() if k!="new"}
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(receipt,ensure_ascii=False,sort_keys=True))


if __name__=="__main__":
    main()
