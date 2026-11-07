from __future__ import annotations

from datetime import datetime,timezone,timedelta
import json
import pytest

from mcp_gateway import multimarket_history_bridge_v1 as fe11

NOW = datetime(2026,11,6,22,0,tzinfo=timezone.utc)
KICKOFF = NOW + timedelta(hours=8)
GENERATED = NOW - timedelta(minutes=5)

def sample_profile(team_id:int):
    return {
        "version":fe11.MODEL_VERSION,"team_id":team_id,
        "target_kickoff":KICKOFF.isoformat(),
        "evidence_scope":"PAST_VERIFIED_FT_ONLY",
        "sports_first_no_market_inputs":True,"descriptive_only":True,
        "calibrated_prediction":False,"bet_eligible":False,
        "splits":{"all":{"matches_with_valid_ft":1,"features":{}},
                  "home":{"matches_with_valid_ft":1,"features":{}},
                  "away":{"matches_with_valid_ft":0,"features":{}}},
        "observations":[{
            "fixture_id":12,
            "kickoff":(NOW-timedelta(days=2)).isoformat(),
            "venue":"home","goals_for":2,"goals_against":1,
            "ft_goals_total":3,"first_half_goals_for":1,
            "first_half_goals_against":0,"first_half_goals_total":1,
            "second_half_goals_for":1,"second_half_goals_against":1,
            "second_half_goals_total":2,"team_corners_ft":5,
            "opponent_corners_ft":3,"total_corners_ft":8,
        }],
        "limitations":["NOT_OPPONENT_ADJUSTED"],
    }

def sample_pattern():
    return {
        variant:{
            "version":fe11.MODEL_VERSION,"venue":"all",
            "eligible_as_joint_model_probability":False,
            "bet_eligible":False,
            "price_decimal":None,"correlated_sgp_quote":None,
            "complete_historical_rows":1,
            "observed_joint_hits":1,"observed_joint_misses":0,
            "asian_handicap_push_records":0,
            "observed_joint_hit_frequency":1.0,
            "legs":[
                {"type":"FT_GOALS_OVER","line":1.5},
                {"type":"FT_CORNERS_OVER","line":6.5},
                {"type":"TEAM_HANDICAP","line":1,"settlement":variant},
            ],
        } for variant in ("ASIAN","THREE_WAY")
    }

def sample_doc():
    return {"source":fe11.SOURCE,"generated_at_utc":GENERATED.isoformat(),
            "reports":[{
                "fixture_id":3501,"kickoff":KICKOFF.isoformat(),
                "home_team_id":101,"away_team_id":202,
                "home_profile":sample_profile(101),
                "away_profile":sample_profile(202),
                "home_pattern":sample_pattern(),
                "away_pattern":sample_pattern(),
            }]}

def test_fe11_sport_first_provenance_and_exact_future_fixture_valid():
    rows=fe11.validate_import(sample_doc(),now=NOW)
    assert len(rows)==1
    assert rows[0]["fixture_id"]==3501
    assert rows[0]["home_id"]==101
    assert rows[0]["kickoff"]==KICKOFF

def test_fe11_never_accept_forged_odds_price_or_bet_eligible():
    doc=sample_doc()
    doc["reports"][0]["home_profile"]["odds"]=2.10
    with pytest.raises(ValueError,match="PROFILE_OR_HANDICAP"):
        fe11.validate_import(doc,now=NOW)
    doc=sample_doc()
    doc["reports"][0]["home_profile"]["bet_eligible"]=True
    with pytest.raises(ValueError,match="PROFILE_OR_HANDICAP"):
        fe11.validate_import(doc,now=NOW)
    doc=sample_doc()
    doc["reports"][0]["home_pattern"]["ASIAN"]["price_decimal"]=2.00
    with pytest.raises(ValueError,match="PROFILE_OR_HANDICAP"):
        fe11.validate_import(doc,now=NOW)

def test_fe11_refuses_postkickoff_history_cross_fixture_and_malformed_periods():
    doc=sample_doc()
    doc["reports"][0]["home_profile"]["observations"][0]["kickoff"]=(KICKOFF+timedelta(minutes=1)).isoformat()
    with pytest.raises(ValueError):
        fe11.validate_import(doc,now=NOW)
    doc=sample_doc()
    doc["reports"][0]["away_profile"]["team_id"]=999
    with pytest.raises(ValueError):
        fe11.validate_import(doc,now=NOW)
    doc=sample_doc()
    doc["reports"][0]["home_profile"]["observations"][0]["total_corners_ft"]=14
    with pytest.raises(ValueError):
        fe11.validate_import(doc,now=NOW)
    doc=sample_doc()
    doc["reports"][0]["home_profile"]["observations"][0]["second_half_goals_total"]=5
    with pytest.raises(ValueError):
        fe11.validate_import(doc,now=NOW)

def test_fe11_rejects_duplicate_fixture_and_missing_source_provenance():
    doc=sample_doc()
    doc["reports"].append(doc["reports"][0].copy())
    with pytest.raises(ValueError):
        fe11.validate_import(doc,now=NOW)
    doc=sample_doc()
    doc["source"]="UNVERIFIED"
    with pytest.raises(ValueError):
        fe11.validate_import(doc,now=NOW)
    doc=sample_doc()
    doc["generated_at_utc"]=(NOW-timedelta(days=9)).isoformat()
    with pytest.raises(ValueError):
        fe11.validate_import(doc,now=NOW)

def test_fe11_db_insert_is_immutable_and_identity_checked():
    assert "ON CONFLICT (fixture_id, source_generated_at) DO NOTHING" in fe11.INSERT_SQL
    assert "f.home_team_id = %s" in fe11.INSERT_SQL
    assert "f.away_team_id = %s" in fe11.INSERT_SQL
    assert "f.status = 'NS'" in fe11.INSERT_SQL
    assert "f.kickoff > now()" in fe11.INSERT_SQL
    assert "SET TRANSACTION READ ONLY" in fe11.read_for_match.__code__.co_consts

def test_fe11_ingest_only_persists_verified_ids(monkeypatch):
    # With real now, provide an appropriate relative event rather than clock fixture.
    now=datetime.now(timezone.utc)
    doc=sample_doc()
    doc["generated_at_utc"]=(now-timedelta(minutes=2)).isoformat()
    ko=(now+timedelta(hours=7)).isoformat()
    doc["reports"][0]["kickoff"]=ko
    for side in ("home_profile","away_profile"):
        doc["reports"][0][side]["target_kickoff"]=ko
    class Cursor:
        def execute(self,sql,args):
            assert sql==fe11.INSERT_SQL and len(args)==11
            self.args=args
        def fetchone(self):return [3501]
        def __enter__(self):return self
        def __exit__(self,*a):pass
    class Connection:
        def transaction(self):return self
        def cursor(self):return Cursor()
        def __enter__(self):return self
        def __exit__(self,*a):pass
    monkeypatch.setattr(fe11.persistence,"ensure_schema",lambda:None)
    monkeypatch.setattr(fe11.persistence,"_connect",lambda:Connection())
    result=fe11.ingest(doc)
    assert result["newly_persisted"]==1
    assert result["bet_eligible"] is False
    assert result["probabilities_calibrated"] is False
    assert result["provider_requests_added"]==0

def test_fe11_non_integer_fixture_fails_closed():
    assert fe11.read_for_match("3501")["status"]=="NOT_VERIFIED"
    assert fe11.read_for_match(None)["bet_eligible"] is False



def test_fe11_inventory_is_aggregate_only_and_never_promotes_bets(monkeypatch):
    class Cursor:
        def execute(self,query,*params):
            assert "SET TRANSACTION READ ONLY" in query or "soccer_multimarket_history_profiles" in query
        def fetchone(self):
            return (3,2,1,datetime.now(timezone.utc))
        def __enter__(self):return self
        def __exit__(self,*args):pass
    class Conn:
        def transaction(self):return self
        def cursor(self):return Cursor()
        def __enter__(self):return self
        def __exit__(self,*args):pass
    monkeypatch.setattr(fe11.persistence,"persistence_configured",lambda:True)
    monkeypatch.setattr(fe11.persistence,"ensure_schema",lambda:None)
    monkeypatch.setattr(fe11.persistence,"_connect",lambda:Conn())
    r=fe11.inventory_readiness()
    assert r["status"]=="VERIFIED_READ"
    assert r["research_rows_persisted"]==3
    assert r["unique_fixtures"]==2
    assert r["current_pregame_profiles"]==1
    assert r["production_bets_authorized"] is False
    assert r["provider_requests_added"]==0
    assert "market_price" not in r and "bet" not in r
