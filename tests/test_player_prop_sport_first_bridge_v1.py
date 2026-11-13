"""No bookmaker-led player shots projections; as-of provenance and exact match."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import time

import pytest

from mcp_gateway import player_prop_sport_first_bridge_v1 as core

NOW = datetime(2026, 11, 7, 12, 0, tzinfo=timezone.utc)


def input_snapshot(*, registry_at=None, captured=None, kickoff=None, player_id=101,
                   historical_n=15, historical_minutes=950, opponent_status=None,
                   team_id=550):
    generated = captured or NOW - timedelta(minutes=30)
    kickoff = kickoff or NOW + timedelta(hours=3)
    registry = registry_at or NOW - timedelta(hours=2)
    opponent = {
        "status": opponent_status or "SHRUNK_OPPONENT_SHOT_SUPPRESSION",
        "opponent_last10_n": 8,
        "factor": 1.11,
    }
    row = {
        "player_id": player_id, "team_id":team_id,"opponent_team_id":770,
        "player":"Test Player", "confirmed_starter":True,
        "status":"LIVE_RESEARCH_SHOTS_DISTRIBUTION",
        "historical_played_matches_last20":historical_n,
        "historical_observed_minutes_last20":historical_minutes,
        "registry_generated_at_utc":registry.isoformat(),
        "base_posterior_mean_shots_per90":2.4,
        "expected_shots":2.10,
        "expected_minutes_if_confirmed_starter":75,
        "opponent_adjustment":opponent,
        "lines":[{"line":1.5,"p_over":0.68,"p_under":0.32},
                 {"line":2.5,"p_over":0.45,"p_under":0.55}],
    }
    return {
        "fixture_id":90001,"kickoff":kickoff,
        "generated_at":generated,"stage":"T-20",
        "home_team_id":550,"away_team_id":770,
        "home_team":"Local","away_team":"Visitante",
        "shots":{"fixture_id":90001,"status":"LIVE_RESEARCH_SHOTS_PROPS",
                 "players":[row]},
        "sot":None,
    }


def offer(price="2.10", selection="Over 2.5",
          line_basis="EXPLICIT_OVER_UNDER", player_id=101, team_id=550,
          xi="MATCHED_CONFIRMED_XI", provider_updated=None):
    timestamp=provider_updated or NOW-timedelta(minutes=12)
    return {
        "fixture_id":90001,"player_id":player_id,"team_id":team_id,
        "xi_alignment_status":xi,"offer_type":"PLAYER_COMPONENT_QUOTE",
        "player_market_family":"SHOTS","bookmaker_id":8,
        "line_basis":line_basis,"parsed_line":2.5,"selection":selection,
        "offered_price_decimal":price,
        "provider_updated_at":timestamp,
        "captured_at":NOW-timedelta(minutes=7),
    }


def test_sport_first_independent_projection_has_actual_verified_history_and_asof():
    report=core.sport_profiles([input_snapshot()], now_utc=NOW)
    assert report["status"]=="VERIFIED_SPORTING_INPUTS_RESEARCH_ONLY"
    assert report["verified_pregame_sport_models"]==1
    one=report["rows"][0]
    assert one["historical_played_matches"]==15
    assert one["historical_observed_minutes"]==950
    assert one["historical_rate_per90"]==2.4
    assert one["projected_count"]==2.1
    assert one["expected_minutes"]==75
    assert one["player_id"]==101
    assert one["source"]=="CANONICAL_PREMATCH_FROZEN_PLAYER_MODEL"
    assert one["oos_validation_complete"] is False
    assert one["bet_eligible"] is False


def test_player_research_never_depends_on_a_market_or_plus110_price():
    origin=input_snapshot()
    m1=core.sport_profiles([origin],now_utc=NOW)
    m2=core.sport_profiles([origin],now_utc=NOW)
    assert m1["rows"]==m2["rows"]
    assert m1["odds_used_to_choose_players"] is False


def test_historical_registry_cannot_be_generated_after_prediction():
    bad=input_snapshot(registry_at=NOW-timedelta(minutes=2))
    out=core.sport_profiles([bad],now_utc=NOW)
    assert out["rows"]==[]
    assert out["unverified_input_reasons"]["PLAYER_REGISTRY_ASOF_NOT_VERIFIED"]==1


@pytest.mark.parametrize("key,value,reason",[
    ("historical_n",3,"HISTORICAL_PLAYER_SAMPLE_INSUFFICIENT"),
    ("historical_minutes",200,"HISTORICAL_PLAYER_MINUTES_INSUFFICIENT"),
    ("team_id",999,"MODEL_PLAYER_ID_XI_NOT_VERIFIED"),
])
def test_missing_real_input_blocks_raw_research(key,value,reason):
    kwargs={key:value}
    result=core.sport_profiles([input_snapshot(**kwargs)],now_utc=NOW)
    assert result["rows"]==[]
    assert result["unverified_input_reasons"][reason]==1


def test_source_kickoff_and_pregame_timestamps_are_strictly_ordered():
    cases=[
        input_snapshot(captured=NOW+timedelta(minutes=1)),
        input_snapshot(kickoff=NOW-timedelta(hours=1)),
        input_snapshot(captured=NOW+timedelta(hours=4)),
    ]
    for item in cases:
        assert core.sport_profiles([item],now_utc=NOW)["rows"]==[]


def test_source_xi_and_sport_model_player_id_must_match():
    source=input_snapshot()
    source["shots"]["players"][0]["confirmed_starter"]=False
    assert core.sport_profiles([source],now_utc=NOW)["rows"]==[]
    source=input_snapshot()
    source["shots"]["players"][0]["player_id"]=None
    assert core.sport_profiles([source],now_utc=NOW)["rows"]==[]


def test_exact_identified_market_may_be_compared_but_not_promoted():
    sport=core.sport_profiles([input_snapshot()],now_utc=NOW)
    out=core.match_offers_to_sport([offer()], sport, now_utc=NOW)
    assert out["raw_model_exact_matches"]==1
    item=out["aligned_research_offers"][0]
    assert item["raw_model_probability"]==0.45
    assert item["break_even_probability"]==round(1/2.10,6)
    assert item["operator"]=="OVER"
    assert item["market_line"]==2.5
    assert item["ev_authorized"] is False
    assert item["bet_eligible"] is False
    assert out["odds_used_to_choose_players"] is False


def test_name_3_reference_is_not_over_2point5_and_cannot_infer_operator():
    sport=core.sport_profiles([input_snapshot()],now_utc=NOW)
    out=core.match_offers_to_sport([offer(selection="Test Player - 3",
                                                line_basis="PLAYER_THRESHOLD_N_PLUS")],
                                   sport,now_utc=NOW)
    assert out["raw_model_exact_matches"]==0
    assert out["unverified_reasons"]["EXACT_MARKET_OPERATOR_OR_HALF_LINE_NOT_VERIFIED"]==1


def test_unidentified_book_offer_cannot_match_sport_by_similar_name():
    sport=core.sport_profiles([input_snapshot()],now_utc=NOW)
    for quote in (
        offer(player_id=None),
        offer(player_id=202),
        offer(team_id=770),
        offer(xi="NOT_VERIFIED"),
    ):
        result=core.match_offers_to_sport([quote],sport,now_utc=NOW)
        assert result["raw_model_exact_matches"]==0


def test_missing_quotes_and_unverified_source_timestamps_fail_closed():
    sport=core.sport_profiles([input_snapshot()],now_utc=NOW)
    assert core.match_offers_to_sport([offer(price="1.00")],sport,now_utc=NOW)["raw_model_exact_matches"]==0
    assert core.match_offers_to_sport(
        [offer(provider_updated=NOW-timedelta(hours=4))],
        sport,now_utc=NOW,
    )["raw_model_exact_matches"]==0
    assert core.match_offers_to_sport(
        [offer(provider_updated=NOW-timedelta(hours=1))],
        sport,now_utc=NOW,
    )["raw_model_exact_matches"]==0  # model captured 30 min ago, quote earlier


def test_no_price_based_player_ranking_even_if_ten_prices_are_same():
    sport=core.sport_profiles([input_snapshot()],now_utc=NOW)
    quotes=[offer(price="2.10",player_id=1000+i) for i in range(10)]
    out=core.match_offers_to_sport(quotes,sport,now_utc=NOW)
    assert out["source_player_offer_count"]==10
    assert out["raw_model_exact_matches"]==0
    assert out["unverified_offer_count"]==10


def test_oos_63_and_61_samples_remain_research_not_calibrated(monkeypatch):
    report={
        "model_version":core.OOS_MODEL_VERSION,
        "generated_at_utc":NOW.isoformat(),
        "families":{
            "SHOTS":{"unique_player_fixtures":63,
                     "binary_probability_rows":150,
                     "brier_score":.10086906, "log_loss":.3315,
                     "oos_validation_complete":False},
            "SOT":{"unique_player_fixtures":61,
                   "binary_probability_rows":150,
                   "brier_score":.08365498,"log_loss":.2957,
                   "oos_validation_complete":False},
        }
    }
    monkeypatch.setitem(core._OOS_READ_MEMO,"payload",report)
    monkeypatch.setitem(core._OOS_READ_MEMO,"expires_at",time.monotonic()+100)
    r=core.get_oos_readiness(now_utc=NOW)
    assert r["status"]=="VERIFIED_RECENT_OOS_SOURCE_FOR_REVIEW"
    assert r["families"]["SHOTS"]["unique_player_fixtures"]==63
    assert r["families"]["SHOTS"]["ready_for_manual_oos_review"] is False
    assert r["families"]["SOT"]["calibration_approved"] is False
    assert r["production_promotion_allowed"] is False


def test_even_2000_samples_do_not_auto_authorize_betting(monkeypatch):
    source={"model_version":core.OOS_MODEL_VERSION,
            "generated_at_utc":NOW.isoformat(),
            "families":{x:{"unique_player_fixtures":2000,
                            "binary_probability_rows":3500,
                            "brier_score":.105, "log_loss":.33,
                            "oos_validation_complete":True}
                        for x in ("SHOTS","SOT")}}
    monkeypatch.setitem(core._OOS_READ_MEMO,"payload",source)
    monkeypatch.setitem(core._OOS_READ_MEMO,"expires_at",time.monotonic()+100)
    out=core.get_oos_readiness(now_utc=NOW)
    assert out["families"]["SHOTS"]["ready_for_manual_oos_review"] is True
    assert out["families"]["SHOTS"]["calibration_approved"] is False
    assert out["production_promotion_allowed"] is False


def test_stale_or_future_dated_oos_blocks_review(monkeypatch):
    for stamp in (NOW-timedelta(hours=73),NOW+timedelta(hours=1)):
        payload={"model_version":core.OOS_MODEL_VERSION,
                 "generated_at_utc":stamp.isoformat(),
                 "families":{k:{"unique_player_fixtures":2000,
                     "binary_probability_rows":3000,"brier_score":.1,"log_loss":.3,
                     "oos_validation_complete":True} for k in ("SHOTS","SOT")}}
        monkeypatch.setitem(core._OOS_READ_MEMO,"payload",payload)
        monkeypatch.setitem(core._OOS_READ_MEMO,"expires_at",time.monotonic()+100)
        assert core.get_oos_readiness(now_utc=NOW)["status"]=="NOT_VERIFIED_CURRENT_OOS_REPORT"


def test_database_query_is_bounded_and_sport_first():
    q=core.FROZEN_MODEL_SQL
    assert "soccer_refresh_events" in q
    assert "r.generated_at < f.kickoff" in q
    assert "r.generated_at <= now()" in q
    assert "WHERE f.status = 'NS'" in q
    assert "LIMIT %s" in q
    assert "soccer_market_snapshots" not in q
    assert "offered_price" not in q


def test_owner_match_adapter_reads_exact_original_shots_and_never_prices():
    original = input_snapshot()
    fx = {k: original[k] for k in ("fixture_id", "kickoff", "home_team_id", "away_team_id", "home_team", "away_team")}
    fx.update({"league_id": 39, "season": 2026, "status": "NS"})
    event = {"fixture": fx, "event_type": "SOCCER_REFRESH", "stage": original["stage"],
             "player_shots_intelligence": original["shots"], "offered_price_decimal": 2.1}
    row = {"event_type": event["event_type"], "stage": event["stage"], "payload": event,
           "generated_at": original["generated_at"]}
    read = core.saved_match_profiles
    report = read([row, row], fx, now_utc=NOW)
    assert len(report["rows"]) == 1
    prop = report["rows"][0]
    assert prop["lines"][0] == {"line": 1.5, "p_over_raw": .68, "p_under_raw": .32}
    assert prop["player_id"] == 101 and not prop["bet_eligible"]
    assert "offered_price_decimal" not in prop
    assert not read([row], {**fx, "season": 2025}, now_utc=NOW)["rows"]
    assert not read([row], fx, now_utc=fx["kickoff"])["rows"]
    assert not read([{**row, "generated_at": NOW + timedelta(minutes=1)}], fx, now_utc=NOW)["rows"]
    original["shots"]["players"][0]["confirmed_starter"] = False
    assert not read([row], fx, now_utc=NOW)["rows"]
