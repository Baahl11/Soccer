from __future__ import annotations
from datetime import datetime, timezone, timedelta
import pytest

from mcp_gateway import soccer_family_research_v1 as sport
from mcp_gateway import soccer_quote_research_v1 as market
from mcp_gateway import soccer_oos_family_validation_v1 as oos
from mcp_gateway import soccer_launch_safety_v1 as safety


FIXTURE={"fixture_id":151,"kickoff_utc":"2026-11-11T18:00:00Z"}
FROZEN="2026-11-11T16:00:00Z"


def _sports_source():
    return {
        "observed_at_utc":"2026-11-11T15:50:00Z",
        "source_updated_at_utc":"2026-11-11T15:40:00Z",
        "goal_rates":{"home_rate":1.7,"away_rate":1.2,"min_sample":10},
        "period_goal_rates":{
            "1H":{"home_rate":0.8,"away_rate":0.5,"min_sample":10},
            "2H":{"home_rate":0.9,"away_rate":0.7,"min_sample":10},
        },
        "corner_rates":{"home_for":5.0,"home_against":3.0,"away_for":4.5,
                        "away_against":5.5,"min_sample":10},
        "card_rates":{"home_for":2.0,"home_against":2.1,
                      "away_for":2.5,"away_against":2.2,"min_sample":10},
        "referee_cards_verified":True,
        "player_shots":{"player_id":991,"shots_per_90":2.7,
                        "expected_minutes":80,"xi_confirmed":True,"sample_matches":14},
    }


def test_all_12_sport_family_models_remain_research_only():
    x=sport.build_family_forecasts(FIXTURE,_sports_source(),frozen_at_utc=FROZEN)
    assert len(x["families"])==12
    assert set(x["families"])==set(sport.MARKETS)
    assert all(v["status"]=="MODELED_LIMITED" for v in x["families"].values())
    assert all(v["bet_eligible"] is False for v in x["families"].values())
    a=x["families"]["1X2"]["selection_probabilities"]
    assert abs(sum(a.values())-1)<1e-6
    assert 0<a["home"]<1
    assert x["families"]["FT_TOTALS"]["selection_probabilities"]["under_2.5"] == pytest.approx(
        1-x["families"]["FT_TOTALS"]["selection_probabilities"]["over_2.5"],abs=1e-7)
    assert x["families"]["TEAM_CORNERS"]["selections_by_team"]["away"]["over_4.5"] > 0
    assert x["families"]["PLAYER_SHOTS"]["player_id"]==991
    assert x["market_queried"] is False


def test_missing_independent_period_corners_cards_and_player_never_imputed():
    src=_sports_source()
    for k in ("period_goal_rates","corner_rates","card_rates","player_shots"):
        src.pop(k)
    x=sport.build_family_forecasts(FIXTURE,src,frozen_at_utc=FROZEN)["families"]
    assert x["1X2"]["status"]=="MODELED_LIMITED"
    for family in ("1H_1X2","1H_TOTALS","2H_1X2","2H_TOTALS",
                   "FT_CORNERS","TEAM_CORNERS","FT_CARDS","PLAYER_SHOTS"):
        assert x[family]["status"]=="NOT_MODELED_DATA_MISSING"


def test_no_imputed_goals_with_incomplete_or_future_source():
    src=_sports_source();src["goal_rates"]["min_sample"]=2
    x=sport.build_family_forecasts(FIXTURE,src,frozen_at_utc=FROZEN)
    assert x["families"]["1X2"]["status"]=="NOT_MODELED_DATA_MISSING"
    src=_sports_source();src["source_updated_at_utc"]="2026-11-11T16:10:00Z"
    x=sport.build_family_forecasts(FIXTURE,src,frozen_at_utc=FROZEN)
    assert all(v["status"]=="NOT_VERIFIED_ASOF" for v in x["families"].values())
    with pytest.raises(ValueError,match="SPORT_PROJECTION_MUST_NOT_SEE_MARKET"):
        sport.build_family_forecasts(FIXTURE,{**_sports_source(),"odds":{"home":1.9}},frozen_at_utc=FROZEN)
    with pytest.raises(ValueError,match="SPORT_FORECAST_NOT_PROSPECTIVE"):
        sport.build_family_forecasts(FIXTURE,_sports_source(),frozen_at_utc=FIXTURE["kickoff_utc"])


def _frozen(family="FT_TOTALS"):
    return {
        "fixture_id":151,"family":family,"frozen_at_utc":FROZEN,
        "kickoff_utc":FIXTURE["kickoff_utc"],
        "selection_probabilities":{
            "over_1.5":0.72,"under_1.5":0.28,"over_2.5":0.54,"under_2.5":0.46,
            "over_3.5":0.31,"under_3.5":0.69,
        },
    }


def _quote():
    return {
        "fixture_id":151,"family":"FT_TOTALS","market_id":"totals:2.5",
        "snapshot_id":"provider-snapshot-1","source_type":"AGGREGATOR_PROXY",
        "captured_at_utc":"2026-11-11T16:05:00Z",
        "source_updated_at_utc":"2026-11-11T16:04:00Z",
        "bookmaker":"Research Source A","line":2.5,"selection":"over_2.5",
        "outcomes":{"over_2.5":1.94,"under_2.5":1.96},
    }


def test_market_exact_line_devig_shrunk_and_ev_still_watch():
    x=market.analyze_frozen_sport_and_quote(
        _frozen(),_quote(),evaluated_at_utc="2026-11-11T16:07:00Z")
    assert x["classification"]=="WATCH"
    assert x["stake_units"]==0
    assert x["direct_executable_quote_verified"] is False
    assert x["true_clv_verified"] is False
    assert x["bet_eligible"] is False
    assert x["line"]==2.5
    assert x["p_raw"]==pytest.approx(0.54)
    assert x["p_breakeven"]==pytest.approx(1/1.94,abs=1e-7)
    assert x["p_market_fair"]+0.0>0
    assert isinstance(x["estimated_ev_research"],float)
    assert x["quote_is_provider_proxy"] is True


def test_market_price_rejects_misaligned_or_partial_instruments():
    bad=_quote();bad["line"]=3.5
    assert market.analyze_frozen_sport_and_quote(
        _frozen(),bad,evaluated_at_utc="2026-11-11T16:07:00Z")["reason"]=="COMPLETE_SAME_LINE_TWO_WAY_OR_THREE_WAY_REQUIRED"
    bad=_quote();bad["outcomes"]={"over_2.5":1.94}
    assert market.analyze_frozen_sport_and_quote(
        _frozen(),bad,evaluated_at_utc="2026-11-11T16:07:00Z")["classification"]=="WATCH"
    bad=_quote();bad["captured_at_utc"]="2026-11-11T15:00:00Z"
    assert market.analyze_frozen_sport_and_quote(
        _frozen(),bad,evaluated_at_utc="2026-11-11T16:07:00Z")["reason"]=="QUOTE_SOURCE_ASOF_NOT_VERIFIED"
    bad=_quote();bad["source_type"]="DIRECT_BOOKMAKER_RESEARCH"
    assert market.analyze_frozen_sport_and_quote(
        _frozen(),bad,evaluated_at_utc="2026-11-11T16:07:00Z")["classification"]=="WATCH"
    bad=_quote();bad["outcomes"]={"over_2.5":1.94,"under_2.5":1.96}
    assert market.analyze_frozen_sport_and_quote(
        {**_frozen(),"family":"TEAM_TOTALS","entity_id":123},
        {**bad,"family":"TEAM_TOTALS"},evaluated_at_utc="2026-11-11T16:07:00Z")["reason"]=="EXACT_TEAM_OR_PLAYER_ID_REQUIRED"


def test_close_requires_matching_direct_attested_quotes():
    a=_quote();b={**a,"price":1.86}
    assert market.compare_close_same_instrument(a,b)["true_clv"] is None
    b["line"]=3.5
    assert market.compare_close_same_instrument(a,b)["reason"]=="CLOSING_QUOTE_INSTRUMENT_MISMATCH"


def _graded_input(family="1X2",selection="home",line=None):
    return {
        "fixture_id":151,"family":family,"selection":selection,
        "p_raw":0.61,"line":line,"entity":None,
        "model_version":"independent-goal-baseline-v1",
        "frozen_at_utc":"2026-11-11T16:00:00Z",
        "kickoff_utc":"2026-11-11T18:00:00Z",
        "official_settlement":{
            "status":"FT","source_type":"OFFICIAL_LEAGUE",
            "independent_auditor_verified":True,
            "official_source_url":"https://official.example/fixture/151",
            "observed_at_utc":"2026-11-11T21:00:00Z",
            "ft_goals":{"home":2,"away":1},
        },
    }


def test_official_ft_family_oos_grades_only_prospective_evidence():
    r=oos.grade_frozen_record(_graded_input())
    assert r["status"]=="OFFICIAL_FT_GRADED_RESEARCH_ONLY"
    assert r["outcome"]==1
    assert r["brier"]==pytest.approx((.61-1)**2)
    row=_graded_input("FT_TOTALS","over_2.5",2.5)
    assert oos.grade_frozen_record(row)["outcome"]==1
    row=_graded_input("BTTS","yes")
    assert oos.grade_frozen_record(row)["outcome"]==1
    row=_graded_input("FT_CORNERS","over_9.5",9.5)
    assert oos.grade_frozen_record(row)["status"]=="NOT_GRADED"
    invalid=_graded_input();invalid["frozen_at_utc"]="2026-11-11T19:00:00Z"
    assert oos.grade_frozen_record(invalid)["status"]=="NOT_GRADED"
    invalid=_graded_input();invalid["official_settlement"]["independent_auditor_verified"]=False
    assert oos.grade_frozen_record(invalid)["status"]=="NOT_GRADED"


def test_oos_scorecard_does_not_transfer_other_market_families_or_promote():
    score=oos.family_oos_scorecards([_graded_input(),_graded_input()])
    assert score["families"]["1X2"]["official_ft_unique_fixtures"]==1
    assert score["families"]["FT_CORNERS"]["official_ft_unique_fixtures"]==0
    assert score["families"]["1X2"]["research_sample_gate"]=="OOS_SAMPLE_INSUFFICIENT"
    assert score["not_graded_reasons"]["DUPLICATE_FROZEN_INSTRUMENT"]==1
    assert score["commercial_promotion_allowed"] is False


def test_launch_denies_public_checkout_with_only_one_flag():
    r=safety.audit_launch_gates({"SOCCER_PUBLIC_BILLING_ENABLED":"true"})
    assert r["billing_enabled"] is False
    assert len(r["missing_gates"])==len(safety.REQUIRED)
    assert r["legal_certification_from_code"] is False
    partly={k:"true" for k in safety.REQUIRED}
    partly["SOCCER_PUBLIC_BILLING_ENABLED"]="true"
    partly["SOCCER_DATA_REDISPLAY_LICENSES_APPROVED"]="false"
    r=safety.audit_launch_gates(partly)
    assert r["billing_enabled"] is False
    assert "SOCCER_DATA_REDISPLAY_LICENSES_APPROVED" in r["missing_gates"]


def test_opt_in_alerts_are_research_only_and_require_rights():
    no=safety.allow_research_alert(
        user_opted_in=False,family_maturity="RESEARCH",identity_verified=True,
        cached_snapshot_fresh=True,notification_deduped=True,
        display_license_approved=False)
    assert no["status"]=="ALERT_BLOCKED"
    assert "USER_OPT_IN_NOT_VERIFIED" in no["reasons"]
    yes=safety.allow_research_alert(
        user_opted_in=True,family_maturity="SHADOW",identity_verified=True,
        cached_snapshot_fresh=True,notification_deduped=True,
        display_license_approved=True)
    assert yes["status"]=="RESEARCH_NOTICE_ALLOWED"
    assert yes["notification_sent"] is False
    assert yes["bet_signal_allowed"] is False


def test_existing_goal_rate_market_cannot_bet_on_provider_proxy_even_with_edge():
    """Regression: old Tier-B path accepted non-executable provider odds."""
    from mcp_gateway import soccer_model

    raw = {
        "status":"MODELED_LIMITED",
        "sample":{"minimum_split_sample":12},
        "raw_home_win_prob":0.555,
        "raw_draw_prob":0.25,
        "raw_away_win_prob":0.195,
        "raw_total_goals":2.70,
        "raw_btts_yes_prob":0.55,
        "_total_dist":{2:0.35,3:0.40,4:0.25},
    }
    proxy = {"markets":[{
        "market":"Match Winner","market_id":1,
        "bookmaker":"Research Provider","bookmaker_id":12,
        "provider_update":"2026-11-11T15:55:00Z",
        "values":[
            {"selection":"Home","price":2.2},
            {"selection":"Draw","price":3.4},
            {"selection":"Away","price":3.5},
        ],
    }]}
    outcome = soccer_model.evaluate_market(
        raw, proxy, {"data_tier":"A"}, 0.94, "T-90", {})
    assert outcome["bet_eligible"] is False
    assert outcome["market_verification_status"]=="DIRECT_EXECUTABLE_QUOTE_NOT_VERIFIED"
    assert all(x["classification"] not in {"BET","LEAN"} for x in outcome["decisions"])
    assert all(x["stake_units"]==0.0 for x in outcome["decisions"])
    assert any(x["tier"]=="B" for x in outcome["decisions"])
    assert all(x["commercial_bet_eligible"] is False for x in outcome["decisions"])


def test_public_stripe_checkout_remains_off_when_operator_flag_set_alone(monkeypatch):
    from mcp_gateway import subscriber_billing_v2

    monkeypatch.setenv("SOCCER_PUBLIC_BILLING_ENABLED","true")
    for key in safety.REQUIRED:
        monkeypatch.delenv(key,raising=False)
    result = subscriber_billing_v2.public_billing_contract()
    assert result["public_launch_enabled"] is False
    assert result["checkout_enabled"] is False
    assert result["launch_safety_status"]=="COMMERCIAL_LAUNCH_BLOCKED"
    assert "SOCCER_PAYMENT_SANDBOX_APPROVED" in result["launch_missing_gate_codes"]
    assert result["launch_operator_evidence_review_required"] is True


def test_binary_quote_rejects_wrong_period_and_disconnected_player():
    frozen={**_frozen("1H_TOTALS"),"period":"1H"}
    quarter={**_quote(),"family":"1H_TOTALS","period":"2H"}
    assert market.analyze_frozen_sport_and_quote(
        frozen,quarter,evaluated_at_utc="2026-11-11T16:07:00Z")["reason"]=="PERIOD_MISMATCH"
    frozen={**_frozen("PLAYER_SHOTS"),"entity_id":999}
    q={**_quote(),"family":"PLAYER_SHOTS","entity_id":1000}
    assert market.analyze_frozen_sport_and_quote(
        frozen,q,evaluated_at_utc="2026-11-11T16:07:00Z")["reason"]=="PLAYER_OR_TEAM_INSTRUMENT_MISMATCH"
