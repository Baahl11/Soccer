from __future__ import annotations
import asyncio
from datetime import datetime, timedelta, timezone
import pytest
from mcp_gateway import soccer_p0_sport_preview_v1 as p0
from mcp_gateway import soccer_p0_persisted_view_v1 as storage
from mcp_gateway import subscriber_contract_v2 as contract
from mcp_gateway import subscriber_frontend_v2 as frontend
from mcp_gateway import automation_v5 as v5

NOW=datetime(2026,11,11,12,15,tzinfo=timezone.utc)
FX={
    "fixture_id":99131,
    "kickoff":(NOW+timedelta(hours=4)).isoformat(),
    "status":"NS",
    "league_id":44,"season":2026,"home_team_id":111,"away_team_id":222,
    "home_team":"Real Local","away_team":"Real Visita",
    "league":"Liga de prueba","country":"México",
}


def fake_raw():
    return {
        "status":"MODELED_LIMITED",
        "model_version":"SOCCER EDGE ENGINE v1.0",
        "sample":{"minimum_split_sample":8},
        "raw_home_goal_rate":1.60,"raw_away_goal_rate":1.10,
        "raw_home_win_prob":0.48,"raw_draw_prob":0.28,"raw_away_win_prob":0.24,
        "raw_btts_yes_prob":0.53,
        "raw_over_1_5_prob":0.75,"raw_over_2_5_prob":0.55,"raw_over_3_5_prob":0.31,
    }


def test_p0_fixture_verifies_kickoff_and_never_models_postkickoff():
    assert p0.eligible(FX,NOW)
    assert not p0.eligible({**FX,"status":"FT"},NOW)
    assert not p0.eligible({**FX,"kickoff":NOW.isoformat()},NOW)
    assert not p0.eligible({**FX,"home_team_id":0},NOW)
    assert not p0.eligible({**FX,"away_team_id":111},NOW)


def test_p0_stores_real_sport_only_asof_and_no_market():
    sporting={"home_stats":{"form":"WLW"},"away_stats":{"form":"LDW"}}
    event=p0.preview_event(FX,NOW,sporting,fake_raw(),
        {"shortlisted":True,"tracks":["SIDE"]})
    assert event["event_type"]=="SOCCER_REFRESH"
    assert event["stage"]=="SPORT_PREVIEW"
    assert event["market"]=="NOT_REQUESTED_SPORT_FIRST"
    assert event["raw_sport_show_as_forecast"] is True
    assert event["quote_requested"] is False
    assert event["bet_eligible"] is False
    assert event["stake_units"]==0.0
    research=p0.asof_research_projection(event,now=NOW+timedelta(minutes=1))
    assert research["raw_probabilities"]["raw_home_win_prob"]==0.48
    assert research["min_split_sample"]==8
    assert research["source_updated_at"]=="NOT_VERIFIED"
    assert research["commercial_display_allowed"] is False


def test_p0_cannot_replay_late_or_missing_or_too_small_samples():
    original=p0.preview_event(FX,NOW,{},fake_raw(),{"shortlisted":False})
    assert p0.asof_research_projection(original,now=NOW+timedelta(hours=5)) is None
    forged={**original,"source_observed_at_utc":(NOW+timedelta(hours=4)).isoformat()}
    assert p0.asof_research_projection(forged,now=NOW+timedelta(hours=5)) is None
    altered={**original,"bet_eligible":True}
    assert p0.asof_research_projection(altered,now=NOW+timedelta(minutes=1)) is None
    low=fake_raw();low["sample"]["minimum_split_sample"]=2
    unqualified=p0.preview_event(FX,NOW,{},low,{"shortlisted":False})
    assert unqualified["raw_sport_show_as_forecast"] is False
    assert p0.asof_research_projection(unqualified,now=NOW+timedelta(minutes=1)) is None
    wrong=fake_raw();wrong["raw_away_win_prob"]=.85
    forged=p0.preview_event(FX,NOW,{},wrong,{"shortlisted":True})
    assert p0.asof_research_projection(forged,now=NOW+timedelta(minutes=1)) is None


def test_p0_budget_daily_cache_and_no_market_calls(monkeypatch):
    cache={}
    provider=[]
    async def team_stats(fx,now):
        provider.append(fx["fixture_id"])
        v5.v2._API_CALLS_THIS_TICK+=2
        return {"home_stats":{},"away_stats":{},"screen_layer":"CHEAP_SEASON_SPLIT"}
    monkeypatch.setattr(v5,"_cheap_sport_bundle",team_stats)
    monkeypatch.setattr(v5,"build_raw_projection",lambda fx,sporting,availability:fake_raw())
    monkeypatch.setattr(v5,"_screen_shortlist",lambda raw:{"shortlisted":True})
    monkeypatch.setattr(v5.v4,"_effective_tick_cap",lambda:35)
    monkeypatch.setattr(v5.base,"_cache_get",lambda ns,k,ttl,now:cache.get((ns,k)))
    monkeypatch.setattr(v5.base,"_cache_set",lambda ns,k,value,now:cache.__setitem__((ns,k),value))
    monkeypatch.setattr(v5,"_shortlist_set",lambda fid,stage,now:None)
    monkeypatch.setattr(v5.v2,"_API_CALLS_THIS_TICK",1)
    monkeypatch.setattr(v5.v2,"_LAST_DAILY_REMAINING",95)
    fixtures=[{**FX,"fixture_id":99131+i,"home_team_id":111+i*2,
            "away_team_id":112+i*2} for i in range(30)]
    first=asyncio.run(v5._daily_p0_sport_previews(fixtures,NOW,"2026-11-11"))
    assert first["attempted"]<=12
    assert first["api_calls_used"]<=24
    assert first["modeled_limited"]==first["attempted"]
    assert all(e["bet_eligible"] is False and e["quote_requested"] is False for e in first["events"])
    calls_before=len(provider)
    second=asyncio.run(v5._daily_p0_sport_previews(fixtures,NOW+timedelta(minutes=10),"2026-11-11"))
    assert second["cached_from_prior_tick"]>=first["attempted"]
    assert all(e["stage"]=="SPORT_PREVIEW" for e in second["events"])
    assert len(provider)>=calls_before


def test_source_owner_scope_no_public_model_research_leak():
    fixture={
        "fixture_id":99131,"home_team":"Real Local","away_team":"Real Visita",
        "kickoff":FX["kickoff"],"league":"Liga de prueba",
    }
    e=p0.preview_event(FX,NOW,{},fake_raw(),{"shortlisted":False})
    row=p0.asof_research_projection(e,now=NOW+timedelta(minutes=1))
    research={"status":"PREMATCH_LIMITED_RESEARCH","rows":[row],
              "attempted_count":1}
    owner={"access":{"owner":True,"admin":False},
        "slate":{"rows":[{"fixture":fixture}]},"radar":{}}
    with_owner=contract._attach_owner_p0_sport_previews(owner,research)
    assert with_owner["sport_previews"]["forecast_count"]==1
    assert with_owner["sport_previews"]["rows"][0]["raw_probabilities"]["raw_home_win_prob"]==0.48
    assert with_owner["sport_previews"]["bet_eligible"] is False
    free={"access":{"owner":False,"admin":False},
        "slate":{"rows":[{"fixture":fixture}]},"radar":{}}
    assert "sport_previews" not in contract._attach_owner_p0_sport_previews(free,research)
    unavailable=contract._attach_owner_p0_sport_previews(
        {"access":{"owner":True},"slate":{"rows":[]}},None)
    assert unavailable["sport_previews"]["forecast_count"] is None


def test_frontend_leads_with_sport_and_does_not_show_market_labs():
    html=frontend.render()
    assert '<html lang="es">' in html
    assert 'id="sportFocus"' in html
    assert "id=\"focusForecasts\"" in html
    assert "id=\"focusSlate\"" in html
    assert '#today > :not(.page-head):not(.kpis):not(#sportFocus)' in html
    assert "function p0Card(x)" in html
    assert "function sportForecastEntries()" in html
    assert "renderSportFocus();void enrichIdentities().then(" in html
    assert "renderSportFocus();void loadResearchDesk()" not in html
    assert "BET" in html
    assert "source_updated_at" not in html or "source_updated_at" in html


def test_no_live_db_contact_when_fixture_ids_empty(monkeypatch):
    monkeypatch.setattr(storage.persistence,"_connect",
        lambda:(_ for _ in ()).throw(AssertionError("Unexpected DB read")))
    assert storage.read_previews([])["status"]=="NO_SLATE_FIXTURES"


def test_active_v7_scheduler_not_orphaning_p0_screen():
    import inspect
    from mcp_gateway import automation_v7 as v7

    source = inspect.getsource(v7.run_tick)
    assert "await v5._daily_p0_sport_previews(" in source
    assert source.index("await v5._daily_p0_sport_previews(") < source.index("due: list[dict")
    assert "events.extend(p0_sport[\"events\"])" in source
    assert 'p0_sport_preview_attempted_count' in source
    assert 'p0_sport_preview_modeled_limited_count' in source
    assert 'p0_sport_preview_deferred_quota_count' in source
    assert 'p0_sport_preview_api_calls' in source
    assert v7.P0_SPORT_MAX_FIXTURES_PER_TICK == 6
    assert v7.P0_SPORT_MIN_REMAINING_FOR_CANONICAL == 8


def test_live_p0_respects_exact_upstream_call_reserve_and_no_odds(monkeypatch):
    cache = {}
    provider = []
    async def sport(fx, now):
        provider.append(fx["fixture_id"])
        v5.v2._API_CALLS_THIS_TICK += 2
        return {"home_stats": {}, "away_stats": {}, "screen_layer": "CHEAP_SEASON_SPLIT"}
    monkeypatch.setattr(v5, "_cheap_sport_bundle", sport)
    monkeypatch.setattr(v5, "build_raw_projection", lambda fx, sport, availability: fake_raw())
    monkeypatch.setattr(v5, "_screen_shortlist", lambda raw: {"shortlisted": True})
    monkeypatch.setattr(v5.v4, "_effective_tick_cap", lambda: 8)
    monkeypatch.setattr(v5.base, "_cache_get", lambda ns, k, ttl, now: cache.get((ns, k)))
    monkeypatch.setattr(v5.base, "_cache_set", lambda ns, k, value, now: cache.__setitem__((ns, k), value))
    monkeypatch.setattr(v5, "_shortlist_set", lambda fid, stage, now: None)
    monkeypatch.setattr(v5.v2, "_API_CALLS_THIS_TICK", 1)
    monkeypatch.setattr(v5.v2, "_LAST_DAILY_REMAINING", 7000)
    fixtures = [
        {**FX, "fixture_id": 99131 + i, "home_team_id": 111 + i * 2,
         "away_team_id": 112 + i * 2}
        for i in range(15)
    ]
    result = asyncio.run(v5._daily_p0_sport_previews(
        fixtures, NOW, "2026-11-11", max_fixtures=2, reserve_calls=3,
    ))
    assert result["attempted"] == 2
    assert result["modeled_limited"] == 2
    assert result["api_calls_used"] == 4
    assert v5.v2._API_CALLS_THIS_TICK == 5
    assert 8 - v5.v2._API_CALLS_THIS_TICK >= 3
    assert all(event["stage"] == "SPORT_PREVIEW"
               and event["quote_requested"] is False
               and event["bet_eligible"] is False
               for event in result["events"])
    monkeypatch.setattr(v5.v2, "_API_CALLS_THIS_TICK", 6)
    denied = asyncio.run(v5._daily_p0_sport_previews(
        fixtures, NOW, "2026-11-11", max_fixtures=2, reserve_calls=3,
    ))
    assert denied["attempted"] == 0
    assert denied["deferred_due_to_quota"] >= 1


def test_research_p0_never_overwrites_canonical_lifecycle_shortlist():
    import inspect
    assert "_shortlist_set" not in inspect.getsource(v5._daily_p0_sport_previews)
