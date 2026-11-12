from datetime import datetime, timedelta, timezone
from mcp_gateway import fixture_owner_recent_context_v1 as recent

NOW=datetime(2026,11,12,17,50,tzinfo=timezone.utc)
KICKOFF=(NOW+timedelta(hours=2)).isoformat()


def source_fixture(fid,team,opponent,date,goals_a,goals_b,*,status="FT",
                   club_home=True,competition="Other league",season=2026):
    home,away=({"id":team,"name":"Team"},{"id":opponent,"name":"Opponent"})
    if not club_home:
        home,away=away,home
    return {
        "fixture":{"id":fid,"date":date,"status":{"short":status}},
        "teams":{"home":home,"away":away},
        "goals":{"home":goals_a,"away":goals_b},
        "league":{"id":99,"name":competition,"season":season},
    }


def clear(monkeypatch):
    monkeypatch.setattr(recent,"_cache",{})
    monkeypatch.setattr(recent,"_hour_begin",0.)
    monkeypatch.setattr(recent,"_hour_count",0)
    monkeypatch.setattr(recent,"_day_begin",0.)
    monkeypatch.setattr(recent,"_day_count",0)


def test_prior_results_verified_and_cross_competition_sample_is_not_model(monkeypatch):
    clear(monkeypatch)
    past=(NOW-timedelta(days=4)).isoformat()
    before=(NOW-timedelta(days=9)).isoformat()
    calls=[]
    def fetch(tid,key):
        calls.append(tid)
        assert key==""
        return {"quota":{"daily_remaining":7000},"response":[
            source_fixture(100+tid,tid,900,past,2,1,competition="National Cup"),
            source_fixture(200+tid,tid,901,before,2,0,
                           club_home=False,competition="League Championship"),
            source_fixture(300+tid,tid,902,
                           (NOW+timedelta(days=1)).isoformat(),3,0,
                           status="NS"),
        ]}
    value=recent.read_for_owner(
        1545869,1508,1499,"Congo DR","Sierra Leone",KICKOFF,
        fetcher=fetch,clock=lambda:12000,utcnow=lambda:NOW,
    )
    assert set(calls)=={1508,1499}
    assert value["status"]=="VERIFIED_RECENT_FORM"
    assert value["provider_requests_added"]==2
    assert value["source"]==recent.SOURCE
    assert value["model_probabilities"] is None
    assert value["bet_eligible"] is False
    assert value["commercial_display_allowed"] is False
    assert value["market_queried"] is False
    assert value["teams"][0]["sample"]==2
    assert value["teams"][0]["multi_competition"] is True
    assert value["teams"][0]["wins"]==1
    assert value["teams"][0]["losses"]==1
    assert value["teams"][0]["goals_for"]==2  # first 2-1 home, second 0-2 away
    assert all(x["kickoff_utc"]<KICKOFF for t in value["teams"]
               for x in t["recent_fixtures"])
    assert any("NO son equivalentes" in x
               for x in value["facts"])
    again=recent.read_for_owner(
        1545869,1508,1499,"Congo DR","Sierra Leone",KICKOFF,
        fetcher=lambda *_: (_ for _ in ()).throw(AssertionError("cache miss")),
        clock=lambda:12100,utcnow=lambda:NOW,
    )
    assert again["provider_requests_added"]==0
    assert again["from_cache"] is True


def test_future_games_and_wrong_team_and_invalid_scores_are_excluded():
    dates=[
        source_fixture(1,1508,88,(NOW-timedelta(days=4)).isoformat(),1,0),
        source_fixture(2,1508,88,(NOW+timedelta(days=8)).isoformat(),2,1),
        source_fixture(3,1508,88,(NOW-timedelta(days=5)).isoformat(),3,2,status="CANC"),
        source_fixture(4,44,55,(NOW-timedelta(days=6)).isoformat(),0,1),
        source_fixture(5,1508,88,(NOW-timedelta(days=9)).isoformat(),None,1),
        source_fixture(1,1508,88,(NOW-timedelta(days=4)).isoformat(),1,0),
    ]
    rows=recent._parse({"response":dates},1508,NOW)
    assert [x["fixture_id"] for x in rows]==[1]


def test_daily_and_hourly_call_budget_blocks_extra_research(monkeypatch):
    clear(monkeypatch)
    monkeypatch.setattr(recent,"_hour_begin",12000.)
    monkeypatch.setattr(recent,"_hour_count",recent.HOURLY_FIXTURE_CAP)
    got=recent.read_for_owner(1545869,1508,1499,"Congo","Sierra",
        KICKOFF,fetcher=lambda *_: (_ for _ in ()).throw(AssertionError("called")),
        clock=lambda:12005.,utcnow=lambda:NOW)
    assert got["status"]=="NOT_VERIFIED"
    assert got["reason"]=="PRIVATE_RECENT_HISTORY_BUDGET_EXHAUSTED"
    assert got["provider_requests_added"]==0


def test_cross_competition_never_reused_for_raw_sport_without_validation():
    from mcp_gateway import soccer_verified_sport_read_v1 as sporting
    # Recent source is intentionally distinct from season context, and cannot
    # pass the identity + league-season validation for the RAW model.
    result=sporting.read_owner_context({
        "status":"VERIFIED_RECENT_FORM","source":recent.SOURCE,
        "owner_admin_only":True,"commercial_display_allowed":False,
        "teams":[],
    },{"fixture_id":1545869,"home_team_id":1508,"away_team_id":1499})
    assert result["status"]=="NO_VERIFIED_SEASON_CONTEXT"
    assert result["model_probabilities"] is None


def test_ui_renders_extra_data_while_preserving_private_boundary():
    from mcp_gateway.subscriber_frontend_v2 import render
    html=render()
    assert "function recentVerifiedForm(packet)" in html
    assert "private_recent_form" in html
    assert "ownerRecent+ownerSportRead+ownerSeasonResearch" in html
    assert "Proyección 1X2: no calculada" in html
    assert "DATOS NUEVOS · ÚLTIMOS PARTIDOS OFICIALES" in html
