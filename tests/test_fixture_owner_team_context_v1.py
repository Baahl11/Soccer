from mcp_gateway import fixture_owner_team_context_v1 as ctx


def season(team, league=667, season=2026, total=9, home=6, away=3):
    return {
        "response": {
            "team": {"id": team, "name":"BK Hacken" if team==367 else "Fredrikstad"},
            "league": {"id":league,"name":"Friendlies Clubs","season":season},
            "fixtures": {
                "played":{"home":home,"away":away,"total":total},
                "wins":{"total":5},
                "draws":{"total":2},
                "loses":{"total":total-7},
            },
            "goals":{
                "for":{"total":{"total":15,"home":11,"away":4},
                       "average":{"home":"1.83","away":"1.33"}},
                "against":{"total":{"total":11,"home":7,"away":4},
                           "average":{"home":"1.17","away":"1.33"}},
            },
        },
        "quota":{"daily_remaining":7300},
    }


def clear(monkeypatch):
    monkeypatch.setattr(ctx, "_cache", {})
    monkeypatch.setattr(ctx, "_hour_begin", 0.0)
    monkeypatch.setattr(ctx, "_hour_count", 0)
    monkeypatch.setattr(ctx, "_day_begin", 0.0)
    monkeypatch.setattr(ctx, "_day_count", 0)


def test_validated_source_stats_do_not_create_an_actionable_projection(monkeypatch):
    clear(monkeypatch)
    calls=[]
    def fetch(team, league, season_year, _api_key):
        calls.append((team,league,season_year))
        return season(team)
    v=ctx.read_for_owner(1646118,667,2026,367,2149,
                         fetcher=fetch,clock=lambda:12000)
    assert v["status"]=="SPORTING_SEASON_CONTEXT_ONLY"
    assert len(v["teams"])==2
    assert v["teams"][0]["played"]==9
    assert v["teams"][0]["goals_for"]==15
    assert v["teams"][0]["raw_xg"] is None
    assert v["teams"][1]["venue"]=="away"
    assert v["projection"] is None
    assert v["bet_eligible"] is False
    assert v["commercial_display_allowed"] is False
    assert v["source_redisplay_rights_verified"] is False
    assert calls==[(367,667,2026),(2149,667,2026)]
    saved=ctx.read_for_owner(1646118,667,2026,367,2149,
                             fetcher=lambda *_: (_ for _ in ()).throw(AssertionError()),
                             clock=lambda:12030)
    assert saved["from_cache"] is True
    assert saved["provider_requests_added"]==0


def test_unknown_or_mismatched_identity_never_shows_source_data(monkeypatch):
    clear(monkeypatch)
    assert ctx._parse(season(367,league=103),367,667,2026,"home") is None
    assert ctx._parse(season(367,total=0,home=0,away=0),367,667,2026,"home") is None
    assert ctx.read_for_owner(1646118,667,2026,367,367,fetcher=lambda *_: {},clock=lambda:12000)["status"]=="NOT_VERIFIED"


def test_strict_owner_research_budget(monkeypatch):
    clear(monkeypatch)
    monkeypatch.setattr(ctx, "_hour_begin", 9000.)
    monkeypatch.setattr(ctx, "_hour_count", ctx.HOURLY_FIXTURE_CAP)
    monkeypatch.setattr(ctx, "_day_begin", 9000.)
    monkeypatch.setattr(ctx, "_day_count", ctx.DAILY_FIXTURE_CAP)
    v=ctx.read_for_owner(1646118,667,2026,367,2149,
        fetcher=lambda *_: (_ for _ in ()).throw(AssertionError("must not call")), clock=lambda:9001.)
    assert v["provider_requests_added"]==0
    assert v["status"]=="NOT_VERIFIED"
