"""Offline SGP capture safety: max two calls, Bet365 ID8, research only."""
from __future__ import annotations

from mcp_gateway import sgp_research_capture_v1 as cap


def test_sgp_request_limit_and_market_whitelist():
    assert cap.MAX_FIXTURES_PER_RUN == 2
    assert cap.BOOKMAKER_ID == 8
    assert {49, 92, 240, 241, 269}.issubset(cap.MARKET_IDS)
    assert "soccer_model_runs" in cap.SPORT_SHORTLIST_SQL
    assert "r.raw_projection IS NOT NULL" in cap.SPORT_SHORTLIST_SQL
    assert "f.status='NS'" in cap.SPORT_SHORTLIST_SQL


def test_sgp_no_api_key_makes_zero_provider_calls(monkeypatch):
    monkeypatch.delenv("API_FOOTBALL_KEY",raising=False)
    monkeypatch.setattr(cap,"_shortlist",lambda: (_ for _ in ()).throw(
        AssertionError("sport shortlist not needed when provider unavailable")))
    r=cap._capture()
    assert r["status"]=="SOURCE_API_KEY_NOT_VERIFIED"
    assert r["api_requests"]==0


def test_sgp_missing_sporting_shortlist_makes_zero_provider_calls(monkeypatch):
    monkeypatch.setenv("API_FOOTBALL_KEY","INTERNAL_FAKE_VALUE")
    monkeypatch.setattr(cap,"_shortlist",lambda:[])
    r=cap._capture()
    assert r["status"]=="NO_PRIOR_SPORTING_SHORTLIST"
    assert r["api_requests"]==0


def test_sgp_capture_uses_at_most_two_sport_first_fixtures_and_no_bet(monkeypatch):
    monkeypatch.setenv("API_FOOTBALL_KEY","INTERNAL_FAKE_VALUE")
    monkeypatch.setattr(cap,"_shortlist",lambda:[101,202,303])
    calls=[]
    queries=[]

    class Response:
        status_code=200
        def raise_for_status(self): pass
        def json(self): return {"response":[]}

    class Provider:
        def __init__(self,**kwargs): assert kwargs["follow_redirects"] is False
        def __enter__(self): return self
        def __exit__(self,*a): pass
        def get(self,url,*,params,headers):
            assert params["bookmaker"]==8 and params["page"]==1
            assert headers["x-apisports-key"]=="INTERNAL_FAKE_VALUE"
            calls.append(params["fixture"])
            return Response()

    class Cursor:
        def __enter__(self): return self
        def __exit__(self,*a): pass
        def execute(self,q,params=None): queries.append((q,params))

    class Conn:
        def __enter__(self): return self
        def __exit__(self,*a): pass
        def transaction(self): return self
        def cursor(self): return Cursor()

    monkeypatch.setattr(cap.httpx,"Client",Provider)
    monkeypatch.setattr(cap.persistence,"_connect",lambda:Conn())
    monkeypatch.setattr(cap.price_resolver_v4,"normalize_api_response",lambda _:[])
    r=cap._capture()
    assert calls==[101,202]
    assert r["api_requests"]==2
    assert r["snapshots_written"]==0
    assert r["bet_eligible"] is False
    assert r["sgp_betslip_price_verified"] is False
    assert r["provider_requests_cap"]==2


def test_sgp_capture_never_persists_other_bookmaker_or_unapproved_markets(monkeypatch):
    monkeypatch.setenv("API_FOOTBALL_KEY","INTERNAL_FAKE_VALUE")
    monkeypatch.setattr(cap,"_shortlist",lambda:[123])
    calls=[]

    class Response:
        status_code=200
        def raise_for_status(self): pass
        def json(self): return {"response":[]}

    class Provider:
        def __init__(self,**kwargs): pass
        def __enter__(self): return self
        def __exit__(self,*a): pass
        def get(self,url,*,params,headers): return Response()

    class Cursor:
        def __enter__(self): return self
        def __exit__(self,*a): pass
        def execute(self,q,params=None):
            assert "INSERT INTO soccer_market_snapshots" not in q

    class Conn:
        def __enter__(self): return self
        def __exit__(self,*a): pass
        def transaction(self): return self
        def cursor(self): return Cursor()

    monkeypatch.setattr(cap.httpx,"Client",Provider)
    monkeypatch.setattr(cap.persistence,"_connect",lambda:Conn())
    monkeypatch.setattr(cap.price_resolver_v4,"normalize_api_response",lambda _:[
       {"fixture_id":123,"bookmaker_id":123,"market_id":49},
       {"fixture_id":123,"bookmaker_id":8,"market_id":999},
       {"fixture_id":777,"bookmaker_id":8,"market_id":49},
    ])
    r=cap._capture()
    assert r["api_requests"]==1
    assert r["snapshots_written"]==0
