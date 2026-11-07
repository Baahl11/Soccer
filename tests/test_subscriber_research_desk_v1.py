from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace

from starlette.requests import Request
from mcp_gateway import subscriber_research_desk_v1 as desk


def sample(price="2.0", classification="WATCH"):
    stamp = datetime(2026, 11, 7, 1, 0, tzinfo=timezone.utc)
    return {
        "fixture_id": 123456, "home_team": "Local", "away_team": "Visitante",
        "league_name": "Liga prueba", "kickoff_utc": stamp,
        "market_family": "1X2", "selection": "Home", "line": None,
        "bookmaker_id": 8, "bookmaker_name": "Bet365",
        "price_decimal": price, "provider_update_at": stamp,
        "captured_at": stamp, "p_raw": .54, "p_shrunk": .5,
        "p_market_fair": .48, "p_breakeven": .5,
        "hypothetical_ev_pct": 0., "edge_vs_breakeven_pp": 0.,
        "value_diagnostic": "NEGATIVE_OR_ZERO_HYPOTHETICAL_EV",
        "classification": classification, "reasons": ["MODEL_NOT_VALIDATED"],
        "independent_model_validated": False, "lineup_verified": False,
        "availability_confidence": .5,
    }


def test_odds_minus_110_or_better_has_no_upper_price_cap():
    x = desk.make_board([sample("1.90"), sample("1.9090909091"), sample("2.50")])
    assert [r["price_bucket"] for r in x["rows"]] == [
        "EDGE_REGULAR", "EDGE_MINUS110_OR_BETTER", "EDGE_MINUS110_OR_BETTER"
    ]
    assert x["bets"] == 0
    assert x["notification_delivery_enabled"] is False


def test_research_never_promotes_source_bet_and_exposes_probability_distinctions():
    x = desk.make_board([sample("2.0", classification="BET")])["rows"][0]
    assert x["classification"] == "WATCH"
    assert x["source_classification"] == "BET"
    assert x["bet_eligible"] is False and x["actionable"] is False
    assert x["raw_sport_probability"] == .54
    assert x["market_shrunk_probability"] == .5
    assert x["fair_market_probability"] == .48
    assert x["breakeven_probability"] == .5
    assert x["hypothetical_ev_pct"] == 0.


def test_query_guards_price_freshness_and_unstarted_fixture():
    assert "provider_update_at >= CURRENT_TIMESTAMP - INTERVAL '90 minutes'" in desk.SQL
    assert "captured_at >= CURRENT_TIMESTAMP - INTERVAL '30 minutes'" in desk.SQL
    assert "kickoff_utc > CURRENT_TIMESTAMP" in desk.SQL
    assert "fixture_status = 'NS'" in desk.SQL
    assert "bookmaker_id = 8" in desk.SQL


def test_anonymous_research_forbidden_before_database_access(monkeypatch):
    async def anonymous(_):
        return {"authenticated": False}, None
    monkeypatch.setattr(desk.subscriber_contract_v2, "_resolve_entitlement", anonymous)
    def prohibited():
        raise AssertionError("DB must not be contacted")
    monkeypatch.setattr(desk, "_query_research_feed", prohibited)
    req = Request({"type": "http", "method": "GET", "path": "/", "headers": []})
    response = asyncio.run(desk.priced_research(req))
    assert response.status_code == 403
    assert json.loads(response.body)["error"] == "OWNER_ONLY_RESEARCH"


def test_owner_database_unavailable_fails_closed(monkeypatch):
    async def owner(_):
        return {"authenticated": True, "owner": True}, None
    monkeypatch.setattr(desk.subscriber_contract_v2, "_resolve_entitlement", owner)
    def unavailable():
        raise RuntimeError("private backend detail")
    monkeypatch.setattr(desk, "_query_research_feed", unavailable)
    req = Request({"type": "http", "method": "GET", "path": "/", "headers": []})
    response = asyncio.run(desk.priced_research(req))
    assert response.status_code == 503
    assert "private backend detail" not in response.body.decode()

def test_render_supabase_bridge_connection_uses_server_only_token(monkeypatch):
    calls = []
    class Reply:
        def raise_for_status(self): pass
        def json(self):
            return {"status":"CONNECTED_RESEARCH_READ_ONLY",
                    "production_bets_authorized":False,
                    "rows":[{"fixture_id":234}]}
    def fake_get(url, *, headers, timeout, follow_redirects):
        calls.append((url,headers,timeout,follow_redirects))
        return Reply()
    monkeypatch.setenv("SOCCER_RESEARCH_BRIDGE_URL",
      "https://pqoobjacewxigngzhbhp.supabase.co/functions/v1/soccer-owner-price-bridge")
    monkeypatch.setenv("SOCCER_RESEARCH_BRIDGE_TOKEN","a"*96)
    monkeypatch.setattr(desk.httpx,"get",fake_get)
    assert desk._query_rows()==[{"fixture_id":234}]
    assert len(calls)==1 and calls[0][1]["x-soccer-bridge-token"]=="a"*96
    assert calls[0][3] is False


def test_bridge_fails_closed_on_partial_config_and_bad_source(monkeypatch):
    monkeypatch.setenv("SOCCER_RESEARCH_BRIDGE_URL",
      "https://pqoobjacewxigngzhbhp.supabase.co/functions/v1/soccer-owner-price-bridge")
    monkeypatch.delenv("SOCCER_RESEARCH_BRIDGE_TOKEN",raising=False)
    import pytest
    with pytest.raises(RuntimeError,match="configuration invalid"):
        desk._query_rows()
    monkeypatch.setenv("SOCCER_RESEARCH_BRIDGE_TOKEN","b"*96)
    class Reply:
        def raise_for_status(self): pass
        def json(self): return {"status":"CONNECTED_RESEARCH_READ_ONLY","rows":[]}
    monkeypatch.setattr(desk.httpx,"get",lambda *a,**k:Reply())
    with pytest.raises(RuntimeError,match="invalid provenance"):
        desk._query_rows()


def test_bridge_iso_datetime_preserves_utc_and_rejects_naive():
    assert desk._time("2026-11-06T14:30:00-06:00")=="2026-11-06T20:30:00+00:00"
    assert desk._time("2026-11-06T20:30:00Z")=="2026-11-06T20:30:00+00:00"
    assert desk._time("2026-11-06T20:30:00") is None

def test_sgp_direct_book_quote_is_not_player_betslip_and_keeps_raw_unvalidated():
    source = {
        "fixture_id": 1556133,"home_team":"FC Tokyo","away_team":"Kashima",
        "league_name":"J1 League","kickoff_utc":"2026-11-07T06:00:00+00:00",
        "bookmaker_id":8,"provider_market":"Total Goals/Both Teams To Score",
        "provider_market_id":49,"selection":"o/yes 2.5","offered_price_decimal":"2.00",
        "provider_updated_at":"2026-11-06T21:03:16+00:00",
        "captured_at":"2026-11-06T22:16:45+00:00",
        "offer_type":"BOOK_QUOTED_COMPOSITE",
        "raw_joint_probability":"0.56267103",
        "raw_hypothetical_ev_pct":"12.534",
        "quote_freshness":"CURRENT_QUOTE",
        "classification":"WATCH","bet_eligible":False,
        "bookmaker_direct_combo_quote":True,
        "research_flags":["MODEL_NOT_OOS_VALIDATED"],
    }
    board=desk.make_board([],[source])
    assert board["bets"]==0
    assert board["sgp_custom_player_ticket_price_verified"] is False
    combo=board["sgp_composites"][0]
    assert combo["price_decimal"]==2.0
    assert combo["raw_joint_probability"]>0.56
    assert combo["price_bucket"]=="MINUS110_TO_PLUS110"
    assert combo["price_meaning"]=="BOOK_QUOTED_SINGLE_COMPOSITE_MARKET"
    assert combo["joint_probability_validated"] is False
    assert combo["bet_eligible"] is False and combo["actionable"] is False
    leg={**source,"offer_type":"PLAYER_COMPONENT_QUOTE",
         "provider_market":"Home Player Shots","provider_market_id":240,
         "selection":"Shion Homma - 3","player_label":"Shion Homma",
         "offered_price_decimal":"2.10","raw_joint_probability":None,
         "raw_hypothetical_ev_pct":None,"bookmaker_direct_combo_quote":False}
    p=desk.make_board([],[source,leg])
    assert len(p["sgp_composites"]) == 1
    assert p["sgp_player_components"] == []
    assert p["sgp_player_quote_catalog_count"] == 1
    assert p["sgp_player_selection_status"] == "BLOCKED_NO_VERIFIED_PLAYER_SPORT_MODEL"
    assert p["sgp_player_quote_price_used_for_ranking"] is False
    assert p["sgp_player_prices_are_picks"] is False


def test_sgp_stale_and_unverified_quotes_are_not_displayed():
    src={"offer_type":"BOOK_QUOTED_COMPOSITE","quote_freshness":"STALE_OR_NOT_ELIGIBLE",
         "bookmaker_id":8,"bet_eligible":False,"classification":"WATCH",
         "bookmaker_direct_combo_quote":True,"offered_price_decimal":"2.10"}
    assert not desk._sgp_market_rows([src])["sgp_composites"]
    src["quote_freshness"]="CURRENT_QUOTE"
    src["bookmaker_id"]=12
    assert not desk._sgp_market_rows([src])["sgp_composites"]
    src["bookmaker_id"]=8
    src["bet_eligible"]=True
    assert not desk._sgp_market_rows([src])["sgp_composites"]


def test_sgp_bridge_fetches_single_source_payload(monkeypatch):
    monkeypatch.setenv("SOCCER_RESEARCH_BRIDGE_URL",
      "https://pqoobjacewxigngzhbhp.supabase.co/functions/v1/soccer-owner-price-bridge")
    monkeypatch.setenv("SOCCER_RESEARCH_BRIDGE_TOKEN","c"*96)
    requests=[]
    class Reply:
        def raise_for_status(self): pass
        def json(self): return {"status":"CONNECTED_RESEARCH_READ_ONLY",
          "production_bets_authorized":False,
          "rows":[],"sgp_rows":[{"offer_type":"BOOK_QUOTED_COMPOSITE"}]}
    monkeypatch.setattr(desk.httpx,"get",lambda *a,**k:(requests.append(1) or Reply()))
    result=desk._query_research_feed()
    assert len(requests)==1
    assert len(result["sgp_rows"])==1


def test_sgp_archived_quote_is_displayed_but_never_claimed_fresh():
    row={"fixture_id":1,"home_team":"Home","away_team":"Away","league_name":"J1",
         "kickoff_utc":"2026-11-07T06:00:00Z","bookmaker_id":8,
         "provider_market":"Total Goals/Both Teams To Score","provider_market_id":49,
         "selection":"o/yes 2.5","offered_price_decimal":"2.00",
         "provider_updated_at":"2026-11-06T21:03:16Z",
         "captured_at":"2026-11-06T22:36:40Z",
         "offer_type":"BOOK_QUOTED_COMPOSITE","raw_joint_probability":"0.56267103",
         "raw_hypothetical_ev_pct":"12.534","quote_freshness":"NEEDS_RECONFIRMATION",
         "classification":"WATCH","bet_eligible":False,
         "bookmaker_direct_combo_quote":True,"research_flags":[]}
    x=desk.make_board([],[row])["sgp_composites"][0]
    assert x["source_quote_fresh"] is False
    assert x["quote_reconfirmation_required"] is True
    assert x["bet_eligible"] is False and x["actionable"] is False
    assert "PRICE_SOURCE_OUTDATED_RECONFIRM_BEFORE_ACTION" in x["research_flags"]


def test_canonical_render_cached_market_quotes_feed_only_fresh_known_bookmaker_and_future_fixture():
    from datetime import timedelta
    now=datetime(2026,11,6,23,40,tzinfo=timezone.utc)
    base={
        "fixture_id":1556133,"league_name":"J1 League",
        "home_team":"FC Tokyo","away_team":"Kashima",
        "kickoff_utc":now+timedelta(hours=6),
        "bookmaker_id":8,"provider_market_id":49,
        "provider_market":"Total Goals/Both Teams To Score",
        "provider_updated_at":now-timedelta(minutes=20),
        "captured_at":now-timedelta(minutes=10),
        "values":[{"raw_selection":"o/yes 2.5","decimal_price":"2.00"},
                  {"raw_selection":"u/no 2.5","decimal_price":"2.75"}],
    }
    v=desk._render_cache_sgp_rows([base],now=now)
    assert len(v)==2
    assert all(x["offer_type"]=="BOOK_QUOTED_COMPOSITE" for x in v)
    assert all(x["quote_freshness"]=="CURRENT_QUOTE" for x in v)
    assert all(x["raw_joint_probability"] is None and x["bet_eligible"] is False for x in v)
    old={**base,"provider_updated_at":now-timedelta(hours=2)}
    a=desk._render_cache_sgp_rows([old],now=now)
    assert len(a)==2 and all(x["quote_freshness"]=="NEEDS_RECONFIRMATION" for x in a)
    assert desk._render_cache_sgp_rows([{**base,"bookmaker_id":10}],now=now)==[]
    assert desk._render_cache_sgp_rows([{**base,"kickoff_utc":now-timedelta(seconds=1)}],now=now)==[]
    assert desk._render_cache_sgp_rows([{**base,"provider_updated_at":None}],now=now)==[]


def test_player_prop_cached_quote_is_single_leg_with_no_custom_ticket_price():
    from datetime import timedelta
    now=datetime(2026,11,6,23,40,tzinfo=timezone.utc)
    row={
        "fixture_id":1556133,"league_name":"J1 League",
        "home_team":"FC Tokyo","away_team":"Kashima",
        "kickoff_utc":now+timedelta(hours=6),
        "bookmaker_id":8,"provider_market_id":240,
        "provider_market":"Home Player Shots",
        "provider_updated_at":now-timedelta(minutes=15),
        "captured_at":now-timedelta(minutes=10),
        "values":[{"raw_selection":"Shion Homma - 3","decimal_price":"2.20"},
                  {"raw_selection":"Unusable - 3","decimal_price":"1.45"}],
    }
    vals=desk._render_cache_sgp_rows([row],now=now)
    assert len(vals)==1 and vals[0]["player_label"]=="Shion Homma"
    assert vals[0]["observed_team_side"]=="HOME"
    board=desk.make_board([] ,vals)
    assert board["sgp_player_components"] == []
    assert board["sgp_player_quote_catalog_count"] == 1
    assert board["sgp_player_selection_status"] == "BLOCKED_NO_VERIFIED_PLAYER_SPORT_MODEL"
    assert board["sgp_custom_player_ticket_price_verified"] is False
    assert board["provider_requests_added"] == 0


def test_sgp_source_duplicate_cache_dedupes_same_offer():
    from datetime import timedelta
    now=datetime(2026,11,6,23,40,tzinfo=timezone.utc)
    e={
      "fixture_id":3,"bookmaker_id":8,"provider_market_id":49,
      "selection":"o/yes 2.5","quote_freshness":"CURRENT_QUOTE",
      "provider_updated_at":now-timedelta(minutes=3),
      "captured_at":now-timedelta(minutes=1),
    }
    old={**e,"quote_freshness":"NEEDS_RECONFIRMATION",
         "captured_at":now-timedelta(minutes=12)}
    merged=desk._dedup_sgp_quotes([old,e])
    assert len(merged)==1
    assert merged[0]["quote_freshness"]=="CURRENT_QUOTE"
    # Same provider quote may be represented as an ISO timestamp from Supabase
    # and a timezone-aware datetime from Render PostgreSQL.
    iso={**old,"provider_updated_at":old["provider_updated_at"].isoformat(),
         "captured_at":old["captured_at"].isoformat()}
    merged=desk._dedup_sgp_quotes([iso,e])
    assert len(merged)==1 and merged[0]["quote_freshness"]=="CURRENT_QUOTE"


def test_render_cached_odds_query_is_bounded_with_no_api_requests():
    assert "soccer_market_snapshots" in desk.RENDER_CACHED_SGP_SQL
    assert "soccer_fixtures" in desk.RENDER_CACHED_SGP_SQL
    assert "bookmaker_id = 8" in desk.RENDER_CACHED_SGP_SQL
    assert "LIMIT 80" in desk.RENDER_CACHED_SGP_SQL
    assert "status = 'NS'" in desk.RENDER_CACHED_SGP_SQL
    assert "now() - interval '12 hours'" in desk.RENDER_CACHED_SGP_SQL


def _sgp_combo_fixture(fixture_id, selection, price, raw_prob, ev, market_id=49):
    return {
        "fixture_id": fixture_id,
        "home_team": "Home", "away_team": "Away", "league_name": "J1",
        "kickoff_utc": "2026-11-07T06:00:00+00:00",
        "bookmaker_id": 8, "provider_market_id": market_id,
        "provider_market": "Total Goals/Both Teams To Score",
        "selection": selection, "offered_price_decimal": price,
        "provider_updated_at": "2026-11-06T21:03:16+00:00",
        "captured_at": "2026-11-06T23:35:00+00:00",
        "offer_type": "BOOK_QUOTED_COMPOSITE",
        "raw_joint_probability": raw_prob, "raw_hypothetical_ev_pct": ev,
        "quote_freshness": "NEEDS_RECONFIRMATION",
        "classification": "WATCH", "bet_eligible": False,
        "bookmaker_direct_combo_quote": True, "research_flags": [],
    }


def test_sgp_exclusive_opposite_outcomes_only_one_positive_raw_signal():
    over = _sgp_combo_fixture(1556133, "o/yes 2.5", "2.00", ".56267103", "12.534")
    under = _sgp_combo_fixture(1556133, "u/no 2.5", "2.75", ".23102597", "-36.468")
    board = desk.make_board([], [under,over])
    assert len(board["sgp_composites"]) == 1
    assert board["sgp_composites"][0]["selection"] == "o/yes 2.5"
    assert board["sgp_composites"][0]["signal_label"] == "RAW_EXPERIMENTAL_NOT_BET"
    assert len(board["sgp_market_catalog"]) == 2
    under_catalog = next(x for x in board["sgp_market_catalog"] if x["selection"] == "u/no 2.5")
    assert under_catalog["catalog_reason"] == "NON_POSITIVE_RAW_HYPOTHETICAL_EV"
    assert under_catalog["catalog_only"] is True and under_catalog["not_a_pick"] is True
    assert all(x["bet_eligible"] is False for x in board["sgp_composites"])
    assert board["bets"] == 0
    assert board["notification_delivery_enabled"] is False


def test_sgp_two_negative_raw_alternatives_show_no_candidate():
    under = _sgp_combo_fixture(1556131, "u/no 2.5", "2.20", ".41796190", "-8.048")
    over = _sgp_combo_fixture(1556131, "o/yes 2.5", "2.40", ".36398839", "-12.643")
    board = desk.make_board([], [under,over])
    assert board["sgp_composites"] == []
    assert len(board["sgp_market_catalog"]) == 2
    assert all(x["catalog_reason"]=="NON_POSITIVE_RAW_HYPOTHETICAL_EV" for x in board["sgp_market_catalog"])
    assert board["status"] == "NO_FRESH_RESEARCH_QUOTES"


def test_sgp_one_signal_per_fixture_even_if_two_markets_happen_to_be_positive():
    main = _sgp_combo_fixture(101,"o/yes 2.5", "2.00", ".60", "20.0")
    alternative = _sgp_combo_fixture(101,"Home/Over 2.5", "3.0", ".40", "20.0",market_id=7)
    other_fixture = _sgp_combo_fixture(102,"o/yes 2.5", "2.00", ".55", "10.0")
    board = desk.make_board([], [main,alternative,other_fixture])
    assert len(board["sgp_composites"]) == 2
    assert {x["fixture_id"] for x in board["sgp_composites"]} == {101,102}
    assert board["sgp_composites_suppressed"] == 1
    assert sum(x["catalog_reason"]=="SUPPRESSED_ALTERNATIVE_SAME_FIXTURE" for x in board["sgp_market_catalog"])==1


def test_sgp_unmodeled_book_quote_is_catalog_only_not_model_candidate():
    unknown = _sgp_combo_fixture(103,"Home/Home", "3.10", None, None, market_id=25)
    board = desk.make_board([], [unknown])
    assert board["sgp_composites"] == []
    assert len(board["sgp_market_catalog"]) == 1
    assert board["sgp_market_catalog"][0]["catalog_reason"] == "NO_JOINT_PROBABILITY_MODEL"
    assert board["sgp_player_components"] == []
