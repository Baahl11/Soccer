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
    monkeypatch.setattr(desk, "_query_rows", prohibited)
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
    monkeypatch.setattr(desk, "_query_rows", unavailable)
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
