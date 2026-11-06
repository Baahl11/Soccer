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
