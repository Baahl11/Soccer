from mcp_gateway import fixture_live_status_v1 as live
from mcp_gateway import soccer_universal_match_v1 as universal


def provider_response(status="CANC", home=367, away=2149):
    return {
        "response": [{
            "fixture": {"id": 1646118, "status": {"short": status}},
            "teams": {"home": {"id": home}, "away": {"id": away}},
        }],
        "quota": {"daily_remaining": 7300},
    }


def test_essential_current_status_is_verified_and_cached_without_market_requests(monkeypatch):
    monkeypatch.setattr(live, "_cache", {})
    monkeypatch.setattr(live, "_hour_begin", 0.0)
    monkeypatch.setattr(live, "_hourly_count", 0)
    monkeypatch.setattr(live, "_day_begin", 0.0)
    monkeypatch.setattr(live, "_daily_count", 0)
    seen = []
    def fetcher(fid, _key):
        seen.append(fid)
        return provider_response()
    first = live.lookup(1646118, 367, 2149, fetcher=fetcher, clock=lambda: 120000.0)
    second = live.lookup(1646118, 367, 2149, fetcher=fetcher, clock=lambda: 120004.0)
    assert first["status"] == "CANC"
    assert first["source"] == "API_FOOTBALL_FIXTURE_STATUS_BY_ID"
    assert first["cancelled_or_postponed"] is True
    assert first["quote_requested"] is False
    assert first["provider_requests_added"] == 1
    assert second["provider_requests_added"] == 0
    assert second["from_cache"] is True
    assert seen == [1646118]


def test_stale_or_unverified_fixture_identity_cannot_override_status(monkeypatch):
    monkeypatch.setattr(live, "_cache", {})
    monkeypatch.setattr(live, "_hour_begin", 0.0)
    monkeypatch.setattr(live, "_hourly_count", 0)
    monkeypatch.setattr(live, "_day_begin", 0.0)
    monkeypatch.setattr(live, "_daily_count", 0)
    bad = live.lookup(1646118, 367, 2149, fetcher=lambda *_: provider_response(home=999), clock=lambda: 140000)
    assert bad["status"] == "NOT_VERIFIED"
    assert bad["provider_requests_added"] == 1
    invalid = live.lookup(1646118, None, 2149, fetcher=lambda *_: (_ for _ in ()).throw(AssertionError()), clock=lambda: 140000)
    assert invalid["status"] == "NOT_VERIFIED"
    assert invalid["provider_requests_added"] == 0


def test_live_lookup_cap_does_not_spend_unbounded_source_quota(monkeypatch):
    monkeypatch.setattr(live, "_cache", {})
    monkeypatch.setattr(live, "_hour_begin", 100000.0)
    monkeypatch.setattr(live, "_hourly_count", live.HOURLY_UNIQUE_CAP)
    monkeypatch.setattr(live, "_day_begin", 100000.0)
    monkeypatch.setattr(live, "_daily_count", live.DAILY_REQUEST_CAP)
    result = live.lookup(1646118, 367, 2149, clock=lambda: 100001.0,
                         fetcher=lambda *_: (_ for _ in ()).throw(AssertionError("must not call")))
    assert result["status"] == "NOT_VERIFIED"
    assert result["provider_requests_added"] == 0


def test_cancelled_fixture_preserves_history_but_blocks_current_opportunity():
    row = {
        "fixture": {"fixture_id":1646118,"fixture_status":"CANC"},
        "coverage": {"analysis_rows":1, "raw_sport_projection_present": True,
                     "evidence_inventory": {"model_runs": 1},
                     "verified_price_markets": ["1X2"]},
        "state": {"classification":"BET"},
        "screening": {"raw_projection_recorded": True},
    }
    packet = universal.add_to_slate_row(row)
    assert packet["sport_analysis"]["status"] == "CANCELLED"
    assert packet["sport_analysis"]["raw_projection_recorded"] is True
    assert packet["opportunity"]["status"] == "EVENTO_NO_VIGENTE"
    assert packet["opportunity"]["recommendation_executable"] is False
    deep = universal.detail_layers({
        "fixture": {"fixture_status":"CANC"},
        "sport_context":{"outcome_probabilities":{"home":.48,"draw":.28,"away":.24}},
        "decision_summary":{"classification":"BET"},
    })
    assert deep["sport_analysis"]["status"] == "CANCELLED"
    assert deep["opportunity"]["status"] == "EVENTO_NO_VIGENTE"
    assert deep["opportunity"]["recommendation_executable"] is False


def test_cancelled_fixture_has_no_current_quote_gate_even_if_historical_bet():
    row = {
        "fixture": {"fixture_id":1646118,"fixture_status":"PST"},
        "coverage": {"verified_price_markets": ["1X2"]},
        "state": {"classification":"BET"},
    }
    packet = universal.add_to_slate_row(row)
    assert packet["sport_analysis"]["status"] == "POSTPONED"
    assert packet["opportunity"]["status"] == "EVENTO_NO_VIGENTE"
