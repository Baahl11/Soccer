import asyncio

from mcp_gateway import automation_v126 as v126


def test_v209_verifies_live_price_reserve_without_changing_budget():
    payload = {
        "api_calls_this_tick": 23,
        "max_api_calls_per_tick": 45,
        "effective_max_api_calls_per_tick": 45,
        "price_resolution_v4": {
            "api_calls_added": 5,
            "max_api_calls": 25,
            "primary_clv_maturation_api_calls_added": 1,
            "primary_clv_maturation_max_calls_per_tick": 8,
            "primary_clv_maturation_candidate_family_counts": {
                "FT_CORNERS": 12,
                "FT_TOTALS": 1,
            },
        },
        "v207_btts_paid_odds_entry_capture": {
            "entry_rows_added": 1,
            "captured_fixtures": 2,
        },
    }

    report = v126._build_post_reserve_verification(payload)

    assert report["status"] == "LIVE_VERIFIED"
    assert report["cap_respected"] is True
    assert report["provider_headroom_after_tick"] == 22
    assert report["price_resolver_api_calls_added"] == 5
    assert report["primary_clv_maturation_api_calls_added"] == 1
    assert report["roadmap_priority_candidate_counts"] == {
        "1X2": 0,
        "BTTS": 0,
        "FT_TOTALS": 1,
    }
    assert report["btts_paid_entry_rows_added"] == 1
    assert report["provider_requests_added"] == 0
    assert report["provider_budget_changed"] is False
    assert report["thresholds_changed"] is False
    assert report["gates_changed"] is False
    assert report["strict_close_semantics_changed"] is False


def test_v209_flags_cap_violation_without_mutating_payload():
    payload = {
        "api_calls_this_tick": 46,
        "max_api_calls_per_tick": 45,
        "effective_max_api_calls_per_tick": 45,
        "price_resolution_v4": {"api_calls_added": 2},
    }

    report = v126._build_post_reserve_verification(payload)

    assert report["status"] == "CAP_VIOLATION_DETECTED"
    assert report["cap_respected"] is False
    assert report["provider_headroom_after_tick"] == 0
    assert payload["price_resolution_v4"] == {"api_calls_added": 2}


def test_v209_run_tick_nests_verification_in_existing_compact_state_path(monkeypatch):
    async def fake_run_tick():
        return {
            "api_calls_this_tick": 23,
            "max_api_calls_per_tick": 45,
            "effective_max_api_calls_per_tick": 45,
            "price_resolution_v4": {
                "api_calls_added": 5,
                "primary_clv_maturation_api_calls_added": 1,
            },
            "v207_btts_paid_odds_entry_capture": {"entry_rows_added": 1},
        }

    monkeypatch.setattr(v126.v125, "run_tick", fake_run_tick)
    payload = asyncio.run(v126.run_tick())

    assert payload["version"] == v126.AUTOMATION_VERSION
    assert payload["model_version"] == v126.MODEL_VERSION
    assert payload["v209_post_reserve_maturation_verification"]["status"] == "LIVE_VERIFIED"
    assert (
        payload["price_resolution_v4"]["v209_post_reserve_verification"]["status"]
        == "LIVE_VERIFIED"
    )
