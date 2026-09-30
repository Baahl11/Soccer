import asyncio

from mcp_gateway import automation_v2 as v2
from mcp_gateway import automation_v4 as v4
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


def test_v209_1_releases_low_water_only_to_declared_global_price_cap(monkeypatch):
    monkeypatch.setattr(v4, "_MONOTONIC_TICK_CAP", 25)
    monkeypatch.setattr(v2, "_API_CALLS_THIS_TICK", 25)
    monkeypatch.setattr(v2, "MAX_API_CALLS_PER_TICK", 45)

    report = v126._release_reserved_price_phase_cap()

    assert report["released"] is True
    assert report["previous_low_water_cap"] == 25
    assert report["declared_global_cap"] == 45
    assert report["effective_cap_after_release"] == 45
    assert report["provider_calls_at_release"] == 25
    assert report["provider_budget_changed"] is False
    assert report["provider_requests_added"] == 0
    assert v4._MONOTONIC_TICK_CAP == 45


def test_v209_1_does_not_raise_cap_when_no_reserved_release_exists(monkeypatch):
    monkeypatch.setattr(v4, "_MONOTONIC_TICK_CAP", 35)
    monkeypatch.setattr(v2, "_API_CALLS_THIS_TICK", 10)
    monkeypatch.setattr(v2, "MAX_API_CALLS_PER_TICK", 35)

    report = v126._release_reserved_price_phase_cap()

    assert report["released"] is False
    assert report["effective_cap_after_release"] == 35
    assert v4._MONOTONIC_TICK_CAP == 35


def test_v209_4_upstream_phase_lock_survives_counter_reset_and_reopen(monkeypatch):
    observations = {}

    async def fake_run_tick():
        # Simulate quota policy tightening the upstream cap after the first
        # provider response.
        v2.MAX_API_CALLS_PER_TICK = 25
        v2._API_CALLS_THIS_TICK = 1
        observations["tightened"] = v4._effective_tick_cap()

        # Simulate a downstream orchestration layer resetting the counter and
        # attempting to reopen the configured cap. v209.4 must retain 25.
        v2._API_CALLS_THIS_TICK = 0
        v2.MAX_API_CALLS_PER_TICK = 45
        observations["after_reset"] = v4._effective_tick_cap()

        return {
            "api_calls_this_tick": 20,
            "max_api_calls_per_tick": 45,
            "effective_max_api_calls_per_tick": 45,
            "price_resolution_v4": {"api_calls_added": 0},
        }

    monkeypatch.setattr(v126.v125, "run_tick", fake_run_tick)
    payload = asyncio.run(v126.run_tick())

    assert observations == {"tightened": 25, "after_reset": 25}
    lock = payload["v209_post_reserve_maturation_verification"]["upstream_phase_lock"]
    assert lock["seed_cap"] == 50
    assert lock["low_water_cap"] == 25
    assert lock["tighten_count"] >= 1
    assert lock["provider_budget_changed"] is False


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
