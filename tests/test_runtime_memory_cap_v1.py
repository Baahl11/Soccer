"""Offline scheduler memory guard contracts. No provider or database calls."""
from __future__ import annotations

from pathlib import Path

import pytest

from mcp_gateway import runtime_memory_cap_v1 as guard


def test_512m_container_tightens_large_existing_upstream_cap_only():
    cap = guard.safe_upstream_phase_cap(50, cgroup_limit_bytes=512*1024*1024)
    assert cap["applied"] is True
    assert cap["existing_upstream_cap"] == 50
    assert cap["effective_upstream_cap"] == 28
    assert cap["deferred_extra_upstream_calls_max"] == 22
    assert cap["provider_budget_increased"] is False
    assert cap["sport_model_weights_changed"] is False
    assert cap["market_gates_changed"] is False


def test_1gb_and_unbounded_cgroups_preserve_provider_cap():
    for limit in (1024*1024*1024, None, 1<<62):
        result = guard.safe_upstream_phase_cap(50, cgroup_limit_bytes=limit)
        assert result["effective_upstream_cap"] == 50
        assert result["applied"] is False


def test_existing_lower_budget_must_not_increase_and_can_still_tighten():
    for original in (1, 4, 12, 27, 28):
        result = guard.safe_upstream_phase_cap(
            original, cgroup_limit_bytes=512*1024*1024
        )
        assert result["effective_upstream_cap"] == original
        assert result["provider_budget_increased"] is False


def test_safe_guard_configuration_range_and_invalid_quota():
    with pytest.raises(ValueError, match="INVALID_EXISTING_UPSTREAM_CAP"):
        guard.safe_upstream_phase_cap(0, cgroup_limit_bytes=512*1024*1024)
    with pytest.raises(ValueError, match="INVALID_SMALL_INSTANCE_UPSTREAM_CAP"):
        guard.safe_upstream_phase_cap(70, cgroup_limit_bytes=512*1024*1024, small_instance_cap=50)
    with pytest.raises(ValueError, match="INVALID_SMALL_INSTANCE_UPSTREAM_CAP"):
        guard.safe_upstream_phase_cap(70, cgroup_limit_bytes=512*1024*1024, small_instance_cap=0)


def test_legacy_memory_cgroup_limit_ignored_if_too_small():
    cap = guard.safe_upstream_phase_cap(60, cgroup_limit_bytes=40 * 1024*1024)
    assert cap["effective_upstream_cap"] == 60


def test_runtime_phase_lock_seeds_protective_limit_not_full_50_calls():
    text = (Path(__file__).resolve().parents[1] / "mcp_gateway" / "automation_v126.py").read_text()
    assert "safe_upstream_phase_cap(" in text
    assert 'initial_upstream_cap = memory_cap_observation["effective_upstream_cap"]' in text
    assert 'v4._MONOTONIC_TICK_CAP = int(initial_upstream_cap)' in text
    assert '"resource_memory_guard": memory_cap_observation' in text
    assert "v4._effective_tick_cap = phase_locked_effective_tick_cap" in text
    assert "price_resolver_v4._fetch_fixture_odds = phase_boundary_fetch" in text


def test_no_bet_or_market_code_modified_in_patch():
    text=(Path(__file__).resolve().parents[1] / "mcp_gateway" / "runtime_memory_cap_v1.py").read_text()
    assert "deferred_extra_upstream_calls_max" in text
    assert "market_gates_changed" in text
    assert "NO_SYNTHETIC_EVENTS" in text
