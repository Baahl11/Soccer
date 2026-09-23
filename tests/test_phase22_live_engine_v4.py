from mcp_gateway import live_engine_v4 as v


def _snapshot(**overrides):
    row = {
        "fixture_id": 1,
        "minute": 56,
        "score": {"home": 1, "away": 0},
        "red_cards": {"home": 0, "away": 0},
        "shots": {"home": 10, "away": 6},
        "shots_on_target": {"home": 4, "away": 2},
        "corners": {"home": 5, "away": 3},
        "possession": {"home": 56, "away": 44},
        "game_state": "HOME_LEAD",
        "pregame_prior": {"p_over_2_5": 0.57, "p_home": 0.51},
    }
    row.update(overrides)
    return row


def test_phase22_blocks_missing_verified_live_inputs():
    snap = _snapshot()
    del snap["red_cards"]
    result = v.build_live_state(snap)
    assert result["status"] == "LIVE_INPUT_BLOCKED"
    assert "MISSING_RED_CARDS" in result["validation"]["blockers"]
    assert result["actionable"] is False


def test_phase22_rejects_unverified_live_xg():
    result = v.build_live_state(_snapshot(
        xg={"home": 1.4, "away": 0.6},
        xg_verified=False,
    ))
    assert result["status"] == "LIVE_INPUT_BLOCKED"
    assert "LIVE_XG_PRESENT_BUT_NOT_VERIFIED" in result["validation"]["blockers"]


def test_phase22_builds_verified_research_state_without_inventing_posterior():
    result = v.build_live_state(_snapshot(
        xg={"home": 1.4, "away": 0.6},
        xg_verified=True,
    ))
    assert result["status"] == "LIVE_RESEARCH_STATE_READY"
    assert result["derived"]["shot_diff_home_minus_away"] == 4
    assert result["derived"]["sot_diff_home_minus_away"] == 2
    assert result["derived"]["corner_diff_home_minus_away"] == 2
    assert result["xg_used"] is True
    assert result["posterior_probability_update"] is None
    assert result["posterior_update_status"] == "BLOCKED_UNTIL_LIVE_MODEL_OOS_CALIBRATED"
    assert result["actionable"] is False
    assert result["decision_weight"] == 0.0


def test_phase22_2h_target_waits_until_halftime():
    result = v.build_live_state(_snapshot(minute=30))
    assert result["target_status"]["live_ft_goals"] == "RESEARCH_INPUT_READY"
    assert result["target_status"]["live_2h_goals"] == "WAIT_HALFTIME_OR_LATER"
