from __future__ import annotations

from mcp_gateway import dynamic_strength_challenger_v4 as v
from mcp_gateway import soccer_model


def _recent_home(team_id: int, scores: list[tuple[int, int]]) -> list[dict]:
    return [
        {
            "home_team_id": team_id,
            "away_team_id": 9000 + index,
            "goals": {"home": home, "away": away},
        }
        for index, (home, away) in enumerate(scores)
    ]


def _recent_away(team_id: int, scores: list[tuple[int, int]]) -> list[dict]:
    return [
        {
            "home_team_id": 8000 + index,
            "away_team_id": team_id,
            "goals": {"home": home, "away": away},
        }
        for index, (home, away) in enumerate(scores)
    ]


def _raw(home_lam: float = 1.20, away_lam: float = 1.10) -> dict:
    probs = soccer_model._probabilities(home_lam, away_lam)
    return {
        "status": "MODELED_LIMITED",
        "raw_home_goal_rate": home_lam,
        "raw_away_goal_rate": away_lam,
        "raw_home_win_prob": probs["home_win"],
        "raw_draw_prob": probs["draw"],
        "raw_away_win_prob": probs["away_win"],
        "raw_btts_yes_prob": probs["btts_yes"],
        "raw_over_1_5_prob": probs["over_1_5"],
        "raw_over_2_5_prob": probs["over_2_5"],
        "raw_over_3_5_prob": probs["over_3_5"],
    }


def _event(*, home_recent_n: int = 3, away_recent_n: int = 3) -> dict:
    home_scores = [(2, 0), (2, 1), (1, 0)][:home_recent_n]
    away_scores = [(1, 2), (2, 2), (0, 1)][:away_recent_n]
    return {
        "event_type": "SOCCER_REFRESH",
        "stage": "T-20",
        "fixture": {"fixture_id": 77, "home_team_id": 1, "away_team_id": 2},
        "raw_projection": _raw(),
        "sporting": {
            "sport_data": "AVAILABLE",
            "home_stats": {
                "goals": {
                    "for": {"average": {"home": "1.20"}},
                    "against": {"average": {"home": "1.10"}},
                },
                "fixtures": {"played": {"home": 10}},
            },
            "away_stats": {
                "goals": {
                    "for": {"average": {"away": "1.00"}},
                    "against": {"average": {"away": "1.40"}},
                },
                "fixtures": {"played": {"away": 10}},
            },
            "home_recent": _recent_home(1, home_scores),
            "away_recent": _recent_away(2, away_scores),
        },
    }


def test_recent_only_counterfactual_builds_from_existing_same_tick_inputs():
    row, reason = v._recent_strength_projection(_event())

    assert reason is None
    assert row is not None
    assert row["fixture_id"] == 77
    assert row["variant"] == "RECENT_ONLY_COUNTERFACTUAL"
    assert row["sample"] == {"home_recent_matches": 3, "away_recent_matches": 3}
    assert row["strength"]["recent_home_goal_rate"] == 1.3333
    assert row["strength"]["recent_away_goal_rate"] == 1.0
    assert row["decision_weight"] == 0.0
    assert row["production_promotion_allowed"] is False
    assert row["provider_requests_added"] == 0


def test_challenger_probability_vector_is_distinct_from_base_when_recent_strength_moves():
    row, _ = v._recent_strength_projection(_event())

    assert row is not None
    deltas = row["probability_delta_pp"]
    assert any(abs(value) > 0.0001 for value in deltas.values())
    assert set(deltas) == {
        "home_win",
        "draw",
        "away_win",
        "btts_yes",
        "over_1_5",
        "over_2_5",
        "over_3_5",
    }


def test_less_than_three_recent_matches_is_not_comparable():
    row, reason = v._recent_strength_projection(_event(home_recent_n=2))

    assert row is None
    assert reason == "INSUFFICIENT_RECENT_SAMPLE"


def test_report_is_research_only_and_does_not_change_production_controls():
    report = v.build_report([_event(), {"event_type": "DAILY_DISCOVERY"}])

    assert report["status"] == "RESEARCH_ONLY"
    assert report["comparable_refresh_events"] == 1
    assert report["by_stage"] == {"T-20": 1}
    assert report["provider_requests_added"] == 0
    assert report["provider_budget_changed"] is False
    assert report["decision_weight"] == 0.0
    assert report["production_promotion_allowed"] is False
    assert report["model_weights_changed"] is False
    assert report["thresholds_changed"] is False
    assert report["gates_changed"] is False
    assert report["canonical_bet_logic_changed"] is False
    assert report["strict_close_semantics_changed"] is False


def test_missing_raw_projection_is_truthfully_excluded():
    event = _event()
    event.pop("raw_projection")
    report = v.build_report([event])

    assert report["status"] == "NOT_VERIFIED_NO_COMPARABLE_ROWS"
    assert report["excluded_refresh_events"] == {"MISSING_BASE_RAW_PROJECTION": 1}
