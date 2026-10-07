from copy import deepcopy

from mcp_gateway import formation_fm5_raw_projection_v1 as fm5
from mcp_gateway import formation_fm6_market_validation_v1 as fm6


def _fm5_report():
    baseline = {
        "status": "MODELED_LIMITED",
        "model_version": "BASE",
        "raw_home_goal_rate": 1.4,
        "raw_away_goal_rate": 1.1,
        "raw_total_goals": 2.5,
    }
    readiness = {
        "model_version": "FORMATION_FM5_READINESS_GATE_V1.0.0",
        "ready_components": ["FM4_STYLE_GOALS"],
        "integration_review_allowed": True,
        "market_prices_consumed": False,
        "production_enabled": False,
    }
    adjustment = {
        "source_component": "FM4_STYLE_GOALS",
        "source_model_version": "FORMATION_MATCHUP_FM4_STYLE_ABLATION_V1.1.0",
        "source_artifact_id": "fixture:123:fm4-style-goals-v1",
        "feature_as_of": "2026-10-07T17:00:00+00:00",
        "fixture_kickoff": "2026-10-07T18:00:00+00:00",
        "prior_only": True,
        "oos_gate_passed": True,
        "market_fields_used": False,
        "manual_research_approval": True,
        "adjusted_home_goal_rate": 1.62,
        "adjusted_away_goal_rate": 1.04,
    }
    return fm5.build_research_candidate(
        baseline,
        readiness,
        adjustment,
        availability_confidence=0.93,
    )


def _market(**overrides):
    row = {
        "captured_at": "2026-10-07T17:20:00+00:00",
        "source": "VERIFIED_SPORTSBOOK_FEED",
        "bookmaker": "Book",
        "market_family": "FT_TOTALS",
        "market": "Goals Over/Under",
        "selection": "OVER",
        "line": 2.5,
        "decimal_price": 1.95,
        "comparison_prices": {"over": 1.95, "under": 1.95},
    }
    row.update(overrides)
    return row


def test_exact_market_signal_is_research_only_and_sport_first():
    report = fm6.build_exact_market_research_signal(_fm5_report(), _market())

    assert report["status"] == "FM6_EXACT_MARKET_RESEARCH_SIGNAL"
    assert report["research_only"] is True
    assert report["decision_weight"] == 0.0
    assert report["bet_eligible"] is False
    assert report["production_enabled"] is False
    assert report["market_can_create_sporting_thesis"] is False
    assert report["market_shrinkage_applied"] is False
    assert report["estimated_ev_used_for_decision"] is False
    assert report["production_promotion_allowed"] is False

    signal = report["signal"]
    assert signal["strict_exact_instrument"] is True
    assert signal["classification"] == "WATCH"
    assert signal["p_raw_sport"] > 0
    assert signal["p_breakeven"] == round(1 / 1.95, 8)
    assert signal["p_market_fair"] == 0.5


def test_market_snapshot_must_follow_raw_projection_and_precede_kickoff():
    early = fm6.build_exact_market_research_signal(
        _fm5_report(),
        _market(captured_at="2026-10-07T16:59:59+00:00"),
    )
    assert early["status"] == "FM6_BLOCKED"
    assert "MARKET_SNAPSHOT_PRECEDES_RAW_SPORT_PROJECTION" in early["blockers"]

    late = fm6.build_exact_market_research_signal(
        _fm5_report(),
        _market(captured_at="2026-10-07T18:00:00+00:00"),
    )
    assert late["status"] == "FM6_BLOCKED"
    assert "MARKET_SNAPSHOT_NOT_STRICTLY_PREKICKOFF" in late["blockers"]


def test_unsupported_exact_total_line_is_blocked_not_invented():
    report = fm6.build_exact_market_research_signal(
        _fm5_report(),
        _market(line=2.0, selection="OVER"),
    )
    assert report["status"] == "FM6_BLOCKED"
    assert "EXACT_RAW_SPORT_PROBABILITY_NOT_AVAILABLE" in report["blockers"]


def test_one_x_two_uses_existing_raw_sport_probability_not_market_to_create_it():
    report = fm6.build_exact_market_research_signal(
        _fm5_report(),
        _market(
            market_family="1X2",
            market="Match Winner",
            selection="HOME",
            line=None,
            decimal_price=2.2,
            comparison_prices={"home": 2.2, "draw": 3.3, "away": 3.2},
        ),
    )
    assert report["status"] == "FM6_EXACT_MARKET_RESEARCH_SIGNAL"
    signal = report["signal"]
    assert signal["p_raw_sport"] == _fm5_report()["fm5_raw_projection_research"]["raw_home_win_prob"]
    assert signal["p_market_fair"] is not None


def test_missing_fm5_candidate_blocks_market_validation():
    blocked = deepcopy(_fm5_report())
    blocked["status"] = "FM5_BLOCKED"
    blocked["research_candidate_available"] = False
    blocked["fm5_raw_projection_research"] = None

    report = fm6.build_exact_market_research_signal(blocked, _market())

    assert report["status"] == "FM6_BLOCKED"
    assert report["signal"] is None
    assert "FM5_RESEARCH_CANDIDATE_NOT_AVAILABLE" in report["blockers"]


def test_inputs_are_not_mutated():
    fm5_report = _fm5_report()
    market = _market()
    before_fm5 = deepcopy(fm5_report)
    before_market = deepcopy(market)

    fm6.build_exact_market_research_signal(fm5_report, market)

    assert fm5_report == before_fm5
    assert market == before_market
