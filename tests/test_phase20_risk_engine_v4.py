from mcp_gateway import risk_engine_v4 as v


POLICY = {
    "fractional_kelly_multiplier": 0.25,
    "max_stake_units": 2.0,
    "max_daily_exposure_units": 8.0,
    "max_correlated_exposure_units": 3.0,
    "max_league_exposure_units": 4.0,
    "max_market_exposure_units": 5.0,
    "max_drawdown_units": 6.0,
}


def test_phase20_blocks_when_market_not_production_validated():
    proposal = v.propose_stake(
        probability=0.60,
        decimal_price=2.0,
        bankroll_units=100,
        production_validated=False,
        policy=POLICY,
    )
    assert proposal["stake_units"] == 0.0
    assert proposal["status"] == "BLOCKED_NOT_PRODUCTION_VALIDATED"
    assert proposal["no_chase"] is True
    assert proposal["no_martingale"] is True


def test_phase20_fractional_kelly_respects_all_caps():
    proposal = v.propose_stake(
        probability=0.60,
        decimal_price=2.0,
        bankroll_units=100,
        production_validated=True,
        policy=POLICY,
        daily_exposure_units=7.5,
        correlated_exposure_units=2.8,
        league_exposure_units=3.9,
        market_exposure_units=4.9,
    )
    assert proposal["stake_units"] == 0.1
    assert proposal["production_execution_enabled"] is False


def test_phase20_recent_losses_do_not_increase_stake():
    base = v.propose_stake(
        probability=0.58,
        decimal_price=2.05,
        bankroll_units=100,
        production_validated=True,
        policy=POLICY,
        recent_results=[],
    )
    losing = v.propose_stake(
        probability=0.58,
        decimal_price=2.05,
        bankroll_units=100,
        production_validated=True,
        policy=POLICY,
        recent_results=["WIN", "LOSS", "LOSS", "LOSS"],
    )
    assert losing["stake_units"] == base["stake_units"]
    assert losing["recent_losing_streak_audit_only"] == 3
    assert losing["recent_results_can_increase_stake"] is False


def test_phase20_drawdown_stop_blocks_stake():
    proposal = v.propose_stake(
        probability=0.60,
        decimal_price=2.0,
        bankroll_units=100,
        production_validated=True,
        policy=POLICY,
        current_drawdown_units=6.0,
    )
    assert proposal["stake_units"] == 0.0
    assert proposal["status"] == "STOP_DRAWDOWN_LIMIT"


def test_phase20_report_locked_with_no_production_markets():
    report = v.build_report(
        {
            "market_family_reviews": [
                {"market_family": "FT_TOTALS", "current_state": "RESEARCH"},
            ]
        },
        {
            "status": "OOS_FRAMEWORK_DATA_GAPS",
            "realized_settlement_metrics": {
                "max_drawdown_units": 2.64,
                "max_losing_streak": 4,
                "return_volatility_stddev": 1.66,
                "settled_rows": 28,
            },
        },
    )
    assert report["status"] == "RISK_ENGINE_LOCKED"
    assert "NO_PRODUCTION_VALIDATED_MARKETS" in report["blockers"]
    assert report["production_execution_enabled"] is False
