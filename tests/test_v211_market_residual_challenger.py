from __future__ import annotations

from mcp_gateway import market_residual_challenger_v4 as v


def _row(*, family="BTTS", p_model=0.60, p_market=0.55, outcome=None):
    row = {
        "fixture_id": 1,
        "market_family": family,
        "market": "Test Market",
        "selection": "Yes",
        "stage": "T-20",
        "p_model_calibrated": p_model,
        "p_market_fair": p_market,
    }
    if outcome is not None:
        row["binary_outcome"] = outcome
    return row


def test_residual_is_model_minus_devig_market():
    report = v.build_report([_row(p_model=0.62, p_market=0.57)])

    assert report["status"] == "RESEARCH_ONLY"
    assert report["comparable_rows"] == 1
    assert report["overall"]["mean_market_residual"] == 0.05
    assert report["family_metrics"]["BTTS"]["mean_absolute_market_residual"] == 0.05


def test_missing_calibrated_probability_is_excluded_truthfully():
    report = v.build_report([_row(p_model=None)])

    assert report["status"] == "NOT_VERIFIED_NO_COMPARABLE_ROWS"
    assert report["excluded_rows"] == {"MISSING_CALIBRATED_MODEL_PROBABILITY": 1}


def test_missing_market_probability_is_excluded_truthfully():
    report = v.build_report([_row(p_market=None)])

    assert report["comparable_rows"] == 0
    assert report["excluded_rows"] == {"MISSING_DEVIG_MARKET_PROBABILITY": 1}


def test_binary_scores_compare_model_to_market_when_explicit_outcome_exists():
    report = v.build_report(
        [
            _row(p_model=0.8, p_market=0.6, outcome=1),
            _row(p_model=0.3, p_market=0.4, outcome=0),
        ]
    )
    overall = report["overall"]

    assert overall["settled_binary_rows"] == 2
    assert overall["model_brier"] is not None
    assert overall["market_brier"] is not None
    assert overall["brier_delta_model_minus_market"] < 0
    assert overall["log_loss_delta_model_minus_market"] < 0


def test_outcome_is_never_inferred_from_unrelated_result_fields():
    row = _row()
    row["result"] = {"home": 3, "away": 0}
    report = v.build_report([row])

    assert report["overall"]["settled_binary_rows"] == 0
    assert report["overall"]["model_brier"] is None


def test_family_and_probability_buckets_are_descriptive_only():
    report = v.build_report(
        [
            _row(family="BTTS", p_model=0.51, p_market=0.50),
            _row(family="FT_TOTALS", p_model=0.73, p_market=0.68),
        ]
    )

    assert set(report["family_metrics"]) == {"BTTS", "FT_TOTALS"}
    assert len(report["reliability_buckets"]) == 2
    assert report["decision_weight"] == 0.0
    assert report["production_promotion_allowed"] is False
    assert report["provider_requests_added"] == 0
    assert report["provider_budget_changed"] is False
    assert report["model_weights_changed"] is False
    assert report["thresholds_changed"] is False
    assert report["gates_changed"] is False
    assert report["canonical_bet_logic_changed"] is False
    assert report["strict_close_semantics_changed"] is False
