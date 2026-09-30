from __future__ import annotations

from mcp_gateway import market_residual_challenger_v4 as v


def _row(
    *,
    fixture_id=1,
    family="BTTS",
    market="Test Market",
    selection="Yes",
    line=None,
    stage="T-20",
    p_model=0.60,
    p_market=0.55,
    outcome=None,
    outcome_class=None,
):
    row = {
        "fixture_id": fixture_id,
        "market_family": family,
        "market": market,
        "selection": selection,
        "line": line,
        "stage": stage,
        "p_model_calibrated": p_model,
        "p_market_fair": p_market,
    }
    if outcome is not None:
        row["binary_outcome"] = outcome
    if outcome_class is not None:
        row["outcome_class"] = outcome_class
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
            _row(fixture_id=1, p_model=0.8, p_market=0.6, outcome=1),
            _row(fixture_id=2, p_model=0.3, p_market=0.4, outcome=0),
        ]
    )
    overall = report["overall"]

    assert overall["settled_binary_rows"] == 2
    assert overall["model_brier"] is not None
    assert overall["market_brier"] is not None
    assert overall["brier_delta_model_minus_market"] < 0
    assert overall["log_loss_delta_model_minus_market"] < 0


def test_outcome_is_never_inferred_from_unrelated_runtime_result_fields():
    row = _row()
    row["result"] = {"home": 3, "away": 0}
    report = v.build_report([row])

    assert report["overall"]["settled_binary_rows"] == 0
    assert report["overall"]["model_brier"] is None


def test_dedicated_settlement_ledger_enriches_binary_outcome_only_for_win_loss():
    source = [_row(family="BTTS", market="Both Teams To Score", selection="Yes")]
    settlement = [
        {
            "fixture_id": 1,
            "market_family": "BTTS",
            "market": "Both Teams To Score",
            "selection": "Yes",
            "line": None,
            "stage": "T-10",
            "settled": True,
            "settlement_status": "WIN",
        }
    ]
    report = v.build_report(source, settlement_rows=settlement)

    assert report["overall"]["settled_binary_rows"] == 1
    assert report["overall"]["model_brier"] == 0.16


def test_rps_compares_complete_1x2_vector_to_market_baseline():
    rows = [
        _row(family="1X2", market="Match Winner", selection="Home", p_model=0.70, p_market=0.50, outcome_class="HOME"),
        _row(family="1X2", market="Match Winner", selection="Draw", p_model=0.20, p_market=0.30, outcome_class="HOME"),
        _row(family="1X2", market="Match Winner", selection="Away", p_model=0.10, p_market=0.20, outcome_class="HOME"),
    ]
    report = v.build_report(rows)
    rps = report["rps_1x2"]

    assert rps["scored_1x2_groups"] == 1
    assert rps["model_rps"] == 0.05
    assert rps["market_rps"] == 0.145
    assert rps["rps_delta_model_minus_market"] == -0.095


def test_rps_requires_complete_1x2_vector():
    rows = [
        _row(family="1X2", market="Match Winner", selection="Home", p_model=0.70, p_market=0.50, outcome_class="HOME"),
        _row(family="1X2", market="Match Winner", selection="Draw", p_model=0.20, p_market=0.30, outcome_class="HOME"),
    ]
    report = v.build_report(rows)

    assert report["rps_1x2"]["scored_1x2_groups"] == 0
    assert report["rps_1x2"]["excluded_groups"]["INCOMPLETE_1X2_VECTOR"] == 1


def test_canonical_ft_settlement_can_supply_1x2_rps_outcome_class():
    rows = [
        _row(fixture_id=7, family="1X2", market="Match Winner", selection="Home", p_model=0.60, p_market=0.50),
        _row(fixture_id=7, family="1X2", market="Match Winner", selection="Draw", p_model=0.25, p_market=0.30),
        _row(fixture_id=7, family="1X2", market="Match Winner", selection="Away", p_model=0.15, p_market=0.20),
    ]
    settlement = [
        {
            "fixture_id": 7,
            "market_family": "FT_1X2",
            "market": "Match Winner",
            "selection": "Home",
            "canonical_ft_market": True,
            "settled": True,
            "settlement_status": "WIN",
            "result": {
                "status": "FT",
                "score": {"fulltime": {"home": 2, "away": 1}},
            },
        }
    ]
    report = v.build_report(rows, settlement_rows=settlement)

    assert report["rps_1x2"]["scored_1x2_groups"] == 1


def test_strict_close_clv_uses_exact_identity_and_existing_canonical_fields():
    source = [
        _row(
            fixture_id=9,
            family="FT_TOTALS",
            market="Goals Over/Under",
            selection="Under",
            line=2.5,
            stage="T-20",
            p_model=0.60,
            p_market=0.55,
        )
    ]
    close = [
        {
            "fixture_id": 9,
            "market_family": "FT_TOTALS",
            "market": "Goals Over/Under",
            "selection": "Under",
            "line": 2.5,
            "stage": "T-20",
            "is_true_closing_line": True,
            "probability_comparable_same_line": True,
            "probability_clv": 3.0,
            "price_clv": 4.5,
            "closing_fair_probability": 0.58,
            "closing_timestamp": "2026-09-30T01:00:00Z",
            "closing_provider_update": "2026-09-30T00:59:00Z",
        }
    ]
    report = v.build_report(source, clv_rows=close)

    assert report["clv"]["matched_true_close_rows"] == 1
    assert report["clv"]["mean_probability_clv_pp"] == 3.0
    assert report["clv"]["mean_price_clv_pct"] == 4.5
    assert report["clv"]["mean_model_residual_to_true_close"] == 0.02


def test_clv_does_not_match_non_true_close_or_different_stage():
    source = [_row(fixture_id=9, family="FT_TOTALS", selection="Under", line=2.5, stage="T-20")]
    closes = [
        {
            "fixture_id": 9,
            "market_family": "FT_TOTALS",
            "selection": "Under",
            "line": 2.5,
            "stage": "T-20",
            "is_true_closing_line": False,
            "probability_clv": 9.0,
        },
        {
            "fixture_id": 9,
            "market_family": "FT_TOTALS",
            "selection": "Under",
            "line": 2.5,
            "stage": "T-10",
            "is_true_closing_line": True,
            "probability_clv": 8.0,
        },
    ]
    report = v.build_report(source, clv_rows=closes)

    assert report["clv"]["matched_true_close_rows"] == 0


def test_family_and_probability_buckets_are_descriptive_only():
    report = v.build_report(
        [
            _row(family="BTTS", p_model=0.51, p_market=0.50),
            _row(fixture_id=2, family="FT_TOTALS", p_model=0.73, p_market=0.68),
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
