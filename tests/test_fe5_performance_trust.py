from __future__ import annotations

from mcp_gateway import subscriber_contract_v2
from mcp_gateway import subscriber_frontend_v2


def _track_record():
    return {
        "model_version": "SOCCER_PUBLIC_PERFORMANCE_V4_1.0.0",
        "status": "ACTIVE",
        "generated_at_utc": "2026-10-07T19:00:00+00:00",
        "bet_only": {
            "classification": "BET",
            "settled": 10,
            "win": 3,
            "loss": 7,
            "push": 0,
            "ungraded": 0,
            "roi_units": -1.6452,
            "hit_rate_ex_push": 0.3,
            "sample_status": "SAMPLE_TOO_SMALL",
            "directional_minimum": 20,
            "families": [
                {
                    "market_family": "FT_TOTALS",
                    "settled": 7,
                    "win": 3,
                    "loss": 4,
                    "push": 0,
                    "ungraded": 0,
                    "hit_rate_ex_push": 3 / 7,
                    "roi_units": -0.5652,
                    "status": "RESEARCH_ONLY_SAMPLE_TOO_SMALL",
                }
            ],
        },
        "research_lean": {
            "classification": "LEAN",
            "settled": 8,
            "win": 8,
            "loss": 0,
            "push": 0,
            "ungraded": 0,
            "roi_units": 3.91,
            "hit_rate_ex_push": 1.0,
            "families": [],
        },
        "settlement": {
            "ledger_rows": 29,
            "source_actionable_rows": 38,
            "coverage_rate": 0.7632,
            "backlog_rows": 9,
        },
        "true_clv": {
            "status": "ACTIVE_TRUE_CLV_SAMPLE",
            "rows": 122,
            "avg_probability_pp": 0.1508,
            "positive": 6,
            "negative": 2,
            "flat": 114,
            "same_book_rows": 122,
            "directional_minimum": 50,
        },
        "historical_reconstruction_performed": False,
        "settlement_backfill_performed": False,
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
    }


def _validation():
    return {
        "model_version": "SUBSCRIBER_VALIDATION_METRICS_V231",
        "status": "ACTIVE",
        "rows": [
            {
                "label": "1X2",
                "sample_n": 50,
                "settled": 10,
                "roi_units": -1.1,
                "avg_clv_pp": 0.2,
                "brier": 0.19,
                "status": "RESEARCH_ONLY",
            }
        ],
        "weighted_avg_clv_pp": 0.2,
        "totals": {"rows": 1},
        "errors": {},
    }


def test_fe5_headline_is_canonical_bet_only():
    payload = subscriber_contract_v2.build_performance_contract(
        _track_record(),
        _validation(),
    )

    bet = payload["bet_track_record"]
    lean = payload["research_lean"]

    assert payload["performance_policy"]["headline_scope"] == (
        "CANONICAL_BET_SETTLEMENT_ONLY"
    )
    assert payload["performance_policy"]["leans_in_headline"] is False
    assert bet["settled"] == 10
    assert bet["win"] == 3
    assert bet["loss"] == 7
    assert bet["roi_units"] == -1.6452
    assert lean["settled"] == 8
    assert lean["win"] == 8
    assert lean["roi_units"] == 3.91
    assert lean["headline_eligible"] is False


def test_fe5_small_sample_warning_is_preserved_not_upgraded_by_ui():
    payload = subscriber_contract_v2.build_performance_contract(
        _track_record(),
        _validation(),
    )

    assert payload["bet_track_record"]["sample_status"] == "SAMPLE_TOO_SMALL"
    assert payload["bet_track_record"]["directional_minimum"] == 20
    assert payload["status"] == "VERIFIED_PERFORMANCE_READY"


def test_fe5_validation_is_explicitly_not_realized_customer_performance():
    payload = subscriber_contract_v2.build_performance_contract(
        _track_record(),
        _validation(),
    )

    validation = payload["validation_evidence"]
    assert validation["kind"] == "PERSISTED_VALIDATION_OOS_EVIDENCE"
    assert payload["performance_policy"][
        "research_oos_is_realized_bet_performance"
    ] is False
    assert any("not realized customer BET performance" in note for note in payload["notes"])


def test_fe5_view_performs_no_historical_reconstruction_or_backfill():
    payload = subscriber_contract_v2.build_performance_contract(
        _track_record(),
        _validation(),
    )

    policy = payload["performance_policy"]
    assert policy["historical_reconstruction_performed"] is False
    assert policy["settlement_backfill_performed"] is False
    assert payload["provider_requests_added"] == 0
    assert payload["canonical_bet_logic_changed"] is False
    assert payload["model_weights_changed"] is False
    assert payload["production_promotion_allowed"] is False


def test_fe5_frontend_separates_bet_lean_and_oos_sections():
    html = subscriber_frontend_v2.render()

    assert "BET-only Track Record" in html
    assert "Canonical settled BET rows only" in html
    assert "LEAN research record" in html
    assert "never included in BET headline" in html
    assert "Research / OOS validation" in html
    assert "not realized customer BET performance" in html


def test_fe5_frontend_contract_declares_performance_scope():
    contract = subscriber_frontend_v2.contract()

    assert contract["performance_headline_scope"] == (
        "CANONICAL_BET_SETTLEMENT_ONLY"
    )
    assert contract["research_oos_separate_from_realized_bets"] is True
    assert contract["provider_requests_added"] == 0
    assert contract["canonical_bet_logic_changed"] is False
    assert contract["model_weights_changed"] is False
