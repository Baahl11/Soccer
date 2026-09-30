from mcp_gateway import public_performance_v4 as v


def _reports():
    return {
        "market_performance": {
            "minimum_sample_policy": {"directional_read": 20},
            "by_market_family_and_classification": {
                "FT_TOTALS": {
                    "BET": {"settled": 7, "win": 3, "loss": 4, "push": 0, "ungraded": 0, "roi_units": -0.5652, "hit_rate_ex_push": 0.4286},
                    "LEAN": {"settled": 8, "win": 8, "loss": 0, "push": 0, "ungraded": 0, "roi_units": 3.91, "hit_rate_ex_push": 1.0},
                },
                "2H_TOTALS": {
                    "BET": {"settled": 2, "win": 0, "loss": 2, "push": 0, "ungraded": 0, "roi_units": -0.72, "hit_rate_ex_push": 0.0},
                },
                "2H_BTTS": {
                    "BET": {"settled": 1, "win": 0, "loss": 1, "push": 0, "ungraded": 0, "roi_units": -0.36, "hit_rate_ex_push": 0.0},
                },
            },
        },
        "settlement_coverage": {
            "settlement_rows": 29,
            "source_actionable_rows": 38,
            "settlement_coverage_rate": 0.7632,
            "backlog_rows": 9,
        },
        "true_clv": {
            "status": "ACTIVE_TRUE_CLV_SAMPLE",
            "true_clv_rows": 122,
            "avg_true_clv_probability_pp": 0.1508,
            "positive_clv": 6,
            "negative_clv": 2,
            "flat_clv": 114,
            "same_book_true_clv_rows": 122,
            "minimum_sample_policy": {"directional_read": 50},
        },
    }


def test_v216_headline_is_bet_only_and_excludes_lean_results():
    snapshot = v._build(_reports(), {})
    bet = snapshot["bet_only"]
    lean = snapshot["research_lean"]
    assert bet["settled"] == 10
    assert bet["win"] == 3
    assert bet["loss"] == 7
    assert bet["roi_units"] == -1.6452
    assert bet["sample_status"] == "SAMPLE_TOO_SMALL"
    assert lean["settled"] == 8
    assert lean["win"] == 8
    assert snapshot["provider_requests_added"] == 0
    assert snapshot["historical_reconstruction_performed"] is False
    assert snapshot["settlement_backfill_performed"] is False


def test_v216_renders_only_verified_metrics_and_sample_warning():
    snapshot = v._build(_reports(), {})
    rendered = v.render_fragment(snapshot)
    assert "BET-only Track Record" in rendered
    assert "SAMPLE_TOO_SMALL" in rendered
    assert ">10</strong>" in rendered
    assert "3-7-0" in rendered
    assert "-1.65u" in rendered
    assert "76.3%" in rendered
    assert ">122</strong>" in rendered
    assert "+0.15 pp" in rendered
    assert "LEAN rows are excluded" in rendered


def test_v216_missing_state_does_not_invent_performance(monkeypatch):
    monkeypatch.delenv("STATE_REPO", raising=False)
    monkeypatch.delenv("STATE_BRANCH", raising=False)
    v._CACHE = None
    snapshot = v.load_snapshot(force=True)
    assert snapshot["status"] == "NO_VERIFIED_SETTLEMENT_SAMPLE"
    assert snapshot["provider_requests_added"] == 0
    assert snapshot["historical_reconstruction_performed"] is False
    assert snapshot["settlement_backfill_performed"] is False
