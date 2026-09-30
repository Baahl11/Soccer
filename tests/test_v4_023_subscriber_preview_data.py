from mcp_gateway.subscriber_preview_data_v231 import build_preview_payload
from mcp_gateway.subscriber_preview_live_v231 import _html


def test_preview_payload_wires_persisted_today_and_edge_feed_without_fake_zeroes():
    payload = {
        "status": "ok",
        "generated_at_utc": "2026-09-30T18:10:00Z",
        "fixture_scan_count": 321,
        "deep_dive_processed_count": 7,
        "maturity_snapshot": {
            "gates": {},
            "generated_at_utc": "2026-09-30T18:09:00Z",
            "reports_loaded": 11,
            "reports_expected": 11,
            "errors": {},
            "maturation_control_tower": {
                "status": "WATCH",
                "families": [
                    {"key": "one_x_two", "label": "1X2", "current": 47, "target": 50, "status": "MATURING"}
                ],
                "monitoring": {
                    "report_freshness": {"status": "WATCH", "reason": "STALE"},
                    "evidence_age": {"status": "WATCH", "evidence": {"age_hours": 25.0}},
                },
            },
        },
        "match_table_rows": [
            {
                "fixture_id": 1001,
                "home_team": "Alpha",
                "away_team": "Beta",
                "status": "NS",
                "market_family": "FT_TOTALS",
                "selection": "Over 2.5",
                "p_model_calibrated": 0.60,
                "p_market_fair": 0.50,
                "model_signal": "STRONG",
                "model_signal_score": 0.9,
                "execution_status": "RESEARCH_ONLY",
                "price": 1.90,
            },
            {
                "fixture_id": 1002,
                "home_team": "Gamma",
                "away_team": "Delta",
                "status": "NS",
                "market_family": "BTTS",
                "selection": "BTTS Yes",
                "execution_status": "WAIT_PRICE",
            },
        ],
        "market_mismatch_rows": [],
    }

    out = build_preview_payload(payload)
    assert out["today"]["metrics"]["matches_scanned"] == 321
    assert out["today"]["metrics"]["deep_analyzed"] == 7
    assert out["today"]["metrics"]["strong_edges"] == 1
    assert out["today"]["top_edge"]["match"]["label"] == "Alpha vs Beta"
    assert round(out["today"]["top_edge"]["pricing"]["edge_pp"], 1) == 10.0

    rows = out["edge_feed"]["rows"]
    assert rows[0]["match"]["label"] == "Alpha vs Beta"
    wait = next(row for row in rows if row["match"]["fixture_id"] == 1002)
    assert wait["model"]["probability"] is None
    assert wait["pricing"]["market_probability"] is None
    assert wait["pricing"]["edge_pp"] is None

    tower = out["control_tower"]
    assert tower["runtime_generated_at_utc"] == "2026-09-30T18:10:00Z"
    assert tower["maturation"]["status"] == "WATCH"
    assert tower["maturation"]["families"][0]["current"] == 47
    assert tower["maturation"]["monitoring"]["evidence_age"]["evidence"]["age_hours"] == 25.0

    assert out["provider_requests_added"] == 0
    assert out["canonical_bet_logic_changed"] is False


def test_live_preview_html_fetches_only_preview_contract_and_marks_unwired_metrics():
    html = _html()
    assert "/app-preview/data" in html
    assert "LIVE DATA PREVIEW" in html
    assert "Maturation evidence" in html
    assert "NEXT WIRING" in html
