from mcp_gateway import product_dashboard_v4 as v


def _payload():
    return {
        "status": "CONTROL_TOWER_V1_CONTRACT",
        "pipeline_version": "4.38.0-test",
        "generated_at_utc": "2026-09-30T03:10:00Z",
        "views": {
            "strong_sport_signals": {
                "total": 1,
                "rows": [{
                    "home": "Alpha",
                    "away": "Beta",
                    "league": "Test League",
                    "market": "Goals Over/Under",
                    "selection": "Over",
                    "line": 2.5,
                    "price": 1.95,
                    "model_signal": "STRONG",
                    "calibrated_edge_pp": 4.8,
                    "execution_status": "READY",
                    "stage": "T-20",
                }],
            },
            "value_plays": {"total": 0, "rows": []},
            "waiting_for_price": {"total": 1, "rows": []},
            "waiting_for_xi": {"total": 2, "rows": []},
            "todays_slate": {"total": 0, "rows": []},
            "team_totals": {"total": 0, "rows": []},
            "first_half": {"total": 0, "rows": []},
            "second_half": {"total": 0, "rows": []},
            "corners": {"total": 0, "rows": []},
            "player_props": {"total": 0, "rows": []},
            "control_tower": {
                "status": "LIVE",
                "system_health": {"api_football_remaining": 7000},
                "pipeline": {
                    "fixtures_scanned": 400,
                    "scheduler_mode": "COVERAGE_CATCHUP",
                    "scheduler_unseen_processed": 10,
                    "scheduler_starvation_count": 3,
                    "team_totals_maturation_candidates": 2,
                    "primary_clv_maturation_candidates": 4,
                    "api_calls": 55,
                    "api_call_cap": 70,
                },
                "errors": {"count": 0, "rows": []},
                "validation_gates": [{
                    "label": "1X2 True CLV",
                    "current": 47,
                    "target": 50,
                    "unit": "rows",
                }],
                "maturity_snapshot": {
                    "status": "OK",
                    "reports_loaded": 12,
                    "reports_expected": 12,
                    "maturation_watchdogs": {
                        "ok_count": 3,
                        "watch_count": 1,
                        "not_verified_count": 0,
                        "watchdogs": {},
                    },
                    "maturation_control_tower": {
                        "families": [{
                            "label": "1X2",
                            "evidence_kind": "TRUE_CLV",
                            "current": 47,
                            "target": 50,
                            "status": "MATURING",
                            "blocker": "1X2_TRUE_CLV_47_LT_50",
                        }],
                    },
                },
                "phases": [],
                "production_valid_market_count": 0,
            },
        },
    }


def test_v214_renders_subscriber_first_decision_surface_without_changing_data_contract():
    html = v.render_dashboard(_payload())
    assert "SUBSCRIBER PREVIEW" in html
    assert "Find the signal." in html
    assert "Decision Desk".upper() in html.upper()
    assert "Strong Sport Signals" in html
    assert "Alpha <span>vs</span> Beta" in html
    assert "+4.8 pp" in html
    assert "Price watch" not in html  # no synthetic waiting row was invented
    assert "Model Maturity" in html
    assert "1X2_TRUE_CLV_47_LT_50" in html
    assert "Control Tower · operator diagnostics" in html
    assert "production_promotion_allowed&quot;: false" in html


def test_v214_keeps_missing_values_explicit_and_escapes_untrusted_text():
    payload = _payload()
    payload["views"]["strong_sport_signals"]["rows"][0]["home"] = "<img src=x onerror=alert(1)>"
    payload["views"]["strong_sport_signals"]["rows"][0]["price"] = None
    payload["views"]["strong_sport_signals"]["rows"][0]["calibrated_edge_pp"] = None

    html = v.render_dashboard(payload)
    assert "&lt;img src=x onerror=alert(1)&gt;" in html
    assert "<img src=x onerror=alert(1)>" not in html
    assert "N/V" in html
