from mcp_gateway import product_dashboard_v4 as v


def test_dashboard_renders_control_tower_master_sections_and_escapes_content():
    payload = {
        "status": "CONTROL_TOWER_V1_CONTRACT",
        "pipeline_version": "4.32.9-test",
        "generated_at_utc": "2026-09-23T18:22:39Z",
        "views": {
            "control_tower": {
                "status": "LIVE",
                "system_health": {
                    "runtime": "HEALTHY",
                    "postgres": "HEALTHY",
                    "scheduler": "TICK_OBSERVED",
                    "api_football": "HEALTHY",
                    "api_football_remaining": 7428,
                    "daily_budget_mode": "NORMAL",
                    "galaxy": "NOT_VERIFIED",
                    "last_tick": "2026-09-25T18:53:06-06:00",
                },
                "pipeline": {
                    "fixtures_scanned": 283,
                    "due": 26,
                    "deep_dives": 18,
                    "events": 21,
                    "research_visible": 19,
                    "api_calls": 70,
                    "api_call_cap": 70,
                    "scheduler_schema_version": "1.2.0",
                    "scheduler_mode": "COVERAGE_CATCHUP",
                    "scheduler_unseen_processed": 8,
                    "scheduler_actionable_processed": 4,
                    "scheduler_starvation_count": 406,
                    "scheduler_due_analyzed_pct": 14.71,
                    "scheduler_planned_unique_leagues": 42,
                    "scheduler_urgent_actionable_count": 3,
                    "team_totals_maturation_candidates": 2,
                    "team_totals_later_quote_refreshes": 1,
                    "primary_clv_maturation_candidates": 1,
                    "primary_clv_maturation_refreshed": 0,
                },
                "errors": {"count": 0, "rows": []},
                "validation_gates": [
                    {
                        "key": "phase16_calibration_sample",
                        "label": "Core calibration sample",
                        "current": 407,
                        "target": 300,
                        "unit": "rows",
                    },
                    {
                        "key": "1x2_true_clv",
                        "label": "1X2 True CLV",
                        "current": None,
                        "target": 50,
                        "unit": "rows",
                    },
                ],
                "phases": [
                    {"phase": "14", "name": "Cards", "status": "VALIDATION_GATE_IMPLEMENTED"},
                    {"phase": "15", "name": "Player Props", "status": "VALIDATION_GATE_IMPLEMENTED"},
                    {"phase": "24", "name": "Product", "status": "READ_ONLY_DASHBOARD_IMPLEMENTED"},
                ],
                "production_valid_market_count": 0,
            },
            "todays_slate": {
                "total": 1,
                "rows": [{
                    "home": "<script>alert(1)</script>",
                    "away": "B",
                    "league": "League",
                    "market": "Goals Over/Under",
                    "selection": "Over",
                    "line": 2.5,
                    "price": 2.0,
                    "model_signal": "STRONG",
                    "execution_status": "WAIT_PRICE",
                }],
            },
            "strong_sport_signals": {"total": 1, "rows": []},
            "value_plays": {"total": 0, "rows": []},
            "waiting_for_price": {"total": 1, "rows": []},
            "waiting_for_xi": {"total": 0, "rows": []},
            "team_totals": {"total": 0, "rows": []},
            "first_half": {"total": 0, "rows": []},
            "second_half": {"total": 0, "rows": []},
            "corners": {"total": 0, "rows": []},
            "player_props": {"total": 0, "rows": []},
        },
    }
    html = v.render_dashboard(payload)
    assert "Control Tower" in html
    assert "System Health" in html
    assert "Model Maturity" in html
    assert "Pipeline Errors" in html
    assert "Phases 14–24" in html
    assert "Today&#x27;s Slate" in html
    assert "Strong Sport Signals" in html
    assert "Player Props" in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<script>alert(1)</script>" not in html
    assert "Research-first dashboard" in html
    assert "Read-only operational view" in html
    assert "N/V / 50" in html
    assert "Production-valid markets: 0" in html
    assert "Scheduler mode" in html
    assert "COVERAGE_CATCHUP" in html
    assert "Unseen processed" in html
    assert "Starvation" in html
    assert "TT close candidates" in html
    assert "Primary close candidates" in html
