from mcp_gateway import product_dashboard_v4 as v


def test_dashboard_renders_master_sections_and_escapes_content():
    payload = {
        "status": "API_VIEW_CONTRACT_IMPLEMENTED_UI_PENDING",
        "pipeline_version": "4.29.0-phase24.product_views",
        "generated_at_utc": "2026-09-23T18:22:39Z",
        "views": {
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
            "strong_sport_signals": {"total": 0, "rows": []},
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
    assert "Today's Slate" in html
    assert "Strong Sport Signals" in html
    assert "Player Props" in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<script>alert(1)</script>" not in html
    assert "Research-first dashboard" in html
    assert "Read-only operational view" in html
