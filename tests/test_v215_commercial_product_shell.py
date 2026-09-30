from mcp_gateway import commercial_shell_v4, product_dashboard_v4


def _payload():
    return {
        "status": "CONTROL_TOWER_V1_CONTRACT",
        "pipeline_version": "4.38.0-test",
        "generated_at_utc": "2026-09-30T03:30:00Z",
        "views": {
            "strong_sport_signals": {"total": 2, "rows": []},
            "value_plays": {"total": 1, "rows": []},
            "waiting_for_price": {"total": 3, "rows": []},
            "waiting_for_xi": {"total": 4, "rows": []},
            "todays_slate": {"total": 0, "rows": []},
            "team_totals": {"total": 0, "rows": []},
            "first_half": {"total": 0, "rows": []},
            "second_half": {"total": 0, "rows": []},
            "corners": {"total": 0, "rows": []},
            "player_props": {"total": 0, "rows": []},
            "performance": {
                "oos_status": "MATURING",
                "promotion_status": "RESEARCH_HOLD",
                "risk_status": "VALIDATION",
            },
            "control_tower": {
                "status": "LIVE",
                "system_health": {},
                "pipeline": {},
                "errors": {"count": 0, "rows": []},
                "validation_gates": [],
                "phases": [],
                "maturity_snapshot": {},
                "production_valid_market_count": 0,
            },
        },
    }


def test_commercial_shell_reports_entitlement_readiness_without_claiming_billing(monkeypatch):
    monkeypatch.setenv("SOCCER_SUPABASE_URL", "https://soccer-edge.supabase.co")
    monkeypatch.setenv("SOCCER_SUPABASE_PUBLISHABLE_KEY", "sb_publishable_example")
    fragment = commercial_shell_v4.render_membership_fragment(_payload())
    assert "Soccer Edge Membership" in fragment
    assert "ENTITLEMENTS READY" in fragment
    assert "Explorer" in fragment
    assert "Edge Pro" in fragment
    assert "Supabase Auth is connected" in fragment
    assert "Billing is intentionally still disabled" in fragment
    assert "&quot;auth_enabled&quot;: true" in fragment
    assert "&quot;billing_enabled&quot;: false" in fragment
    assert "&quot;entitlements_enforced&quot;: true" in fragment
    assert "&quot;provider_requests_added&quot;: 0" in fragment


def test_commercial_shell_does_not_claim_auth_when_runtime_is_unconfigured(monkeypatch):
    monkeypatch.delenv("SOCCER_SUPABASE_URL", raising=False)
    monkeypatch.delenv("SOCCER_SUPABASE_PUBLISHABLE_KEY", raising=False)
    fragment = commercial_shell_v4.render_membership_fragment(_payload())
    assert "AUTH REQUIRED" in fragment
    assert "no Supabase Auth configuration" in fragment
    assert "&quot;auth_enabled&quot;: false" in fragment
    assert "&quot;entitlements_enforced&quot;: true" in fragment


def test_v215_wraps_existing_operator_dashboard_instead_of_replacing_it():
    assert hasattr(product_dashboard_v4, "_v215_operator_render_dashboard")
    rendered = product_dashboard_v4.render_dashboard(_payload())
    assert "Soccer Edge Membership" in rendered
    assert "Control Tower" in rendered
    assert "System Health" in rendered
    assert "Pipeline Errors" in rendered
    assert rendered.count("Soccer Edge Membership") == 1


def test_v215_metrics_come_only_from_product_payload():
    rendered = commercial_shell_v4.render_membership_fragment(_payload())
    assert ">2</strong>" in rendered
    assert ">1</strong>" in rendered
    assert ">3</strong>" in rendered
    assert ">4</strong>" in rendered
    assert "MATURING" in rendered
    assert "RESEARCH_HOLD" in rendered
    assert "VALIDATION" in rendered
