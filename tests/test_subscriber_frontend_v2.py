from __future__ import annotations

from mcp_gateway import subscriber_frontend_v2


def test_fe2_shell_contains_no_design_time_mock_data():
    html = subscriber_frontend_v2.render()

    for forbidden in (
        "Arsenal vs Brighton",
        "Man City vs Nottm Forest",
        "const rows=[",
        "const markets=[",
        "Mockup Preview",
    ):
        assert forbidden not in html


def test_fe2_primary_information_architecture_is_customer_first():
    html = subscriber_frontend_v2.render()

    for label in (
        "Today",
        "Picks",
        "Leans",
        "Matches",
        "Performance",
        "My Edge",
        "Account",
    ):
        assert label in html

    assert "Control Tower" not in html
    assert "Research Lab" not in html


def test_fe2_visually_preserves_sport_first_market_second_separation():
    html = subscriber_frontend_v2.render()

    assert "RAW SPORT" in html
    assert "MARKET SHRUNK" in html
    assert "MARKET FAIR" in html
    assert "PROB EDGE" in html
    assert "Sport first. Market second." in html


def test_fe2_consumes_only_v2_customer_contract_for_decision_data():
    html = subscriber_frontend_v2.render()

    assert '"/app/api/v2"' in html
    assert "api('/today')" in html
    assert "api('/performance')" in html
    assert "api('/match/'" in html
    assert "/app-preview/data" not in html
    assert "/app/data" not in html


def test_fe2_preview_does_not_enable_billing_mutations():
    html = subscriber_frontend_v2.render()

    assert "create-checkout-session" not in html
    assert "create-customer-portal" not in html
    assert "Subscription infrastructure already exists" in html


def test_fe2_has_one_customer_bootstrap_controller():
    html = subscriber_frontend_v2.render()

    assert html.count('id="soccer-edge-v2-app"') == 1
    assert html.count('id="boot"') == 1
    assert "MutationObserver" not in html
    assert "setTimeout(" not in html


def test_fe2_contract_preserves_engine_firewall():
    contract = subscriber_frontend_v2.contract()

    assert contract["mock_data"] is False
    assert contract["single_bootstrap_controller"] is True
    assert contract["operator_surfaces_in_primary_navigation"] is False
    assert contract["billing_mutations_enabled"] is False
    assert contract["frontend_creates_bet_or_lean"] is False
    assert contract["provider_requests_added"] == 0
    assert contract["canonical_bet_logic_changed"] is False
    assert contract["model_weights_changed"] is False
    assert contract["production_promotion_allowed"] is False
