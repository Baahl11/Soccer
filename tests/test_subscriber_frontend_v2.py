from __future__ import annotations

from mcp_gateway import subscriber_frontend_v2


def test_fe2_shell_contains_no_design_time_mock_data():
    html = subscriber_frontend_v2.render()

    for forbidden in (
        "Arsenal vs Brighton",
        "Man City vs Nottm Forest",
        "Mockup Preview",
        "const markets=[['1X2'",
        "const rows=[['Arsenal vs Brighton'",
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


def test_fe2_billing_surface_keeps_price_ids_out_and_launch_default_off():
    html = subscriber_frontend_v2.render()
    contract = subscriber_frontend_v2.contract()

    assert "price_1ULUp1BE8LWeAYkQBNijFI03" not in html
    assert "price_1ULUp7BE8LWeAYkQbanCnbG7" not in html
    assert contract["billing_mutations_enabled"] is False
    assert contract["public_billing_launch_default"] is False
    assert contract["browser_sends_stripe_price_id"] is False

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

def test_fe2_full_slate_exposes_coverage_and_insufficient_data_copy():
    html = subscriber_frontend_v2.render()

    assert "Every eligible fixture stays visible, even when analysis is incomplete" in html
    assert "Insufficient data · fixture only" in html
    assert "Sport data: " in html
    assert 'data-intel="' in html

def test_fe2_every_slate_fixture_is_openable_for_data_review():
    html = subscriber_frontend_v2.render()

    assert "Open data →" in html
    assert "Sport-first read" in html
    assert "Scoring profile" in html
    assert "Key takeaways" in html
    assert 'class="slate-row" href="' in html
    assert "matchPath(f.fixture_id||'')" in html
    assert '[data-intel="1"]' not in html

def test_fe2_match_navigation_uses_dedicated_url_and_direct_boot_route():
    html = subscriber_frontend_v2.render()

    assert "'/app/match/'" in html
    assert "routeMatchId()" in html
    assert "history.pushState" in html
    assert "← Matches" in html
    assert "matchesBrowsePanel" in html

def test_fe2_identity_enrichment_chunks_large_slates_and_shows_relational_evidence():
    html = subscriber_frontend_v2.render()

    assert "for(let i=0;i<ids.length;i+=80)" in html
    assert "Latest market activity" in html
    assert "Market snapshots" in html
    assert "Model runs" in html
    assert "Evidence quality" in html


def test_fe2_match_detail_is_summary_first_and_mobile_safe():
    html = subscriber_frontend_v2.render()

    assert "intel-metrics" in html
    assert "intel-lens" in html
    assert "intel-tabs" in html
    assert "intel-accordion" in html
    assert "overflow-x:auto" in html
    assert "Raw current-snapshot market table" in html
    assert "Input coverage" in html
    assert "Freshness" in html
    assert "Match result probability" in html
    assert "Expected goals" in html


def test_fe2_match_intelligence_is_sport_first_not_market_first():
    html = subscriber_frontend_v2.render()

    assert "Sport-first read" in html
    assert "Scoring profile" in html
    assert "Team form & scoring" in html
    assert "Home form" in html
    assert "Market layer" in html
    assert "context only · not an edge by itself" in html
    assert "Market data exists, but verified sporting evidence is not sufficient" in html


def test_fe2_visible_sport_input_count_matches_rendered_inputs():
    html = subscriber_frontend_v2.render()

    assert "CUSTOMER_SPORT_FEATURES" in html
    assert "Verified sporting inputs" in html
    assert "visible inputs" in html
    assert "core sport inputs" in html
    assert "No customer-facing sporting inputs are verified" in html


def test_fe2_kickoff_display_is_pinned_to_mexico_central_time():
    html = subscriber_frontend_v2.render()

    assert "APP_TIMEZONE='America/Mexico_City'" in html
    assert "timeZone:APP_TIMEZONE" in html


def test_fe2_match_center_visual_components_follow_blueprint_phase_1_and_2():
    html = subscriber_frontend_v2.render()

    for component in (
        "visual-grid",
        "coverage-gauge",
        "rate-bars",
        "form-dots",
        "prob-tiles",
        "prob-stack",
        "takeaway-list",
        "sport-stat-grid",
    ):
        assert component in html

    assert "Market rows do not increase it." in html
    assert "values are not labeled xG unless a verified xG source exists." in html
    assert "No synthetic probabilities." not in html


def test_fe2_match_center_rejects_zero_sample_team_stats_as_verified():
    html = subscriber_frontend_v2.render()

    assert "source==='API_FOOTBALL_TEAM_STATS'" in html
    assert "n!==null&&n<=0" in html
    assert "Market rows do not increase it." in html


def test_fe2_match_center_mobile_uses_premium_dense_composition():
    html = subscriber_frontend_v2.render()

    assert "match-brand-row" in html
    assert "mini-visual-grid" in html
    assert "form-spark" in html
    assert "goal-card" in html
    assert "body:has(#matches.active .match-detail) .topbar" in html
