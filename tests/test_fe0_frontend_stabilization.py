from __future__ import annotations

from mcp_gateway import (
    subscriber_preview_v230,
    subscriber_product_v233,
    subscriber_product_v234,
    subscriber_product_v235,
    subscriber_today_v236,
    subscriber_visual_v237,
)


def test_fe0_first_paint_is_gated_and_mock_bootstrap_is_retired():
    html = subscriber_preview_v230._html()
    assert 'class="se-booting"' in html
    assert 'id="seBoot"' in html
    assert 'SOCCER_EDGE_STATIC_SHELL' in html
    assert "const rows=[['Arsenal vs Brighton'" not in html
    assert "const markets=[['1X2'" not in html


def test_fe0_final_product_uses_readiness_events_instead_of_timer_stack():
    html = subscriber_product_v235.product_html()
    for marker in (
        "soccer-edge:preview-ready",
        "soccer-edge:app-data-ready",
        "soccer-edge:v234-ready",
        "soccer-edge:v235-ready",
        "soccer-edge:today-ready",
        "soccer-edge:visual-ready",
        "soccer-edge:boot-ready",
    ):
        assert marker in html
    for legacy in (
        "setTimeout(refreshFullSlate,220)",
        "setTimeout(refreshFullSlate,1200)",
        "[260,1450,3000].forEach",
        "[250,900,1800].forEach",
    ):
        assert legacy not in html


def test_fe0_today_and_visual_layers_have_no_mutation_observer_bootstrap():
    assert "MutationObserver" not in subscriber_today_v236._SCRIPT
    assert "MutationObserver" not in subscriber_visual_v237._SCRIPT


def test_fe0_single_document_and_single_initial_active_page():
    html = subscriber_product_v235.product_html()
    assert html.count("<body") == 1
    assert html.count("</body>") == 1
    assert html.count('class="page active"') == 1


def test_fe0_engine_firewall_remains_intact():
    for contract in (
        subscriber_product_v233.contract(),
        subscriber_product_v234.contract(),
        subscriber_product_v235.contract(),
        subscriber_today_v236.contract(),
        subscriber_visual_v237.contract(),
    ):
        assert contract["provider_requests_added"] == 0
        assert contract["canonical_bet_logic_changed"] is False
        assert contract["model_weights_changed"] is False
        assert contract["production_promotion_allowed"] is False
