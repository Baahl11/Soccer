from __future__ import annotations

from mcp_gateway import subscriber_product_v235


def test_v235_mobile_nav_exposes_complete_product_information_architecture():
    html = subscriber_product_v235.product_html()
    for label in ("Today", "Feed", "Match", "Markets", "Performance", "My Edge", "Tower", "Lab", "Account"):
        assert label in html
    assert "SOCCER_V235_VISUAL_MOBILE_STYLE" in html
    assert "SOCCER_V235_VISUAL_MOBILE_SCRIPT" in html
    assert "safe-area-inset-bottom" in html
    assert "scroll-snap-type" in html


def test_v235_owner_gets_pro_presentation_without_mutating_input_entitlement():
    source = {
        "authenticated": True,
        "effective_plan": "FREE",
        "owner": True,
        "user": {"role": "OWNER"},
    }
    result = subscriber_product_v235._owner_view_entitlement(source)
    assert source["effective_plan"] == "FREE"
    assert result["effective_plan"] == "PRO"
    assert result["owner"] is True
    assert result["effective_plan_reason"] == "OWNER_ADMIN_PRESENTATION_ACCESS"


def test_v235_regular_free_user_is_not_promoted_in_presentation():
    source = {
        "authenticated": True,
        "effective_plan": "FREE",
        "owner": False,
        "admin": False,
        "user": {"role": "USER"},
    }
    result = subscriber_product_v235._owner_view_entitlement(source)
    assert result["effective_plan"] == "FREE"
    assert result.get("effective_plan_reason") != "OWNER_ADMIN_PRESENTATION_ACCESS"


def test_v235_preserves_frontend_firewall():
    c = subscriber_product_v235.contract()
    assert c["desktop_mockup_shell_preserved"] is True
    assert c["mobile_horizontal_navigation"] is True
    assert c["safe_area_support"] is True
    assert c["owner_admin_presentation_access"] == "PRO_VIEW_WITHOUT_BILLING_MUTATION"
    assert c["provider_requests_added"] == 0
    assert c["canonical_bet_logic_changed"] is False
    assert c["model_weights_changed"] is False
    assert c["production_promotion_allowed"] is False
