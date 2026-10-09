from __future__ import annotations

from mcp_gateway import subscriber_contract_v2, subscriber_frontend_v2


def test_maturation_contract_preserves_research_firewall():
    source = {
        "status": "PARTIAL",
        "families": [{
            "label": "Corners", "stage": "FORMATION MATURATION + CLV",
            "model_evidence": {"current": 44, "target": 100, "unit": "formation-adjusted fixtures"},
            "mapped_rows": 0, "priced_rows": 0, "true_clv_rows": 0,
            "true_clv_target": 50, "blockers": ["FORMATION_ADJUSTED_44_LT_100"],
            "source": "v4_022_corners_oos_validation.json",
        }],
        "errors": {"Player Props": "source unavailable"},
        "comparable_true_clv_rows": 595,
        "minimum_true_close_rows": 50,
    }
    result = subscriber_contract_v2.build_maturity_contract(source)
    assert result["status"] == "PARTIAL"
    assert result["families"][0]["model_evidence"]["current"] == 44
    assert result["families"][0]["true_clv_rows"] == 0
    assert result["errors"]["Player Props"] == "source unavailable"
    assert result["evidence_scope"] == "RESEARCH_ONLY_NOT_PRODUCTION_ELIGIBILITY"
    assert result["provider_requests_added"] == 0
    assert result["production_promotion_allowed"] is False
    assert result["canonical_bet_logic_changed"] is False
    assert result["model_weights_changed"] is False


def test_maturation_contract_handles_missing_research_without_fake_zeros():
    d = subscriber_contract_v2.build_maturity_contract({"status": "UNAVAILABLE", "families": []})
    assert d["status"] == "UNAVAILABLE"
    assert d["families"] == []
    assert d["comparable_true_clv_rows"] is None


def test_v2_renders_read_only_maturation_subsection():
    html = subscriber_frontend_v2.render()
    assert "api('/maturity')" in html
    assert "Market Maturation" in html
    assert "Research Only" in html
    assert "Production promotion:" in html
    assert "NOT VERIFIED" in html
    assert "loadMaturity()" in html
    assert "MutationObserver" not in html
