from mcp_gateway import ui_golden_master_v1


def test_match_center_golden_master_is_visual_prototype_only():
    html = ui_golden_master_v1._html()
    assert "VISUAL GOLDEN MASTER · SAMPLE DATA" in html
    assert "DESIGN LAB ONLY" in html
    assert "illustrative values" in html
    assert "Match Result Probability" in html
    assert "Expected Goals (λ)" in html
    assert "Sport Profile" in html
    assert "Score Matrix (FT)" in html
    assert "Goal Distribution" in html
    assert "MODEL VS MARKET" in html
