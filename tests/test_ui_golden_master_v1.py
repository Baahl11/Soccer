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


def test_golden_master_matches_approved_dashboard_structure():
    html = ui_golden_master_v1._html()
    assert "Edge Gap" in html
    assert "Over / Under 2.5 Goals" in html
    assert "Last update 2m ago" in html
    assert "Back to matches" in html
    assert "FOOTBALL ONLY" in html


def test_golden_master_uses_restrained_glass_depth():
    html = ui_golden_master_v1._html()
    assert "backdrop-filter:blur(12px)" in html
    assert "backdrop-filter:blur(16px)" in html
    assert "rgba(55,112,140,.34)" in html
    assert "linear-gradient(180deg,#58b8e8,#2e96c7)" in html


def test_golden_master_includes_team_gf_ga_heatmaps():
    html = ui_golden_master_v1._html()
    assert "FT + SCORING PROFILE" in html
    assert "Arsenal" in html and "Brighton" in html
    assert "GF × GA" in html
    assert "X = GF · Y = GA" in html
