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


def test_golden_master_compacts_matrices_and_refines_xg_and_goals():
    html = ui_golden_master_v1._html()
    assert "TOTAL XG" in html
    assert "HOME DELTA" in html
    assert "MODEL OVER 2.5" in html
    assert "Market fair 55.3%" in html
    assert "aspect-ratio:1.34/1" in html


def test_golden_master_probability_hierarchy_and_color_contrast_pass():
    html = ui_golden_master_v1._html()
    assert "grid-template-columns:1.18fr .91fr .91fr" in html
    assert "prob.home b{font-size:15px" in html
    assert "home-p5{background:#42a087" in html
    assert "away-p5{background:#3d83a1" in html


def test_golden_master_final_polish_pass():
    html = ui_golden_master_v1._html()
    assert "Golden Master final polish" in html
    assert "align-items:center" in html
    assert ".ou-gauge .lab.market{left:51%" in html
    assert "grid-template-columns:minmax(0,1fr) 146px" in html
