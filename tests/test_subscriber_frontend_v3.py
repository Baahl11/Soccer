from mcp_gateway import subscriber_frontend_v3


def test_v3_contract_is_preview_only_and_sport_first():
    c = subscriber_frontend_v3.contract()
    assert c["surface"] == "/app-v3"
    assert c["sport_first"] is True
    assert c["market_second"] is True
    assert c["provider_requests_added"] == 0
    assert c["canonical_bet_logic_changed"] is False
    assert c["model_weights_changed"] is False
    assert c["production_promotion_allowed"] is False


def test_v3_html_contains_real_visual_match_components():
    html = subscriber_frontend_v3._html()
    assert "SPORT-FIRST READ" in html
    assert "Score Matrix (FT)" in html
    assert "Goal Distribution" in html
    assert "Match Result Probability" in html
    assert "Expected Goals" in html
    assert "MARKET CONTEXT · SECONDARY LAYER" in html
    assert "America/Mexico_City" in html
    assert "/app/api/v2/match/" in html
    assert "/app/api/v2/today" in html


def test_v3_normalizes_nested_registry_fixture_rows():
    html = subscriber_frontend_v3._html()
    assert "url.includes('/app/api/v2/today')" in html
    assert "Object.assign({}, row, row.fixture)" in html
    assert "/app-v3/match/" in html


def test_v3_injects_mockup_match_intelligence_board():
    html = subscriber_frontend_v3._html()
    assert "MATCH INTELLIGENCE" in html
    assert "Sport model snapshot" in html
    assert "Result Probability" in html
    assert "Edge Gap" in html
    assert "Sport Profile" in html
    assert "Goal Distribution" in html
    assert "url.includes('/app/api/v2/match/')" in html
