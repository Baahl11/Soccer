from mcp_gateway.presentation_tables import render_match_tables


def test_v4_004_research_board_surfaces_model_and_execution_separately():
    rows = [{
        "classification": "WATCH",
        "kickoff": "2026-09-22T20:00:00+00:00",
        "country": "England",
        "competition": "EFL Trophy",
        "home": "Walsall",
        "away": "Stevenage",
        "data_tier": "B",
        "model_signal": "VERY_STRONG",
        "side_score": 82.0,
        "goals_score": 61.0,
        "two_way_score": 58.0,
        "execution_status": "WAIT_PRICE",
        "blockers": ["WAIT_PRICE", "SPORTING_SCREEN_PASS"],
        "availability_confidence": 0.8,
        "market": None,
        "price": None,
        "reason": "SPORTING_SCREEN_PASS",
    }]

    rendered = render_match_tables(rows)

    assert "MODEL_SIGNAL" in rendered
    assert "EXECUTION_STATUS" in rendered
    assert "Blockers" in rendered
    assert "VERY_STRONG" in rendered
    assert "WAIT_PRICE" in rendered
    assert "WAIT_PRICE; SPORTING_SCREEN_PASS" in rendered
    assert "S 82.0 / G 61.0 / 2W 58.0" in rendered
