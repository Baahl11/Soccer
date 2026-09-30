from mcp_gateway import match_detail_v4 as v


def _row(**overrides):
    base = {
        "fixture_id": 1640509,
        "kickoff": "2026-09-29T22:00:00-06:00",
        "stage": "T-60",
        "league": "Friendlies",
        "country": "World",
        "home": "Papua New Guinea",
        "away": "Solomon Islands",
        "data_tier": "B",
        "market_family": "BTTS",
        "market": "Both Teams Score",
        "selection": "Yes",
        "line": None,
        "price": 1.7,
        "bookmaker": "1xBet",
        "prob_edge_pp": -4.8791,
        "p_market_fair": 0.545455,
        "p_raw": 0.496664,
        "p_model_calibrated": 0.55063869,
        "reason": "SPORTING_SCREEN_PASS",
        "model_signal": "STRONG",
        "model_signal_score": 72.6,
        "execution_status": "RESEARCH_ONLY",
        "blockers": ["RESEARCH_ONLY", "SPORTING_SCREEN_PASS"],
        "model_signal_basis": "SPORTING_SCREEN_ONLY_NO_PRICE",
        "execution_status_basis": "READINESS_BLOCKERS_AND_MARKET_AVAILABILITY",
        "price_resolution_status": "PRICE_API_RESOLVED",
        "price_resolution_source": "API_FOOTBALL_ODDS_V3",
        "price_resolution_provider_update": "2026-09-30T02:20:14+00:00",
        "price_resolution_reference_policy": "MEDIAN_PRICE_NEAREST_BOOKMAKER",
        "phase16_calibration_status": "RESEARCH_CALIBRATION_APPLIED",
    }
    base.update(overrides)
    return base


def _payload(rows_by_view):
    views = {
        "strong_sport_signals": {"rows": [], "total": 0},
        "value_plays": {"rows": [], "total": 0},
        "waiting_for_price": {"rows": [], "total": 0},
        "waiting_for_xi": {"rows": [], "total": 0},
        "todays_slate": {"rows": [], "total": 0},
    }
    for key, rows in rows_by_view.items():
        views[key] = {"rows": rows, "total": len(rows)}
    return {"views": views}


def test_v217_uses_confirmed_persisted_fields_and_probability_units():
    rendered = v.render_fragment(_payload({"strong_sport_signals": [_row()]}))
    assert "Premium Signal Detail" in rendered
    assert "Papua New Guinea vs Solomon Islands" in rendered
    assert "1xBet" in rendered
    assert "49.7%" in rendered
    assert "55.1%" in rendered
    assert "54.5%" in rendered
    assert "-4.88 pp" in rendered
    assert "API_FOOTBALL_ODDS_V3" in rendered
    assert "RESEARCH_ONLY · SPORTING_SCREEN_PASS" in rendered
    assert "RESEARCH_CALIBRATION_APPLIED" in rendered


def test_v217_deduplicates_same_market_row_across_views_and_respects_limit():
    duplicate = _row()
    rows = [duplicate] + [_row(fixture_id=2000 + i, home=f"H{i}", away=f"A{i}") for i in range(12)]
    payload = _payload({
        "strong_sport_signals": rows,
        "todays_slate": [duplicate],
    })
    selected = v.build_detail_rows(payload)
    assert len(selected) == v.MAX_DETAIL_ROWS
    identities = [v._identity(row) for row in selected]
    assert len(identities) == len(set(identities))
    assert selected[0]["_detail_source_view"] == "strong_sport_signals"


def test_v217_missing_values_remain_nv_and_html_is_escaped():
    row = _row(
        home="<script>alert(1)</script>",
        price=None,
        bookmaker=None,
        p_model_calibrated=None,
        p_market_fair=None,
        p_raw=None,
        prob_edge_pp=None,
        blockers=[],
        price_resolution_source=None,
    )
    rendered = v.render_fragment(_payload({"waiting_for_price": [row]}))
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in rendered
    assert "<script>alert(1)</script>" not in rendered
    assert rendered.count("N/V") >= 6
    assert "PREMIUM PREVIEW" in rendered


def test_v217_does_not_add_provider_or_auth_behavior():
    module_text = open("mcp_gateway/match_detail_v4.py", encoding="utf-8").read()
    assert "API_FOOTBALL" not in module_text.replace("API_FOOTBALL_ODDS_V3", "")
    assert "httpx" not in module_text
    assert "requests" not in module_text
    assert "billing" not in module_text.lower()
    assert "entitlement" in module_text.lower()
