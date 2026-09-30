from __future__ import annotations

from mcp_gateway import content_factory_v4


def _product():
    ready = {
        "fixture_id": 101,
        "home": "Home FC",
        "away": "Away FC",
        "league": "Test League",
        "kickoff": "2026-10-01T20:00:00Z",
        "market_family": "BTTS",
        "selection": "YES",
        "price": 1.91,
        "bookmaker": "Verified Book",
        "p_market_fair": 0.535,
        "p_model_calibrated": 0.614,
        "calibrated_edge_pp": 7.9,
        "model_signal": "STRONG",
        "execution_status": "BET",
        "reason": "VERIFIED_TEST_FIXTURE",
    }
    wait = {
        "fixture_id": 102,
        "home": "Alpha",
        "away": "Beta",
        "market_family": "1X2",
        "execution_status": "WAIT_XI",
        "reason": "LINEUP_NOT_CONFIRMED",
        "blockers": ["WAIT_XI"],
    }
    return {
        "views": {
            "strong_sport_signals": {"rows": [ready], "total": 1},
            "value_plays": {"rows": [], "total": 0},
            "waiting_for_price": {"rows": [], "total": 0},
            "waiting_for_xi": {"rows": [wait], "total": 1},
            "todays_slate": {"rows": [ready, wait], "total": 2},
        }
    }


def test_model_vs_market_package_uses_only_persisted_numeric_facts():
    result = content_factory_v4.build_content_packages(_product())
    item = next(p for p in result["packages"] if p["format"] == "MODEL_VS_MARKET")
    facts = item["facts"]
    assert facts["market_probability"] == 0.535
    assert facts["model_probability"] == 0.614
    assert round(facts["edge_pp"], 6) == 7.9
    assert facts["price"] == 1.91
    assert "53.5%" in item["copy"]["en"]["voiceover"]
    assert "61.4%" in item["copy"]["es"]["voiceover"]
    assert item["evidence_policy"]["ai_may_modify_numeric_facts"] is False
    assert item["evidence_policy"]["provider_requests_added"] == 0
    assert item["evidence_policy"]["probability_gap_recomputed_from_persisted_probabilities"] is True


def test_public_gap_ignores_ambiguous_upstream_edge_field():
    product = _product()
    row = product["views"]["strong_sport_signals"]["rows"][0]
    row["p_market_fair"] = 0.1579
    row["p_model_calibrated"] = 0.26963821
    row["calibrated_edge_pp"] = 1.158
    item = next(
        p for p in content_factory_v4.build_content_packages(product)["packages"]
        if p["format"] == "MODEL_VS_MARKET"
    )
    assert round(item["facts"]["edge_pp"], 4) == 11.1738
    assert item["facts"]["edge_display"] == "+11.2 pp"
    assert item["facts"]["source_edge_value"] == 1.158
    assert "+11.2 pp" in item["copy"]["en"]["voiceover"]
    assert "+11.2 pp" in item["copy"]["es"]["voiceover"]


def test_missing_probability_prevents_numeric_content_invention():
    product = _product()
    row = product["views"]["strong_sport_signals"]["rows"][0]
    row.pop("p_model_calibrated")
    product["views"]["todays_slate"]["rows"][0] = row
    result = content_factory_v4.build_content_packages(product)
    assert all(p["format"] != "MODEL_VS_MARKET" for p in result["packages"])


def test_why_we_passed_uses_verified_execution_context():
    result = content_factory_v4.build_content_packages(_product())
    item = next(p for p in result["packages"] if p["format"] == "WHY_WE_PASSED")
    assert item["facts"]["execution_status"] == "WAIT_XI"
    assert item["facts"]["blockers"] == ["WAIT_XI"]
    assert "LINEUP_NOT_CONFIRMED" in item["copy"]["en"]["voiceover"]
    assert "WAIT_XI" in item["copy"]["en"]["voiceover"]
    assert "Pasar también es una decisión" in item["copy"]["es"]["voiceover"]


def test_content_package_has_multi_platform_render_contract():
    item = content_factory_v4.build_content_packages(_product())["packages"][0]
    assert item["render_spec"]["width"] == 1080
    assert item["render_spec"]["height"] == 1920
    assert item["render_spec"]["duration_seconds"] == 30
    assert item["platforms"]["tiktok"]["aspect_ratio"] == "9:16"
    assert item["platforms"]["instagram_reels"]["aspect_ratio"] == "9:16"
    assert item["platforms"]["youtube_shorts"]["aspect_ratio"] == "9:16"
    assert item["platforms"]["x"]["card_aspect_ratio"] == "16:9"


def test_content_factory_has_no_provider_or_runtime_mutation_path():
    source = open(content_factory_v4.__file__, "r", encoding="utf-8").read().lower()
    for forbidden in ("api-football", "requests.get", "threshold =", "decision_weight =", "model_weights ="):
        assert forbidden not in source
    result = content_factory_v4.build_content_packages(_product())
    assert result["provider_requests_added"] == 0
    assert result["canonical_bet_logic_changed"] is False
    assert result["model_weights_changed"] is False
    assert result["production_promotion_allowed"] is False
