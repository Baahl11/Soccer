from mcp_gateway import alert_engine_v4 as v


def _snap(**overrides):
    row = {
        "fixture_id": 1,
        "home": "A",
        "away": "B",
        "league": "Test League",
        "kickoff": "2026-09-23T18:00:00-06:00",
        "model_signal": "MODERATE",
        "model_signal_score": 68.0,
        "data_tier": "A",
        "classification": "WATCH",
        "execution_status": "WAIT_PRICE",
        "market": "Goals Over/Under",
        "selection": "Over",
        "line": 2.5,
        "price": None,
        "p_market_fair": None,
        "prob_edge_pp": None,
        "estimated_ev": None,
        "bookmaker": None,
        "blockers": ["WAIT_PRICE"],
        "xi_confirmed": False,
        "gk_confirmed": False,
    }
    row.update(overrides)
    return row


def test_phase21_detects_requested_change_types():
    previous = _snap(
        model_signal="MODERATE",
        price=None,
        prob_edge_pp=3.0,
        classification="BET",
        execution_status="WAIT_FRESH_QUOTE",
    )
    current = _snap(
        model_signal="STRONG",
        price=2.10,
        bookmaker="Book",
        prob_edge_pp=7.0,
        classification="LEAN",
        execution_status="MODEL_DISAGREEMENT",
        xi_confirmed=True,
        gk_confirmed=True,
        blockers=[],
        model_disagreement="HIGH",
    )
    changes = v.detect_changes(previous, current, edge_threshold_pp=5.0)
    kinds = {change["type"] for change in changes}
    assert "SIGNAL_UPGRADED" in kinds
    assert "XI_CONFIRMED" in kinds
    assert "GK_CONFIRMED" in kinds
    assert "FRESH_QUOTE_APPEARED" in kinds
    assert "EDGE_CROSSED_THRESHOLD" in kinds
    assert "MODEL_DISAGREEMENT_APPEARED" in kinds
    assert "PICK_DEMOTED" in kinds


def test_phase21_price_improved_and_stale_are_distinct_changes():
    previous = _snap(price=1.90, execution_status="READY", blockers=[])
    current = _snap(price=2.05, execution_status="STALE_QUOTE", blockers=["STALE_QUOTE"])
    changes = v.detect_changes(previous, current)
    kinds = {change["type"] for change in changes}
    assert "PRICE_IMPROVED" in kinds
    assert "MARKET_BECAME_STALE" in kinds


def test_phase21_edge_threshold_is_not_invented():
    previous = _snap(prob_edge_pp=1.0)
    current = _snap(prob_edge_pp=20.0)
    changes = v.detect_changes(previous, current)
    assert "EDGE_CROSSED_THRESHOLD" not in {change["type"] for change in changes}


def test_phase21_alert_format_matches_master_sections_and_missing_values():
    current = _snap(
        model_signal="VERY_STRONG",
        classification="WATCH",
        execution_status="WAIT_PRICE",
    )
    payload = v.build_alert_payload(
        current,
        changes=[{"type": "SIGNAL_UPGRADED"}],
        sport_probabilities={"p_home": 0.55, "p_draw": 0.25, "p_away": 0.20},
        model_context={"agreement": "HIGH", "uncertainty": 0.18},
    )
    text = payload["text"]
    assert "🔥 STRONG SPORT SIGNAL" in text
    assert "SPORT" in text
    assert "MODEL" in text
    assert "BEST MARKET" in text
    assert "EXECUTION" in text
    assert "FINAL" in text
    assert "P(O2.5): N/V" in text
    assert payload["webhook_delivery_enabled"] is False
    assert payload["production_action_enabled"] is False
