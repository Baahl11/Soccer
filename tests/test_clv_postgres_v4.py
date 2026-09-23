from mcp_gateway import clv_postgres_v4 as v


def test_group_fair_probability_devigs_exact_same_line():
    values = [
        {"selection": "Over 2.5", "line": 2.5, "decimal_price": 2.0},
        {"selection": "Under 2.5", "line": 2.5, "decimal_price": 1.8},
    ]
    fair, price = v._group_fair_probability(values, "Over 2.5", 2.5)
    assert price == 2.0
    assert fair is not None
    assert round(fair, 6) == round((1 / 2.0) / ((1 / 2.0) + (1 / 1.8)), 6)


def test_closing_line_candidate_requires_unambiguous_side():
    values = [
        {"selection": "Over 2.75", "line": 2.75, "decimal_price": 1.95},
        {"selection": "Under 2.75", "line": 2.75, "decimal_price": 1.95},
    ]
    line, price = v._closing_line_candidate(values, "Over 2.5")
    assert line == 2.75
    assert price == 1.95


def test_family_mapping_uses_phase16_canonical_mapping():
    assert v._family({"family": "TOTAL", "market": "Goals Over/Under", "selection": "Over 2.5"}) == "FT_TOTALS"
    assert v._family({"family": "BTTS", "market": "Both Teams To Score", "selection": "Yes"}) == "BTTS"
    assert v._family({"market": "First Half Goals Over/Under", "selection": "Over 1.5"}) == "1H"
    assert v._family({"market": "Home Team Total Goals", "selection": "Over 1.5"}) == "HOME_TT"


def test_model_signal_fallback_is_sport_only():
    event = {
        "sporting_shortlist": {
            "side_edge_score": 78,
            "goal_environment_score": 66,
            "two_way_scoring_score": 70,
        }
    }
    assert v._model_signal_from_event(event) == "STRONG"
