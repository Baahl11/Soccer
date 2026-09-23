from datetime import datetime, timezone

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


def test_group_fair_probability_matches_over_side_and_only_target_line():
    values = [
        {"selection": "Over 2.5", "line": 2.5, "decimal_price": 2.0},
        {"selection": "Under 2.5", "line": 2.5, "decimal_price": 1.8},
        {"selection": "Over 3.5", "line": 3.5, "decimal_price": 3.1},
        {"selection": "Under 3.5", "line": 3.5, "decimal_price": 1.4},
    ]
    fair, price = v._group_fair_probability(values, "Over", 2.5)
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
    assert v._family({"market_family": "CORNERS", "market": "Corners Over/Under", "selection": "Over", "line": 9.5}) == "FT_CORNERS"
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


def test_pipeline_market_rows_take_precedence_over_legacy_best_market():
    generated = datetime(2026, 9, 23, 10, 0, tzinfo=timezone.utc)
    pipeline = [{
        "fixture_id": 1,
        "generated_at": generated,
        "market_candidate": {
            "market_family": "TOTAL",
            "market": "Goals Over/Under",
            "selection": "Over",
            "line": 2.5,
            "price": 2.0,
            "bookmaker": "Book",
        },
        "signal_source": "PIPELINE_MATCH_TABLE",
    }]
    legacy = [{
        "fixture_id": 1,
        "generated_at": generated,
        "event_payload": {
            "best_market": {
                "family": "TOTAL",
                "market": "Goals Over/Under",
                "selection": "Over",
                "line": 2.5,
                "price": 2.0,
                "bookmaker": "Book",
            }
        },
        "signal_source": "LEGACY_BEST_MARKET",
    }]
    merged = v._merge_signals(pipeline, legacy, max_rows=10)
    assert len(merged) == 1
    assert merged[0]["signal_source"] == "PIPELINE_MATCH_TABLE"


def test_merge_keeps_independent_market_families_from_same_fixture_and_run():
    generated = datetime(2026, 9, 23, 10, 0, tzinfo=timezone.utc)
    pipeline = [
        {
            "fixture_id": 1,
            "generated_at": generated,
            "market_candidate": {
                "market_family": "TOTAL",
                "market": "Goals Over/Under",
                "selection": "Over",
                "line": 2.5,
                "price": 2.0,
                "bookmaker": "Book",
            },
            "signal_source": "PIPELINE_MATCH_TABLE",
        },
        {
            "fixture_id": 1,
            "generated_at": generated,
            "market_candidate": {
                "market_family": "CORNERS",
                "market": "Corners Over/Under",
                "selection": "Over",
                "line": 9.5,
                "price": 1.95,
                "bookmaker": "Book",
            },
            "signal_source": "PIPELINE_MATCH_TABLE",
        },
    ]
    merged = v._merge_signals(pipeline, [], max_rows=10)
    assert len(merged) == 2
    assert {v._family(row["market_candidate"]) for row in merged} == {"FT_TOTALS", "FT_CORNERS"}
