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
    merged = v._merge_signals(pipeline, [], legacy, max_rows=10)
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
    merged = v._merge_signals(pipeline, [], [], max_rows=10)
    assert len(merged) == 2
    assert {v._family(row["market_candidate"]) for row in merged} == {"FT_TOTALS", "FT_CORNERS"}


def test_derivative_signals_extract_observed_research_markets():
    event = {
        "fixture_id": 10,
        "generated_at": datetime(2026, 9, 23, 10, 0, tzinfo=timezone.utc),
        "stage": "T-20",
        "classification": "WATCH",
        "event_payload": {
            "team_totals_intelligence": {
                "observed_exact_market_rows": [
                    {
                        "market": "Home Team Total Goals",
                        "selection": "OVER",
                        "line": 1.5,
                        "decimal_price": 1.95,
                        "bookmaker": "Book",
                    }
                ]
            },
            "one_h_goals_intelligence": {
                "observed_market_rows": [
                    {
                        "market": "Goals Over/Under First Half",
                        "selection": "UNDER",
                        "line": 1.5,
                        "decimal_price": 1.9,
                        "bookmaker": "Book",
                    }
                ]
            },
            "corners_intelligence": {
                "observed_market_rows": [
                    {
                        "market": "Corners Over/Under",
                        "selection": "OVER",
                        "line": 9.5,
                        "decimal_price": 2.0,
                        "bookmaker": "Book",
                    }
                ]
            },
        },
    }
    rows = v._derivative_signals_from_events([event])
    assert len(rows) == 3
    assert {v._family(row["market_candidate"]) for row in rows} == {
        "HOME_TT",
        "1H",
        "FT_CORNERS",
    }
    assert all(row["signal_source"].startswith("DERIVATIVE_INTELLIGENCE:") for row in rows)


def test_period_team_totals_do_not_contaminate_generic_half_families():
    first_half = {
        "signal_source": "DERIVATIVE_INTELLIGENCE:team_totals_intelligence",
        "market_candidate": {
            "market_family": "TEAM_TOTALS",
            "market": "Home Team Total Goals First Half",
            "selection": "OVER",
            "line": 0.5,
            "decimal_price": 1.9,
        },
    }
    full_time = {
        "signal_source": "DERIVATIVE_INTELLIGENCE:team_totals_intelligence",
        "market_candidate": {
            "market_family": "TEAM_TOTALS",
            "market": "Home Team Total Goals",
            "selection": "OVER",
            "line": 1.5,
            "decimal_price": 1.95,
        },
    }

    assert v._is_period_team_total_signal(first_half) is True
    assert v._is_period_team_total_signal(full_time) is False
    assert v._family(full_time["market_candidate"]) == "HOME_TT"


def test_derivative_team_totals_keep_home_and_away_family_identity():
    events = [{
        "fixture_id": 101,
        "generated_at": "2026-09-24T10:00:00+00:00",
        "event_payload": {
            "team_totals_intelligence": {
                "observed_exact_market_rows": [
                    {
                        "market": "Home Team Total Goals",
                        "selection": "OVER",
                        "line": 1.5,
                        "decimal_price": 1.95,
                    },
                    {
                        "market": "Away Team Total Goals",
                        "selection": "UNDER",
                        "line": 1.5,
                        "decimal_price": 1.90,
                    },
                ]
            }
        },
    }]
    rows = v._derivative_signals_from_events(events)
    assert len(rows) == 2
    assert [row["signal_source"] for row in rows] == [
        "DERIVATIVE_INTELLIGENCE:team_totals_intelligence",
        "DERIVATIVE_INTELLIGENCE:team_totals_intelligence",
    ]
    assert {v._family(row["market_candidate"]) for row in rows} == {"HOME_TT", "AWAY_TT"}


class _FakeCursor:
    def __init__(self):
        self.query = ""
        self.description = []

    def execute(self, query, params):
        self.query = query

    def fetchall(self):
        return []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeConn:
    def __init__(self):
        self.cursor_instance = _FakeCursor()

    def cursor(self):
        return self.cursor_instance


def test_pipeline_loader_does_not_duplicate_full_event_payload():
    conn = _FakeConn()
    rows = v._load_pipeline_market_signals(conn, lookback_days=30, max_rows=10)
    query = conn.cursor_instance.query

    assert rows == []
    assert "e.payload AS event_payload" not in query
    assert "jsonb_build_object" in query
    assert "'sporting_shortlist'" in query
    assert "'model_signal'" in query


def test_legacy_loader_keeps_only_best_market_and_sport_metadata():
    conn = _FakeConn()
    rows = v._load_legacy_signals(conn, lookback_days=30, max_rows=10)
    query = conn.cursor_instance.query

    assert rows == []
    assert "e.payload AS event_payload" not in query
    assert "'best_market'" in query
    assert "'sporting_shortlist'" in query
    assert "'model_signal'" in query


def test_minimal_legacy_payload_preserves_candidate_and_confidence():
    signal = {
        "event_payload": {
            "best_market": {
                "family": "TOTAL",
                "market": "Goals Over/Under",
                "selection": "Over",
                "line": 2.5,
                "price": 2.0,
            },
            "sporting_shortlist": {
                "side_edge_score": 78,
                "goal_environment_score": 66,
                "two_way_scoring_score": 70,
            },
        }
    }
    converted = v._legacy_to_signal(signal)

    assert converted is not None
    assert converted["market_candidate"]["market"] == "Goals Over/Under"
    assert v._model_signal_from_candidate(converted["market_candidate"], signal["event_payload"]) == "STRONG"
