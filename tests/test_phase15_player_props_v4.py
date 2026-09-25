from mcp_gateway import player_props_phase15_v4 as v
from mcp_gateway import player_props_clv_postgres_v4 as prop_clv
from mcp_gateway import player_props_oos_postgres_v4 as prop_oos
from mcp_gateway import player_props_phase15_coverage_audit as coverage_audit
from mcp_gateway import player_props_postgame_backfill_v4 as prop_backfill
from mcp_gateway import player_trend_registry_backfill_v4 as registry_backfill
from mcp_gateway import gk_saves_intelligence as gk_saves
from mcp_gateway import analyze_player_trends as trend_analysis
from mcp_gateway import automation as base_automation


BASE = {
    "status": "PASS",
    "failures": [],
    "profiles_checked": 1000,
    "profiles_valid": 1000,
    "oos_validation_complete": False,
    "actionable": False,
    "decision_weight": 0,
    "validation_scope": "STRUCTURAL_ONLY_NOT_OOS_PERFORMANCE",
}


def test_phase15_blocks_structural_only_props():
    gk = dict(BASE)
    gk["profiles_checked"] = 22
    gk["profiles_valid"] = 22

    report = v.build_report(
        dict(BASE), dict(BASE), dict(BASE), dict(BASE), dict(BASE), gk, []
    )
    assert report["status"] == "RESEARCH_HOLD"
    assert "SHOTS_OOS_VALIDATION_INCOMPLETE" in report["blockers"]
    assert "GK_SAVES_VALID_PROFILES_22_LT_100" in report["blockers"]
    assert "PLAYER_PROP_TRUE_CLV_0_LT_50" in report["blockers"]
    assert "SHOTS_OBSERVED_MARKET_PRICE_HISTORY_MISSING" in report["blockers"]
    assert "SHOTS_EXACT_LINE_HISTORY_MISSING" in report["blockers"]
    assert "ASSISTS_CONFIRMED_XI_MARKET_OVERLAP_MISSING" in report["blockers"]
    assert report["production_promotion_allowed"] is False


def test_phase15_prop_clv_matcher_is_prop_specific():
    summary = v.summarize_true_clv([
        {"market": "Player Shots", "selection": "Player A Over 2.5", "fixture_id": 1, "clv_probability_pp": 0.02},
        {"market": "Goalkeeper Saves", "selection": "Player B Over 3.5", "fixture_id": 2, "clv_probability_pp": -0.01},
        {"market": "Goals Over/Under", "selection": "Over 2.5", "fixture_id": 3, "clv_probability_pp": 0.20},
    ])
    assert summary["rows"] == 2
    assert summary["unique_fixtures"] == 2
    assert summary["avg_probability_clv_pp"] == 0.005



def test_phase15_market_audit_is_family_specific_and_line_aware():
    audit = {
        "model_version": "SOCCER_RESEARCH_DERIVATIVE_MARKET_AUDIT_V4_1.0.0",
        "families": {
            "SHOTS": {
                "market_snapshot_rows": 10,
                "priced_value_rows": 20,
                "exact_line_value_rows": 20,
                "unique_fixtures": 4,
                "pre_kickoff_unique_fixtures": 4,
                "provider_update_unique_fixtures": 4,
                "confirmed_xi_pre_kickoff_unique_fixtures": 2,
                "bookmaker_count": 2,
                "xi_aligned_value_rows": 20,
                "xi_aligned_priced_value_rows": 20,
                "xi_aligned_exact_line_value_rows": 20,
                "confirmed_xi_player_aligned_unique_fixtures": 2,
                "exact_observed_market_history_materialized": True,
                "confirmed_xi_overlap_materialized": True,
                "player_xi_alignment_materialized": True,
            },
            "ASSISTS": {
                "market_snapshot_rows": 35,
                "priced_value_rows": 57,
                "exact_line_value_rows": 0,
                "unique_fixtures": 4,
                "pre_kickoff_unique_fixtures": 4,
                "provider_update_unique_fixtures": 4,
                "confirmed_xi_pre_kickoff_unique_fixtures": 1,
                "bookmaker_count": 2,
                "xi_aligned_value_rows": 57,
                "xi_aligned_priced_value_rows": 57,
                "xi_aligned_exact_line_value_rows": 0,
                "confirmed_xi_player_aligned_unique_fixtures": 1,
                "exact_observed_market_history_materialized": True,
                "confirmed_xi_overlap_materialized": True,
                "player_xi_alignment_materialized": True,
            },
        },
    }
    report = v.build_report(
        dict(BASE), dict(BASE), dict(BASE), dict(BASE), dict(BASE), dict(BASE), [], audit
    )

    assert report["prop_families"]["shots"]["market_evidence"]["exact_line_value_rows"] == 20
    assert "SHOTS_OBSERVED_MARKET_PRICE_HISTORY_MISSING" not in report["blockers"]
    assert "SHOTS_CONFIRMED_XI_MARKET_OVERLAP_MISSING" not in report["blockers"]
    assert "SHOTS_EXACT_LINE_HISTORY_MISSING" not in report["blockers"]
    assert "SHOTS_CONFIRMED_XI_PLAYER_PRICE_OVERLAP_MISSING" not in report["blockers"]
    assert "SHOTS_XI_ALIGNED_EXACT_LINE_HISTORY_MISSING" not in report["blockers"]

    # Assists is a binary player-event market; lack of a numeric O/U line is not
    # itself a blocker when an exact priced market is observed.
    assert report["prop_families"]["assists"]["market_evidence"]["priced_value_rows"] == 57
    assert "ASSISTS_OBSERVED_MARKET_PRICE_HISTORY_MISSING" not in report["blockers"]
    assert "ASSISTS_CONFIRMED_XI_MARKET_OVERLAP_MISSING" not in report["blockers"]
    assert "ASSISTS_CONFIRMED_XI_PLAYER_PRICE_OVERLAP_MISSING" not in report["blockers"]
    assert "ASSISTS_EXACT_LINE_HISTORY_MISSING" not in report["blockers"]

    # Player cards must use PLAYER_CARDS evidence, never generic match-card rows.
    assert "CARDS_OBSERVED_MARKET_PRICE_HISTORY_MISSING" in report["blockers"]


def test_phase15_rejects_fixture_level_xi_without_player_level_alignment():
    audit = {
        "families": {
            "SHOTS": {
                "market_snapshot_rows": 10,
                "priced_value_rows": 20,
                "exact_line_value_rows": 20,
                "unique_fixtures": 4,
                "pre_kickoff_unique_fixtures": 4,
                "provider_update_unique_fixtures": 4,
                "confirmed_xi_pre_kickoff_unique_fixtures": 2,
                "bookmaker_count": 2,
                "xi_aligned_value_rows": 0,
                "xi_aligned_priced_value_rows": 0,
                "xi_aligned_exact_line_value_rows": 0,
                "confirmed_xi_player_aligned_unique_fixtures": 0,
                "exact_observed_market_history_materialized": True,
                "confirmed_xi_overlap_materialized": True,
                "player_xi_alignment_materialized": False,
            }
        }
    }
    report = v.build_report(
        dict(BASE), dict(BASE), dict(BASE), dict(BASE), dict(BASE), dict(BASE), [], audit
    )

    assert "SHOTS_CONFIRMED_XI_MARKET_OVERLAP_MISSING" not in report["blockers"]
    assert "SHOTS_CONFIRMED_XI_PLAYER_PRICE_OVERLAP_MISSING" in report["blockers"]
    assert "SHOTS_XI_ALIGNED_EXACT_LINE_HISTORY_MISSING" in report["blockers"]
    assert report["production_promotion_allowed"] is False


def _xi_lineup():
    return {
        "both_xi_confirmed": True,
        "teams": [
            {
                "team_id": 10,
                "team": "Home",
                "starters": [{"id": 501, "name": "Player A", "pos": "F"}],
            },
            {
                "team_id": 20,
                "team": "Away",
                "starters": [{"id": 601, "name": "Keeper B", "pos": "G"}],
            },
        ],
    }


def test_player_prop_true_clv_extracts_xi_aligned_shadow_signal_and_devigs_line():
    event = {
        "fixture_id": 9001,
        "generated_at": "2026-09-25T10:00:00+00:00",
        "stage": "T-20",
        "kickoff": "2026-09-25T11:00:00+00:00",
        "event_payload": {
            "stage": "T-20",
            "fixture": {"fixture_id": 9001, "kickoff": "2026-09-25T11:00:00+00:00"},
            "lineups": _xi_lineup(),
            "player_shots_intelligence": {
                "players": [{
                    "player_id": 501,
                    "player": "Player A",
                    "team_id": 10,
                    "confirmed_starter": True,
                    "expected_minutes_if_confirmed_starter": 82.0,
                    "lines": [{"line": 2.5, "p_over": 0.57, "p_under": 0.43}],
                }]
            },
            "market": {
                "research_cards_props_markets": [{
                    "research_family": "PLAYER_PROPS",
                    "research_subfamily": "SHOTS",
                    "market": "Player Shots",
                    "market_id": 801,
                    "bookmaker_id": 1,
                    "bookmaker": "Book",
                    "provider_update": "2026-09-25T09:58:00+00:00",
                    "values": [
                        {
                            "selection": "Player A Over 2.5",
                            "price": "2.00",
                            "parsed_line": 2.5,
                            "xi_alignment_status": "MATCHED_CONFIRMED_XI",
                            "player_id": 501,
                            "player_name": "Player A",
                            "team_id": 10,
                        },
                        {
                            "selection": "Player A Under 2.5",
                            "price": "1.80",
                            "parsed_line": 2.5,
                            "xi_alignment_status": "MATCHED_CONFIRMED_XI",
                            "player_id": 501,
                            "player_name": "Player A",
                            "team_id": 10,
                        },
                    ],
                }]
            },
        },
    }

    signals = prop_clv.extract_shadow_signals([event])

    assert len(signals) == 2
    over = next(row for row in signals if row["side"] == "OVER")
    assert over["market_family"] == "SHOTS"
    assert over["player_id"] == 501
    assert over["line"] == 2.5
    assert over["model_probability"] == 0.57
    assert over["entry_market_fair_basis"] == "DEVIGGED_TWO_WAY"
    assert round(over["entry_market_fair_probability"], 6) == round((1/2.0) / ((1/2.0)+(1/1.8)), 6)
    assert over["decision_weight"] == 0.0
    assert over["production_promotion_allowed"] is False


def test_player_prop_true_clv_pairs_exact_player_line_side_with_strict_later_provider_update():
    signal = {
        "fixture_id": 9001,
        "kickoff": "2026-09-25T11:00:00+00:00",
        "signal_timestamp": "2026-09-25T10:00:00+00:00",
        "stage": "T-20",
        "market_family": "SHOTS",
        "market": "Player Shots",
        "bookmaker_id": 1,
        "bookmaker": "Book",
        "player_id": 501,
        "player_name": "Player A",
        "selection": "Player A Over 2.5",
        "side": "OVER",
        "line": 2.5,
        "entry_price": 2.0,
        "entry_market_fair_probability": (1/2.0) / ((1/2.0)+(1/1.8)),
        "entry_market_fair_basis": "DEVIGGED_TWO_WAY",
        "model_probability": 0.57,
        "decision_weight": 0.0,
        "production_promotion_allowed": False,
    }
    snapshot = {
        "fixture_id": 9001,
        "captured_at": "2026-09-25T10:45:00+00:00",
        "provider_update": "2026-09-25T10:40:00+00:00",
        "bookmaker_id": 1,
        "bookmaker": "Book",
        "market": "Player Shots",
        "confirmed_lineup_payload": _xi_lineup(),
        "values": [
            {"selection": "Player A Over 2.5", "price": "1.80", "parsed_line": 2.5},
            {"selection": "Player A Under 2.5", "price": "2.00", "parsed_line": 2.5},
        ],
    }

    rows, skip = prop_clv.pair_signals_to_closes([signal], [snapshot])

    assert skip == {}
    assert len(rows) == 1
    row = rows[0]
    assert row["is_true_closing_line"] is True
    assert row["strict_later_provider_update"] is True
    assert row["probability_comparable_same_line"] is True
    assert row["same_book_preferred"] is True
    assert row["closing_price"] == 1.8
    assert row["clv_probability_pp"] is not None
    assert row["price_clv_pct"] > 0
    assert row["decision_weight"] == 0.0
    assert row["production_promotion_allowed"] is False


def test_player_prop_true_clv_one_way_market_tracks_price_without_fake_devig():
    signal = {
        "fixture_id": 9002,
        "kickoff": "2026-09-25T11:00:00+00:00",
        "signal_timestamp": "2026-09-25T10:00:00+00:00",
        "stage": "T-20",
        "market_family": "GOALSCORER_ANYTIME",
        "market": "Anytime Goal Scorer",
        "bookmaker_id": 1,
        "bookmaker": "Book",
        "player_id": 501,
        "player_name": "Player A",
        "selection": "Player A",
        "side": "PLAYER_EVENT",
        "line": None,
        "entry_price": 3.0,
        "entry_market_fair_probability": None,
        "entry_market_fair_basis": "ONE_WAY_OR_UNPAIRED",
        "model_probability": 0.36,
    }
    snapshot = {
        "fixture_id": 9002,
        "captured_at": "2026-09-25T10:45:00+00:00",
        "provider_update": "2026-09-25T10:40:00+00:00",
        "bookmaker_id": 1,
        "bookmaker": "Book",
        "market": "Anytime Goal Scorer",
        "confirmed_lineup_payload": _xi_lineup(),
        "values": [{"selection": "Player A", "price": "2.70"}],
    }

    rows, _ = prop_clv.pair_signals_to_closes([signal], [snapshot])
    assert len(rows) == 1
    assert rows[0]["is_true_closing_line"] is True
    assert rows[0]["probability_comparable_same_line"] is False
    assert rows[0]["clv_probability_pp"] is None
    assert rows[0]["price_clv_pct"] > 0
    assert rows[0]["closing_line_status"] == "TRUE_PREKICKOFF_PLAYER_PROP_CLOSE_PRICE_ONLY"


def test_player_prop_true_clv_rejects_cache_replay_or_stale_provider_quote():
    signal = {
        "fixture_id": 9003,
        "kickoff": "2026-09-25T11:00:00+00:00",
        "signal_timestamp": "2026-09-25T10:00:00+00:00",
        "market_family": "SHOTS",
        "bookmaker_id": 1,
        "bookmaker": "Book",
        "player_id": 501,
        "selection": "Player A Over 2.5",
        "side": "OVER",
        "line": 2.5,
        "entry_price": 2.0,
        "entry_market_fair_probability": 0.48,
        "entry_market_fair_basis": "DEVIGGED_TWO_WAY",
    }
    stale = {
        "fixture_id": 9003,
        "captured_at": "2026-09-25T10:40:00+00:00",
        "provider_update": "2026-09-25T09:55:00+00:00",
        "bookmaker_id": 1,
        "bookmaker": "Book",
        "market": "Player Shots",
        "confirmed_lineup_payload": _xi_lineup(),
        "values": [
            {"selection": "Player A Over 2.5", "price": "1.9", "parsed_line": 2.5},
            {"selection": "Player A Under 2.5", "price": "1.9", "parsed_line": 2.5},
        ],
    }

    rows, skip = prop_clv.pair_signals_to_closes([signal], [stale])
    assert rows == []
    assert skip["NO_LATER_STRICT_PLAYER_PROP_CLOSE"] == 1


def test_player_prop_true_clv_summary_requires_rows_and_fixture_diversity_per_family():
    rows = [
        {
            "fixture_id": i,
            "player_id": 1000 + i,
            "market_family": "SHOTS",
            "is_true_closing_line": True,
            "probability_comparable_same_line": True,
        }
        for i in range(20)
    ]
    summary = prop_clv.summarize_tracking(rows)
    shots = summary["families"]["SHOTS"]
    assert shots["true_clv_rows"] == 20
    assert shots["unique_fixtures"] == 20
    assert shots["fixture_diversity_target_met"] is True
    assert shots["row_target_met"] is False
    assert shots["review_ready"] is False


def test_phase15_true_clv_cannot_mature_from_mixed_or_low_diversity_rows():
    rows = [
        {
            "fixture_id": 1 + (i % 2),
            "player_id": 1000 + i,
            "market_family": "SHOTS",
            "market": "Player Shots",
            "selection": "Player A Over 2.5",
            "is_true_closing_line": True,
            "clv_probability_pp": 0.2,
            "price_clv_pct": 0.4,
        }
        for i in range(60)
    ]
    summary = v.summarize_true_clv(rows)

    assert summary["rows"] == 60
    assert summary["by_family"]["shots"]["rows"] == 60
    assert summary["by_family"]["shots"]["unique_fixtures"] == 2
    assert summary["by_family"]["shots"]["row_target_met"] is True
    assert summary["by_family"]["shots"]["fixture_diversity_target_met"] is False
    assert summary["by_family"]["sot"]["rows"] == 0

    report = v.build_report(
        dict(BASE), dict(BASE), dict(BASE), dict(BASE), dict(BASE), dict(BASE), rows, {}
    )
    assert "PLAYER_PROP_TRUE_CLV_60_LT_50" not in report["blockers"]
    assert "SHOTS_TRUE_CLV_60_LT_50" not in report["blockers"]
    assert "SHOTS_TRUE_CLV_FIXTURES_2_LT_20" in report["blockers"]
    assert "SOT_TRUE_CLV_0_LT_50" in report["blockers"]
    assert report["production_promotion_allowed"] is False


def test_phase15_counts_exact_one_way_price_clv_without_faking_probability_clv():
    rows = [{
        "fixture_id": 44,
        "player_id": 501,
        "market_family": "GOALSCORER_ANYTIME",
        "market": "Anytime Goal Scorer",
        "selection": "Player A",
        "is_true_closing_line": True,
        "clv_probability_pp": None,
        "price_clv_pct": 4.2,
    }]
    summary = v.summarize_true_clv(rows)
    scorer = summary["by_family"]["goalscorer"]

    assert scorer["rows"] == 1
    assert scorer["price_clv_rows"] == 1
    assert scorer["probability_clv_rows"] == 0
    assert scorer["avg_probability_clv_pp"] is None


def test_player_prop_clv_gk_signal_does_not_require_outfield_confirmed_starter_flag():
    lineup = {
        "both_xi_confirmed": True,
        "both_goalkeepers_confirmed": True,
        "teams": [
            {
                "team_id": 10,
                "team": "Home",
                "starters": [{"id": 501, "name": "Player A", "pos": "F"}],
                "goalkeepers": [{"id": 701, "name": "Keeper A", "pos": "G"}],
            },
            {
                "team_id": 20,
                "team": "Away",
                "starters": [{"id": 601, "name": "Player B", "pos": "F"}],
                "goalkeepers": [{"id": 702, "name": "Keeper B", "pos": "G"}],
            },
        ],
    }
    event = {
        "fixture_id": 9100,
        "generated_at": "2026-09-25T10:00:00+00:00",
        "stage": "T-20",
        "kickoff": "2026-09-25T11:00:00+00:00",
        "event_payload": {
            "stage": "T-20",
            "fixture": {"fixture_id": 9100, "kickoff": "2026-09-25T11:00:00+00:00"},
            "lineups": lineup,
            "gk_saves_intelligence": {
                "goalkeepers": [{
                    "player_id": 701,
                    "player": "Keeper A",
                    "team_id": 10,
                    "expected_minutes": 90.0,
                    "lines": [{"line": 3.5, "p_over": 0.45, "p_under": 0.55}],
                }]
            },
            "market": {
                "research_cards_props_markets": [{
                    "research_family": "PLAYER_PROPS",
                    "research_subfamily": "GK_SAVES",
                    "market": "Goalkeeper Saves",
                    "bookmaker": "Book",
                    "provider_update": "2026-09-25T09:58:00+00:00",
                    "values": [
                        {
                            "selection": "Keeper A Over 3.5",
                            "price": "2.05",
                            "parsed_line": 3.5,
                            "xi_alignment_status": "MATCHED_CONFIRMED_XI",
                            "player_id": 701,
                            "player_name": "Keeper A",
                            "team_id": 10,
                        },
                        {
                            "selection": "Keeper A Under 3.5",
                            "price": "1.75",
                            "parsed_line": 3.5,
                            "xi_alignment_status": "MATCHED_CONFIRMED_XI",
                            "player_id": 701,
                            "player_name": "Keeper A",
                            "team_id": 10,
                        },
                    ],
                }]
            },
        },
    }

    signals = prop_clv.extract_shadow_signals([event])
    assert len(signals) == 2
    assert {row["market_family"] for row in signals} == {"GK_SAVES"}
    assert all(isinstance(row["provider_update"], str) for row in signals)
    assert next(row for row in signals if row["side"] == "OVER")["model_probability"] == 0.45


def test_player_props_oos_chooses_one_latest_priority_pregame_event_per_fixture():
    rows = [
        {
            "fixture_id": 1,
            "stage": "T-40",
            "generated_at": "2026-09-25T10:00:00+00:00",
            "kickoff": "2026-09-25T11:00:00+00:00",
            "event_payload": {"fixture": {"fixture_id": 1, "kickoff": "2026-09-25T11:00:00+00:00"}},
        },
        {
            "fixture_id": 1,
            "stage": "T-20",
            "generated_at": "2026-09-25T10:30:00+00:00",
            "kickoff": "2026-09-25T11:00:00+00:00",
            "event_payload": {"fixture": {"fixture_id": 1, "kickoff": "2026-09-25T11:00:00+00:00"}},
        },
        {
            "fixture_id": 1,
            "stage": "T-10",
            "generated_at": "2026-09-25T10:50:00+00:00",
            "kickoff": "2026-09-25T11:00:00+00:00",
            "event_payload": {"fixture": {"fixture_id": 1, "kickoff": "2026-09-25T11:00:00+00:00"}},
        },
    ]

    out = prop_oos.choose_canonical_pregame_events(rows)
    assert len(out) == 1
    assert out[0]["stage"] == "T-10"


def test_player_props_oos_joins_confirmed_model_prediction_to_final_player_result():
    pregame = [{
        "fixture_id": 7001,
        "stage": "T-10",
        "generated_at": "2026-09-25T10:50:00+00:00",
        "kickoff": "2026-09-25T11:00:00+00:00",
        "event_payload": {
            "stage": "T-10",
            "fixture": {"fixture_id": 7001, "kickoff": "2026-09-25T11:00:00+00:00"},
            "player_shots_intelligence": {
                "players": [{
                    "player_id": 501,
                    "player": "Player A",
                    "team_id": 10,
                    "position": "F",
                    "confirmed_starter": True,
                    "expected_minutes_if_confirmed_starter": 82.0,
                    "expected_shots": 2.8,
                    "lines": [
                        {"line": 1.5, "p_over": 0.72, "p_under": 0.28},
                        {"line": 2.5, "p_over": 0.51, "p_under": 0.49},
                    ],
                }]
            },
            "player_goalscorer_intelligence": {
                "players": [{
                    "player_id": 501,
                    "player": "Player A",
                    "team_id": 10,
                    "position": "F",
                    "confirmed_starter": True,
                    "expected_minutes_if_confirmed_starter": 82.0,
                    "expected_goals": 0.44,
                    "p_anytime_goal": 0.36,
                }]
            },
        },
    }]
    postgame = [{
        "fixture_id": 7001,
        "stage": "POSTGAME",
        "generated_at": "2026-09-25T13:00:00+00:00",
        "event_payload": {
            "fixture": {"fixture_id": 7001},
            "postgame_player_stats": {
                "status": "RESEARCH_ONLY_PLAYER_FIXTURE_STATS",
                "teams": [{
                    "team_id": 10,
                    "team": "Home",
                    "players": [{
                        "player_id": 501,
                        "name": "Player A",
                        "minutes": 90,
                        "position": "F",
                        "shots": 3,
                        "shots_on_target": 2,
                        "goals": 1,
                        "assists": 0,
                        "yellow_cards": 0,
                        "saves": 0,
                    }],
                }],
            },
        },
    }]

    rows = prop_oos.build_oos_rows(pregame, postgame)
    assert {row["market_family"] for row in rows} == {"SHOTS", "GOALSCORER_ANYTIME"}

    shots = next(row for row in rows if row["market_family"] == "SHOTS")
    assert shots["expected_count"] == 2.8
    assert shots["actual_count"] == 3
    assert shots["actual_minutes"] == 90
    assert shots["expected_minutes"] == 82.0
    assert len(shots["binary_rows"]) == 2
    assert shots["binary_rows"][0]["outcome"] == 1

    scorer = next(row for row in rows if row["market_family"] == "GOALSCORER_ANYTIME")
    assert scorer["actual_count"] == 1
    assert scorer["binary_rows"][0]["probability"] == 0.36
    assert scorer["binary_rows"][0]["outcome"] == 1


def test_player_props_oos_metrics_are_family_specific_and_not_promotional():
    rows = [
        {
            "fixture_id": i,
            "market_family": "SHOTS",
            "player_id": 1000 + i,
            "expected_count": 2.0,
            "actual_count": 3.0 if i % 2 == 0 else 1.0,
            "count_error": 1.0 if i % 2 == 0 else -1.0,
            "expected_minutes": 80.0,
            "actual_minutes": 90.0,
            "minutes_error": 10.0,
            "binary_rows": [{
                "fixture_id": i,
                "market_family": "SHOTS",
                "player_id": 1000 + i,
                "line": 1.5,
                "probability": 0.60,
                "outcome": 1 if i % 2 == 0 else 0,
            }],
        }
        for i in range(20)
    ]

    report = prop_oos.summarize_oos(rows)
    shots = report["families"]["SHOTS"]
    assert shots["unique_fixtures"] == 20
    assert shots["unique_player_fixtures"] == 20
    assert shots["brier_score"] is not None
    assert shots["log_loss"] is not None
    assert shots["expected_count_mae"] == 1.0
    assert shots["expected_minutes_mae"] == 10.0
    assert shots["oos_validation_complete"] is False
    assert shots["minimum_player_games_for_review"] == 500


def test_phase15_uses_dedicated_oos_report_instead_of_static_sanity_flag():
    oos = {
        "families": {
            family: {
                "status": "OOS_REVIEW_READY",
                "player_game_rows": 1200,
                "binary_probability_rows": 2400,
                "unique_fixtures": 120,
                "unique_player_fixtures": 1200,
                "minimum_player_games_for_review": 500,
                "sample_target_met": True,
                "oos_validation_complete": True,
                "brier_score": 0.20,
                "log_loss": 0.60,
                "expected_count_mae": 0.9,
                "expected_count_rmse": 1.2,
                "expected_minutes_mae": 8.0,
                "calibration_ece": 0.04,
                "calibration_bins": [],
            }
            for family in ("SHOTS", "SOT", "GOALSCORER_ANYTIME", "ASSISTS", "PLAYER_CARDS", "GK_SAVES")
        }
    }
    audit = {
        "families": {
            family: {
                "market_snapshot_rows": 10,
                "priced_value_rows": 20,
                "exact_line_value_rows": 20,
                "unique_fixtures": 5,
                "pre_kickoff_unique_fixtures": 5,
                "provider_update_unique_fixtures": 5,
                "confirmed_xi_pre_kickoff_unique_fixtures": 5,
                "bookmaker_count": 2,
                "xi_aligned_value_rows": 20,
                "xi_aligned_priced_value_rows": 20,
                "xi_aligned_exact_line_value_rows": 20,
                "confirmed_xi_player_aligned_unique_fixtures": 5,
                "exact_observed_market_history_materialized": True,
                "confirmed_xi_overlap_materialized": True,
                "player_xi_alignment_materialized": True,
            }
            for family in ("SHOTS", "SOT", "GOALSCORER_ANYTIME", "ASSISTS", "PLAYER_CARDS", "GK_SAVES")
        }
    }

    report = v.build_report(
        dict(BASE), dict(BASE), dict(BASE), dict(BASE), dict(BASE), dict(BASE),
        [], audit, oos
    )

    for prop in v.PROP_KEYS:
        assert report["prop_families"][prop]["oos_validation_complete"] is True
        assert report["prop_families"][prop]["oos_evidence"]["status"] == "OOS_REVIEW_READY"
        assert f"{prop.upper()}_OOS_VALIDATION_INCOMPLETE" not in report["blockers"]

    assert "PLAYER_PROP_OOS_LEDGER_NOT_MATERIALIZED" not in report["blockers"]
    assert "EXPECTED_MINUTES_OOS_VALIDATION_NOT_MATERIALIZED" not in report["blockers"]
    assert "PROP_SPECIFIC_CALIBRATION_NOT_MATERIALIZED" not in report["blockers"]
    assert report["production_promotion_allowed"] is False


def test_player_props_oos_excludes_confirmed_but_unmodelable_players_from_sample():
    pregame = [{
        "fixture_id": 7100,
        "stage": "T-10",
        "generated_at": "2026-09-25T10:50:00+00:00",
        "kickoff": "2026-09-25T11:00:00+00:00",
        "event_payload": {
            "stage": "T-10",
            "fixture": {"fixture_id": 7100, "kickoff": "2026-09-25T11:00:00+00:00"},
            "player_shots_intelligence": {
                "players": [
                    {
                        "player_id": 501,
                        "player": "Modelable",
                        "team_id": 10,
                        "confirmed_starter": True,
                        "expected_minutes_if_confirmed_starter": 80.0,
                        "expected_shots": 2.2,
                        "lines": [{"line": 1.5, "p_over": 0.61, "p_under": 0.39}],
                    },
                    {
                        "player_id": 502,
                        "player": "Unmodelable",
                        "team_id": 10,
                        "confirmed_starter": True,
                        "status": "PROFILE_NOT_MODELABLE",
                        "lines": [],
                    },
                ]
            },
        },
    }]
    postgame = [{
        "fixture_id": 7100,
        "stage": "POSTGAME",
        "generated_at": "2026-09-25T13:00:00+00:00",
        "event_payload": {
            "fixture": {"fixture_id": 7100},
            "postgame_player_stats": {
                "teams": [{
                    "team_id": 10,
                    "players": [
                        {"player_id": 501, "minutes": 90, "shots": 2},
                        {"player_id": 502, "minutes": 90, "shots": 1},
                    ],
                }],
            },
        },
    }]

    rows = prop_oos.build_oos_rows(pregame, postgame)
    assert len(rows) == 1
    assert rows[0]["player_id"] == 501
    assert len(rows[0]["binary_rows"]) == 1


def test_phase15_coverage_audit_separates_future_capture_from_provider_backfill():
    pregame = {
        "fixture_id": 8001,
        "stage": "T-10",
        "generated_at": "2026-09-25T10:50:00+00:00",
        "kickoff": "2026-09-25T11:00:00+00:00",
        "event_payload": {
            "stage": "T-10",
            "fixture": {"fixture_id": 8001, "kickoff": "2026-09-25T11:00:00+00:00"},
            "coverage": {"statistics_players": True},
            "lineups": _xi_lineup(),
            "player_shots_intelligence": {
                "players": [{
                    "player_id": 501,
                    "player": "Player A",
                    "team_id": 10,
                    "confirmed_starter": True,
                    "lines": [{"line": 2.5, "p_over": 0.55, "p_under": 0.45}],
                }]
            },
        },
    }

    shots = coverage_audit.classify_fixture_family(
        pregame,
        family="SHOTS",
        postgame_row=None,
        finalized_result_exists=True,
    )
    assert shots["oos_reason"] == "POSTGAME_EVENT_MISSING"
    assert shots["oos_recoverability"] == "PROVIDER_BACKFILL_CANDIDATE"
    assert shots["clv_reason"] == "PLAYER_PROP_MARKET_NOT_CAPTURED"
    assert shots["clv_recoverability"] == "FUTURE_CAPTURE_ONLY"

    cards = coverage_audit.classify_fixture_family(
        pregame,
        family="PLAYER_CARDS",
        postgame_row=None,
        finalized_result_exists=True,
    )
    assert cards["oos_reason"] == "NO_PREGAME_MODEL_SIGNAL"
    assert cards["oos_recoverability"] == "FUTURE_CAPTURE_ONLY"
    assert cards["clv_reason"] == "NO_PREGAME_MODEL_SIGNAL"


def test_phase15_coverage_audit_flags_reconciliation_when_player_ids_do_not_overlap():
    pregame = {
        "fixture_id": 8002,
        "stage": "T-10",
        "generated_at": "2026-09-25T10:50:00+00:00",
        "kickoff": "2026-09-25T11:00:00+00:00",
        "event_payload": {
            "stage": "T-10",
            "fixture": {"fixture_id": 8002, "kickoff": "2026-09-25T11:00:00+00:00"},
            "coverage": {"statistics_players": True},
            "lineups": _xi_lineup(),
            "player_shots_intelligence": {
                "players": [{
                    "player_id": 501,
                    "player": "Player A",
                    "team_id": 10,
                    "confirmed_starter": True,
                    "lines": [{"line": 2.5, "p_over": 0.55, "p_under": 0.45}],
                }]
            },
        },
    }
    postgame = {
        "fixture_id": 8002,
        "stage": "POSTGAME",
        "generated_at": "2026-09-25T13:00:00+00:00",
        "event_payload": {
            "fixture": {"fixture_id": 8002},
            "postgame_player_stats": {
                "teams": [{
                    "team_id": 10,
                    "players": [{"player_id": 999, "minutes": 90, "shots": 3}],
                }]
            },
        },
    }

    audit = coverage_audit.classify_fixture_family(
        pregame,
        family="SHOTS",
        postgame_row=postgame,
        finalized_result_exists=True,
    )
    assert audit["oos_reason"] == "PLAYER_ID_OVERLAP_MISSING"
    assert audit["oos_recoverability"] == "RECONCILIATION_CANDIDATE"


def test_phase15_coverage_audit_marks_ready_oos_and_entry_signal_when_evidence_exists():
    pregame = {
        "fixture_id": 8003,
        "stage": "T-10",
        "generated_at": "2026-09-25T10:50:00+00:00",
        "kickoff": "2026-09-25T11:00:00+00:00",
        "event_payload": {
            "stage": "T-10",
            "fixture": {"fixture_id": 8003, "kickoff": "2026-09-25T11:00:00+00:00"},
            "coverage": {"statistics_players": True},
            "lineups": _xi_lineup(),
            "player_shots_intelligence": {
                "players": [{
                    "player_id": 501,
                    "player": "Player A",
                    "team_id": 10,
                    "confirmed_starter": True,
                    "lines": [{"line": 2.5, "p_over": 0.55, "p_under": 0.45}],
                }]
            },
            "market": {
                "research_cards_props_markets": [{
                    "research_family": "PLAYER_PROPS",
                    "research_subfamily": "SHOTS",
                    "market": "Player Shots",
                    "values": [
                        {
                            "selection": "Player A Over 2.5",
                            "price": "1.90",
                            "parsed_line": 2.5,
                        },
                        {
                            "selection": "Player A Under 2.5",
                            "price": "1.90",
                            "parsed_line": 2.5,
                        },
                    ],
                }]
            },
        },
    }
    postgame = {
        "fixture_id": 8003,
        "stage": "POSTGAME",
        "generated_at": "2026-09-25T13:00:00+00:00",
        "event_payload": {
            "fixture": {"fixture_id": 8003},
            "postgame_player_stats": {
                "teams": [{
                    "team_id": 10,
                    "players": [{"player_id": 501, "minutes": 90, "shots": 3}],
                }]
            },
        },
    }

    audit = coverage_audit.classify_fixture_family(
        pregame,
        family="SHOTS",
        postgame_row=postgame,
        finalized_result_exists=True,
    )
    assert audit["oos_reason"] == "READY_OOS"
    assert audit["oos_recoverability"] == "ALREADY_MATERIALIZED"
    assert audit["clv_reason"] == "ENTRY_SIGNAL_READY"
    assert audit["model_price_player_overlap_count"] == 1


def test_phase15_backfill_selects_unique_provider_candidates_only():
    audit = {
        "fixtures": [
            {
                "fixture_id": 1,
                "families": {
                    "SHOTS": {"oos_recoverability": "PROVIDER_BACKFILL_CANDIDATE"},
                    "SOT": {"oos_recoverability": "PROVIDER_BACKFILL_CANDIDATE"},
                },
            },
            {
                "fixture_id": 2,
                "families": {
                    "SHOTS": {"oos_recoverability": "FUTURE_CAPTURE_ONLY"},
                },
            },
            {
                "fixture_id": 3,
                "families": {
                    "PLAYER_CARDS": {"oos_recoverability": "PROVIDER_BACKFILL_CANDIDATE"},
                },
            },
        ]
    }
    assert prop_backfill.select_candidate_fixture_ids(audit, max_fixtures=5) == [1, 3]


def test_phase15_backfill_event_never_creates_retroactive_pregame_evidence():
    compact = {
        "status": "RESEARCH_ONLY_PLAYER_FIXTURE_STATS",
        "teams": [{
            "team_id": 10,
            "team": "Home",
            "players": [{"player_id": 501, "name": "Player A", "shots": 3}],
        }],
    }
    event = prop_backfill.make_backfill_event(
        9001,
        compact,
        provider_daily_remaining=7000,
    )
    assert event["stage"] == "POSTGAME_BACKFILL"
    assert event["event_type"] == "RESEARCH_BACKFILL"
    assert event["postgame_player_stats"]["capture_phase"] == "POSTGAME_BACKFILL"
    assert event["backfill"]["pregame_signal_required"] is True
    assert event["backfill"]["retroactive_pregame_signal_created"] is False
    assert event["backfill"]["retroactive_market_created"] is False
    assert event["actionable"] is False
    assert event["decision_weight"] == 0.0


def test_player_props_clv_accepts_t30_and_n_plus_line_as_real_prediction_point():
    lineup = {
        "both_xi_confirmed": True,
        "both_goalkeepers_confirmed": True,
        "teams": [
            {
                "team_id": 10,
                "team": "Home",
                "starters": [{"id": 501, "name": "Player A", "pos": "F"}],
            },
            {
                "team_id": 20,
                "team": "Away",
                "starters": [{"id": 601, "name": "Keeper B", "pos": "G"}],
            },
        ],
    }
    market = base_automation._compact_odds(
        {
            "response": [{
                "update": "2026-09-25T10:28:00+00:00",
                "bookmakers": [{
                    "id": 1,
                    "name": "Book",
                    "bets": [{
                        "id": 801,
                        "name": "Player Shots",
                        "values": [{"value": "Player A - 3", "odd": "1.90"}],
                    }],
                }],
            }],
        },
        lineup=lineup,
    )
    event = {
        "fixture_id": 9100,
        "generated_at": "2026-09-25T10:30:00+00:00",
        "stage": "T-30",
        "kickoff": "2026-09-25T11:00:00+00:00",
        "event_payload": {
            "stage": "T-30",
            "fixture": {"fixture_id": 9100, "kickoff": "2026-09-25T11:00:00+00:00"},
            "lineups": lineup,
            "player_shots_intelligence": {
                "players": [{
                    "player_id": 501,
                    "player": "Player A",
                    "team_id": 10,
                    "confirmed_starter": True,
                    "expected_minutes_if_confirmed_starter": 82.0,
                    "lines": [{"line": 2.5, "p_over": 0.54, "p_under": 0.46}],
                }]
            },
            "market": market,
        },
    }

    signals = prop_clv.extract_shadow_signals([event])
    assert len(signals) == 1
    assert signals[0]["stage"] == "T-30"
    assert signals[0]["line"] == 2.5
    assert signals[0]["side"] == "OVER"
    assert signals[0]["model_probability"] == 0.54
    assert signals[0]["player_id"] == 501


def test_player_props_oos_t30_is_valid_but_t20_and_t10_remain_higher_priority():
    base_payload = {
        "fixture": {"fixture_id": 9200, "kickoff": "2026-09-25T11:00:00+00:00"},
    }
    rows = [
        {
            "fixture_id": 9200,
            "stage": "T-40",
            "generated_at": "2026-09-25T10:20:00+00:00",
            "kickoff": "2026-09-25T11:00:00+00:00",
            "event_payload": {**base_payload, "stage": "T-40"},
        },
        {
            "fixture_id": 9200,
            "stage": "T-30",
            "generated_at": "2026-09-25T10:30:00+00:00",
            "kickoff": "2026-09-25T11:00:00+00:00",
            "event_payload": {**base_payload, "stage": "T-30"},
        },
    ]
    out = prop_oos.choose_canonical_pregame_events(rows)
    assert len(out) == 1
    assert out[0]["stage"] == "T-30"

    rows.append({
        "fixture_id": 9200,
        "stage": "T-20",
        "generated_at": "2026-09-25T10:40:00+00:00",
        "kickoff": "2026-09-25T11:00:00+00:00",
        "event_payload": {**base_payload, "stage": "T-20"},
    })
    out = prop_oos.choose_canonical_pregame_events(rows)
    assert out[0]["stage"] == "T-20"


def test_clv_probability_reconciliation_distinguishes_missing_profile_from_line_mismatch():
    missing_player = {
        "player_id": 1,
        "player": "Missing Profile",
        "status": "PROFILE_NOT_MODELABLE",
        "confirmed_starter": True,
    }
    reason, detail = prop_clv._probability_reconciliation_reason(
        missing_player,
        mode="LINES",
        line=2.5,
        side="OVER",
    )
    assert reason == "PLAYER_MODEL_LINES_MISSING"
    assert detail["player_status"] == "PROFILE_NOT_MODELABLE"

    modeled_player = {
        "player_id": 2,
        "player": "Modeled",
        "status": "LIVE_RESEARCH_SHOTS_DISTRIBUTION",
        "confirmed_starter": True,
        "lines": [
            {"line": 0.5, "p_over": 0.80, "p_under": 0.20},
            {"line": 1.5, "p_over": 0.55, "p_under": 0.45},
        ],
    }
    reason, detail = prop_clv._probability_reconciliation_reason(
        modeled_player,
        mode="LINES",
        line=2.5,
        side="OVER",
    )
    assert reason == "MODEL_LINE_NOT_AVAILABLE"
    assert detail["available_lines"] == [0.5, 1.5]


def test_clv_probability_reconciliation_accepts_anytime_probability():
    player = {
        "player_id": 3,
        "player": "Scorer",
        "status": "LIVE_RESEARCH_GOALSCORER_DISTRIBUTION",
        "p_anytime_goal": 0.31,
    }
    reason, detail = prop_clv._probability_reconciliation_reason(
        player,
        mode="ANYTIME",
        line=None,
        side="PLAYER_EVENT",
    )
    assert reason == "OK"
    assert detail["probability_key"] == "p_anytime_goal"
    assert detail["probability"] == 0.31


def test_registry_backfill_event_is_future_only_and_never_reconstructs_oos():
    compact = {
        "status": "RESEARCH_ONLY_PLAYER_FIXTURE_STATS",
        "teams": [{
            "team_id": 10,
            "team": "Home",
            "players": [{"player_id": 501, "name": "Player A", "minutes": 90, "shots": 3}],
        }],
    }
    event = registry_backfill.make_registry_backfill_event(
        99001,
        "2026-09-25T18:00:00+00:00",
        compact,
        provider_daily_remaining=3500,
    )
    assert event["stage"] == "POSTGAME_REGISTRY_BACKFILL"
    assert event["event_type"] == "RESEARCH_BACKFILL"
    assert event["postgame_player_stats"]["capture_phase"] == "POSTGAME_REGISTRY_BACKFILL"
    assert event["postgame_player_stats"]["future_registry_use_only"] is True
    assert event["backfill"]["future_registry_use_only"] is True
    assert event["backfill"]["eligible_for_historical_oos_reconstruction"] is False
    assert event["backfill"]["retroactive_pregame_signal_created"] is False
    assert event["backfill"]["retroactive_market_created"] is False
    assert event["actionable"] is False
    assert event["decision_weight"] == 0.0


def test_registry_backfill_infers_conceded_only_for_full_match_goalkeeper():
    teams = [
        {
            "team_id": 10,
            "team": "Home",
            "players": [
                {
                    "player_id": 1,
                    "name": "Home GK",
                    "position": "G",
                    "minutes": 90,
                    "saves": 4,
                    "goals_conceded": None,
                },
                {
                    "player_id": 2,
                    "name": "Partial GK",
                    "position": "G",
                    "minutes": 45,
                    "saves": 2,
                    "goals_conceded": None,
                },
            ],
        },
        {
            "team_id": 20,
            "team": "Away",
            "players": [
                {
                    "player_id": 3,
                    "name": "Away GK",
                    "position": "GK",
                    "minutes": 90,
                    "saves": 5,
                    "goals_conceded": None,
                }
            ],
        },
    ]

    out = registry_backfill._enrich_full_match_goalkeeper_conceded(
        teams,
        home_team_id=10,
        away_team_id=20,
        home_goals=2,
        away_goals=1,
    )

    home = out[0]["players"]
    away = out[1]["players"]
    assert home[0]["goals_conceded"] == 1.0
    assert home[0]["goals_conceded_source"] == "FINAL_SCORE_FULL_MATCH_GK_FALLBACK"
    assert home[0]["goals_conceded_inferred_for_registry_only"] is True
    assert home[1]["goals_conceded"] is None
    assert "goals_conceded_source" not in home[1]
    assert away[0]["goals_conceded"] == 2.0


def test_registry_backfill_preserves_provider_goalkeeper_conceded_value():
    teams = [{
        "team_id": 10,
        "players": [{
            "player_id": 1,
            "position": "G",
            "minutes": 90,
            "saves": 3,
            "goals_conceded": 4,
        }],
    }]
    out = registry_backfill._enrich_full_match_goalkeeper_conceded(
        teams,
        home_team_id=10,
        away_team_id=20,
        home_goals=0,
        away_goals=1,
    )
    assert out[0]["players"][0]["goals_conceded"] == 4
    assert "goals_conceded_source" not in out[0]["players"][0]


def test_clv_close_diagnostics_distinguish_no_later_capture_from_stale_provider_update():
    signal = {
        "fixture_id": 5001,
        "kickoff": "2026-09-25T12:00:00+00:00",
        "signal_timestamp": "2026-09-25T11:30:00+00:00",
        "stage": "T-30",
        "market_family": "SHOTS",
        "player_id": 100,
        "side": "OVER",
        "line": 1.5,
        "entry_price": 1.9,
        "bookmaker_id": 1,
        "bookmaker": "Book",
    }
    earlier = {
        "fixture_id": 5001,
        "captured_at": "2026-09-25T11:29:00+00:00",
        "provider_update": "2026-09-25T11:28:00+00:00",
        "market": "Player Shots",
        "values": [{
            "player_id": 100,
            "xi_alignment_status": "MATCHED_CONFIRMED_XI",
            "selection": "Player A - 2",
            "parsed_line": 1.5,
            "price": 1.8,
        }],
        "confirmed_lineup_payload": {},
        "bookmaker_id": 1,
        "bookmaker": "Book",
    }

    diagnostics = {}
    rows, skip = prop_clv.pair_signals_to_closes(
        [signal],
        [earlier],
        diagnostics=diagnostics,
    )
    assert rows == []
    assert skip["NO_LATER_STRICT_PLAYER_PROP_CLOSE"] == 1
    assert diagnostics["failure_reasons"]["NO_LATER_CAPTURE_BEFORE_KICKOFF"] == 1

    stale = {
        **earlier,
        "captured_at": "2026-09-25T11:40:00+00:00",
        "provider_update": "2026-09-25T11:29:00+00:00",
    }
    diagnostics = {}
    rows, skip = prop_clv.pair_signals_to_closes(
        [signal],
        [stale],
        diagnostics=diagnostics,
    )
    assert rows == []
    assert diagnostics["failure_reasons"]["LATER_CAPTURE_PROVIDER_UPDATE_NOT_NEWER"] == 1


def test_clv_close_diagnostics_count_strict_close():
    signal = {
        "fixture_id": 5002,
        "kickoff": "2026-09-25T12:00:00+00:00",
        "signal_timestamp": "2026-09-25T11:30:00+00:00",
        "stage": "T-30",
        "market_family": "SHOTS",
        "player_id": 101,
        "side": "OVER",
        "line": 1.5,
        "entry_price": 1.9,
        "bookmaker_id": 1,
        "bookmaker": "Book",
    }
    snapshot = {
        "fixture_id": 5002,
        "captured_at": "2026-09-25T11:45:00+00:00",
        "provider_update": "2026-09-25T11:44:00+00:00",
        "market": "Player Shots",
        "values": [{
            "player_id": 101,
            "xi_alignment_status": "MATCHED_CONFIRMED_XI",
            "selection": "Player B - 2",
            "parsed_line": 1.5,
            "price": 1.8,
        }],
        "confirmed_lineup_payload": {},
        "bookmaker_id": 1,
        "bookmaker": "Book",
    }
    diagnostics = {}
    rows, skip = prop_clv.pair_signals_to_closes(
        [signal],
        [snapshot],
        diagnostics=diagnostics,
    )
    assert len(rows) == 1
    assert skip == {}
    assert diagnostics["by_family"]["SHOTS"]["strict_close_rows"] == 1


def test_clv_reclassifies_legacy_score_or_assist_row_out_of_goalscorer():
    event = {
        "market": {
            "research_cards_props_markets": [{
                "research_family": "PLAYER_PROPS",
                "research_subfamily": "GOALSCORER_ANYTIME",
                "market": "Player to Score or Assist",
                "values": [{"selection": "Player A", "price": 2.0}],
            }]
        }
    }
    assert prop_clv._event_market_rows(event, "GOALSCORER_ANYTIME") == []


def test_audit_does_not_classify_score_or_assist_as_anytime_scorer():
    from mcp_gateway import research_derivative_postgres_audit as derivative_audit
    assert derivative_audit.classify_market("Player to Score or Assist") is None
    assert derivative_audit.classify_market("Anytime Goal Scorer") == "GOALSCORER_ANYTIME"



def test_assists_and_player_cards_binary_probability_mapping_is_side_and_rung_aware():
    assists = {
        "p_1plus_assist": 0.30,
        "p_2plus_assists": 0.08,
    }
    cards = {
        "p_player_booked_yellow": 0.25,
        "p_2plus_yellow_cards": 0.04,
    }

    assert prop_clv._prob_from_model(assists, mode="ASSISTS", line=None, side="YES") == 0.30
    assert prop_clv._prob_from_model(assists, mode="ASSISTS", line=None, side="NO") == 0.70
    assert prop_clv._prob_from_model(assists, mode="ASSISTS", line=1.5, side="OVER") == 0.08
    assert prop_clv._prob_from_model(assists, mode="ASSISTS", line=1.5, side="UNDER") == 0.92
    assert prop_clv._prob_from_model(assists, mode="ASSISTS", line=2.5, side="OVER") is None

    assert prop_clv._prob_from_model(cards, mode="CARDS", line=None, side="YES") == 0.25
    assert prop_clv._prob_from_model(cards, mode="CARDS", line=None, side="NO") == 0.75
    assert prop_clv._prob_from_model(cards, mode="CARDS", line=1.5, side="OVER") == 0.04
    assert prop_clv._prob_from_model(cards, mode="CARDS", line=1.5, side="UNDER") == 0.96


def test_assists_binary_yes_no_shadow_signals_use_explicit_player_identity():
    lineup = {
        "both_xi_confirmed": True,
        "teams": [
            {
                "team_id": 10,
                "team": "Home",
                "starters": [{"id": 501, "name": "Player A", "pos": "M"}],
            },
            {
                "team_id": 20,
                "team": "Away",
                "starters": [{"id": 601, "name": "Player B", "pos": "F"}],
            },
        ],
    }
    event = {
        "fixture_id": 9200,
        "generated_at": "2026-09-25T10:00:00+00:00",
        "stage": "T-20",
        "kickoff": "2026-09-25T11:00:00+00:00",
        "event_payload": {
            "stage": "T-20",
            "fixture": {"fixture_id": 9200, "kickoff": "2026-09-25T11:00:00+00:00"},
            "lineups": lineup,
            "player_assists_intelligence": {
                "players": [{
                    "player_id": 501,
                    "player": "Player A",
                    "team_id": 10,
                    "confirmed_starter": True,
                    "expected_minutes_if_confirmed_starter": 82.0,
                    "p_1plus_assist": 0.30,
                    "p_2plus_assists": 0.08,
                }]
            },
            "market": {
                "research_cards_props_markets": [{
                    "research_family": "PLAYER_PROPS",
                    "research_subfamily": "ASSISTS",
                    "market": "Player Assists",
                    "bookmaker": "Book",
                    "provider_update": "2026-09-25T09:58:00+00:00",
                    "values": [
                        {
                            "selection": "Yes",
                            "decimal_price": 3.20,
                            "player_id": 501,
                            "player_name": "Player A",
                        },
                        {
                            "selection": "No",
                            "decimal_price": 1.30,
                            "player_id": 501,
                            "player_name": "Player A",
                        },
                    ],
                }]
            },
        },
    }

    signals = prop_clv.extract_shadow_signals([event])
    assert len(signals) == 2
    yes = next(row for row in signals if row["side"] == "YES")
    no = next(row for row in signals if row["side"] == "NO")
    assert yes["market_family"] == "ASSISTS"
    assert yes["model_probability"] == 0.30
    assert no["model_probability"] == 0.70
    assert yes["entry_market_fair_basis"] == "DEVIGGED_TWO_WAY"
    assert no["entry_market_fair_basis"] == "DEVIGGED_TWO_WAY"



def test_v156_gk_saves_uses_global_prior_without_claiming_player_specific_quality():
    prior = {"save_probability_proxy": 0.674157}
    row = gk_saves._keeper_save_probability(None, prior)

    assert row["status"] == "GLOBAL_PRIOR_ONLY_SAVE_RESULT_PROXY"
    assert row["save_probability"] == 0.674157
    assert row["player_specific_evidence_applied"] is False
    assert row["player_specific_weight"] == 0.0


def test_v156_gk_saves_models_confirmed_unknown_goalkeepers_with_full_market_ladder():
    event = {
        "fixture": {
            "fixture_id": 9300,
            "home_team_id": 10,
            "away_team_id": 20,
        },
        "lineups": {
            "both_goalkeepers_confirmed": True,
            "teams": [
                {
                    "team_id": 10,
                    "team": "Home",
                    "goalkeepers": [{"id": 501, "name": "Home GK"}],
                },
                {
                    "team_id": 20,
                    "team": "Away",
                    "goalkeepers": [{"id": 601, "name": "Away GK"}],
                },
            ],
        },
    }
    registry = {
        "goalkeepers": {
            "999": {
                "sample_band": "LOW",
                "windows": {
                    "last_20": {
                        "save_result_proxy_saves": 6.0,
                        "save_result_proxy_goals_conceded": 3.0,
                    }
                },
            }
        }
    }
    trends = {
        "global_context": {"avg_team_sot": 4.5},
        "teams": [],
    }

    report = gk_saves.build(event, registry, trends)

    assert report["status"] == "LIVE_RESEARCH_GK_SAVES"
    assert report["modeled_goalkeepers"] == 2
    assert report["prior_only_modeled_goalkeepers"] == 2
    assert report["player_specific_modeled_goalkeepers"] == 0
    for keeper in report["goalkeepers"]:
        assert keeper["status"] == "LIVE_RESEARCH_GK_SAVES_GLOBAL_PRIOR_DISTRIBUTION"
        assert keeper["save_profile"]["player_specific_evidence_applied"] is False
        lines = [row["line"] for row in keeper["lines"]]
        assert lines == [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
        overs = [row["p_over"] for row in keeper["lines"]]
        assert all(overs[i] >= overs[i + 1] for i in range(len(overs) - 1))


def test_v156_registry_backfill_marks_gk_saves_priority_future_only():
    compact = {
        "status": "RESEARCH_ONLY_PLAYER_FIXTURE_STATS",
        "teams": [{
            "team_id": 10,
            "team": "Home",
            "players": [{
                "player_id": 501,
                "name": "Home GK",
                "position": "G",
                "minutes": 90,
                "saves": 4,
                "goals_conceded": 1,
            }],
        }],
    }
    event = registry_backfill.make_registry_backfill_event(
        9301,
        "2026-09-25T20:00:00+00:00",
        compact,
        provider_daily_remaining=7000,
        priority_family="GK_SAVES",
    )

    assert registry_backfill.MAX_FIXTURES_PER_RUN == 16
    assert event["backfill"]["priority_family"] == "GK_SAVES"
    assert event["backfill"]["future_registry_use_only"] is True
    assert event["backfill"]["retroactive_pregame_signal_created"] is False
    assert event["backfill"]["eligible_for_historical_oos_reconstruction"] is False



def test_v156_unknown_goalkeeper_reaches_player_prop_shadow_signal_pipeline():
    lineup = {
        "both_goalkeepers_confirmed": True,
        "both_xi_confirmed": True,
        "teams": [
            {
                "team_id": 10,
                "team": "Home",
                "goalkeepers": [{"id": 501, "name": "Home GK"}],
                "starters": [{"id": 501, "name": "Home GK", "pos": "G"}],
            },
            {
                "team_id": 20,
                "team": "Away",
                "goalkeepers": [{"id": 601, "name": "Away GK"}],
                "starters": [{"id": 601, "name": "Away GK", "pos": "G"}],
            },
        ],
    }
    registry = {
        "goalkeepers": {
            "999": {
                "sample_band": "LOW",
                "windows": {
                    "last_20": {
                        "save_result_proxy_saves": 8.0,
                        "save_result_proxy_goals_conceded": 4.0,
                    }
                },
            }
        }
    }
    trends = {
        "global_context": {"avg_team_sot": 4.5},
        "teams": [],
    }
    model = gk_saves.build(
        {
            "fixture": {
                "fixture_id": 9400,
                "home_team_id": 10,
                "away_team_id": 20,
            },
            "lineups": lineup,
        },
        registry,
        trends,
    )
    home_model = next(row for row in model["goalkeepers"] if row["player_id"] == 501)
    expected = next(row["p_over"] for row in home_model["lines"] if row["line"] == 1.5)

    event = {
        "fixture_id": 9400,
        "generated_at": "2026-09-25T10:00:00+00:00",
        "stage": "T-20",
        "kickoff": "2026-09-25T11:00:00+00:00",
        "event_payload": {
            "stage": "T-20",
            "fixture": {"fixture_id": 9400, "kickoff": "2026-09-25T11:00:00+00:00"},
            "lineups": lineup,
            "gk_saves_intelligence": model,
            "market": {
                "research_cards_props_markets": [{
                    "research_family": "PLAYER_PROPS",
                    "research_subfamily": "GK_SAVES",
                    "market": "Goalkeeper Saves",
                    "bookmaker": "Book",
                    "provider_update": "2026-09-25T09:58:00+00:00",
                    "values": [{
                        "selection": "Home GK - 2",
                        "decimal_price": 1.95,
                        "parsed_line": 1.5,
                        "player_id": 501,
                        "player_name": "Home GK",
                        "xi_alignment_status": "MATCHED_CONFIRMED_XI",
                    }],
                }]
            },
        },
    }

    signals = prop_clv.extract_shadow_signals([event])
    assert len(signals) == 1
    assert signals[0]["market_family"] == "GK_SAVES"
    assert signals[0]["player_id"] == 501
    assert signals[0]["line"] == 1.5
    assert signals[0]["side"] == "OVER"
    assert signals[0]["model_probability"] == expected



def test_v156_goalkeeper_taxonomy_ignores_outfield_zero_save_placeholders():
    assert trend_analysis._is_goalkeeper_row({
        "position": "M",
        "saves": 0,
        "goals_conceded": None,
    }) is False
    assert trend_analysis._is_goalkeeper_row({
        "position": None,
        "saves": 0,
        "goals_conceded": None,
    }) is False
    assert trend_analysis._is_goalkeeper_row({
        "position": "G",
        "saves": 0,
        "goals_conceded": None,
    }) is True
    assert trend_analysis._is_goalkeeper_row({
        "position": None,
        "saves": 2,
        "goals_conceded": None,
    }) is True
    assert trend_analysis._is_goalkeeper_row({
        "position": None,
        "saves": 0,
        "goals_conceded": 1,
    }) is True
