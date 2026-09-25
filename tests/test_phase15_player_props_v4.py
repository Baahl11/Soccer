from mcp_gateway import player_props_phase15_v4 as v
from mcp_gateway import player_props_clv_postgres_v4 as prop_clv


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
