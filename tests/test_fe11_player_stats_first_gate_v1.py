"""Regression: a bookmaker quote is not a verified player prop projection.

Synthetic test inputs are NOT sporting observations or real odds. This suite
checks the system cannot rank/publish player choices based on price alone.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from mcp_gateway import subscriber_research_desk_v1 as desk
from mcp_gateway import subscriber_frontend_v2


def book_quote(i: int, price: str, *, fresh: bool = True) -> dict:
    at = datetime(2026, 11, 7, 12, 0, tzinfo=timezone.utc)
    return {
        "fixture_id": 1800000 + i % 2,
        "home_team": "TEST HOME", "away_team": "TEST AWAY",
        "league_name": "TEST ONLY", "kickoff_utc": at,
        "bookmaker_id": 8, "provider_market": "Home Player Shots",
        "provider_market_id": 240, "selection": f"TEST PLAYER {i} - 3",
        "player_label": f"TEST PLAYER {i}", "observed_team_side": "HOME",
        "offered_price_decimal": price,
        "provider_updated_at": at, "captured_at": at,
        "offer_type": "PLAYER_COMPONENT_QUOTE",
        "quote_freshness": "CURRENT_QUOTE" if fresh else "NEEDS_RECONFIRMATION",
        "classification": "WATCH", "bet_eligible": False,
        "bookmaker_direct_combo_quote": False, "research_flags": [],
    }


def test_ten_near_plus110_prices_cannot_create_one_player_pick():
    rows = [book_quote(i, "2.10") for i in range(10)]
    board = desk.make_board([], rows)
    assert board["sgp_player_quote_catalog_count"] == 10
    assert board["sgp_player_components"] == []
    assert board["sgp_player_statistically_validated_count"] == 0
    assert board["sgp_player_quote_price_used_for_ranking"] is False
    assert board["sgp_player_prices_are_picks"] is False
    assert board["bets"] == 0 and board["sgp_player_builder_enabled"] is False
    assert board["status"] == "PLAYER_QUOTES_ONLY_NO_VERIFIED_SPORTING_EDGE"


def test_price_diversity_or_quote_freshness_cannot_choose_an_unmodeled_player():
    rows = [
        book_quote(1, "1.45"),
        book_quote(2, "1.80"),
        book_quote(3, "2.10"),
        book_quote(4, "2.35"),
        book_quote(5, "3.90"),
        book_quote(6, "2.10", fresh=False),
    ]
    result = desk._sgp_market_rows(rows)
    assert result["sgp_player_quote_catalog_count"] == len(rows)
    assert result["sgp_player_components"] == []
    assert result["sgp_player_selection_policy"] == "SPORT_FIRST_STATS_AND_OOS_BEFORE_MARKET"
    assert "INDEPENDENT_OOS_PROBABILITY_VALIDATION" in result["sgp_player_verification_needed"]
    assert "PLAYER_SHOTS_OR_SOT_RATE_AND_EXPECTED_MINUTES" in result["sgp_player_verification_needed"]


def test_spoofed_player_quote_cannot_self_certify_stats_or_become_pick():
    quote = book_quote(1, "2.10")
    quote.update(
        model_validated=True, oos_validation_complete=True,
        historical_stats_verified=True, available=True, player_id=54321,
        expected_shots=3.5, raw_sport_probability=.67,
        starting_xi_verified=True,
    )
    board = desk.make_board([], [quote])
    assert board["sgp_player_components"] == []
    assert board["sgp_player_quote_catalog_count"] == 1
    assert board["sgp_player_prices_are_picks"] is False


def test_pricing_floor_is_never_used_to_rank_player_candidates():
    source = Path(desk.__file__).read_text()
    assert 'abs(x["price_decimal"]-2.10)' not in source.split("verified_player_candidates:")[0].split("# HARD SPORT-FIRST GATE:")[-1]
    assert "players.sort(" not in source
    assert "if is_direct and price < 21/11:" in source
    assert "sgp_player_components\": verified_player_candidates" in source
    assert '"sgp_player_quote_price_used_for_ranking": False' in source


def test_frontend_rejects_price_only_data_and_shows_no_unmodeled_picks():
    html = subscriber_frontend_v2.render()
    assert "rawPlayerRows.filter(playerStatEvidenceEligible)" in html
    assert "function playerStatEvidenceEligible(x)" in html
    assert "e.projection_before_market===true" in html
    assert "e.oos_validation_complete===true" in html
    assert "e.historical_stats_verified===true" in html
    assert "e.exact_market_operator_verified===true" in html
    assert "e.starting_xi_verified===true" in html
    assert "0 jugadores con ventaja estadística verificada" in html
    assert "Se detectaron " in html
    assert "Una cuota +110 NO es una proyección" in html
    assert "No se crea borrador con cotizaciones aisladas" in html
