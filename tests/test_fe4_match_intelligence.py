from __future__ import annotations

from mcp_gateway import subscriber_contract_v2
from mcp_gateway import subscriber_frontend_v2


def _row(family: str, market: str, **extra):
    row = {
        "fixture_id": 9911,
        "kickoff": "2026-10-07T22:30:00Z",
        "league": "Test League",
        "country": "Test",
        "home_team_id": 101,
        "home_team": "Alpha FC",
        "away_team_id": 202,
        "away_team": "Beta FC",
        "market_family": family,
        "market": market,
        "execution_status": "WAIT_MARKET",
        "classification": "WATCH",
        "stage": "T-40",
        "reason": "WAIT_MARKET",
        "model_version": "soccer-v-test",
    }
    row.update(extra)
    return row


def _payload():
    return {
        "status": "ok",
        "version": "pipeline-test",
        "model_version": "runtime-test",
        "generated_at_utc": "2026-10-07T21:45:00Z",
        "match_table_rows": [
            _row(
                "1X2",
                "Match Winner",
                selection="Alpha FC",
                p_raw=0.58,
                p_shrunk=0.55,
                p_market_fair=0.51,
                p_breakeven=0.50,
                prob_edge_pp=4.0,
                decimal_price=2.00,
                bookmaker="Verified Book",
                availability_confidence=0.91,
                data_tier="A",
                lineup_status="CONFIRMED",
                prob_home_win=0.58,
                prob_draw=0.25,
                prob_away_win=0.17,
                lambda_home=1.70,
                lambda_away=0.92,
                score_matrix={"1-0": 0.14, "2-0": 0.12, "1-1": 0.10},
                sport_profile={
                    "attack_strength": 78,
                    "defense_strength": 66,
                    "form_score": 72,
                },
            ),
            _row("FT_TOTALS", "Goals Over/Under", selection="Over", line=2.5),
            _row("CORNERS", "Total Corners", selection="Over", line=9.5),
            _row("CARDS", "Total Cards", selection="Over", line=4.5),
            _row("PLAYER_PROP", "Player Shots", selection="Player A Over", line=2.5),
        ],
    }


def test_fe4_match_contract_groups_persisted_market_families():
    result = subscriber_contract_v2.build_match_contract(_payload(), 9911)
    assert result is not None

    groups = result["market_context"]["groups"]
    assert len(groups["GOALS"]) == 1
    assert len(groups["CORNERS"]) == 1
    assert len(groups["CARDS"]) == 1
    assert len(groups["PLAYERS"]) == 1
    assert len(groups["GENERAL"]) == 1


def test_fe4_match_contract_keeps_sport_market_and_availability_layers_distinct():
    result = subscriber_contract_v2.build_match_contract(_payload(), 9911)
    assert result is not None

    ladder = result["projection_ladder"]
    assert ladder["raw_sport_probability"] == 0.58
    assert ladder["market_shrunk_probability"] == 0.55
    assert ladder["fair_market_probability"] == 0.51
    assert ladder["breakeven_probability"] == 0.50
    assert ladder["probability_edge_pp"] == 4.0

    availability = result["availability"]
    assert availability["confidence"] == 0.91
    assert availability["data_tier"] == "A"

    sport = result["sport_context"]
    assert sport["expected_goals"]["home"] == 1.70
    assert sport["expected_goals"]["away"] == 0.92
    assert sport["outcome_probabilities"]["home"] == 0.58
    assert sport["score_matrix"]


def test_fe4_match_contract_reports_missing_sections_instead_of_fabricating_them():
    payload = {
        "status": "ok",
        "generated_at_utc": "2026-10-07T21:45:00Z",
        "match_table_rows": [_row("1X2", "Match Winner")],
    }
    result = subscriber_contract_v2.build_match_contract(payload, 9911)
    assert result is not None

    missing = set(result["data_disclosure"]["missing_sections"])
    assert "OUTCOME_PROBABILITIES" in missing
    assert "EXPECTED_GOALS" in missing
    assert "SCORE_MATRIX" in missing
    assert "SPORT_PROFILE" in missing
    assert "AVAILABILITY_CONFIDENCE" in missing
    assert "EXACT_PRICE" in missing
    assert result["data_disclosure"]["unknown_policy"] == "NOT VERIFIED"


def test_fe4_match_contract_is_read_only_and_adds_no_provider_requests():
    result = subscriber_contract_v2.build_match_contract(_payload(), 9911)
    assert result is not None

    assert result["provider_requests_added"] == 0
    assert result["data_disclosure"]["provider_requests_added"] == 0
    assert result["canonical_bet_logic_changed"] is False
    assert result["model_weights_changed"] is False
    assert result["production_promotion_allowed"] is False


def test_fe4_match_contract_returns_none_for_unknown_fixture():
    assert subscriber_contract_v2.build_match_contract(_payload(), 123456) is None


def test_fe4_frontend_contains_all_match_intelligence_tabs():
    html = subscriber_frontend_v2.render()

    for tab in (
        "Overview",
        "Sport",
        "Goals",
        "Corners",
        "Cards",
        "Players",
        "Availability",
        "Market",
        "Model",
    ):
        assert f">{tab}</button>" in html

    assert "data-detail-tab" in html
    assert "data-detail-section" in html
    assert "Raw Sport" in html or "RAW SPORT" in html
    assert "Market shrink" in html or "MARKET SHRUNK" in html


def test_fe4_frontend_preserves_missing_data_language():
    html = subscriber_frontend_v2.render()

    assert "NOT VERIFIED" in html
    assert "Material unverified availability blocks BET eligibility" in html
    assert "No persisted market rows" in html
