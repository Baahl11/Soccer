import asyncio

from mcp_gateway import automation_v122 as v


def test_v122_recomputes_phase16_after_price_resolution(monkeypatch):
    payload = {
        "events": [{
            "fixture": {
                "fixture_id": 1001,
                "kickoff": "2026-09-24T18:00:00+00:00",
                "league": "League A",
                "home_team": "Home",
                "away_team": "Away",
            },
            "raw_projection": {"raw_home_win_prob": 0.62, "raw_away_win_prob": 0.20},
            "stage": "T-20",
            "event_type": "SOCCER_REFRESH",
            "classification": "WATCH",
        }],
        "match_table_rows": [{
            "row_index": 0,
            "fixture_id": 1001,
            "stage": "T-20",
            "classification": "WATCH",
            "event_classification": "WATCH",
            "market_family": "FT_1X2_RESEARCH",
            "market": "Match Winner / Side",
            "selection": "Side research",
            "price": None,
            "data_tier": "A",
            "side_score": 80.0,
            "goals_score": 40.0,
            "two_way_score": 30.0,
            "execution_status": "WAIT_PRICE",
            "p_model_calibrated": 0.60,
            "blockers": ["WAIT_PRICE"],
            "reason": "SPORTING_SCREEN_PASS",
        }],
    }

    async def fake_run_tick():
        return payload

    async def fake_resolve(target):
        row = target["match_table_rows"][0]
        row.update({
            "market_family": "1X2",
            "market": "Match Winner",
            "selection": "Home",
            "price": 2.0,
            "bookmaker": "Book",
            "p_market_fair": 0.48,
            "price_resolution_status": "PRICE_API_RESOLVED",
        })
        target["events"][0]["raw_projection"].update({
            "raw_home_goal_rate": 1.7,
            "raw_away_goal_rate": 1.1,
        })
        target["events"][0]["fixture"].update({
            "home_team_id": 10,
            "away_team_id": 20,
        })
        target["events"][0]["market"] = {
            "source": "API_FOOTBALL_ODDS_V3",
            "resolution_status": "PRICE_API_RESOLVED",
            "markets": [{
                "market": "Home Team Total Goals",
                "bookmaker": "Book",
                "bookmaker_id": 99,
                "market_id": 10,
                "values": [
                    {"selection": "Over", "line": 1.5, "decimal_price": 1.95},
                    {"selection": "Under", "line": 1.5, "decimal_price": 1.85},
                ],
            }],
        }
        target["price_resolution_v4"] = {
            "status": "ACTIVE_PRICE_RESOLVER",
            "candidate_rows": 1,
            "unique_candidate_fixtures": 1,
            "api_calls_added": 1,
            "resolution_counts": {"PRICE_API_RESOLVED": 1},
        }
        return target["price_resolution_v4"]

    monkeypatch.setattr(v.v121, "run_tick", fake_run_tick)
    monkeypatch.setattr(v.price_resolver_v4, "resolve_payload", fake_resolve)

    out = asyncio.run(v.run_tick())

    assert out["match_table_rows"][0]["execution_status"] == "RESEARCH_ONLY"
    assert out["phase16_market_mismatch_finder"]["rankable_rows"] == 1
    assert out["phase16_market_mismatch_finder"]["primary_candidate_count"] == 1
    assert out["market_mismatch_rows"][0]["selection"] == "Home"
    assert out["price_resolution_checkpoint"]["phase16_recomputed_after_price_resolution"] is True
    assert out["price_resolution_checkpoint"]["team_totals_recomputed_after_price_resolution"] is True
    assert out["price_resolution_checkpoint"]["calibrated_probability_fabricated"] is False
    assert out["team_totals_post_resolution"]["events_with_observed_team_total_markets"] == 1
    assert out["team_totals_post_resolution"]["observed_exact_market_rows"] == 2
    rows = out["events"][0]["team_totals_intelligence"]["observed_exact_market_rows"]
    assert {(row["selection"], row["line"]) for row in rows} == {("OVER", 1.5), ("UNDER", 1.5)}
    assert out["price_resolution_v4"]["api_calls_added"] == 1
    assert out["version"] == "4.31.1-price-resolver-team-totals-refresh"
