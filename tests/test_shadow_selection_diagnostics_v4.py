from mcp_gateway import shadow_selection_diagnostics_v4 as v


def _watch(fid, stage, selection, price, edge=4.0, ev=6.0):
    return {
        "fixture_id": fid,
        "generated_at_local": "2026-09-20T17:50:00+00:00",
        "kickoff_local": "2026-09-20T18:00:00+00:00",
        "stage": stage,
        "classification": "WATCH",
        "league": "League A",
        "home_team": "Home",
        "away_team": "Away",
        "best_market": {
            "market": "Match Winner",
            "selection": selection,
            "decimal_price": price,
            "prob_edge_pp": edge,
            "ev_pct": ev,
        },
    }


def _final(fid, home=2, away=1):
    return {
        "fixture_id": fid,
        "generated_at_local": "2026-09-20T20:00:00+00:00",
        "kickoff_local": "2026-09-20T18:00:00+00:00",
        "stage": "POSTGAME",
        "classification": "CLOSE",
        "result": {"goals": {"home": home, "away": away}},
    }


def test_selection_diagnostics_segments_stage_side_price_edge_and_ev():
    rows = [
        _watch(1, "T-10", "home", 1.70, edge=1.0, ev=2.0),
        _final(1),
        _watch(2, "T-20", "away", 3.20, edge=6.0, ev=12.0),
        _final(2, home=0, away=1),
    ]
    report = v.build(rows)

    assert report["rows"] == 2
    assert report["by_stage"]["T-10"]["by_selection_side"]["HOME"]["settled"] == 1
    assert report["by_stage"]["T-10"]["by_price_band"]["LT_1_80"]["settled"] == 1
    assert report["by_stage"]["T-10"]["by_edge_band"]["0_TO_1_99PP"]["settled"] == 1
    assert report["by_stage"]["T-20"]["by_ev_band"]["GE_10PCT"]["settled"] == 1


def test_selection_diagnostics_is_research_only():
    report = v.build([_watch(1, "T-10", "home", 2.0), _final(1)])
    assert report["provider_requests_added"] == 0
    assert report["production_promotion_allowed"] is False
    assert report["runtime_logic_changed"] is False
