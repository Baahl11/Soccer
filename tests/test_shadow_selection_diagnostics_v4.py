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


def test_discrepancy_recheck_bucket_uses_raw_market_gap():
    row = _watch(1, "T-10", "home", 3.5, edge=5.0, ev=8.0)
    row["best_market"]["p_raw"] = 0.45
    row["best_market"]["p_market_fair"] = 0.25
    rows = [row, _final(1)]
    report = v.build(rows)
    stage = report["by_stage"]["T-10"]
    assert stage["discrepancy_recheck_rows"] == 1
    assert stage["by_watch_quality_bucket"]["DISCREPANCY_RECHECK"]["settled"] == 1
    assert stage["by_raw_market_gap_band"]["GE_20PP_RECHECK"]["settled"] == 1


def test_only_clean_advanced_tier_cap_counts_as_promotion_evaluable():
    clean = _watch(1, "T-10", "home", 2.0, edge=6.0, ev=8.0)
    clean["availability_confidence"] = 0.90
    clean["lineups"] = {"both_xi_confirmed": True, "both_goalkeepers_confirmed": True}
    clean["best_market"]["tier"] = "A"
    clean["best_market"]["p_raw"] = 0.50
    clean["best_market"]["p_market_fair"] = 0.45

    recheck = _watch(2, "T-10", "away", 3.0, edge=8.0, ev=10.0)
    recheck["availability_confidence"] = 0.90
    recheck["lineups"] = {"both_xi_confirmed": True, "both_goalkeepers_confirmed": True}
    recheck["best_market"]["tier"] = "A"
    recheck["best_market"]["p_raw"] = 0.50
    recheck["best_market"]["p_market_fair"] = 0.30

    report = v.build([clean, _final(1), recheck, _final(2, home=0, away=1)])

    assert report["promotion_evaluable"]["settled"] == 1
    assert report["promotion_evaluable"]["unique_fixtures"] == 1
    assert report["watch_alert"]["settled"] == 1
    assert report["sample_policy"]["promotion_evidence_requires_quality_bucket"] == "ADVANCED_TIER_CAP"
