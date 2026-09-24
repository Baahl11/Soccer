from datetime import datetime, timezone

from mcp_gateway import promotion_shadow_postgres_v4 as v


def _row(
    fid,
    ts,
    stage,
    selection,
    price,
    p_model,
    p_market,
    *,
    family="1X2",
    line=None,
    goals=(2, 1),
    model="SOCCER EDGE ENGINE v1.7",
    automation="4.31.0-price-resolver-v4",
):
    return {
        "fixture_id": fid,
        "generated_at": datetime.fromisoformat(ts).replace(tzinfo=timezone.utc),
        "stage": stage,
        "phase16_candidate": {
            "fixture_id": fid,
            "stage": stage,
            "market_family": family,
            "market": "Match Winner" if family == "1X2" else "Goals Over/Under" if family == "FT_TOTALS" else "Both Teams Score",
            "selection": selection,
            "line": line,
            "price": price,
            "bookmaker": "Book",
            "calibrated_probability": p_model,
            "market_fair_probability": p_market,
            "calibrated_edge_pp": (p_model - p_market) * 100.0,
            "mismatch_score": 70,
            "sport_confidence_score": 70,
            "data_quality_score": 75,
            "price_quality_score": 80,
            "uncertainty": 0.2,
            "rankable": True,
        },
        "kickoff": datetime(2026, 9, 20, 18, 0, tzinfo=timezone.utc),
        "league": "League A",
        "home_team": "Home",
        "away_team": "Away",
        "home_goals": None if goals is None else goals[0],
        "away_goals": None if goals is None else goals[1],
        "source_model_version": model,
        "automation_version": automation,
    }


def test_persisted_1x2_candidate_is_promotion_evaluable():
    report = v.build_report_from_rows([
        _row(1, "2026-09-20T17:40:00", "T-20", "home", 2.0, 0.60, 0.50),
    ])
    assert report["families"]["1X2"]["promotion_evaluable"]["settled"] == 1
    assert report["rows"][0]["signal_source"] == "PERSISTED_PHASE16_MARKET_MISMATCH"


def test_ft_totals_and_btts_settle_exactly():
    report = v.build_report_from_rows([
        _row(1, "2026-09-20T17:40:00", "T-20", "Over", 2.0, 0.60, 0.50, family="FT_TOTALS", line=2.5, goals=(2, 1)),
        _row(2, "2026-09-20T17:40:00", "T-20", "Under", 1.9, 0.60, 0.50, family="FT_TOTALS", line=2.5, goals=(1, 0)),
        _row(3, "2026-09-20T17:40:00", "T-20", "Yes", 1.8, 0.60, 0.50, family="BTTS", goals=(2, 1)),
        _row(4, "2026-09-20T17:40:00", "T-20", "Yes", 1.8, 0.60, 0.50, family="BTTS", goals=(2, 0)),
    ])
    totals = report["families"]["FT_TOTALS"]["promotion_evaluable"]
    btts = report["families"]["BTTS"]["promotion_evaluable"]
    assert totals["settled"] == 2 and totals["win"] == 2
    assert btts["settled"] == 2 and btts["win"] == 1 and btts["loss"] == 1


def test_integer_total_push_is_zero_roi_settlement():
    report = v.build_report_from_rows([
        _row(1, "2026-09-20T17:40:00", "T-20", "Over", 2.0, 0.60, 0.50, family="FT_TOTALS", line=3.0, goals=(2, 1)),
    ])
    row = report["rows"][0]
    assert row["outcome"] == "PUSH"
    assert row["roi_units"] == 0.0
    assert report["families"]["FT_TOTALS"]["promotion_evaluable"]["push"] == 1


def test_candidate_is_visible_pending_before_result_exists():
    report = v.build_report_from_rows([
        _row(1, "2026-09-20T17:40:00", "T-20", "Over", 2.0, 0.60, 0.50, family="FT_TOTALS", line=2.5, goals=None),
    ])
    evidence = report["families"]["FT_TOTALS"]["promotion_evaluable"]
    assert evidence["rows"] == 1
    assert evidence["settled"] == 0
    assert evidence["pending"] == 1
    assert report["rows"][0]["settlement_status"] == "PENDING"


def test_latest_pre_kickoff_candidate_wins_per_fixture_and_family():
    rows = [
        _row(1, "2026-09-20T17:20:00", "T-40", "home", 2.0, 0.60, 0.50),
        _row(1, "2026-09-20T17:50:00", "T-10", "home", 2.1, 0.58, 0.48),
        _row(1, "2026-09-20T17:45:00", "T-20", "Over", 1.9, 0.60, 0.50, family="FT_TOTALS", line=2.5),
    ]
    report = v.build_report_from_rows(rows)
    assert report["promotion_evaluable"]["unique_fixtures"] == 1
    assert len(report["rows"]) == 2
    one_x_two = next(row for row in report["rows"] if row["market_family"] == "1X2")
    assert one_x_two["stage"] == "T-10"


def test_latest_versioned_regime_prevents_mixed_history():
    rows = [
        _row(1, "2026-09-20T17:40:00", "T-20", "home", 2.0, 0.60, 0.50, model="SOCCER EDGE ENGINE v1.6"),
        _row(2, "2026-09-20T17:40:00", "T-20", "away", 3.0, 0.40, 0.30, goals=(0, 1), model="SOCCER EDGE ENGINE v1.7"),
    ]
    report = v.build_report_from_rows(rows)
    assert report["source_regime"] == "SOCCER EDGE ENGINE v1.7"
    assert report["promotion_evaluable"]["unique_fixtures"] == 1
    assert report["rows"][0]["fixture_id"] == 2


def test_no_provider_requests_or_runtime_mutation():
    report = v.build_report_from_rows([
        _row(1, "2026-09-20T17:50:00", "T-10", "home", 2.0, 0.60, 0.50),
    ])
    assert report["provider_requests_added"] == 0
    assert report["production_promotion_allowed"] is False
    assert report["runtime_logic_changed"] is False
