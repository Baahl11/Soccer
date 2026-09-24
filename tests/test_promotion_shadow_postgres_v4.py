from datetime import datetime, timezone

from mcp_gateway import promotion_shadow_postgres_v4 as v


def _row(fid, ts, stage, selection, price, p_model, p_market, goals=(2, 1), model="SOCCER EDGE ENGINE v1.7", automation="SOCCER_EDGE_V121"):
    return {
        "fixture_id": fid,
        "generated_at": datetime.fromisoformat(ts).replace(tzinfo=timezone.utc),
        "stage": stage,
        "phase16_candidate": {
            "fixture_id": fid,
            "stage": stage,
            "market_family": "1X2",
            "market": "Match Winner",
            "selection": selection,
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
        "home_goals": goals[0],
        "away_goals": goals[1],
        "source_model_version": model,
        "automation_version": automation,
    }


def test_persisted_phase16_candidate_is_promotion_evaluable():
    report = v.build_report_from_rows([
        _row(1, "2026-09-20T17:40:00", "T-20", "home", 2.0, 0.60, 0.50),
    ])
    assert report["promotion_evaluable"]["settled"] == 1
    assert report["rows"][0]["signal_source"] == "PERSISTED_PHASE16_MARKET_MISMATCH"


def test_latest_pre_kickoff_candidate_wins_per_fixture():
    rows = [
        _row(1, "2026-09-20T17:20:00", "T-40", "home", 2.0, 0.60, 0.50),
        _row(1, "2026-09-20T17:50:00", "T-10", "home", 2.1, 0.58, 0.48),
    ]
    report = v.build_report_from_rows(rows)
    assert report["promotion_evaluable"]["unique_fixtures"] == 1
    assert report["rows"][0]["stage"] == "T-10"


def test_latest_versioned_regime_prevents_mixed_history():
    rows = [
        _row(1, "2026-09-20T17:40:00", "T-20", "home", 2.0, 0.60, 0.50, model="SOCCER EDGE ENGINE v1.6"),
        _row(2, "2026-09-20T17:40:00", "T-20", "away", 3.0, 0.40, 0.30, goals=(0, 1), model="SOCCER EDGE ENGINE v1.7"),
    ]
    report = v.build_report_from_rows(rows)
    assert report["source_regime"] == "SOCCER EDGE ENGINE v1.7"
    assert report["promotion_evaluable"]["unique_fixtures"] == 1
    assert report["rows"][0]["fixture_id"] == 2


def test_unversioned_history_is_not_dropped_when_no_regime_exists():
    row = _row(1, "2026-09-20T17:50:00", "T-10", "home", 2.0, 0.60, 0.50)
    row["source_model_version"] = None
    row["automation_version"] = None
    report = v.build_report_from_rows([row])
    assert report["source_regime"] is None
    assert report["promotion_evaluable"]["settled"] == 1


def test_no_provider_requests_or_runtime_mutation():
    report = v.build_report_from_rows([
        _row(1, "2026-09-20T17:50:00", "T-10", "home", 2.0, 0.60, 0.50),
    ])
    assert report["provider_requests_added"] == 0
    assert report["production_promotion_allowed"] is False
    assert report["runtime_logic_changed"] is False
