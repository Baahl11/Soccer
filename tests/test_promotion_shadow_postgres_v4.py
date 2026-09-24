from datetime import datetime, timezone

from mcp_gateway import promotion_shadow_postgres_v4 as v


def _row(fid, ts, stage, selection, price, p_model, p_market, goals=(2, 1), model="SOCCER EDGE ENGINE v1.7"):
    return {
        "fixture_id": fid,
        "generated_at": datetime.fromisoformat(ts).replace(tzinfo=timezone.utc),
        "stage": stage,
        "match_table_row": {
            "fixture_id": fid,
            "stage": stage,
            "market_family": "1X2",
            "market": "Match Winner",
            "selection": selection,
            "price": price,
            "bookmaker": "Book",
            "model_signal_score": 70,
            "data_tier": "A",
            "p_market_fair": p_market,
            "p_model_calibrated": p_model,
            "uncertainty": 0.2,
            "blockers": [],
        },
        "kickoff": datetime(2026, 9, 20, 18, 0, tzinfo=timezone.utc),
        "league": "League A",
        "home_team": "Home",
        "away_team": "Away",
        "home_goals": goals[0],
        "away_goals": goals[1],
        "source_model_version": model,
        "automation_version": "1.0",
    }


def test_phase16_replay_keeps_one_primary_1x2_per_fixture_run():
    rows = [
        _row(1, "2026-09-20T17:40:00", "T-20", "home", 2.0, 0.60, 0.50),
        _row(1, "2026-09-20T17:40:00", "T-20", "away", 4.0, 0.30, 0.25),
    ]
    report = v.build_report_from_rows(rows)
    assert report["promotion_evaluable"]["settled"] == 1
    assert len(report["rows"]) == 1
    assert report["rows"][0]["selection"] == "home"


def test_latest_pre_kickoff_candidate_wins_per_fixture():
    rows = [
        _row(1, "2026-09-20T17:20:00", "T-40", "home", 2.0, 0.60, 0.50),
        _row(1, "2026-09-20T17:50:00", "T-10", "home", 2.1, 0.58, 0.48),
    ]
    report = v.build_report_from_rows(rows)
    assert len(report["rows"]) == 1
    assert report["rows"][0]["stage"] == "T-10"
    assert report["rows"][0]["decimal_price"] == 2.1


def test_current_model_only_prevents_mixed_regimes():
    rows = [
        _row(1, "2026-09-20T17:40:00", "T-20", "home", 2.0, 0.60, 0.50, model="SOCCER EDGE ENGINE v1.6"),
        _row(2, "2026-09-20T17:40:00", "T-20", "away", 3.0, 0.40, 0.30, goals=(0, 1), model="SOCCER EDGE ENGINE v1.7"),
    ]
    report = v.build_report_from_rows(rows)
    assert report["source_model_version"] == "SOCCER EDGE ENGINE v1.7"
    assert report["promotion_evaluable"]["unique_fixtures"] == 1
    assert report["rows"][0]["fixture_id"] == 2


def test_no_provider_requests_or_runtime_mutation():
    report = v.build_report_from_rows([
        _row(1, "2026-09-20T17:50:00", "T-10", "home", 2.0, 0.60, 0.50),
    ])
    assert report["provider_requests_added"] == 0
    assert report["production_promotion_allowed"] is False
    assert report["runtime_logic_changed"] is False
