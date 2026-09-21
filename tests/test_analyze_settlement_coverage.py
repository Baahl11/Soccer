from mcp_gateway.analyze_settlement_coverage import build_report


def _row(event_key, classification="LEAN", fixture_id=1, generated="2026-09-21T10:00:00-06:00", result=None, best_market=None):
    return {
        "event_key": event_key,
        "fixture_id": fixture_id,
        "classification": classification,
        "generated_at_local": generated,
        "stage": "T-30",
        "league": "Test League",
        "home_team": "Home FC",
        "away_team": "Away FC",
        "result": result,
        "best_market": best_market,
    }


def _market(selection="Over 2.5"):
    return {"market": "Goals Over/Under", "selection": selection, "line": 2.5, "decimal_price": 1.9}


def test_coverage_marks_pending_final_result():
    source = [_row("a", result=None, best_market=_market())]
    report, backlog = build_report(source, [], [])
    assert report["source_actionable_rows"] == 1
    assert report["settlement_rows"] == 0
    assert report["by_reason"]["PENDING_FINAL_RESULT"] == 1
    assert backlog[0]["coverage_reason"] == "PENDING_FINAL_RESULT"


def test_coverage_marks_missing_market_with_final():
    source = [_row("a", result={"goals": {"home": 2, "away": 1}}, best_market=None)]
    report, backlog = build_report(source, [], [])
    assert report["by_reason"]["MISSING_BEST_MARKET_WITH_FINAL"] == 1
    assert backlog[0]["next_action"].startswith("store exact market")


def test_coverage_marks_in_settlement_ledger():
    source = [_row("a", result={"goals": {"home": 2, "away": 1}}, best_market=_market())]
    settlement = [{
        "fixture_id": 1,
        "classification": "LEAN",
        "market_family": "FT_TOTALS",
        "market": "Goals Over/Under",
        "selection": "Over 2.5",
        "line": 2.5,
        "settlement_status": "WIN",
    }]
    report, backlog = build_report(source, [], settlement)
    assert report["settlement_rows"] == 1
    assert report["backlog_rows"] == 0
    assert report["by_reason"]["IN_SETTLEMENT_LEDGER"] == 1


def test_coverage_marks_duplicate_older_snapshot():
    source = [
        _row("old", result={"goals": {"home": 2, "away": 1}}, best_market=_market(), generated="2026-09-21T10:00:00-06:00"),
        _row("new", result={"goals": {"home": 2, "away": 1}}, best_market=_market(), generated="2026-09-21T10:05:00-06:00"),
    ]
    settlement = [{
        "fixture_id": 1,
        "classification": "LEAN",
        "market_family": "FT_TOTALS",
        "market": "Goals Over/Under",
        "selection": "Over 2.5",
        "line": 2.5,
        "settlement_status": "WIN",
    }]
    report, backlog = build_report(source, [], settlement)
    assert report["by_reason"]["DUPLICATE_OLDER_SNAPSHOT"] == 1
    assert report["by_reason"]["IN_SETTLEMENT_LEDGER"] == 1
    assert len(backlog) == 1
