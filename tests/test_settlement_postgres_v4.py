from __future__ import annotations

from mcp_gateway.settlement_postgres_v4 import build_report_from_rows


def _row(
    *,
    fixture_id: int,
    generated_at: str,
    kickoff: str,
    classification: str,
    market: str,
    selection: str,
    line=None,
    price: float = 2.0,
    home_goals: int = 2,
    away_goals: int = 1,
    quote_timestamp=None,
    quote_provider_update=None,
    lineup_timestamp=None,
    feature_timestamp=None,
):
    best = {
        "market": market,
        "selection": selection,
        "line": line,
        "decimal_price": price,
        "bookmaker": "TestBook",
    }
    return {
        "fixture_id": fixture_id,
        "generated_at": generated_at,
        "kickoff": kickoff,
        "stage": "T-10",
        "classification": classification,
        "availability_confidence": 0.9,
        "bet_eligible": True,
        "data_tier": "A",
        "quote_timestamp": quote_timestamp,
        "quote_provider_update": quote_provider_update,
        "lineup_timestamp": lineup_timestamp,
        "feature_timestamp": feature_timestamp,
        "event_payload": {
            "tier": "B",
            "stake_units": 1.0,
            "best_market": best,
        },
        "league": "Test League",
        "home_team_id": 10,
        "home_team": "Home",
        "away_team_id": 20,
        "away_team": "Away",
        "final_status": "FT",
        "home_goals": home_goals,
        "away_goals": away_goals,
        "final_score": {
            "fulltime": {"home": home_goals, "away": away_goals},
            "halftime": {"home": 1, "away": 0},
        },
        "result_event_payload": {
            "result": {
                "status": "FT",
                "goals": {"home": home_goals, "away": away_goals},
                "score": {
                    "fulltime": {"home": home_goals, "away": away_goals},
                    "halftime": {"home": 1, "away": 0},
                },
            }
        },
    }


def test_postgres_settlement_dedupes_latest_prekickoff_decision():
    rows = [
        _row(
            fixture_id=1,
            generated_at="2026-09-24T16:00:00+00:00",
            kickoff="2026-09-24T18:00:00+00:00",
            classification="BET",
            market="Goals Over/Under",
            selection="Over 2.5",
            line=2.5,
            price=1.90,
        ),
        _row(
            fixture_id=1,
            generated_at="2026-09-24T17:30:00+00:00",
            kickoff="2026-09-24T18:00:00+00:00",
            classification="BET",
            market="Goals Over/Under",
            selection="Over 2.5",
            line=2.5,
            price=2.05,
        ),
    ]

    report = build_report_from_rows(rows)

    assert report["deduped_actionable_decisions"] == 1
    assert report["settled_decisions"] == 1
    row = report["rows"][0]
    assert row["market_family"] == "FT_TOTALS"
    assert row["settlement_status"] == "WIN"
    assert row["decimal_price"] == 2.05
    assert row["roi_units"] == 1.05


def test_postgres_settlement_excludes_postkickoff_and_grades_1x2():
    rows = [
        _row(
            fixture_id=2,
            generated_at="2026-09-24T17:00:00+00:00",
            kickoff="2026-09-24T18:00:00+00:00",
            classification="LEAN",
            market="Match Winner",
            selection="Away",
            price=3.0,
            home_goals=2,
            away_goals=1,
        ),
        _row(
            fixture_id=3,
            generated_at="2026-09-24T18:01:00+00:00",
            kickoff="2026-09-24T18:00:00+00:00",
            classification="BET",
            market="Goals Over/Under",
            selection="Under 2.5",
            line=2.5,
            price=1.8,
        ),
    ]

    report = build_report_from_rows(rows)

    assert report["deduped_actionable_decisions"] == 1
    assert report["rows"][0]["market_family"] == "FT_1X2"
    assert report["rows"][0]["settlement_status"] == "LOSS"
    assert report["rows"][0]["roi_units"] == -1.0


def test_market_performance_summary_matches_existing_contract():
    rows = [
        _row(
            fixture_id=4,
            generated_at="2026-09-24T17:00:00+00:00",
            kickoff="2026-09-24T18:00:00+00:00",
            classification="BET",
            market="Goals Over/Under",
            selection="Over 2.5",
            line=2.5,
            price=1.9,
        ),
        _row(
            fixture_id=5,
            generated_at="2026-09-24T17:00:00+00:00",
            kickoff="2026-09-24T18:00:00+00:00",
            classification="LEAN",
            market="Goals Over/Under",
            selection="Under 3.5",
            line=3.5,
            price=1.8,
        ),
    ]

    report = build_report_from_rows(rows)
    summary = report["market_performance_summary"]

    assert summary["schema_version"] == "1.2.0"
    assert summary["settlement_decisions"] == 2
    assert summary["by_market_family"]["FT_TOTALS"]["settled"] == 2
    assert set(summary["by_market_family_and_classification"]["FT_TOTALS"]) == {"BET", "LEAN"}
    assert summary["promotion_gate_review"]["FT_TOTALS"]["settled"] == 2



def test_postgres_settlement_preserves_point_in_time_provenance():
    report = build_report_from_rows([
        _row(
            fixture_id=6,
            generated_at="2026-09-24T17:30:00+00:00",
            kickoff="2026-09-24T18:00:00+00:00",
            classification="BET",
            market="Goals Over/Under",
            selection="Over 2.5",
            line=2.5,
            price=1.95,
            quote_timestamp="2026-09-24T17:29:00+00:00",
            quote_provider_update="2026-09-24T17:28:30+00:00",
            lineup_timestamp="2026-09-24T17:20:00+00:00",
            feature_timestamp="2026-09-24T17:30:00+00:00",
        )
    ])

    row = report["rows"][0]
    assert report["schema_version"] == "1.1.0"
    assert report["model_version"] == "SOCCER_SETTLEMENT_POSTGRES_V4_1.1.0"
    assert row["quote_timestamp"] == "2026-09-24T17:29:00+00:00"
    assert row["bookmaker_timestamp"] == "2026-09-24T17:29:00+00:00"
    assert row["lineup_captured_at"] == "2026-09-24T17:20:00+00:00"
    assert row["feature_captured_at"] == "2026-09-24T17:30:00+00:00"


class _FakeCursor:
    def __init__(self):
        self.query = ""
        self.description = []

    def execute(self, query, params):
        self.query = query

    def fetchall(self):
        return []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeConn:
    def __init__(self):
        self.cursor_instance = _FakeCursor()

    def cursor(self):
        return self.cursor_instance


def test_postgres_settlement_snapshot_joins_are_strictly_point_in_time():
    from mcp_gateway import settlement_postgres_v4 as v

    conn = _FakeConn()
    assert v._load_rows(conn, lookback_days=30, max_rows=100) == []
    query = " ".join(conn.cursor_instance.query.split())

    assert "m.captured_at <= e.generated_at" in query
    assert "l.captured_at <= e.generated_at" in query
    assert "s.captured_at <= e.generated_at" in query
    assert "soccer_market_snapshots" in query
    assert "soccer_lineup_snapshots" in query
    assert "soccer_feature_snapshots" in query
