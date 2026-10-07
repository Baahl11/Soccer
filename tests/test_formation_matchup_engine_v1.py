import json
from datetime import datetime, timezone

from mcp_gateway import formation_matchup_engine_v1 as engine


def _fixture(lineup_obs, tactical_stats=None):
    return {
        1: {
            "fixture_id": 1,
            "kickoff_local": "2026-10-01T12:00:00+00:00",
            "league_id": 39,
            "league": "Test League",
            "home_team_id": 10,
            "home_team": "Home",
            "away_team_id": 20,
            "away_team": "Away",
            "lineup_obs": lineup_obs,
            "result": {
                "goals": {"home": 2, "away": 1},
                "score": {"halftime": {"home": 1, "away": 0}},
            },
            "tactical_stats": tactical_stats
            or {
                "teams": [
                    {
                        "team_id": 10,
                        "corners": 7,
                        "total_shots": 14,
                        "shots_on_goal": 6,
                    },
                    {
                        "team_id": 20,
                        "corners": 3,
                        "total_shots": 9,
                        "shots_on_goal": 2,
                    },
                ],
                "totals": {},
            },
        }
    }


def test_side_specific_metrics_and_direction(monkeypatch):
    pre = datetime(2026, 10, 1, 11, 30, tzinfo=timezone.utc)
    monkeypatch.setattr(
        engine.formation_v2,
        "_enhanced_load_history",
        lambda _: _fixture([(pre, "4-3-3", "5-4-1", "T-40")]),
    )
    rows = engine.build_rows("unused")
    assert len(rows) == 1
    row = rows[0]
    assert row["matchup_key"] == "4-3-3 vs 5-4-1"
    assert row["home_corners"] == 7.0
    assert row["away_corners"] == 3.0
    assert row["total_corners"] == 10.0
    assert row["home_shots"] == 14.0
    assert row["away_shots"] == 9.0
    assert row["home_sot"] == 6.0
    assert row["away_sot"] == 2.0
    assert row["home_goals"] == 2
    assert row["away_goals"] == 1


def test_postkickoff_only_formation_is_excluded(monkeypatch):
    post = datetime(2026, 10, 1, 12, 1, tzinfo=timezone.utc)
    monkeypatch.setattr(
        engine.formation_v2,
        "_enhanced_load_history",
        lambda _: _fixture([(post, "4-3-3", "5-4-1", "POST")]),
    )
    assert engine.build_rows("unused") == []


def test_missing_metric_is_not_zero(monkeypatch):
    pre = datetime(2026, 10, 1, 11, 30, tzinfo=timezone.utc)
    stats = {
        "teams": [
            {"team_id": 10, "corners": 5},
            {"team_id": 20, "corners": 4},
        ],
        "totals": {},
    }
    monkeypatch.setattr(
        engine.formation_v2,
        "_enhanced_load_history",
        lambda _: _fixture([(pre, "4-2-3-1", "4-4-2", "T-40")], stats),
    )
    rows = engine.build_rows("unused")
    assert rows[0]["home_shots"] is None
    assert rows[0]["away_shots"] is None
    summary = engine.summarize_group(rows)
    assert summary["metric_completeness"]["home_shots"]["observed_n"] == 0
    assert summary["avg_home_shots"] is None


def test_report_is_research_only_and_market_independent(monkeypatch):
    pre = datetime(2026, 10, 1, 11, 30, tzinfo=timezone.utc)
    monkeypatch.setattr(
        engine.formation_v2,
        "_enhanced_load_history",
        lambda _: _fixture([(pre, "4-3-3", "4-4-2", "T-40")]),
    )
    report = engine.build_report("unused")
    assert report["production_enabled"] is False
    assert report["decision_weight"] == 0.0
    assert report["health"]["odds_consumed"] is False


def test_history_retains_latest_confirmed_prekickoff_xi_and_ignores_postkickoff(tmp_path):
    kickoff = "2026-10-01T12:00:00+00:00"

    def team(team_id, coach_id, starter_start):
        return {
            "team_id": team_id,
            "team": f"T{team_id}",
            "formation": "4-3-3",
            "coach_id": coach_id,
            "coach": f"C{coach_id}",
            "starters": [
                {
                    "id": starter_start + i,
                    "name": f"P{starter_start + i}",
                    "pos": "G" if i == 0 else "D",
                    "grid": f"1:{i+1}",
                }
                for i in range(11)
            ],
        }

    base_fixture = {
        "fixture_id": 1,
        "kickoff": kickoff,
        "league_id": 39,
        "league": "Test",
        "home_team_id": 10,
        "home_team": "Home",
        "away_team_id": 20,
        "away_team": "Away",
    }
    pre = {
        "generated_at_local": "2026-10-01T11:40:00+00:00",
        "events": [
            {
                "fixture": base_fixture,
                "stage": "T-20",
                "lineups": {
                    "both_xi_confirmed": True,
                    "lineup_state": "CONFIRMED_API",
                    "teams": [team(10, 1000, 100), team(20, 2000, 200)],
                },
            }
        ],
    }
    post = {
        "generated_at_local": "2026-10-01T12:05:00+00:00",
        "events": [
            {
                "fixture": base_fixture,
                "stage": "POST",
                "lineups": {
                    "both_xi_confirmed": True,
                    "lineup_state": "CONFIRMED_API",
                    "teams": [team(10, 9999, 900), team(20, 8888, 800)],
                },
            }
        ],
    }
    history = tmp_path / "2026-10-01.jsonl"
    history.write_text(
        json.dumps(pre) + "\n" + json.dumps(post) + "\n",
        encoding="utf-8",
    )

    fixtures = engine.formation_v2.base.load_history(str(tmp_path))
    rec = fixtures[1]

    assert rec["lineup_detail"] is not None
    assert rec["lineup_detail_stage"] == "T-20"
    by_team = {row["team_id"]: row for row in rec["lineup_detail"]["teams"]}
    assert by_team[10]["coach_id"] == 1000
    assert by_team[20]["coach_id"] == 2000
    assert by_team[10]["starters"][0]["id"] == 100
    assert by_team[20]["starters"][0]["id"] == 200
