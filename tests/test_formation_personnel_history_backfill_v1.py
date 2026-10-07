from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from mcp_gateway import analyze_formation_intelligence_v2 as formation_v2
from mcp_gateway import formation_matchup_fm4_style_ablation_v1 as fm4
from mcp_gateway import formation_personnel_history_backfill_v1 as v


def _candidate():
    return {
        "fixture_id": 7001,
        "kickoff": datetime(2026, 9, 1, 12, tzinfo=timezone.utc),
        "league_id": 39,
        "league": "Test League",
        "country": "Test",
        "season": 2026,
        "round": "Round 1",
        "home_team_id": 1,
        "home_team": "Home FC",
        "away_team_id": 2,
        "away_team": "Away FC",
        "final_status": "FT",
        "home_goals": 2,
        "away_goals": 1,
    }


def _lineup():
    def team(team_id, name, coach_id):
        starters = [
            {
                "id": team_id * 100 + i,
                "name": f"{name} Player {i}",
                "number": i,
                "pos": "G" if i == 1 else "D" if i <= 5 else "M" if i <= 9 else "F",
                "grid": f"{1 if i == 1 else 2}-{i}",
            }
            for i in range(1, 12)
        ]
        return {
            "team_id": team_id,
            "team": name,
            "formation": "4-3-3",
            "coach_id": coach_id,
            "coach": f"Coach {coach_id}",
            "starters": starters,
            "goalkeepers": [starters[0]],
            "substitutes_count": 9,
        }

    return {
        "teams": [team(1, "Home FC", 10), team(2, "Away FC", 20)],
        "both_xi_confirmed": True,
        "both_goalkeepers_confirmed": True,
        "lineup_state": "CONFIRMED_API",
    }


def test_personnel_backfill_requires_complete_xi_and_goalkeepers():
    assert v._sufficient_lineup(_lineup()) is True
    broken = _lineup()
    broken["both_goalkeepers_confirmed"] = False
    assert v._sufficient_lineup(broken) is False


def test_personnel_candidate_selection_prioritizes_undercovered_teams():
    now = datetime(2026, 9, 10, tzinfo=timezone.utc)
    rows = [
        {
            "fixture_id": 1,
            "kickoff": now,
            "home_team_id": 10,
            "away_team_id": 20,
        },
        {
            "fixture_id": 2,
            "kickoff": now,
            "home_team_id": 30,
            "away_team_id": 40,
        },
    ]
    selected = v.select_candidates(
        rows,
        team_ids={10, 20, 30, 40},
        prior_counts=Counter({10: 2, 20: 2, 30: 0, 40: 0}),
        max_fixtures=1,
    )
    assert selected[0]["fixture_id"] == 2


def test_personnel_fact_uses_conservative_postmatch_availability_and_never_retrofits_current_lineup():
    candidate = _candidate()
    retrieved = datetime(2026, 10, 7, 14, tzinfo=timezone.utc)
    event = v.make_event(
        candidate,
        _lineup(),
        retrieved_at=retrieved,
        provider_daily_remaining=5000,
    )
    fact = event["historical_personnel_fact"]

    expected_available = candidate["kickoff"] + timedelta(
        hours=v.HISTORICAL_FACT_DELAY_HOURS
    )
    assert fact["historical_fact_available_at"] == expected_available.isoformat()
    assert fact["retrieved_at"] == retrieved.isoformat()
    assert fact["prior_history_use_only"] is True
    assert fact["retroactive_current_fixture_allowed"] is False
    assert event["backfill"]["retroactive_prediction_rewrite"] is False
    assert event["decision_weight"] == 0.0


def test_history_loader_keeps_backfilled_personnel_fact_separate_from_pregame_lineup(tmp_path: Path):
    candidate = _candidate()
    event = v.make_event(
        candidate,
        _lineup(),
        retrieved_at=datetime(2026, 10, 7, 14, tzinfo=timezone.utc),
        provider_daily_remaining=5000,
    )
    tick = {
        "generated_at_utc": "2026-10-07T14:00:00+00:00",
        "generated_at_local": "2026-10-07T08:00:00-06:00",
        "events": [event],
    }
    (tmp_path / "personnel.jsonl").write_text(
        json.dumps(tick) + "\n",
        encoding="utf-8",
    )

    fixtures = formation_v2._enhanced_load_history(str(tmp_path))
    rec = fixtures[7001]

    assert rec.get("lineup_detail") is None
    facts = rec.get("historical_personnel_facts") or []
    assert len(facts) == 1
    assert facts[0]["retroactive_current_fixture_allowed"] is False
    assert facts[0]["both_xi_confirmed"] is True


def test_fm4_uses_backfilled_personnel_only_as_prior_history(tmp_path: Path):
    candidate = _candidate()
    event = v.make_event(
        candidate,
        _lineup(),
        retrieved_at=datetime(2026, 10, 7, 14, tzinfo=timezone.utc),
        provider_daily_remaining=5000,
    )
    tick = {
        "generated_at_utc": "2026-10-07T14:00:00+00:00",
        "generated_at_local": "2026-10-07T08:00:00-06:00",
        "events": [event],
    }
    (tmp_path / "personnel.jsonl").write_text(
        json.dumps(tick) + "\n",
        encoding="utf-8",
    )

    _, personnel = fm4._history_events(str(tmp_path))

    assert len(personnel) == 1
    row = personnel[0]
    assert row["fixture_id"] == 7001
    assert row["stage"] == "FM4_PERSONNEL_BACKFILL"
    assert row["retroactive_current_fixture_allowed"] is False
    assert row["history_available_at"] == (
        candidate["kickoff"] + timedelta(hours=4)
    )


def test_personnel_backfill_model_is_research_only():
    assert v.MODEL_VERSION == "SOCCER_FM4_PERSONNEL_HISTORY_BACKFILL_V1.0.0"
    assert v.MAX_LINEUP_REQUESTS_PER_RUN == 8
    assert v.HISTORICAL_FACT_DELAY_HOURS == 4
