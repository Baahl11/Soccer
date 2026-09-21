import json
from pathlib import Path

from mcp_gateway import run_postgame_final_backfill as runner


def _queue_row(fixture_id=111):
    return {
        "schema_version": "1.0.0",
        "queue_reason": "PENDING_FINAL_RESULT",
        "fixture_id": fixture_id,
        "kickoff_local": "2026-09-13T01:30:00-06:00",
        "league": "Test League",
        "home_team": "Home FC",
        "away_team": "Away FC",
        "classifications": ["BET"],
        "market_families": ["FT_TOTALS"],
        "stages": ["T-10"],
        "source_backlog_rows": 1,
        "decision_lines": [{"classification": "BET", "market_family": "FT_TOTALS"}],
    }


def test_dry_run_does_not_call_provider(tmp_path, monkeypatch):
    queue = tmp_path / "queue.jsonl"
    queue.write_text(json.dumps(_queue_row()) + "\n", encoding="utf-8")
    called = {"value": False}

    def fake_api(*args, **kwargs):
        called["value"] = True
        raise AssertionError("provider should not be called in dry-run")

    monkeypatch.setattr(runner, "api_get_fixture", fake_api)
    report = runner.run_backfill(
        queue_path=str(queue),
        history_output=str(tmp_path / "history.jsonl"),
        report_output=str(tmp_path / "report.json"),
        api_key=None,
        base_url="https://v3.football.api-sports.io",
        max_calls=5,
        execute=False,
    )
    assert report["status"] == "FINAL_BACKFILL_DRY_RUN"
    assert report["provider_calls_attempted"] == 0
    assert called["value"] is False
    assert not (tmp_path / "history.jsonl").exists()


def test_execute_without_api_key_blocks(tmp_path):
    queue = tmp_path / "queue.jsonl"
    queue.write_text(json.dumps(_queue_row()) + "\n", encoding="utf-8")
    report = runner.run_backfill(
        queue_path=str(queue),
        history_output=str(tmp_path / "history.jsonl"),
        report_output=str(tmp_path / "report.json"),
        api_key=None,
        base_url="https://v3.football.api-sports.io",
        max_calls=5,
        execute=True,
    )
    assert report["status"] == "FINAL_BACKFILL_BLOCKED_NO_API_KEY"
    assert report["final_results_appended"] == 0
    assert not (tmp_path / "history.jsonl").exists()


def test_final_fixture_appends_postgame_event(tmp_path, monkeypatch):
    queue = tmp_path / "queue.jsonl"
    queue.write_text(json.dumps(_queue_row(222)) + "\n", encoding="utf-8")

    def fake_api(fixture_id, api_key, base_url, timeout):
        assert fixture_id == 222
        return {
            "response": [
                {
                    "fixture": {"id": 222, "status": {"short": "FT"}},
                    "league": {"id": 9, "name": "Test League", "country": "X", "season": 2026},
                    "teams": {"home": {"id": 1, "name": "Home FC"}, "away": {"id": 2, "name": "Away FC"}},
                    "goals": {"home": 2, "away": 1},
                    "score": {"halftime": {"home": 1, "away": 0}, "fulltime": {"home": 2, "away": 1}},
                }
            ]
        }

    monkeypatch.setattr(runner, "api_get_fixture", fake_api)
    history = tmp_path / "history.jsonl"
    report = runner.run_backfill(
        queue_path=str(queue),
        history_output=str(history),
        report_output=str(tmp_path / "report.json"),
        api_key="key",
        base_url="https://v3.football.api-sports.io",
        max_calls=5,
        execute=True,
        sleep_seconds=0,
    )
    assert report["final_results_appended"] == 1
    tick = json.loads(history.read_text(encoding="utf-8").strip())
    event = tick["events"][0]
    assert event["event_type"] == "POSTGAME_FINAL_BACKFILL"
    assert event["classification"] == "POSTGAME"
    assert event["fixture"]["status"] == "FT"
    assert event["result"]["goals"] == {"home": 2, "away": 1}
    assert event["bet_eligible"] is False
    assert event["stake_units"] is None
