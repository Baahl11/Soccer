from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]

# Load the legacy helper module without executing mcp_gateway.__init__.
legacy_spec = importlib.util.spec_from_file_location(
    "mcp_gateway.build_signal_ledger",
    ROOT / "mcp_gateway" / "build_signal_ledger.py",
)
assert legacy_spec is not None and legacy_spec.loader is not None
legacy_module = importlib.util.module_from_spec(legacy_spec)
legacy_spec.loader.exec_module(legacy_module)

fake_pkg = types.ModuleType("mcp_gateway")
fake_pkg.__path__ = [str(ROOT / "mcp_gateway")]
fake_pkg.build_signal_ledger = legacy_module
fake_pkg.persistence = types.SimpleNamespace()
sys.modules["mcp_gateway"] = fake_pkg
sys.modules["mcp_gateway.build_signal_ledger"] = legacy_module

spec = importlib.util.spec_from_file_location(
    "signal_ledger_postgres_v4_under_test",
    ROOT / "mcp_gateway" / "signal_ledger_postgres_v4.py",
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_mutable_fixture_status_never_leaks_into_historical_row() -> None:
    raw = {
        "event_id": 99,
        "fixture_id": 123,
        "generated_at": "2026-09-30T18:00:00+00:00",
        "stage": "T-40",
        "event_type": "SOCCER_REFRESH",
        "classification": "WATCH",
        "availability_confidence": 0.8,
        "bet_eligible": False,
        "data_tier": "A",
        "event_payload": {
            "event_type": "SOCCER_REFRESH",
            "stage": "T-40",
            "classification": "WATCH",
            "fixture": {
                "fixture_id": 123,
                "kickoff": "2026-09-30T19:00:00+00:00",
                "home_team": "Home",
                "away_team": "Away",
            },
        },
        "kickoff": "2026-09-30T19:00:00+00:00",
        "league_id": 1,
        "league": "League",
        "country": "Country",
        "season": 2026,
        "home_team_id": 10,
        "home_team": "Home",
        "away_team_id": 20,
        "away_team": "Away",
        # Deliberately final/latest DB state. This must NOT enter the row.
        "fixture_status": "FT",
        "pipeline_run_id": 500,
        "pipeline_generated_at_local": "2026-09-30T12:00:00-06:00",
        "pipeline_timezone": "America/Mexico_City",
        "pipeline_payload": {
            "version": "v-test",
            "model_version": "m-test",
            "match_table_rows": [],
        },
    }

    row = module._ledger_row(raw)

    assert row is not None
    assert row["fixture_status"] is None
    assert row["result"] is None
    assert row["ledger_source"] == module.SOURCE
    assert row["postgres_event_id"] == 99


def test_point_in_time_event_status_is_preserved_when_it_really_existed() -> None:
    raw = {
        "event_id": 100,
        "fixture_id": 124,
        "generated_at": "2026-09-30T20:00:00+00:00",
        "event_payload": {
            "event_type": "SOCCER_REFRESH",
            "stage": "POSTGAME",
            "classification": "POSTGAME",
            "fixture": {
                "fixture_id": 124,
                "kickoff": "2026-09-30T18:00:00+00:00",
                "status": "FT",
                "goals": {"home": 2, "away": 1},
                "score": {},
            },
        },
        "pipeline_run_id": None,
        "pipeline_timezone": "America/Mexico_City",
        "pipeline_payload": {},
    }

    row = module._ledger_row(raw)

    assert row is not None
    assert row["fixture_status"] == "FT"
    assert row["result"] is not None
