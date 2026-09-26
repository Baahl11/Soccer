from __future__ import annotations

import json
from pathlib import Path

from mcp_gateway import build_signal_ledger as v


def test_phase16_calibration_snapshot_matches_same_fixture_line_and_direction():
    tick = {
        "match_table_rows": [
            {
                "fixture_id": 100,
                "market_family": "FT_TOTALS",
                "market": "Goals Over/Under",
                "selection": "Over",
                "line": 2.5,
                "price": 1.91,
                "bookmaker": "Book A",
                "phase16_calibration_status": "BINARY_DISCRIMINATION_NOT_READY",
                "phase16_calibration_promotion_shadow_eligible": False,
                "phase16_binary_calibration_diagnostics": {
                    "target": "over_2_5",
                    "source_model_version": "M1",
                    "requested_model_version": "M1",
                    "source_model_version_matches": True,
                    "rows": 407,
                    "positive_count": 241,
                    "negative_count": 166,
                    "auc": 0.522,
                    "auc_lower_95": 0.465,
                    "brier_delta": -0.04,
                    "log_loss_delta": -0.16,
                    "eligible_for_phase16_research": False,
                    "calibrator_status": "RESEARCH_CALIBRATOR_FITTED",
                },
            },
            {
                "fixture_id": 100,
                "market_family": "FT_TOTALS",
                "market": "Goals Over/Under",
                "selection": "Under",
                "line": 2.5,
                "price": 1.99,
                "bookmaker": "Book B",
            },
        ]
    }
    best = {
        "family": "FT_TOTALS",
        "market": "Goals Over/Under",
        "selection": "Over",
        "line": 2.5,
        "decimal_price": 1.90,
    }

    out = v._phase16_calibration_snapshot(tick, 100, best)

    assert out is not None
    assert out["selection"] == "Over"
    assert out["line"] == 2.5
    assert out["price"] == 1.91
    assert out["calibration_target"] == "over_2_5"
    assert out["calibration_status"] == "BINARY_DISCRIMINATION_NOT_READY"
    assert out["promotion_shadow_eligible"] is False
    assert out["binary_calibration_diagnostics"]["auc_lower_95"] == 0.465


def test_build_rows_persists_provenance_without_changing_market(tmp_path: Path):
    history = tmp_path / "history"
    history.mkdir()
    tick = {
        "generated_at_local": "2026-09-25T22:00:00-06:00",
        "generated_at_utc": "2026-09-26T04:00:00+00:00",
        "version": "test",
        "model_version": "M1",
        "match_table_rows": [
            {
                "fixture_id": 123,
                "market_family": "FT_TOTALS",
                "market": "Goals Over/Under",
                "selection": "Over",
                "line": 2.5,
                "price": 1.80,
                "bookmaker": "Book A",
                "phase16_calibration_status": "BINARY_DISCRIMINATION_NOT_READY",
                "phase16_calibration_promotion_shadow_eligible": False,
                "phase16_binary_calibration_diagnostics": {"target": "over_2_5"},
            }
        ],
        "events": [
            {
                "fixture": {
                    "fixture_id": 123,
                    "kickoff": "2026-09-25T23:00:00-06:00",
                    "league_id": 39,
                    "league": "Test League",
                    "home_team": "A",
                    "away_team": "B",
                    "status": "NS",
                },
                "event_type": "SOCCER_REFRESH",
                "stage": "T-40",
                "classification": "WATCH",
                "coverage": {"data_tier": "B"},
                "best_market": {
                    "family": "FT_TOTALS",
                    "market": "Goals Over/Under",
                    "selection": "Over",
                    "line": 2.5,
                    "decimal_price": 1.80,
                    "p_raw": 0.62,
                    "p_shrunk": 0.56,
                },
            }
        ],
    }
    (history / "2026-09-25.jsonl").write_text(json.dumps(tick) + "\n", encoding="utf-8")

    rows, summary = v.build_rows(str(history))

    assert len(rows) == 1
    assert rows[0]["best_market"]["decimal_price"] == 1.80
    assert rows[0]["classification"] == "WATCH"
    assert rows[0]["phase16_calibration_provenance"]["calibration_target"] == "over_2_5"
    assert summary["schema_version"] == "1.3.0"
    assert summary["rows_with_phase16_calibration_provenance"] == 1
    assert summary["provider_requests_added"] == 0
    assert summary["canonical_bet_logic_changed"] is False
