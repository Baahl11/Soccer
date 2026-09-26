from __future__ import annotations

from mcp_gateway import oos_calibration_scope_v4 as scope


def _row(fixture_id: int, model_version: str, index: int) -> dict:
    # Balanced deterministic outcomes ensure calibration readiness for every target.
    y = index % 2
    p = 0.25 if y == 0 else 0.75
    return {
        "fixture_id": fixture_id,
        "model_version": model_version,
        "predictions": {
            "home_win": p,
            "draw": 1.0 - p,
            "away_win": p,
            "btts": p,
            "over_2_5": p,
        },
        "outcomes": {
            "home_win": y,
            "draw": 1 - y,
            "away_win": y,
            "btts": y,
            "over_2_5": y,
        },
    }


def test_current_model_version_uses_semantic_version_not_row_majority() -> None:
    rows = [_row(i, "v1.7", i) for i in range(300)]
    rows += [_row(1000 + i, "v1.10", i) for i in range(210)]
    assert scope.current_model_version(rows) == "v1.10"


def test_apply_current_model_scope_preserves_history_but_targets_current_only() -> None:
    old_rows = [_row(i, "v1.0", i) for i in range(220)]
    current_rows = [_row(1000 + i, "v1.7", i) for i in range(210)]
    rows = old_rows + current_rows

    historical_targets, historical_ready, historical_improving = scope.build_targets(
        rows,
        source_model_version="ALL_MODELS",
    )
    source = {
        "schema_version": "1.0.0",
        "status": "OOS_CALIBRATION_MATERIALIZED",
        "fixture_rows": len(rows),
        "model_version_counts": {"v1.0": 220, "v1.7": 210},
        "targets": historical_targets,
        "ready_target_count": historical_ready,
        "targets_improving_brier_and_log_loss": historical_improving,
        "rows": rows,
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
    }

    out = scope.apply_current_model_scope(source)

    assert out["current_source_model_version"] == "v1.7"
    assert out["canonical_target_scope"] == "CURRENT_SOURCE_MODEL_VERSION_ONLY"
    assert out["all_model_rows"] == 430
    assert out["current_model_rows"] == 210
    assert len(out["rows"]) == 430
    assert out["fixture_rows"] == 430
    assert out["targets"]["btts"]["rows"] == 210
    assert out["targets"]["btts"]["source_model_version"] == "v1.7"
    assert out["historical_all_models_targets"]["btts"]["rows"] == 430
    assert out["provider_requests_added"] == 0
    assert out["production_promotion_allowed"] is False
    assert out["runtime_prediction_weight"] == 0.0
    assert out["canonical_bet_logic_changed"] is False
