from mcp_gateway import one_x_two_multiclass_oos_v4 as v


def _row(i, version, probs, target):
    outcomes = {
        "home_win": int(target == 0),
        "draw": int(target == 1),
        "away_win": int(target == 2),
    }
    return {
        "fixture_id": i,
        "run_timestamp": f"2026-09-{1 + (i // 24):02d}T{(i % 24):02d}:00:00+00:00",
        "model_version": version,
        "predictions": {
            "home_win": probs[0],
            "draw": probs[1],
            "away_win": probs[2],
        },
        "outcomes": outcomes,
    }


def test_temperature_scaling_preserves_simplex_and_argmax():
    probs = (0.8, 0.15, 0.05)
    scaled = v.temperature_scale(probs, 1.5)
    assert abs(sum(scaled) - 1.0) < 1e-12
    assert max(range(3), key=lambda i: probs[i]) == max(range(3), key=lambda i: scaled[i])


def test_walk_forward_temperature_scaling_can_improve_overconfident_current_model():
    rows = []
    for i in range(360):
        # Current model is systematically overconfident toward home.
        target = 0 if i % 5 < 3 else 1 if i % 5 == 3 else 2
        rows.append(_row(i, "M2", (0.85, 0.10, 0.05), target))
    report = v.build_report(rows, min_train_rows=200, batch_size=40)

    assert report["source_model_version"] == "M2"
    assert report["evaluated_rows"] == 160
    assert report["improves_brier_and_log_loss"] is True
    assert report["brier_delta"] < 0
    assert report["log_loss_delta"] < 0
    assert report["production_promotion_allowed"] is False


def test_only_latest_model_version_is_used():
    rows = []
    for i in range(250):
        rows.append(_row(i, "M1", (0.6, 0.2, 0.2), i % 3))
    for i in range(250, 510):
        rows.append(_row(i, "M2", (0.65, 0.2, 0.15), i % 3))
    report = v.build_report(rows, min_train_rows=200, batch_size=30)

    assert report["source_model_version"] == "M2"
    assert report["source_rows_current_model"] == 260
    assert report["evaluated_rows"] == 60
    assert report["status"] == "INSUFFICIENT_WALK_FORWARD_EVAL"
    assert report["anti_leakage"]["current_model_version_only"] is True
