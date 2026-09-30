from mcp_gateway.subscriber_validation_metrics_v231 import _metric_row
from mcp_gateway.subscriber_preview_performance_live_v231 import _html


def test_validation_metric_row_maps_real_persisted_shapes():
    one_x_two = _metric_row(
        "1X2",
        {
            "status": "RESEARCH_HOLD",
            "model_version": "MODEL",
            "canonical_multiclass_oos": {
                "temperature_scaled": {
                    "multiclass_brier": 0.655,
                    "multiclass_log_loss": 1.083,
                    "n": 670,
                }
            },
            "true_clv": {
                "rows": 47,
                "minimum_rows": 50,
                "avg_probability_clv_pp": 0.13,
            },
            "blockers": ["1X2_TRUE_CLV_47_LT_50"],
        },
    )
    assert one_x_two["sample_n"] == 670
    assert one_x_two["brier"] == 0.655
    assert one_x_two["log_loss"] == 1.083
    assert one_x_two["true_clv_rows"] == 47
    assert one_x_two["true_clv_target"] == 50
    assert one_x_two["avg_clv_pp"] == 0.13


def test_validation_metric_row_does_not_invent_missing_metrics():
    row = _metric_row("Corners", {"status": "RESEARCH_HOLD", "true_clv": {"rows": 0}})
    assert row["brier"] is None
    assert row["log_loss"] is None
    assert row["ece"] is None
    assert row["avg_clv_pp"] is None


def test_layered_preview_fetches_persisted_performance_endpoint():
    html = _html()
    assert "/app-preview/performance" in html
    assert "Persisted Validation Performance" in html
    assert "True CLV by Family" in html
