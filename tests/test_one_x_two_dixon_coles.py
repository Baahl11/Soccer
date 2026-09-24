from mcp_gateway import analyze_1x2_dixon_coles as dc
from mcp_gateway import analyze_1x2_model_selection as selection


def _same_cohort_rows():
    rows = []
    for i in range(100):
        if i < 30:
            actual = "D"
            challenger = [0.15, 0.70, 0.15]
        elif i % 2 == 0:
            actual = "H"
            challenger = [0.75, 0.10, 0.15]
        else:
            actual = "A"
            challenger = [0.15, 0.10, 0.75]
        rows.append({
            "actual": actual,
            "baseline": [0.40, 0.30, 0.30],
            "challenger": challenger,
        })
    return rows


def test_same_cohort_draw_auc_detects_real_discrimination():
    rows = _same_cohort_rows()
    baseline = dc.metrics(rows, "baseline")
    challenger = dc.metrics(rows, "challenger")

    assert baseline["class_discrimination"]["D"]["auc"] == 0.5
    assert baseline["class_discrimination"]["D"]["discrimination_ready"] is False
    assert challenger["class_discrimination"]["D"]["auc"] == 1.0
    assert challenger["class_discrimination"]["D"]["auc_lower_95"] == 1.0
    assert challenger["class_discrimination"]["D"]["discrimination_ready"] is True
    assert challenger["class_discrimination"]["D"]["positive_count"] == 30
    assert challenger["class_discrimination"]["D"]["negative_count"] == 70


def _selection_report(draw_ready):
    return {
        "walk_forward_evaluated": 470,
        "baseline": {"brier": 0.65, "log_loss": 1.08},
        "challenger": {"brier": 0.64, "log_loss": 1.07},
        "improvement": {
            "brier_delta": -0.01,
            "log_loss_delta": -0.01,
            "accuracy_delta_pp": 0.0,
            "class_discrimination": {
                "H": {"challenger_discrimination_ready": True, "challenger_auc_lower_95": 0.56},
                "D": {
                    "challenger_discrimination_ready": draw_ready,
                    "challenger_auc_lower_95": 0.53 if draw_ready else 0.48,
                    "auc_lower_95_delta": 0.04 if draw_ready else -0.01,
                },
                "A": {"challenger_discrimination_ready": True, "challenger_auc_lower_95": 0.55},
            },
        },
    }


def test_model_selection_blocks_quality_win_when_draw_discrimination_is_not_ready():
    row = selection.challenger_row("DIXON_COLES", _selection_report(False))
    assert row["quality_better_on_matched_report"] is True
    assert row["class_discrimination_available"] is True
    assert row["draw_discrimination_ready"] is False
    assert row["status"] == "DRAW_DISCRIMINATION_NOT_READY"


def test_model_selection_allows_formal_review_only_after_draw_discrimination_gate():
    row = selection.challenger_row("DIXON_COLES", _selection_report(True))
    assert row["draw_discrimination_ready"] is True
    assert row["status"] == "PROMOTION_REVIEW_ELIGIBLE_RESEARCH_ONLY"
