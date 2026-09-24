from mcp_gateway import analyze_1x2_draw_signal as v


def _row(fid, timestamp, actual, home_lambda, away_lambda, probs, *, availability=None, relative_strength_shadow=None):
    goals = {"H": (2, 0), "D": (1, 1), "A": (0, 2)}[actual]
    return {
        "fixture_id": fid,
        "generated_at_local": timestamp,
        "kickoff_local": "2026-12-31T18:00:00+00:00",
        "availability_confidence": availability,
        "raw_projection": {
            "raw_home_goal_rate": home_lambda,
            "raw_away_goal_rate": away_lambda,
            "raw_home_win_prob": probs[0],
            "raw_draw_prob": probs[1],
            "raw_away_win_prob": probs[2],
            "relative_strength_shadow": relative_strength_shadow,
        },
        "result": {"goals": {"home": goals[0], "away": goals[1]}},
    }


def test_signal_values_have_draw_positive_direction():
    row = {
        "home_lambda": 1.1,
        "away_lambda": 1.0,
        "home_probability": 0.36,
        "draw_probability": 0.34,
        "away_probability": 0.30,
    }
    balanced = v.signal_values(row)
    row2 = {
        "home_lambda": 2.1,
        "away_lambda": 0.5,
        "home_probability": 0.75,
        "draw_probability": 0.15,
        "away_probability": 0.10,
    }
    unbalanced = v.signal_values(row2)
    assert balanced["lambda_closeness"] > unbalanced["lambda_closeness"]
    assert balanced["home_away_probability_balance"] > unbalanced["home_away_probability_balance"]
    assert balanced["weak_favorite"] > unbalanced["weak_favorite"]
    assert balanced["one_x_two_entropy"] > unbalanced["one_x_two_entropy"]


def test_build_report_detects_strong_lambda_closeness_signal_on_same_cohort():
    rows = []
    for i in range(80):
        draw = i % 4 == 0
        actual = "D" if draw else ("H" if i % 2 == 0 else "A")
        if draw:
            home_lambda, away_lambda = 1.10, 1.08
            probs = (0.34, 0.33, 0.33)
        else:
            home_lambda, away_lambda = (2.0, 0.7) if actual == "H" else (0.7, 2.0)
            probs = (0.65, 0.20, 0.15) if actual == "H" else (0.15, 0.20, 0.65)
        rows.append(_row(
            i + 1,
            f"2026-01-{(i % 28) + 1:02d}T{(i % 12):02d}:00:00+00:00",
            actual,
            home_lambda,
            away_lambda,
            probs,
        ))

    report = v.build_report(rows)
    assert report["evaluated_fixtures"] == 50
    assert report["signals"]["lambda_closeness"]["auc"] == 1.0
    assert report["signals"]["lambda_closeness"]["discrimination_ready"] is True
    assert "lambda_closeness" in report["ready_nonbaseline_signals"]
    assert report["recommendation"] == "BUILD_DRAW_CHALLENGER_FROM_VERIFIED_SIGNAL"


def test_safe_features_carry_forward_point_in_time_before_kickoff():
    rows = []
    for i in range(40):
        actual = "D" if i % 4 == 0 else "H"
        base = _row(
            i + 1,
            f"2026-03-{(i % 28) + 1:02d}T08:00:00+00:00",
            actual,
            1.3,
            1.1,
            (0.45, 0.25, 0.30),
            relative_strength_shadow={"early_signal": 0.9 if actual == "D" else 0.1},
        )
        rows.append(base)
        rows.append({
            "fixture_id": i + 1,
            "generated_at_local": f"2026-03-{(i % 28) + 1:02d}T10:00:00+00:00",
            "kickoff_local": "2026-12-31T18:00:00+00:00",
            "sporting_shortlist": {"side_edge_score": 55 + i},
            "result": base["result"],
        })

    cohort, eligible = v.extract_same_cohort(rows)
    assert eligible == 40
    assert len(cohort) == 10
    for row in cohort:
        assert "raw_projection.relative_strength_shadow.early_signal" in row["safe_feature_values"]
        assert "sporting_shortlist.side_edge_score" in row["safe_feature_values"]


def test_persisted_safe_feature_audit_detects_exploratory_signal_without_market_data():
    rows = []
    for i in range(100):
        draw = i % 4 == 0
        actual = "D" if draw else ("H" if i % 2 == 0 else "A")
        strength = 0.9 if draw else 0.1
        rows.append(_row(
            i + 1,
            f"2026-02-{(i % 28) + 1:02d}T{(i % 12):02d}:00:00+00:00",
            actual,
            1.4,
            1.2,
            (0.45, 0.25, 0.30),
            availability=0.95,
            relative_strength_shadow={"draw_pressure": strength, "market_price": 2.5},
        ))

    report = v.build_report(rows)
    audit = report["persisted_safe_feature_audit"]
    feature = audit["features"]["raw_projection.relative_strength_shadow.draw_pressure"]
    assert feature["coverage_ready"] is True
    assert feature["ci_excludes_random"] is True
    assert feature["two_sided_direction"] == "HIGHER_VALUE_MORE_DRAW"
    assert "raw_projection.relative_strength_shadow.draw_pressure" in audit["candidate_features"]
    assert "raw_projection.relative_strength_shadow.market_price" not in audit["features"]


def test_audit_does_not_change_runtime_or_promotion():
    report = v.build_report([])
    assert report["provider_requests_added"] == 0
    assert report["production_promotion_allowed"] is False
    assert report["model_weights_changed"] is False
    assert report["canonical_bet_logic_changed"] is False
