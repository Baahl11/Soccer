from mcp_gateway.subscriber_maturity_v232 import _build_family_rows
from mcp_gateway.subscriber_preview_data_v231 import build_preview_payload
from mcp_gateway.subscriber_preview_live_v231 import _html
from mcp_gateway.subscriber_preview_maturity_live_v232 import _html as maturity_html


def _payload():
    return {
        "status": "ok",
        "generated_at_utc": "2026-09-30T18:10:00Z",
        "fixture_scan_count": 321,
        "deep_dive_processed_count": 7,
        "model_version": "SOCCER EDGE ENGINE v1.7",
        "version": "4.38.0-dynamic-strength-challenger",
        "maturity_snapshot": {
            "gates": {},
            "generated_at_utc": "2026-09-30T18:09:00Z",
            "reports_loaded": 11,
            "reports_expected": 11,
            "errors": {},
            "maturation_control_tower": {
                "status": "WATCH",
                "families": [
                    {"key": "one_x_two", "label": "1X2", "current": 47, "target": 50, "status": "MATURING"},
                    {"key": "ft_totals", "label": "FT Totals", "current": 4, "target": 50, "status": "MATURING", "blocker": "FT_TOTALS_TRUE_CLV_4_LT_50"},
                ],
                "monitoring": {
                    "report_freshness": {"status": "WATCH", "reason": "STALE"},
                    "evidence_age": {"status": "WATCH", "evidence": {"age_hours": 25.0}},
                },
            },
        },
        "match_table_rows": [
            {
                "fixture_id": 1001,
                "home_team": "Alpha",
                "away_team": "Beta",
                "league": "Test League",
                "kickoff": "2026-09-30T20:00:00Z",
                "status": "NS",
                "stage": "T-35",
                "market_family": "FT_TOTALS",
                "selection": "Over 2.5",
                "p_model_calibrated": 0.60,
                "p_market_fair": 0.50,
                "model_signal": "STRONG",
                "model_signal_score": 84,
                "execution_status": "RESEARCH_ONLY",
                "price": 1.90,
                "fair_price": 2.00,
                "data_tier": "A",
                "confirmed_xi": True,
                "lambda_home": 1.72,
                "lambda_away": 0.94,
                "prob_home_win": 0.55,
                "prob_draw": 0.26,
                "prob_away_win": 0.19,
                "score_matrix": {"1-0": 0.14, "2-0": 0.13, "1-1": 0.12},
                "sport_profile": {"attack_score": 0.81, "defense_score": 0.66},
                "provider_update": "2026-09-30T18:09:30Z",
            },
            {
                "fixture_id": 1002,
                "home_team": "Gamma",
                "away_team": "Delta",
                "status": "NS",
                "market_family": "BTTS",
                "selection": "BTTS Yes",
                "execution_status": "WAIT_PRICE",
            },
        ],
        "market_mismatch_rows": [],
    }


def test_preview_payload_wires_all_live_surfaces_without_fake_zeroes():
    out = build_preview_payload(_payload())
    assert out["today"]["metrics"]["matches_scanned"] == 321
    assert out["today"]["metrics"]["deep_analyzed"] == 7
    assert out["today"]["metrics"]["strong_edges"] == 1
    assert out["today"]["top_edge"]["match"]["label"] == "Alpha vs Beta"
    assert round(out["today"]["top_edge"]["pricing"]["edge_pp"], 1) == 10.0

    rows = out["edge_feed"]["rows"]
    wait = next(row for row in rows if row["match"]["fixture_id"] == 1002)
    assert wait["model"]["probability"] is None
    assert wait["pricing"]["market_probability"] is None
    assert wait["pricing"]["edge_pp"] is None

    detail = out["match_center"]["detail"]
    assert detail["expected_goals"] == {"home": 1.72, "away": 0.94, "total": 2.66}
    assert round(detail["outcome_probabilities"]["home"], 2) == 0.55
    assert detail["score_matrix"][0]["score"] == "1-0"
    assert detail["model_context"]["data_quality"] == "A"
    assert detail["model_context"]["lineup"] == "CONFIRMED"
    assert detail["sport_profile"][0]["label"] == "Attack"

    catalog = {row["label"]: row for row in out["markets"]["families"]}
    assert catalog["FT Totals"]["live_rows"] == 1
    assert catalog["BTTS"]["live_rows"] == 1
    assert catalog["FT Totals"]["maturity_current"] == 4

    alerts = out["my_edge"]["system_alerts"]
    assert any(alert["type"] == "PRICE" for alert in alerts)
    assert any(alert["type"] == "MATURATION" for alert in alerts)

    tower = out["control_tower"]
    assert tower["runtime_generated_at_utc"] == "2026-09-30T18:10:00Z"
    assert tower["maturation"]["status"] == "WATCH"
    assert tower["maturation"]["monitoring"]["evidence_age"]["evidence"]["age_hours"] == 25.0

    firewall = out["research_lab"]["firewall"]
    assert firewall["decision_weight"] == 0.0
    assert firewall["production_promotion_allowed"] is False
    assert out["provider_requests_added"] == 0
    assert out["canonical_bet_logic_changed"] is False


def test_live_preview_html_wires_three_advanced_surfaces_and_neutralizes_mock_data():
    html = _html()
    assert "/app-preview/data" in html
    assert "/app-preview/performance" in html
    assert "renderMatch" in html
    assert "renderPerformance" in html
    assert "renderResearch" in html
    assert "renderMyEdge" in html
    assert "neutralizeMocks" in html
    assert "DEVICE PREVIEW" in html
    assert "Missing data stays missing" in html


def test_multi_gate_maturity_does_not_call_zero_true_clv_no_evidence():
    clv_report = {
        "mapped_family_counts": {
            "1H": 123,
            "2H": 123,
            "HOME_TT": 124,
            "AWAY_TT": 124,
            "FT_CORNERS": 82,
            "TEAM_CORNERS": 18,
        },
        "priced_entry_family_counts": {
            "1H": 123,
            "2H": 123,
            "HOME_TT": 124,
            "AWAY_TT": 124,
            "FT_CORNERS": 82,
            "TEAM_CORNERS": 18,
        },
        "family_counts": {"1X2": 47, "BTTS": 15, "FT_TOTALS": 4},
        "team_totals_maturation_funnel": {"modeled_signal_unique_fixtures": 29},
    }
    reports = {
        "Team Totals": {
            "status": "RESEARCH_HOLD",
            "oos_sample": {"evaluated_fixtures": 500, "minimum_actionable_review_fixtures": 200},
            "true_clv": {"rows": 0, "minimum_rows": 50, "unique_fixtures": 0},
            "blockers": ["TEAM_TOTALS_TRUE_CLV_0_LT_50"],
        },
        "1H": {
            "status": "RESEARCH_HOLD",
            "calibration": {"n": 220, "minimum_calibrated_oos": 100},
            "true_clv": {"rows": 0, "minimum_rows": 50, "unique_fixtures": 0},
            "blockers": ["CHALLENGER_BRIER_NOT_BETTER_THAN_BASELINE", "1H_TRUE_CLV_0_LT_50"],
        },
        "Corners": {
            "status": "RESEARCH_HOLD",
            "ft_corners": {"formation_adjusted_evaluations": 39},
            "true_clv": {"rows": 0, "minimum_rows": 50, "unique_fixtures": 0},
            "blockers": ["FORMATION_ADJUSTED_39_LT_100", "CORNERS_TRUE_CLV_0_LT_50"],
        },
    }

    rows = {row["label"]: row for row in _build_family_rows(clv_report, reports)}
    team = rows["Team Totals"]
    assert team["model_evidence"]["current"] == 500
    assert team["model_evidence"]["target"] == 200
    assert team["model_evidence"]["ready"] is True
    assert team["mapped_rows"] == 248
    assert team["priced_rows"] == 248
    assert team["modeled_signal_fixtures"] == 29
    assert team["true_clv_rows"] == 0
    assert team["true_clv_target"] == 50
    assert team["stage"] == "TRUE CLV COLLECTION"
    assert "NO EVIDENCE" not in team["stage"]

    one_h = rows["1H"]
    assert one_h["model_evidence"]["current"] == 220
    assert one_h["model_evidence"]["target"] == 100
    assert one_h["priced_rows"] == 123
    assert one_h["true_clv_rows"] == 0
    assert one_h["stage"] == "MODEL REVIEW + CLV COLLECTION"

    corners = rows["Corners"]
    assert corners["model_evidence"]["current"] == 39
    assert corners["model_evidence"]["target"] == 100
    assert corners["priced_rows"] == 100
    assert corners["true_clv_rows"] == 0
    assert corners["true_clv_target"] == 50
    assert corners["stage"] == "FORMATION MATURATION + CLV"


def test_v232_html_labels_true_clv_as_one_gate_not_total_maturity():
    html = maturity_html()
    assert "/app-preview/maturity" in html
    assert "Model / OOS" in html
    assert "Market evidence" in html
    assert "Strict True CLV" in html
    assert "LIVE MATURITY · MULTI-GATE" in html
    assert "1X2 True CLV" in html
    assert "formationPrimary=m.label==='Corners'" in html
    assert "Strict CLV ${clvText(m)}" in html
