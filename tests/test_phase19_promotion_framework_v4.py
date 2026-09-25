from mcp_gateway import promotion_framework_v4 as v


def test_phase19_sample_policy_matches_master_roadmap():
    assert v.DIRECTIONAL_READ_MIN == 20
    assert v.TIER_B_REVIEW_MIN == 50
    assert v.TIER_A_REVIEW_MIN == 100
    assert v.TIER_S_REVIEW_MIN == 200
    assert v.MODEL_WEIGHT_CHANGE_MIN == 200


def test_raw_clv_rows_cannot_inflate_promotion_sample():
    review = v.review_market(
        market_family="1H",
        unique_fixtures=17,
        settled=100,
        roi_per_settled_unit=0.10,
        clv_rows=992,
        avg_clv_pp=0.5,
        stability_status="DATA_BLOCKED",
        shadow_settled=100,
        shadow_roi_per_settled_unit=0.05,
        shadow_sample_status="SHADOW_REVIEW_READY",
        validation_blockers=[],
    )
    assert review["recommended_state"] == "RESEARCH"
    assert review["tier_review_eligibility"]["directional_read"] is False


def test_phase19_never_auto_promotes_without_manual_approval():
    review = v.review_market(
        market_family="FT_TOTALS",
        unique_fixtures=80,
        settled=80,
        roi_per_settled_unit=0.12,
        clv_rows=160,
        avg_clv_pp=1.0,
        stability_status="STABILITY_REVIEW_READY",
        shadow_settled=80,
        shadow_roi_per_settled_unit=0.05,
        shadow_sample_status="SHADOW_REVIEW_READY",
        validation_blockers=[],
        manual_approval=False,
    )
    assert review["recommended_state"] == "LEAN_ELIGIBLE"
    assert "MANUAL_APPROVAL_REQUIRED_FOR_PRODUCTION_TIER" in review["warnings"]


def test_phase19_manual_approval_can_recommend_tier_by_unique_fixture_and_settlement_sample():
    review = v.review_market(
        market_family="FT_TOTALS",
        unique_fixtures=120,
        settled=120,
        roi_per_settled_unit=0.08,
        clv_rows=240,
        avg_clv_pp=0.5,
        stability_status="STABILITY_REVIEW_READY",
        shadow_settled=120,
        shadow_roi_per_settled_unit=0.04,
        shadow_sample_status="SHADOW_REVIEW_READY",
        validation_blockers=[],
        manual_approval=True,
    )
    assert review["recommended_state"] == "TIER_A"


def test_phase19_validation_blocker_holds_market_in_shadow_after_directional_sample():
    review = v.review_market(
        market_family="FT_CORNERS",
        unique_fixtures=60,
        settled=60,
        roi_per_settled_unit=0.05,
        clv_rows=200,
        avg_clv_pp=0.8,
        stability_status="STABILITY_REVIEW_READY",
        shadow_settled=60,
        shadow_roi_per_settled_unit=0.03,
        shadow_sample_status="SHADOW_REVIEW_READY",
        validation_blockers=["FORMATION_ADJUSTED_39_LT_100"],
    )
    assert review["recommended_state"] == "SHADOW"
    assert "VALIDATION:FORMATION_ADJUSTED_39_LT_100" in review["blockers"]


def test_phase19_flags_safety_demotion_only_for_production_collapse():
    review = v.review_market(
        market_family="FT_TOTALS",
        unique_fixtures=60,
        settled=60,
        roi_per_settled_unit=-0.05,
        clv_rows=120,
        avg_clv_pp=-0.2,
        stability_status="STABILITY_REVIEW_READY",
        shadow_settled=60,
        shadow_roi_per_settled_unit=-0.03,
        shadow_sample_status="SHADOW_REVIEW_READY",
        validation_blockers=[],
        current_state="TIER_B",
    )
    assert review["automatic_demotion_candidate"] is True
    assert review["recommended_state"] == "DEMOTED"


def test_build_report_uses_g5_unique_fixtures():
    report = v.build_report(
        {"by_market_family": {"FT_1X2": {"settled": 10, "roi_per_decision_units": 1.12}}},
        {
            "families": {
                "1X2": {
                    "status": "DATA_BLOCKED",
                    "overall": {
                        "rows": 27,
                        "unique_fixtures": 16,
                        "fixture_weighted_avg_probability_clv_pp": 0.159329,
                    },
                }
            }
        },
        {"1X2": {"status": "RESEARCH_HOLD", "blockers": ["ACTIONABLE_SAMPLE_LT_20"]}},
        {
            "by_market_family": {
                "FT_1X2": {
                    "rows": 30,
                    "settled": 30,
                    "shadow_roi_hypothetical_units": 1.5,
                }
            }
        },
    )
    reviews = {row["market_family"]: row for row in report["market_family_reviews"]}
    assert reviews["1X2"]["unique_fixtures"] == 16
    assert reviews["1X2"]["true_clv_rows"] == 27
    assert reviews["1X2"]["recommended_state"] == "RESEARCH"
    assert report["automatic_promotion_allowed"] is False
    assert report["runtime_state_mutation_enabled"] is False


def test_negative_shadow_roi_blocks_lean_eligibility():
    review = v.review_market(
        market_family="1X2",
        unique_fixtures=80,
        settled=80,
        roi_per_settled_unit=0.10,
        clv_rows=100,
        avg_clv_pp=0.4,
        stability_status="STABILITY_REVIEW_READY",
        shadow_settled=170,
        shadow_roi_per_settled_unit=-0.16,
        shadow_sample_status="SHADOW_REVIEW_READY",
        validation_blockers=[],
    )
    assert review["recommended_state"] == "SHADOW"
    assert "PROMOTION_SHADOW_ROI_NOT_POSITIVE" in review["blockers"]


def test_combined_family_performance_aggregates_aliases():
    perf = v._performance_for_family(
        {
            "by_market_family": {
                "2H_BTTS": {"n": 1, "settled": 1, "roi_units": -0.36},
                "2H_TOTALS": {"n": 2, "settled": 2, "roi_units": -0.72},
            }
        },
        ("2H", "2H_TOTALS", "2H_BTTS"),
    )
    assert perf["n"] == 3
    assert perf["settled"] == 3
    assert perf["roi_units"] == -1.08


def test_watch_alert_roi_does_not_count_as_promotion_shadow():
    report = v.build_report(
        {"by_market_family": {"FT_1X2": {"n": 80, "settled": 80, "roi_units": 8.0}}},
        {
            "families": {
                "1X2": {
                    "status": "STABILITY_REVIEW_READY",
                    "overall": {
                        "rows": 100,
                        "unique_fixtures": 80,
                        "fixture_weighted_avg_probability_clv_pp": 0.4,
                    },
                }
            }
        },
        {"1X2": {"status": "CALIBRATION_REVIEW_ELIGIBLE", "blockers": []}},
        {
            "by_market_family": {
                "FT_1X2": {
                    "rows": 170,
                    "settled": 170,
                    "shadow_roi_hypothetical_units": -28.22,
                    "shadow_roi_per_settled_unit": -0.166,
                    "sample_status": "SHADOW_REVIEW_READY",
                }
            }
        },
        {
            "market_family": "FT_1X2",
            "promotion_evaluable": {
                "settled": 0,
                "roi_per_settled_unit": None,
                "sample_status": "DATA_BLOCKED",
            },
        },
    )
    review = next(row for row in report["market_family_reviews"] if row["market_family"] == "1X2")
    assert review["recommended_state"] == "SHADOW"
    assert "PROMOTION_SHADOW_SETTLED_0_LT_DIRECTIONAL_20" in review["blockers"]
    assert "PROMOTION_SHADOW_ROI_NOT_POSITIVE" not in review["blockers"]
    assert review["watch_shadow_settled"] == 170
    assert review["watch_shadow_roi_per_settled_unit"] == -0.166


def test_clean_promotion_shadow_can_support_lean_eligibility():
    report = v.build_report(
        {"by_market_family": {"FT_1X2": {"n": 80, "settled": 80, "roi_units": 8.0}}},
        {
            "families": {
                "1X2": {
                    "status": "STABILITY_REVIEW_READY",
                    "overall": {
                        "rows": 100,
                        "unique_fixtures": 80,
                        "fixture_weighted_avg_probability_clv_pp": 0.4,
                    },
                }
            }
        },
        {"1X2": {"status": "CALIBRATION_REVIEW_ELIGIBLE", "blockers": []}},
        {},
        {
            "market_family": "FT_1X2",
            "promotion_evaluable": {
                "settled": 60,
                "roi_per_settled_unit": 0.08,
                "sample_status": "SHADOW_REVIEW_READY",
                "family_discrimination_ready": True,
                "not_ready_classes": [],
            },
        },
    )
    review = next(row for row in report["market_family_reviews"] if row["market_family"] == "1X2")
    assert review["recommended_state"] == "LEAN_ELIGIBLE"
    assert review["promotion_shadow_settled"] == 60


def test_postgres_phase16_shadow_replay_has_priority():
    report = v.build_report(
        {"by_market_family": {"FT_1X2": {"n": 80, "settled": 80, "roi_units": 8.0}}},
        {
            "families": {
                "1X2": {
                    "status": "STABILITY_REVIEW_READY",
                    "overall": {
                        "rows": 100,
                        "unique_fixtures": 80,
                        "fixture_weighted_avg_probability_clv_pp": 0.4,
                    },
                }
            }
        },
        {"1X2": {"status": "CALIBRATION_REVIEW_ELIGIBLE", "blockers": []}},
        {},
        {
            "market_family": "FT_1X2",
            "promotion_evaluable": {
                "settled": 0,
                "roi_per_settled_unit": None,
                "sample_status": "DATA_BLOCKED",
            },
        },
        {
            "market_family": "1X2",
            "promotion_evaluable": {
                "settled": 60,
                "roi_per_settled_unit": 0.08,
                "sample_status": "SHADOW_REVIEW_READY",
                "negative_directional_stages": [],
                "family_discrimination_ready": True,
                "not_ready_classes": [],
            },
        },
    )
    review = next(row for row in report["market_family_reviews"] if row["market_family"] == "1X2")
    assert review["recommended_state"] == "LEAN_ELIGIBLE"
    assert review["promotion_shadow_settled"] == 60
    assert review["promotion_shadow_evidence_source"] == "POSTGRES_PHASE16_REPLAY:PROMOTION_EVALUABLE"


def test_1x2_partial_class_readiness_blocks_family_advancement():
    report = v.build_report(
        {"by_market_family": {"FT_1X2": {"n": 80, "settled": 80, "roi_units": 8.0}}},
        {
            "families": {
                "1X2": {
                    "status": "STABILITY_REVIEW_READY",
                    "overall": {
                        "rows": 100,
                        "unique_fixtures": 80,
                        "fixture_weighted_avg_probability_clv_pp": 0.4,
                    },
                }
            }
        },
        {"1X2": {"status": "CALIBRATION_REVIEW_ELIGIBLE", "blockers": []}},
        {},
        {},
        {
            "families": {
                "1X2": {
                    "promotion_evaluable": {
                        "settled": 60,
                        "pending": 0,
                        "roi_per_settled_unit": 0.08,
                        "sample_status": "SHADOW_REVIEW_READY",
                        "negative_directional_stages": [],
                        "family_discrimination_ready": False,
                        "not_ready_classes": ["DRAW"],
                        "class_discrimination_diagnostics": {
                            "home_win": {"rows": 400, "auc_lower_95": 0.56, "ready": True},
                            "draw": {"rows": 400, "auc_lower_95": 0.48, "ready": False},
                            "away_win": {"rows": 400, "auc_lower_95": 0.55, "ready": True},
                        },
                    }
                }
            }
        },
    )
    review = next(row for row in report["market_family_reviews"] if row["market_family"] == "1X2")
    assert review["recommended_state"] == "SHADOW"
    assert "PROMOTION_SHADOW_1X2_CLASS_DISCRIMINATION_NOT_READY:DRAW" in review["blockers"]
    assert review["promotion_shadow_family_discrimination_ready"] is False
    assert review["promotion_shadow_not_ready_classes"] == ["DRAW"]
    assert review["tier_review_eligibility"]["promotion_shadow_family_discrimination"] is False
    readiness = report["promotion_readiness"]["families"]["1X2"]
    assert readiness["class_discrimination"]["family_ready"] is False
    assert readiness["class_discrimination"]["classes"]["draw"]["rows"] == 400
    assert readiness["class_discrimination"]["classes"]["draw"]["auc_lower_95"] == 0.48


def test_promotion_readiness_exposes_exact_remaining_counts_and_validator_deficits():
    report = v.build_report(
        {"by_market_family": {"FT_CORNERS": {"n": 35, "settled": 35, "roi_units": 2.1}}},
        {
            "families": {
                "FT_CORNERS": {
                    "status": "STABILITY_REVIEW_READY",
                    "overall": {
                        "rows": 60,
                        "unique_fixtures": 60,
                        "fixture_weighted_avg_probability_clv_pp": 0.3,
                    },
                }
            }
        },
        {
            "FT_CORNERS": {
                "status": "RESEARCH_HOLD",
                "true_clv": {"rows": 12, "minimum_rows": 50, "avg_probability_clv_pp": 0.2},
                "blockers": [
                    "FORMATION_ADJUSTED_39_LT_100",
                    "SOURCE_FT_CORNERS_PROMOTION_GATE_DISABLED",
                ],
            }
        },
        {},
        {},
        {
            "families": {
                "FT_CORNERS": {
                    "promotion_evaluable": {
                        "settled": 18,
                        "pending": 3,
                        "roi_per_settled_unit": 0.04,
                        "sample_status": "DATA_BLOCKED",
                        "negative_directional_stages": [],
                    }
                }
            }
        },
    )
    readiness = report["promotion_readiness"]["families"]["FT_CORNERS"]
    assert readiness["tier_samples"]["directional"]["sample_ready"] is True
    assert readiness["tier_samples"]["tier_b"]["settled_remaining"] == 15
    assert readiness["tier_samples"]["tier_a"]["unique_fixtures_remaining"] == 40
    assert readiness["tier_samples"]["tier_s"]["settled_remaining"] == 165
    assert readiness["promotion_shadow"]["directional_remaining"] == 2
    assert readiness["promotion_shadow"]["review_remaining"] == 32
    assert readiness["true_clv"]["rows_remaining"] == 38
    assert readiness["validation"]["numeric_deficits"][0]["gate"] == "FORMATION_ADJUSTED_39_LT_100"
    assert readiness["validation"]["numeric_deficits"][0]["remaining"] == 61.0
    assert readiness["validation"]["qualitative_blockers"] == ["SOURCE_FT_CORNERS_PROMOTION_GATE_DISABLED"]


def test_1x2_zero_shadow_rows_use_oos_class_diagnostics_for_blocker_identity_only():
    report = v.build_report(
        {"by_market_family": {"FT_1X2": {"n": 10, "settled": 10, "roi_units": 2.0}}},
        {
            "families": {
                "1X2": {
                    "status": "STABILITY_REVIEW_READY",
                    "overall": {
                        "rows": 72,
                        "unique_fixtures": 61,
                        "fixture_weighted_avg_probability_clv_pp": 0.1,
                    },
                }
            }
        },
        {"1X2": {"status": "CALIBRATION_REVIEW_ELIGIBLE", "blockers": []}},
        {},
        {},
        {
            "families": {
                "1X2": {
                    "promotion_evaluable": {
                        "rows": 0,
                        "settled": 0,
                        "pending": 0,
                        "roi_per_settled_unit": None,
                        "sample_status": "DATA_BLOCKED",
                        "negative_directional_stages": [],
                    }
                }
            }
        },
        {
            "current_model_deployment_calibrators": {
                "home_win": {
                    "rows": 407,
                    "eligible_for_phase16_research": True,
                    "brier_delta": -0.01,
                    "log_loss_delta": -0.02,
                    "discrimination": {
                        "auc": 0.61,
                        "auc_lower_95": 0.55,
                        "positive_count": 176,
                        "negative_count": 231,
                    },
                },
                "draw": {
                    "rows": 407,
                    "eligible_for_phase16_research": False,
                    "brier_delta": -0.007,
                    "log_loss_delta": -0.015,
                    "discrimination": {
                        "auc": 0.499,
                        "auc_lower_95": 0.433,
                        "positive_count": 96,
                        "negative_count": 311,
                    },
                },
                "away_win": {
                    "rows": 407,
                    "eligible_for_phase16_research": True,
                    "brier_delta": -0.008,
                    "log_loss_delta": -0.01,
                    "discrimination": {
                        "auc": 0.614,
                        "auc_lower_95": 0.554,
                        "positive_count": 135,
                        "negative_count": 272,
                    },
                },
            }
        },
    )
    review = next(row for row in report["market_family_reviews"] if row["market_family"] == "1X2")
    readiness = report["promotion_readiness"]["families"]["1X2"]

    assert "PROMOTION_SHADOW_1X2_CLASS_DISCRIMINATION_NOT_READY:DRAW" in review["blockers"]
    assert review["promotion_shadow_settled"] == 0
    assert review["promotion_shadow_roi_per_settled_unit"] is None
    assert review["promotion_shadow_not_ready_classes"] == ["DRAW"]
    assert "OOS_STAGE_DIAGNOSTICS_FALLBACK" in review["promotion_shadow_evidence_source"]
    assert readiness["class_discrimination"]["classes"]["draw"]["auc_lower_95"] == 0.433
    assert readiness["class_discrimination"]["classes"]["draw"]["ready"] is False


def test_multimarket_postgres_shadow_is_used_for_totals_and_btts():
    promotion_shadow = {
        "families": {
            "FT_TOTALS": {
                "promotion_evaluable": {
                    "settled": 25,
                    "pending": 4,
                    "roi_per_settled_unit": 0.06,
                    "sample_status": "DIRECTIONAL_SHADOW",
                    "negative_directional_stages": [],
                }
            },
            "BTTS": {
                "promotion_evaluable": {
                    "settled": 21,
                    "pending": 2,
                    "roi_per_settled_unit": 0.03,
                    "sample_status": "DIRECTIONAL_SHADOW",
                    "negative_directional_stages": [],
                }
            },
        }
    }
    totals = v._promotion_shadow_for_family("FT_TOTALS", {}, promotion_shadow)
    btts = v._promotion_shadow_for_family("BTTS", {}, promotion_shadow)
    assert totals["settled"] == 25
    assert totals["pending"] == 4
    assert totals["evidence_source"] == "POSTGRES_PHASE16_REPLAY:PROMOTION_EVALUABLE"
    assert btts["settled"] == 21
    assert btts["pending"] == 2


def test_load_validation_reports_uses_family_view_for_shared_corners_validator(tmp_path):
    corners = {
        "model_version": "SOCCER_CORNERS_OOS_VALIDATION_V4_1.1.0",
        "status": "RESEARCH_HOLD",
        "blockers": ["SHARED"],
        "family_views": {
            "FT_CORNERS": {
                "status": "OOS_REVIEW_ELIGIBLE",
                "blockers": [],
                "true_clv": {"rows": 60, "minimum_rows": 50},
            },
            "TEAM_CORNERS": {
                "status": "RESEARCH_HOLD",
                "blockers": ["TEAM_CORNERS_TRUE_CLV_10_LT_50"],
                "true_clv": {"rows": 10, "minimum_rows": 50},
            },
        },
    }
    for family, spec in v.FAMILY_SPECS.items():
        path = tmp_path / str(spec["validation_file"])
        if path.exists():
            continue
        path.write_text("{}", encoding="utf-8")
    (tmp_path / "v4_022_corners_oos_validation.json").write_text(__import__("json").dumps(corners), encoding="utf-8")

    reports = v._load_validation_reports(str(tmp_path))
    assert reports["FT_CORNERS"]["status"] == "OOS_REVIEW_ELIGIBLE"
    assert reports["FT_CORNERS"]["blockers"] == []
    assert reports["FT_CORNERS"]["validation_family_view"] == "FT_CORNERS"
    assert reports["TEAM_CORNERS"]["blockers"] == ["TEAM_CORNERS_TRUE_CLV_10_LT_50"]


def test_phase19_exposes_1x2_selection_progress_without_relaxing_family_gate():
    promotion_shadow = {
        "families": {
            "1X2": {
                "promotion_evaluable": {
                    "settled": 0,
                    "pending": 2,
                    "roi_per_settled_unit": None,
                    "sample_status": "DATA_BLOCKED",
                    "negative_directional_stages": [],
                    "family_discrimination_ready": False,
                    "not_ready_classes": ["DRAW"],
                    "class_discrimination_diagnostics": {
                        "home_win": {"rows": 407, "auc_lower_95": 0.5518, "ready": True},
                        "draw": {"rows": 407, "auc_lower_95": 0.4333, "ready": False},
                        "away_win": {"rows": 407, "auc_lower_95": 0.5544, "ready": True},
                    },
                    "by_selection": {
                        "HOME": {"rows": 1, "settled": 0, "pending": 1, "directional_remaining": 20, "review_remaining": 50},
                        "DRAW": {"rows": 0, "settled": 0, "pending": 0, "directional_remaining": 20, "review_remaining": 50},
                        "AWAY": {"rows": 1, "settled": 0, "pending": 1, "directional_remaining": 20, "review_remaining": 50},
                    },
                }
            }
        }
    }
    report = v.build_report(
        {"by_market_family": {"FT_1X2": {"n": 10, "settled": 10, "roi_units": 2.0}}},
        {
            "families": {
                "1X2": {
                    "status": "STABILITY_REVIEW_READY",
                    "overall": {
                        "rows": 82,
                        "unique_fixtures": 67,
                        "fixture_weighted_avg_probability_clv_pp": 0.122891,
                    },
                }
            }
        },
        {"1X2": {"status": "CALIBRATION_REVIEW_ELIGIBLE", "blockers": []}},
        {},
        {},
        promotion_shadow,
    )
    review = next(row for row in report["market_family_reviews"] if row["market_family"] == "1X2")
    readiness = report["promotion_readiness"]["families"]["1X2"]

    assert review["promotion_shadow_selection_progress"]["HOME"]["pending"] == 1
    assert review["promotion_shadow_selection_progress"]["AWAY"]["pending"] == 1
    assert review["promotion_shadow_selection_progress"]["DRAW"]["rows"] == 0
    assert readiness["class_discrimination"]["promotion_shadow_by_selection"]["HOME"]["rows"] == 1
    assert "PROMOTION_SHADOW_1X2_CLASS_DISCRIMINATION_NOT_READY:DRAW" in review["blockers"]
    assert review["tier_review_eligibility"]["promotion_shadow_family_discrimination"] is False


def test_combined_team_totals_uses_exact_fixture_union_without_double_counting():
    stability = {
        "families": {
            "HOME_TT": {
                "status": "DATA_BLOCKED",
                "overall": {
                    "rows": 4,
                    "unique_fixtures": 2,
                    "fixture_ids": [1, 2],
                    "fixture_weighted_avg_probability_clv_pp": 0.2,
                },
            },
            "AWAY_TT": {
                "status": "DATA_BLOCKED",
                "overall": {
                    "rows": 5,
                    "unique_fixtures": 2,
                    "fixture_ids": [2, 3],
                    "fixture_weighted_avg_probability_clv_pp": 0.4,
                },
            },
        }
    }
    combined = v._stability_for_family(stability, ("TEAM_TOTALS", "HOME_TT", "AWAY_TT"))
    assert combined["overall"]["rows"] == 9
    assert combined["overall"]["unique_fixtures"] == 3
    assert combined["overall"]["unique_fixture_count_source"] == "EXACT_FIXTURE_ID_UNION"


def test_combined_team_totals_legacy_report_keeps_conservative_max_fallback():
    stability = {
        "families": {
            "HOME_TT": {
                "status": "DATA_BLOCKED",
                "overall": {
                    "rows": 4,
                    "unique_fixtures": 2,
                    "fixture_weighted_avg_probability_clv_pp": 0.2,
                },
            },
            "AWAY_TT": {
                "status": "DATA_BLOCKED",
                "overall": {
                    "rows": 5,
                    "unique_fixtures": 3,
                    "fixture_weighted_avg_probability_clv_pp": 0.4,
                },
            },
        }
    }
    combined = v._stability_for_family(stability, ("TEAM_TOTALS", "HOME_TT", "AWAY_TT"))
    assert combined["overall"]["unique_fixtures"] == 3
    assert combined["overall"]["unique_fixture_count_source"] == "LEGACY_CONSERVATIVE_MAX"
