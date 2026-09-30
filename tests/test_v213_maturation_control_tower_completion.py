from mcp_gateway import maturity_snapshot_v4 as v


def _reports():
    return {
        "one_x_two": {"true_clv": {"rows": 47, "minimum_rows": 50, "unique_fixtures": 47}},
        "btts": {"true_clv": {"rows": 15, "minimum_rows": 50, "unique_fixtures": 15}},
        "ft_totals": {
            "true_clv": {"rows": 4, "minimum_rows": 50, "unique_fixtures": 4},
            "blockers": ["FT_TOTALS_TRUE_CLV_4_LT_50"],
        },
        "team_totals": {"true_clv": {"rows": 0, "minimum_rows": 50, "unique_fixtures": 0}},
        "one_h": {"true_clv": {"rows": 0, "minimum_rows": 50, "unique_fixtures": 0}},
        "corners": {"minimum_formation_adjusted": 100, "ft_corners": {"formation_adjusted_evaluations": 39}},
        "two_h": {"true_clv": {"rows": 0, "minimum_rows": 50, "unique_fixtures": 0}},
        "cards": {"true_clv": {"rows": 0, "minimum_rows": 50}},
        "player_props": {"true_clv": {"minimum_rows": 50, "by_family": {"PLAYER_CARDS": {"rows": 0}, "SHOTS": {"rows": 3}}}},
    }


def test_v213_summary_exposes_ft_totals_true_clv_gate():
    summary = v._build_summary(_reports(), {})
    gate = summary["gates"]["ft_totals_true_clv"]

    assert gate["current"] == 4
    assert gate["target"] == 50
    assert gate["unique_fixtures"] == 4
    assert gate["source"] == "v4_016_ft_totals_production_validation.json"
    assert summary["provider_requests_added"] == 0
    assert summary["production_promotion_allowed"] is False
    assert summary["canonical_bet_logic_changed"] is False
    assert summary["model_weights_changed"] is False


def test_v213_maturation_tower_matches_official_roadmap_order_and_ft_totals_state():
    tower = v._build_maturation_control_tower(
        _reports(),
        {"status": "OK", "watchdogs": {}},
        {"status": "OK", "source": "TEST"},
    )

    assert [row["key"] for row in tower["families"]] == [
        "one_x_two",
        "btts",
        "ft_totals",
        "team_totals",
        "one_h",
        "corners",
        "two_h",
        "cards",
        "player_props",
    ]

    ft = next(row for row in tower["families"] if row["key"] == "ft_totals")
    assert ft["label"] == "FT Totals"
    assert ft["current"] == 4
    assert ft["target"] == 50
    assert ft["unique_fixtures"] == 4
    assert ft["status"] == "MATURING"
    assert ft["blocker"] == "FT_TOTALS_TRUE_CLV_4_LT_50"
    assert tower["provider_requests_added"] == 0
    assert tower["production_promotion_allowed"] is False
    assert tower["decision_weight"] == 0.0
