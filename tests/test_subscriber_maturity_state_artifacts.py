"""Contract reconciliation against the real persisted soccer-edge-state research artifacts.

No provider calls, no fixture writes, and no deployment. This check reads the
same canonical report names as the customer maturity endpoint.
"""
from __future__ import annotations

import json
from pathlib import Path

from mcp_gateway import subscriber_maturity_v232 as m

ROOT = Path("state/soccer_edge_state/analysis")


def test_real_canonical_research_sources_match_customer_contract():
    assert ROOT.is_dir(), "Canonical soccer-edge-state branch was not checked out"
    clv_path = ROOT / "clv_v4_postgres_report.json"
    assert clv_path.is_file(), "Canonical strict True CLV artifact absent"
    clv = json.loads(clv_path.read_text(encoding="utf-8"))
    reports = {}
    for family, filename in {**m._REPORTS, **m._MARKET_ADDITIONAL_REPORTS}.items():
        path = ROOT / filename
        assert path.is_file(), f"Missing source for {family}: {filename}"
        reports[family] = json.loads(path.read_text(encoding="utf-8"))

    source_family_counts = clv.get("family_counts") or {}
    assert isinstance(source_family_counts, dict)
    assert clv["comparable_true_clv_rows"] == sum(source_family_counts.values())
    assert set(source_family_counts) <= set(k for keys in m._COLLECTION_KEYS.values() for k in keys)

    parents = m._build_family_rows(clv, reports)
    rows = m._build_market_inventory(parents, clv, reports)
    assert len(rows) == 21 and len({row["key"] for row in rows}) == 21
    by_key = {row["key"]: row for row in rows}
    for key, row in by_key.items():
        # Canonical strict True CLV counts must never be manufactured.
        assert row["true_clv_rows"] == source_family_counts.get(key)
        assert row["production_promotion_allowed"] is False
        assert row["source_temporal_provenance"] == "NOT VERIFIED"
        expected_source_model = (
            reports["Player Props OOS"].get("model_version")
            if row["parent_family"] == "Player Props"
            else reports[row["parent_family"]].get("model_version")
            if row["parent_family"] in reports else None
        )
        assert row["source_model_version"] == expected_source_model
    assert by_key["1X2"]["model_evidence"]["current"] == (
        reports["1X2"]["canonical_multiclass_oos"]["temperature_scaled"]["n"]
    )
    assert by_key["BTTS"]["model_evidence"]["current"] == reports["BTTS"]["canonical_oos_calibration"]["rows"]
    assert by_key["FT_TOTALS"]["model_evidence"]["independent_oos"] is False
    assert "not independently verified OOS" in by_key["FT_TOTALS"]["model_evidence"]["unit"]
    assert by_key["HOME_TT"]["model_evidence"]["current"] == reports["Team Totals"]["by_team_role"]["HOME"]["n"]
    assert by_key["AWAY_TT"]["model_evidence"]["current"] == reports["Team Totals"]["by_team_role"]["AWAY"]["n"]
    assert by_key["HOME_TT"]["true_clv_target"] is None
    assert by_key["AWAY_TT"]["true_clv_target"] is None
    assert by_key["YELLOW_CARDS"]["model_evidence"]["current"] == reports["Cards"]["yellow_cards"]["oos_n"]
    assert by_key["RED_CARDS"]["model_evidence"]["target"] == reports["Cards"]["red_cards"]["minimum_market_review"]

    prop_key_map = {
        "SHOTS": ("shots", "SHOTS"), "SOT": ("sot", "SOT"),
        "GOALSCORER": ("goalscorer", "GOALSCORER_ANYTIME"),
        "ASSISTS": ("assists", "ASSISTS"),
        "PLAYER_CARDS": ("cards", "PLAYER_CARDS"),
        "GK_SAVES": ("gk_saves", "GK_SAVES"),
    }
    prop_oos = reports["Player Props OOS"]
    assert prop_oos["review_ready_family_count"] == 0
    assert sum(prop_oos["families"][audit]["player_game_rows"] for _, audit in prop_key_map.values()) == prop_oos["player_game_rows"]
    for key, (old_key, new_key) in prop_key_map.items():
        source = prop_oos["families"][new_key]
        row = by_key[key]
        assert row["model_evidence"]["current"] == source["player_game_rows"]
        assert row["model_evidence"]["target"] == source["minimum_player_games_for_review"]
        assert row["model_evidence"]["unique_fixtures"] == source["unique_fixtures"]
        assert row["model_evidence"]["ready"] is False
        assert row["market_specific_evidence_verified"] is False
        assert "MARKET_OOS_CALIBRATION_INCOMPLETE" in row["blockers"]
        assert "MARKET_OOS_SAMPLE_BELOW_REVIEW_MINIMUM" in row["blockers"]
        assert row["source"] == "player_props_oos_v4_report.json"
        assert row["clv_report_source"] == "phase15_player_props_validation.json"
        assert row["report_true_clv_rows"] == reports["Player Props"]["true_clv"]["by_family"][old_key]["rows"]
        assert row["true_clv_rows"] is None

    for key in ("1H", "2H", "FT_CORNERS", "TEAM_CORNERS", "YELLOW_CARDS", "RED_CARDS"):
        assert by_key[key]["true_clv_rows"] is None
        assert "MARKET_TRUE_CLV_NOT_VERIFIED" in by_key[key]["blockers"]
    for key in ("DOUBLE_CHANCE", "DNB", "ASIAN_HANDICAP", "CORRECT_SCORE"):
        assert by_key[key]["source"] is None
        assert by_key[key]["model_evidence"]["current"] is None
        assert by_key[key]["report_status"] == "NOT VERIFIED"
        assert by_key[key]["production_promotion_allowed"] is False
