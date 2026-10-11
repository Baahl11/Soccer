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
    for family, filename in m._REPORTS.items():
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
        assert row["source_model_version"] == (
            reports[row["parent_family"]].get("model_version")
            if row["parent_family"] in reports else None
        )
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
        "SHOTS": "shots", "SOT": "sot", "GOALSCORER": "goalscorer",
        "ASSISTS": "assists", "PLAYER_CARDS": "cards", "GK_SAVES": "gk_saves",
    }
    for key, subkey in prop_key_map.items():
        source = reports["Player Props"]["prop_families"][subkey]["oos_evidence"]
        assert by_key[key]["model_evidence"]["current"] == source["player_game_rows"]
        assert by_key[key]["model_evidence"]["ready"] is source["oos_validation_complete"]
        assert by_key[key]["report_true_clv_rows"] == reports["Player Props"]["true_clv"]["by_family"][subkey]["rows"]
        assert by_key[key]["true_clv_rows"] is None

    for key in ("1H", "2H", "FT_CORNERS", "TEAM_CORNERS", "YELLOW_CARDS", "RED_CARDS"):
        assert by_key[key]["true_clv_rows"] is None
        assert "MARKET_TRUE_CLV_NOT_VERIFIED" in by_key[key]["blockers"]
    for key in ("DOUBLE_CHANCE", "DNB", "ASIAN_HANDICAP", "CORRECT_SCORE"):
        assert by_key[key]["source"] is None
        assert by_key[key]["model_evidence"]["current"] is None
        assert by_key[key]["report_status"] == "NOT VERIFIED"
        assert by_key[key]["production_promotion_allowed"] is False
