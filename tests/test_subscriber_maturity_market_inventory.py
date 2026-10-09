"""Frontend maturity contract regression: no synthesized CLV, no production promotion."""
from mcp_gateway import subscriber_maturity_v232 as maturity


def test_maturity_market_inventory_is_complete_but_not_promoted():
    families = [
        {
            "label": "1X2", "source": "1x2_report.json",
            "model_evidence": {"current": 670, "target": None, "ready": True, "unit": "OOS"},
            "true_clv_target": 50, "stage": "MODEL REVIEW + CLV COLLECTION",
            "blockers": ["TRUE_CLV_GATE_PENDING"], "next_gate": "TRUE_CLV_GATE_PENDING",
        },
        {
            "label": "Team Totals", "source": "team_totals.json",
            "model_evidence": {"current": 500, "target": 500, "ready": True, "unit": "OOS"},
            "true_clv_target": 50, "stage": "REVIEW READY",
        },
    ]
    clv = {
        "family_counts": {"1X2": 53, "HOME_TT": 248, "AWAY_TT": 266},
        "priced_entry_family_counts": {"1X2": 79, "HOME_TT": 310},
        "mapped_family_counts": {"1X2": 100, "HOME_TT": 330},
    }
    rows = maturity._build_market_inventory(families, clv, {})
    assert len(rows) == 21
    by_key = {row["key"]: row for row in rows}
    assert by_key["1X2"]["true_clv_rows"] == 53
    assert by_key["1X2"]["model_evidence"]["current"] == 670
    assert by_key["HOME_TT"]["true_clv_rows"] == 248
    assert by_key["AWAY_TT"]["true_clv_rows"] == 266
    # Parent's 500 OOS fixtures are NOT copied to each team-total side.
    assert by_key["HOME_TT"]["model_evidence"]["current"] is None
    assert by_key["AWAY_TT"]["model_evidence"]["current"] is None
    assert by_key["CORRECT_SCORE"]["true_clv_rows"] is None
    assert by_key["PLAYER_CARDS"]["model_evidence"]["current"] is None
    assert all(row["production_promotion_allowed"] is False for row in rows)
    assert all(row["classification"] == "RESEARCH_ONLY" for row in rows)


def test_clv_unknown_vs_actual_zero_and_card_oos_provenance():
    families = [{"label": "Cards", "source": "cards.json", "blockers": ["REFEREE_ADJUSTED_MISSING"]}]
    reports = {"Cards": {"yellow_cards": {"oos_n": 254, "minimum_oos": 100, "referee_adjusted_n": 0}, "red_cards": {"oos_n": 184, "minimum_oos": 500}}}
    missing = maturity._build_market_inventory(families, {}, reports)
    by_key = {row["key"]: row for row in missing}
    assert by_key["1X2"]["true_clv_rows"] is None
    assert by_key["YELLOW_CARDS"]["model_evidence"]["current"] == 254
    assert by_key["RED_CARDS"]["model_evidence"]["current"] == 184
    assert by_key["RED_CARDS"]["model_evidence"]["ready"] is False
    # A generic cards CLV 0 must not be attributed to yellow and red twice.
    assert by_key["YELLOW_CARDS"]["true_clv_rows"] is None
    assert by_key["RED_CARDS"]["true_clv_rows"] is None


def test_sparse_canonical_clv_preserves_unknown_in_every_market_and_family():
    """A report proving 1X2=0 does not prove BTTS or other markets=0."""
    partial_clv = {"family_counts": {"1X2": 0}}
    family_rows = maturity._build_family_rows(partial_clv, {})
    by_family = {row["label"]: row for row in family_rows}
    assert by_family["1X2"]["true_clv_rows"] == 0
    assert by_family["BTTS"]["true_clv_rows"] is None
    assert by_family["Player Props"]["model_evidence"]["current"] is None
    markets = maturity._build_market_inventory(family_rows, partial_clv, {})
    by_key = {row["key"]: row for row in markets}
    assert by_key["1X2"]["true_clv_rows"] == 0
    assert by_key["BTTS"]["true_clv_rows"] is None
    assert by_key["HOME_TT"]["true_clv_rows"] is None
    assert by_key["YELLOW_CARDS"]["true_clv_rows"] is None
    assert by_key["SHOTS"]["true_clv_rows"] is None
    assert by_key["1X2"]["source"] is None
    assert by_key["1X2"]["parent_research_stage"] is None
    assert by_key["DOUBLE_CHANCE"]["next_gate"] == "SOURCE_REPORT_NOT_VERIFIED"
    assert "MARKET_TRUE_CLV_NOT_VERIFIED" in by_key["BTTS"]["blockers"]
    assert all(not row["production_promotion_allowed"] for row in markets)


def test_child_market_does_not_inherit_parent_review_ready_or_clv_target():
    families = [{
        "label": "Team Totals", "stage": "REVIEW READY",
        "source": "team_totals.json", "true_clv_target": 50,
        "blockers": [], "model_evidence": {"current": 500, "target": 500},
    }]
    rows = maturity._build_market_inventory(families, {"family_counts": {"HOME_TT": 0}}, {
        "Team Totals": {"status": "OK", "oos_sample": {"evaluated_fixtures": 500}},
    })
    home = next(row for row in rows if row["key"] == "HOME_TT")
    away = next(row for row in rows if row["key"] == "AWAY_TT")
    assert home["true_clv_rows"] == 0
    assert away["true_clv_rows"] is None
    assert home["true_clv_target"] is None
    assert away["true_clv_target"] is None
    assert home["model_evidence"]["current"] is None
    assert not home["market_specific_evidence_verified"]
    assert home["next_gate"] == "MARKET_OOS_NOT_VERIFIED"
    assert home["parent_research_stage"] == "REVIEW READY"
    assert home["production_promotion_allowed"] is False


def test_authenticated_maturity_route_blocks_anonymous_and_free(monkeypatch):
    import asyncio
    from starlette.requests import Request
    from mcp_gateway import subscriber_preview_maturity_v232 as route

    def request():
        return Request({"type": "http", "method": "GET", "headers": []})

    def deny_data_access(*args, **kwargs):
        raise AssertionError("Research must not be loaded for unauthorized users")

    monkeypatch.setattr(route.subscriber_maturity_v232, "load_maturity_evidence", deny_data_access)
    monkeypatch.setattr(route.supabase_auth_v4, "bearer_token", lambda header: None)
    anonymous = asyncio.run(route.preview_maturity(request()))
    assert anonymous.status_code == 401

    monkeypatch.setattr(route.supabase_auth_v4, "bearer_token", lambda header: "test-token")
    monkeypatch.setattr(route.subscription_entitlements_v4, "resolve_entitlement",
                        lambda token: {"ok": True, "authenticated": True, "effective_plan": "FREE", "owner": False})
    free = asyncio.run(route.preview_maturity(request()))
    assert free.status_code == 403


def test_authenticated_maturity_route_returns_read_only_pro_evidence(monkeypatch):
    import asyncio
    import json
    from starlette.requests import Request
    from mcp_gateway import subscriber_preview_maturity_v232 as route

    monkeypatch.setattr(route.supabase_auth_v4, "bearer_token", lambda header: "test-token")
    monkeypatch.setattr(route.subscription_entitlements_v4, "resolve_entitlement",
                        lambda token: {"ok": True, "authenticated": True,
                                       "effective_plan": route.subscription_entitlements_v4.PRO_PLAN, "owner": False})
    monkeypatch.setattr(route.subscriber_maturity_v232, "load_maturity_evidence",
                        lambda: {"status": "PARTIAL", "market_rows": [], "production_promotion_allowed": False})
    response = asyncio.run(route.preview_maturity(Request({"type": "http", "method": "GET", "headers": []})))
    payload = json.loads(response.body)
    assert response.status_code == 200
    assert payload["status"] == "PARTIAL"
    assert payload["production_promotion_allowed"] is False


def test_total_source_failure_is_unavailable_not_partial(monkeypatch):
    class FailedSource:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def get(self, *args, **kwargs):
            raise ConnectionError("not available")

    monkeypatch.setattr(maturity, "_state_config", lambda: ("sample-state-repo", "sample-state-branch"))
    monkeypatch.setattr(maturity.httpx, "Client", lambda **kwargs: FailedSource())
    result = maturity.load_maturity_evidence(force=True)
    assert result["status"] == "UNAVAILABLE"
    assert result["errors"]
    assert len(result["market_rows"]) == 21
    assert all(row["source"] is None for row in result["market_rows"])
    assert all(row["true_clv_rows"] is None for row in result["market_rows"])
    assert all(row["production_promotion_allowed"] is False for row in result["market_rows"])


def test_only_relevant_parent_blockers_are_attached_to_each_submarket():
    family = [{
        "label": "Player Props",
        "blockers": [
            "SHOTS_OOS_VALIDATION_INCOMPLETE",
            "SOT_OOS_VALIDATION_INCOMPLETE",
            "ASSISTS_TRUE_CLV_0_LT_50",
            "PLAYER_PROP_TRUE_CLV_0_LT_50",
        ],
    }]
    report = {"Player Props": {"prop_families": {
        "shots": {"oos_evidence": {"player_game_rows": 0, "oos_validation_complete": False}},
        "sot": {"oos_evidence": {"player_game_rows": 0, "oos_validation_complete": False}},
    }}}
    rows = maturity._build_market_inventory(family, {}, report)
    shots = next(row for row in rows if row["key"] == "SHOTS")
    sot = next(row for row in rows if row["key"] == "SOT")
    assert "SHOTS_OOS_VALIDATION_INCOMPLETE" in shots["blockers"]
    assert "SOT_OOS_VALIDATION_INCOMPLETE" not in shots["blockers"]
    assert "ASSISTS_TRUE_CLV_0_LT_50" not in shots["blockers"]
    assert "PLAYER_PROP_TRUE_CLV_0_LT_50" in shots["blockers"]
    assert "SOT_OOS_VALIDATION_INCOMPLETE" in sot["blockers"]
    assert "SHOTS_OOS_VALIDATION_INCOMPLETE" not in sot["blockers"]
