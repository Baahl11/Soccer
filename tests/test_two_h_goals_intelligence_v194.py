from __future__ import annotations

import re
from pathlib import Path

from mcp_gateway import two_h_goals_intelligence as two_h


def _value(selection: str, line: float, decimal_price: float) -> dict:
    return {"selection": selection, "line": line, "decimal_price": decimal_price}


def test_v194_observed_market_scope_is_only_second_half_goals_over_under() -> None:
    event = {
        "derivative_research_market_snapshot": {
            "groups": [
                {
                    "family": "2H_GOALS",
                    "market": "Goals Over/Under - Second Half",
                    "bookmaker": "scope-test",
                    "values": [_value("OVER", 1.5, 2.05), _value("UNDER", 1.5, 1.80)],
                },
                {
                    "family": "2H_GOALS",
                    "market": "Team Total Goals Over/Under - Second Half",
                    "bookmaker": "scope-test",
                    "values": [_value("OVER", 0.5, 1.90), _value("UNDER", 0.5, 1.90)],
                },
                {
                    "family": "2H_GOALS",
                    "market": "Second Half Result 1X2",
                    "bookmaker": "scope-test",
                    "values": [_value("OVER", 1.5, 2.00)],
                },
                {
                    "family": "2H_CARDS",
                    "market": "Cards Over/Under - Second Half",
                    "bookmaker": "scope-test",
                    "values": [_value("OVER", 1.5, 2.00), _value("UNDER", 1.5, 1.80)],
                },
                {
                    "family": "1H_GOALS",
                    "market": "Goals Over/Under - First Half",
                    "bookmaker": "scope-test",
                    "values": [_value("OVER", 1.5, 2.00), _value("UNDER", 1.5, 1.80)],
                },
            ]
        }
    }

    rows, unsupported = two_h._observed(event, 1.20)

    assert len(rows) == 2
    assert unsupported == []
    assert {row["selection"] for row in rows} == {"OVER", "UNDER"}
    assert {row["market"] for row in rows} == {"Goals Over/Under - Second Half"}
    assert all(row["decision_weight"] == 0.0 for row in rows)
    assert all(row["actionable"] is False for row in rows)
    assert all(row["research_only"] is True for row in rows)


def test_v194_build_uses_existing_2h_pregame_baseline_and_is_never_promotable(monkeypatch) -> None:
    calls: list[str] = []

    def fake_model_fixture(fixture, period, registry):
        calls.append(period)
        return {"model": "V194_TEST_PERIOD_MODEL", "period": period, "total_lambda": 1.20}

    monkeypatch.setattr(two_h.period_rate_registry, "model_fixture", fake_model_fixture)

    intel = two_h.build(
        {
            "fixture": {"fixture_id": 194},
            "stage": "T-90",
            "derivative_research_market_snapshot": {"groups": []},
        },
        {"registry": "existing"},
    )

    assert calls == ["2H"]
    assert intel["status"] == "LIVE_RESEARCH_MODELED"
    assert intel["model_timing"] == "PREGAME_ONLY_NOT_HALFTIME_CONDITIONED"
    assert intel["decision_weight"] == 0.0
    assert intel["production_promotion_allowed"] is False
    assert intel["provider_requests_added"] == 0
    assert intel["actionable"] is False


def test_v194_attach_publishes_two_h_goals_intelligence_with_zero_request_budget(monkeypatch) -> None:
    monkeypatch.setattr(two_h.period_rate_registry, "load_registry", lambda: {"registry": "existing"})
    monkeypatch.setattr(
        two_h,
        "build",
        lambda event, registry: {
            "status": "LIVE_RESEARCH_MODELED",
            "observed_market_count": 0,
            "observed_market_rows": [],
            "model_inputs": {"period": "2H", "total_lambda": 1.20},
            "model_timing": "PREGAME_ONLY_NOT_HALFTIME_CONDITIONED",
            "calibration_gate": {},
            "actionable": False,
            "decision_weight": 0.0,
            "production_promotion_allowed": False,
            "provider_requests_added": 0,
        },
    )
    payload = {
        "events": [
            {
                "event_type": "SOCCER_REFRESH",
                "stage": "T-90",
                "fixture": {"fixture_id": 194},
                "match_intelligence": {"areas": {}},
            }
        ]
    }

    telemetry = two_h.attach(payload)
    event = payload["events"][0]

    assert "two_h_goals_intelligence" in event
    assert event["two_h_goals_intelligence"]["decision_weight"] == 0.0
    assert event["two_h_goals_intelligence"]["production_promotion_allowed"] is False
    assert event["two_h_goals_intelligence"]["provider_requests_added"] == 0
    assert event["match_intelligence"]["areas"]["goals_second_half_pregame"]["production_promotion_allowed"] is False
    assert telemetry["decision_weight"] == 0.0
    assert telemetry["production_promotion_allowed"] is False
    assert telemetry["provider_requests_added"] == 0


def test_v194_product_wrapper_chain_reaches_existing_v47_2h_attach_without_new_derivative_wiring() -> None:
    version = 123
    visited: list[int] = []
    while version != 47:
        assert version not in visited, f"automation wrapper cycle at v{version}"
        visited.append(version)
        source = Path(f"mcp_gateway/automation_v{version}.py").read_text(encoding="utf-8")
        match = re.search(r"from mcp_gateway import automation_v(\d+) as v\d+", source)
        assert match, f"automation_v{version}.py does not delegate to a prior wrapper"
        next_version = int(match.group(1))
        assert next_version < version, f"automation_v{version}.py does not move backward"
        version = next_version

    source_v47 = Path("mcp_gateway/automation_v47.py").read_text(encoding="utf-8")
    assert "two_h_goals_intelligence.attach(payload)" in source_v47
    assert "two_h_cards" not in source_v47.lower()
    assert "two_h_1x2" not in source_v47.lower()
    assert len(visited) < 100
