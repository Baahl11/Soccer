from datetime import datetime, timedelta, timezone

from mcp_gateway import maturation_watchdogs_v4 as watchdogs
from mcp_gateway import maturity_snapshot_v4 as snapshot
from mcp_gateway import product_dashboard_v4 as dashboard


def _reports(now):
    return {
        "signal_summary": {
            "report_updated_at_utc": now.isoformat(),
            "last_evidence_at_local": (now - timedelta(hours=60)).isoformat(),
            "stage_counts": {"CLOSE": 12},
        },
        "one_x_two": {"true_clv": {"rows": 47, "minimum_rows": 50, "unique_fixtures": 47}},
        "btts": {"true_clv": {"rows": 15, "minimum_rows": 50, "unique_fixtures": 15}},
        "team_totals": {"true_clv": {"rows": 0, "minimum_rows": 50, "unique_fixtures": 0}},
        "one_h": {"true_clv": {"rows": 0, "minimum_rows": 50, "unique_fixtures": 0}},
        "two_h": {"true_clv": {"rows": 0, "minimum_rows": 50, "unique_fixtures": 0}},
        "corners": {"minimum_formation_adjusted": 100, "ft_corners": {"formation_adjusted_evaluations": 39}},
        "cards": {"market_evidence": {"match_cards": {"priced_value_rows": 0, "market_snapshot_rows": 0, "unique_fixtures": 0}}},
        "player_props": {"true_clv": {"rows": 0, "minimum_rows": 50}, "prop_families": {"assists": {"market_evidence": {"priced_value_rows": 10, "confirmed_xi_pre_kickoff_unique_fixtures": 0, "confirmed_xi_player_aligned_unique_fixtures": 0}}}},
        "settlement_coverage": {"settlement_rows": 0, "by_market_family_reason": {}},
        "api_efficiency": {"totals": {"api_calls_this_tick": 55}},
    }


def test_signal_report_freshness_is_separate_from_evidence_age():
    now = datetime(2026, 9, 29, 18, 0, tzinfo=timezone.utc)
    bundle, _ = watchdogs.build_watchdogs(_reports(now), now=now)
    nodes = bundle["watchdogs"]
    assert nodes["signal_close_freshness"]["status"] == "OK"
    assert nodes["signal_close_freshness"]["reason"] == "SIGNAL_REPORT_ARTIFACT_FRESH"
    assert nodes["signal_evidence_age"]["status"] == "WATCH"
    assert nodes["signal_evidence_age"]["reason"] == "SIGNAL_EVIDENCE_AGE_OVER_48H"


def test_maturation_control_tower_exposes_eight_families_and_blockers():
    now = datetime(2026, 9, 29, 18, 0, tzinfo=timezone.utc)
    reports = _reports(now)
    bundle, _ = watchdogs.build_watchdogs(reports, now=now)
    tower = snapshot._build_maturation_control_tower(
        reports,
        bundle,
        {"status": "OK", "source": "POSTGRES:maturation_watchdog_state"},
    )
    by_key = {row["key"]: row for row in tower["families"]}
    assert len(by_key) == 8
    assert by_key["one_x_two"]["current"] == 47
    assert by_key["corners"]["current"] == 39
    assert by_key["two_h"]["status"] == "WATCH"
    assert by_key["player_props"]["blocker"] == "PLAYER_PROP_XI_ALIGNMENT_GAPS"
    assert tower["provider_requests_added"] == 0
    assert tower["production_promotion_allowed"] is False


def test_dashboard_renders_maturation_control_tower():
    payload = {
        "views": {
            "control_tower": {
                "status": "LIVE",
                "system_health": {},
                "pipeline": {},
                "errors": {"count": 0, "rows": []},
                "validation_gates": [],
                "phases": [],
                "production_valid_market_count": 0,
                "maturity_snapshot": {
                    "status": "OK",
                    "reports_loaded": 11,
                    "reports_expected": 11,
                    "watchdog_baseline_store": {"status": "OK"},
                    "maturation_watchdogs": {"ok_count": 0, "watch_count": 0, "not_verified_count": 0, "watchdogs": {}},
                    "maturation_control_tower": {
                        "families": [{"key": "one_x_two", "label": "1X2", "evidence_kind": "TRUE_CLV", "current": 47, "target": 50, "unique_fixtures": 47, "status": "MATURING", "source": "1x2.json"}]
                    },
                },
            }
        }
    }
    html = dashboard.render_dashboard(payload)
    assert "Maturation Control Tower" in html
    assert "Baseline OK · read-only" in html
    assert "47 / 50" in html
    assert "TRUE_CLV" in html
