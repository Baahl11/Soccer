from __future__ import annotations

from datetime import datetime, timedelta, timezone

from mcp_gateway import maturation_watchdogs_v4 as w


def _reports(now: datetime) -> dict:
    return {
        "one_x_two": {"true_clv": {"rows": 10, "unique_fixtures": 5}},
        "btts": {"true_clv": {"rows": 15, "unique_fixtures": 8}},
        "team_totals": {"true_clv": {"rows": 3, "unique_fixtures": 2}},
        "one_h": {"true_clv": {"rows": 4, "unique_fixtures": 3}},
        "two_h": {"true_clv": {"rows": 0, "unique_fixtures": 0, "minimum_rows": 50}},
        "corners": {"ft_corners": {"formation_adjusted_evaluations": 39}},
        "cards": {
            "market_evidence": {
                "match_cards": {
                    "market_snapshot_rows": 0,
                    "priced_value_rows": 0,
                    "unique_fixtures": 0,
                }
            },
            "true_clv": {"rows": 0},
        },
        "player_props": {
            "prop_families": {
                "shots": {
                    "market_evidence": {
                        "market_snapshot_rows": 10,
                        "priced_value_rows": 100,
                        "confirmed_xi_pre_kickoff_unique_fixtures": 5,
                        "confirmed_xi_player_aligned_unique_fixtures": 0,
                    }
                },
                "goalscorer": {
                    "market_evidence": {
                        "market_snapshot_rows": 10,
                        "priced_value_rows": 100,
                        "confirmed_xi_pre_kickoff_unique_fixtures": 5,
                        "confirmed_xi_player_aligned_unique_fixtures": 4,
                    }
                },
            }
        },
        "signal_summary": {
            "last_generated_at_local": (now - timedelta(hours=72)).isoformat(),
            "stage_counts": {"CLOSE": 100},
        },
        "settlement_coverage": {
            "settlement_rows": 20,
            "by_market_family_reason": {"FT_TOTALS": {"IN_SETTLEMENT_LEDGER": 20}},
        },
        "api_efficiency": {"totals": {"api_calls_this_tick": 110}},
    }


def test_watchdogs_surface_current_maturation_gaps_without_runtime_changes():
    now = datetime(2026, 9, 29, 17, 0, tzinfo=timezone.utc)
    summary, baseline = w.build_watchdogs(_reports(now), now=now)

    assert summary["watchdogs"]["signal_close_freshness"]["status"] == "WATCH"
    assert summary["watchdogs"]["two_h_market_maturation"]["status"] == "WATCH"
    assert summary["watchdogs"]["corners_formation_join"]["status"] == "OK"
    assert summary["watchdogs"]["cards_settlement"]["status"] == "WATCH"
    assert summary["watchdogs"]["player_props_xi_confirmed"]["status"] == "WATCH"
    assert summary["watchdogs"]["evidence_growth_48h"]["status"] == "NOT_VERIFIED"
    assert summary["watchdogs"]["provider_requests_without_evidence_growth"]["status"] == "NOT_VERIFIED"
    assert summary["provider_requests_added"] == 0
    assert summary["production_promotion_allowed"] is False
    assert summary["decision_weights_changed"] is False
    assert baseline["provider_calls"] == 110


def test_watchdog_flags_requests_without_evidence_growth_after_48h():
    now = datetime(2026, 9, 29, 17, 0, tzinfo=timezone.utc)
    reports = _reports(now)
    counters = w.evidence_counters(reports)
    baseline = {
        "since_ts": (now - timedelta(hours=49)).timestamp(),
        "counters": counters,
        "provider_calls": 100,
    }

    summary, _ = w.build_watchdogs(reports, baseline=baseline, now=now)

    assert summary["watchdogs"]["evidence_growth_48h"]["status"] == "WATCH"
    provider = summary["watchdogs"]["provider_requests_without_evidence_growth"]
    assert provider["status"] == "WATCH"
    assert provider["evidence"]["provider_call_delta"] == 10


def test_evidence_growth_resets_stagnation_baseline():
    now = datetime(2026, 9, 29, 17, 0, tzinfo=timezone.utc)
    reports = _reports(now)
    current = w.evidence_counters(reports)
    previous = dict(current)
    previous["corners.formation_adjusted_evaluations"] = 38
    baseline = {
        "since_ts": (now - timedelta(hours=60)).timestamp(),
        "counters": previous,
        "provider_calls": 100,
    }

    summary, next_baseline = w.build_watchdogs(reports, baseline=baseline, now=now)

    assert summary["watchdogs"]["evidence_growth_48h"]["status"] == "OK"
    assert summary["watchdogs"]["provider_requests_without_evidence_growth"]["status"] == "OK"
    assert next_baseline["since_ts"] == now.timestamp()
    assert next_baseline["provider_calls"] == 110
