from __future__ import annotations

from mcp_gateway.build_postgame_backfill_queue import build_queue


def test_build_queue_dedupes_pending_rows_by_fixture_and_prioritizes_bet():
    backlog = [
        {
            "fixture_id": 11,
            "coverage_reason": "PENDING_FINAL_RESULT",
            "classification": "LEAN",
            "stage": "T-20",
            "generated_at_local": "2026-09-13T10:00:00-06:00",
            "kickoff_local": "2026-09-13T10:30:00-06:00",
            "league": "League A",
            "home_team": "Home A",
            "away_team": "Away A",
            "market_family": "FT_TOTALS",
            "market": "Goals Over/Under",
            "selection": "under",
            "line": 2.5,
            "decimal_price": 1.9,
        },
        {
            "fixture_id": 11,
            "coverage_reason": "PENDING_FINAL_RESULT",
            "classification": "BET",
            "stage": "T-10",
            "generated_at_local": "2026-09-13T10:15:00-06:00",
            "kickoff_local": "2026-09-13T10:30:00-06:00",
            "league": "League A",
            "home_team": "Home A",
            "away_team": "Away A",
            "market_family": "2H_TOTALS",
            "market": "Goals Over/Under - Second Half",
            "selection": "over",
            "line": 1.5,
            "decimal_price": 2.4,
        },
        {
            "fixture_id": 22,
            "coverage_reason": "DUPLICATE_OLDER_SNAPSHOT",
            "classification": "BET",
            "market_family": "FT_TOTALS",
        },
    ]

    queue, summary = build_queue(backlog)

    assert summary["pending_backlog_rows"] == 2
    assert summary["unique_fixtures"] == 1
    assert summary["max_provider_calls_needed"] == 1
    assert summary["by_classification"] == {"BET": 1, "LEAN": 1}
    assert summary["by_market_family"] == {"2H_TOTALS": 1, "FT_TOTALS": 1}
    assert len(queue) == 1
    assert queue[0]["fixture_id"] == 11
    assert queue[0]["classifications"] == ["BET", "LEAN"]
    assert queue[0]["market_families"] == ["2H_TOTALS", "FT_TOTALS"]
    assert queue[0]["provider_hint"]["endpoint"] == "fixtures"
    assert queue[0]["provider_hint"]["params"] == {"id": 11}
    assert queue[0]["source_backlog_rows"] == 2
    assert queue[0]["priority_score"] >= 100


def test_build_queue_ignores_non_pending_rows():
    queue, summary = build_queue([
        {"fixture_id": 1, "coverage_reason": "IN_SETTLEMENT_LEDGER", "classification": "BET"},
        {"fixture_id": 2, "coverage_reason": "DUPLICATE_OLDER_SNAPSHOT", "classification": "LEAN"},
    ])

    assert queue == []
    assert summary["pending_backlog_rows"] == 0
    assert summary["unique_fixtures"] == 0
    assert summary["max_provider_calls_needed"] == 0
