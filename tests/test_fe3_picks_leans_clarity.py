from __future__ import annotations

from mcp_gateway import subscriber_contract_v2
from mcp_gateway import subscriber_frontend_v2


def _row(**extra):
    row = {
        "fixture_id": 7001,
        "kickoff": "2026-10-07T22:00:00Z",
        "league": "Test League",
        "home_team_id": 1,
        "home_team": "Home",
        "away_team_id": 2,
        "away_team": "Away",
        "market_family": "BTTS",
        "market": "Both Teams Score",
        "selection": "Yes",
    }
    row.update(extra)
    return row


def test_fe3_research_codes_are_presented_as_customer_language_without_changing_classification():
    candidate = subscriber_contract_v2.adapt_candidate(
        _row(
            classification="WATCH",
            execution_status="RESEARCH_ONLY",
            reason="BTTS_PAID_ODDS_RESEARCH_ENTRY",
        )
    )
    decision = candidate["decision"]

    assert decision["classification"] == "WATCH"
    assert decision["display_bucket"] == "WATCH"
    assert decision["reason_code"] == "BTTS_PAID_ODDS_RESEARCH_ENTRY"
    assert decision["reason_display"] == (
        "Research-only market evaluation; production BET gate not passed."
    )
    assert decision["is_canonical_bet"] is False
    assert decision["is_canonical_lean"] is False


def test_fe3_api_budget_reason_hides_internal_call_path_from_customer_copy():
    reason = (
        "Research screen · Per-tick API budget reached (25); lower-priority work "
        "deferred; endpoint=teams/statistics; call_path=automation_v5.py:_priority_event"
    )
    candidate = subscriber_contract_v2.adapt_candidate(
        _row(
            classification="WATCH",
            execution_status="WAIT_MARKET",
            reason=reason,
        )
    )

    assert candidate["decision"]["reason_code"] == reason
    assert candidate["decision"]["reason_display"] == (
        "Data refresh deferred by the request budget; required inputs are "
        "not fully verified yet."
    )
    assert "automation_v5.py" not in candidate["decision"]["reason_display"]


def test_fe3_wait_state_is_user_facing_watch_without_becoming_canonical_bet():
    candidate = subscriber_contract_v2.adapt_candidate(
        _row(
            execution_status="WAIT_XI",
            reason="WAIT_XI",
        )
    )

    decision = candidate["decision"]
    assert decision["classification"] is None
    assert decision["display_bucket"] == "WATCH"
    assert decision["reason_display"] == "Waiting for confirmed starting XI."
    assert decision["is_canonical_bet"] is False


def test_fe3_public_watch_redacts_reason_code_but_keeps_safe_reason():
    candidate = subscriber_contract_v2.adapt_candidate(
        _row(
            classification="WATCH",
            execution_status="RESEARCH_ONLY",
            reason="SPORTING_SCREEN_PASS",
        )
    )
    public = subscriber_contract_v2._public_watch(candidate)

    assert "reason_code" not in public["decision"]
    assert public["decision"]["classification"] == "WATCH"
    assert public["decision"]["reason_display"].startswith("Sporting screen passed")


def test_fe3_public_slate_uses_customer_display_status_but_preserves_state_code():
    public = subscriber_contract_v2._public_fixture(
        _row(
            execution_status="WAIT_MARKET",
            reason="Waiting for price",
        )
    )

    assert public["state"]["display_status"] == "WATCH"
    assert public["state"]["status_code"] == "WAIT_MARKET"
    assert public["state"]["classification"] is None


def test_fe3_frontend_does_not_paywall_zero_bets_or_zero_leans():
    html = subscriber_frontend_v2.render()

    assert "(picks.locked&&picks.total>0)" in html
    assert "(leans.locked&&leans.total>0)" in html
    assert "No BETS" in html
    assert "Zero bets is a valid outcome." in html


def test_fe3_frontend_prefers_customer_safe_watch_copy_and_bucket():
    html = subscriber_frontend_v2.render()

    assert "d.reason_display" in html
    assert "d.classification||d.display_bucket||'WATCH'" in html
    assert "s.display_status||'NOT VERIFIED'" in html


def test_fe3_frontend_enriches_presentation_identity_without_provider_calls():
    html = subscriber_frontend_v2.render()
    contract = subscriber_frontend_v2.contract()

    assert "/app/fixture-identities?ids=" in html
    assert "presentation enrichment is optional" in html
    assert "api-sports.io" not in html
    assert contract["provider_requests_added"] == 0
    assert contract["canonical_bet_logic_changed"] is False


def test_fe3_frontend_uses_inline_favicon_instead_of_external_favicon_request():
    html = subscriber_frontend_v2.render()

    assert '<link rel="icon" href="data:image/svg+xml,' in html
