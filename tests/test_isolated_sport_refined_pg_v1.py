from copy import deepcopy
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from mcp_gateway import isolated_sport_refined_pg_v1 as v


NOW = datetime(2026, 11, 10, 19, 0, tzinfo=timezone.utc)
RUNTIME = "a" * 40


def sample_row():
    return {
        "schema_version": "1.0.0",
        "status": v.STATUS,
        "fixture_id": 42,
        "league_id": 77,
        "home_team_id": 12,
        "away_team_id": 13,
        "kickoff_utc": (NOW + timedelta(hours=1)).isoformat(),
        "sport_analysis_started_at_utc": (NOW - timedelta(minutes=3)).isoformat(),
        "sport_analysis_finished_at_utc": (NOW - timedelta(minutes=2)).isoformat(),
        "source_runtime_commit": RUNTIME,
        "model_version": "SOCCER EDGE ENGINE v1.0",
        "projection_model": v.MODEL,
        "sporting_inputs_sha256": "b" * 64,
        "source_stage": "T-60",
        "availability": {"availability_confidence": 0.6,
                         "lineup_state": "NOT VERIFIED"},
        "raw_sport_projection": {
            "status": "MODELED_LIMITED",
            "projection_model": v.MODEL,
            "raw_home_win_prob": 0.45, "raw_draw_prob": 0.25,
            "raw_away_win_prob": 0.30,
            "raw_btts_yes_prob": 0.57,
            "raw_over_2_5_prob": 0.55,
        },
        "full_deep_dive_verified": False,
        "availability_verified_for_bet": False,
        "market_used_to_generate_raw_probability": False,
        "market_price_verified": False,
        "canonical_clv_eligible": False,
        "bet_eligible": False,
        "production_promotion_allowed": False,
        "retroactive_forecast_created": False,
    }


def sample_batch():
    return {
        "transport_proof": {
            "runtime_sha": RUNTIME,
            "actual_provider_calls": 22,
            "quota_headers_verified": 22,
            "last_daily_remaining": 7180,
            "transport_budget_denied": 0,
            "markets_queried": 0,
            "price_resolver_invocations": 0,
            "clv_rows_added": 0,
            "db_writes": 0,
            "bet_eligible": False,
            "production_promotion_allowed": False,
            "full_deep_dives_verified": 0,
            "provider_calls_by_endpoint": {
                "/fixtures": 9, "/fixtures/lineups": 4,
                "/leagues": 1, "/teams/statistics": 8,
            },
        },
        "snapshots": [sample_row()],
    }


def test_valid_batch_and_source_free_persistence_payload():
    result = v.validate_batch(sample_batch(), now=NOW)
    assert len(result) == 1
    assert result[0]["fixture_id"] == 42
    assert result[0]["source_analysis_finished_at"] < result[0]["kickoff"]
    assert result[0]["model_git_sha"] == RUNTIME
    assert result[0]["payload_sha256"] == v.validate_batch(sample_batch(), now=NOW)[0]["payload_sha256"]
    assert result[0]["sporting_inputs_sha256"] == "b" * 64
    assert "sporting_inputs" not in result[0]["research_forecast"]
    assert "soccer_refresh_events" not in v._CREATE
    assert "soccer_primary_clv" not in v._CREATE
    assert "5 minutes" in v._CREATE
    assert "ON CONFLICT" not in v._CREATE


@pytest.mark.parametrize("mutate", [
    lambda x: x["snapshots"][0].update({"status": "FT"}),
    lambda x: x["snapshots"][0].update({"bet_eligible": True}),
    lambda x: x["snapshots"][0].update({"production_promotion_allowed": True}),
    lambda x: x["snapshots"][0].update({"availability_verified_for_bet": True}),
    lambda x: x["snapshots"][0].update({"market_used_to_generate_raw_probability": True}),
    lambda x: x["snapshots"][0].update({"source_runtime_commit": "c"*40}),
    lambda x: x["snapshots"][0].update({"source_stage": "POSTGAME"}),
    lambda x: x["snapshots"][0].update({"sporting_inputs": {"team_stats": {}}}),
    lambda x: x["snapshots"][0].update({"sporting_inputs_sha256": "BAD"}),
    lambda x: x["snapshots"][0].update({"kickoff_utc": (NOW + timedelta(minutes=3)).isoformat()}),
    lambda x: x["snapshots"][0].update({"sport_analysis_started_at_utc": (NOW - timedelta(hours=2)).isoformat()}),
    lambda x: x["snapshots"][0].update({"sport_analysis_finished_at_utc": (NOW + timedelta(minutes=3)).isoformat()}),
    lambda x: x["snapshots"][0]["raw_sport_projection"].update({"raw_home_win_prob": 0.99}),
    lambda x: x["snapshots"][0]["raw_sport_projection"].update({"raw_btts_yes_prob": float("nan")}),
    lambda x: x["transport_proof"].update({"actual_provider_calls": 41}),
    lambda x: x["transport_proof"].update({"transport_budget_denied": 1}),
    lambda x: x["transport_proof"].update({"markets_queried": 1}),
    lambda x: x["transport_proof"].update({"clv_rows_added": 1}),
    lambda x: x["transport_proof"].update({"provider_calls_by_endpoint": {"/odds": 22}}),
    lambda x: x["transport_proof"].update({"quota_headers_verified": 0}),
    lambda x: x["transport_proof"].update({"last_daily_remaining": 999}),
])
def test_rejected_no_markets_lookahead_or_quota_bypass(mutate):
    payload = sample_batch()
    mutate(payload)
    with pytest.raises((ValueError, TypeError)):
        v.validate_batch(payload, now=NOW)


def test_duplicate_row_rejected():
    payload = sample_batch()
    payload["snapshots"].append(deepcopy(payload["snapshots"][0]))
    with pytest.raises(ValueError, match="DUPLICATE"):
        v.validate_batch(payload, now=NOW)


class FakeCursor:
    def __init__(self):
        self.queries = []
        self.current = None
        self.digest = None

    def execute(self, sql, params=None):
        self.queries.append((sql, params))
        if "INSERT INTO soccer_isolated_sport_refined_research_v1" in sql:
            self.current = ((params["payload_sha256"],)
                            if self.digest is None else None)
            if self.digest is None:
                self.digest = params["payload_sha256"]
        elif "SELECT payload_sha256" in sql:
            self.current = (self.digest,)
        else:
            self.current = None

    def fetchone(self):
        return self.current

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class FakeConnection:
    def __init__(self):
        self.cur = FakeCursor()

    def cursor(self):
        return self.cur

    def transaction(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def test_immutable_idempotent_and_research_only():
    conn = FakeConnection()
    with patch.object(v.persistence, "persistence_configured", return_value=True), \
         patch.object(v.persistence, "_connect", return_value=conn), \
         patch.object(v, "validate_batch",
                      side_effect=lambda x: v.validate_batch(x, now=NOW)):
        # First insert and identical repeat do not touch canonical tables.
        assert v.store_frozen(sample_batch())["new_research_rows"] == 1
        again = v.store_frozen(sample_batch())
    assert again["new_research_rows"] == 0
    assert again["identical_existing_rows"] == 1
    assert again["bet_eligible"] is False
    assert again["production_promotion_allowed"] is False
    assert again["market_prices_added"] == 0
    assert again["canonical_clv_rows_added"] == 0
    sql = " ".join(statement for statement, _ in conn.cur.queries)
    assert "INSERT INTO soccer_refresh_events" not in sql
    assert "INSERT INTO soccer_primary_clv" not in sql


def test_immutable_conflict_rejected():
    conn = FakeConnection()
    p = sample_batch()
    mutated = deepcopy(p)
    mutated["snapshots"][0]["raw_sport_projection"]["raw_btts_yes_prob"] = 0.58
    with patch.object(v.persistence, "persistence_configured", return_value=True), \
         patch.object(v.persistence, "_connect", return_value=conn), \
         patch.object(v, "validate_batch",
                      side_effect=lambda x: v.validate_batch(x, now=NOW)):
        v.store_frozen(p)
        with pytest.raises(RuntimeError, match="IMMUTABILITY_CONFLICT"):
            v.store_frozen(mutated)
