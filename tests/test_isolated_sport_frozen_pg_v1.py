from copy import deepcopy
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
import pytest

from mcp_gateway import isolated_sport_frozen_pg_v1 as v


def snapshot():
    kickoff=datetime(2026,11,12,20,tzinfo=timezone.utc)
    captured=kickoff-timedelta(hours=2)
    return {
        "schema_version":"1.0.0",
        "status":"FROZEN_RESEARCH_PREMATCH_SPORT_PROJECTION",
        "source":"LIVE_API_FOOTBALL_OFFICIAL_PREGAME_SPORT_INPUTS",
        "fixture_id":42,
        "fixture":{"fixture_id":42,"league_id":55,"season":2026,
                   "home_team_id":11,"away_team_id":12,"source_status":"NS",
                   "kickoff":kickoff.isoformat()},
        "kickoff":kickoff.isoformat(),
        "captured_at_utc":captured.isoformat(),
        "generated_at_utc":captured.isoformat(),
        "canonical_runtime_model_sha":"a"*40,
        "model_file_sha256":"b"*64,
        "projection_model":"GOAL_RATE_POISSON_LIMITED_v0.1",
        "raw_sport_projection":{
            "status":"MODELED_LIMITED",
            "projection_model":"GOAL_RATE_POISSON_LIMITED_v0.1",
            "raw_home_win_prob":0.52,"raw_draw_prob":0.27,
            "raw_away_win_prob":0.21,"raw_btts_yes_prob":0.49,
            "raw_over_2_5_prob":0.54},
        "sporting_source":{"availability_confidence":0.50},
        "model_inputs_missing":["CONFIRMED_XI","INJURIES"],
        "raw_projection_before_market":True,
        "provider_observed_before_kickoff":True,
        "historical_predictions_recomputed":False,
        "provider_odds_requests_added":0,
        "oos_validated":False,"clv_validated":False,
        "bet_eligible":False,"production_promotion_allowed":False,
    }


def test_asof_research_model_valid_and_deterministic():
    first=v.validated_row(snapshot())
    second=v.validated_row(snapshot())
    assert first["payload_sha256"]==second["payload_sha256"]
    assert first["source_generated_at"] < first["kickoff"]
    assert first["model_git_sha"]=="a"*40
    assert first["fixture_id"]==42
    assert len(v.validate_batch({"snapshots":[snapshot()]}))==1
    assert "soccer_refresh_events" not in v._CREATE
    assert "soccer_primary_clv_signal_anchors" not in v._CREATE
    assert "source_generated_at <= kickoff - INTERVAL '45 minutes'" in v._CREATE


@pytest.mark.parametrize("change", [
    lambda x:x.update({"status":"FT"}),
    lambda x:x.update({"bet_eligible":True}),
    lambda x:x.update({"production_promotion_allowed":True}),
    lambda x:x.update({"provider_odds_requests_added":1}),
    lambda x:x.update({"oos_validated":True}),
    lambda x:x.update({"raw_projection_before_market":False}),
    lambda x:x.update({"canonical_runtime_model_sha":"NOT VERIFIED"}),
    lambda x:x.update({"generated_at_utc":x["kickoff"]}),
    lambda x:x.update({"captured_at_utc":x["kickoff"]}),
    lambda x:x["fixture"].update({"home_team_id":12}),
    lambda x:x["fixture"].update({"source_status":"FT"}),
    lambda x:x["raw_sport_projection"].update({"raw_home_win_prob":0.99}),
    lambda x:x["sporting_source"].update({"availability_confidence":1.0}),
])
def test_fail_closed_invalid_or_retroactive(change):
    row=snapshot()
    change(row)
    with pytest.raises(ValueError):
        v.validated_row(row)


def test_duplicate_and_wrong_batch_rejected():
    with pytest.raises(ValueError):
        v.validate_batch({"snapshots":[snapshot(),snapshot()]})
    with pytest.raises(ValueError):
        v.validate_batch({"snapshots":[]})
    with pytest.raises(ValueError):
        v.validate_batch({"snapshots":[snapshot()],"promote":True})


class FakeCursor:
    def __init__(self):
        self.queries=[]
        self.row=None
        self.digest=None

    def execute(self,sql,params=None):
        self.queries.append((sql,params))
        if "INSERT INTO soccer_isolated_sport_frozen_research_v1" in sql:
            self.row=(params["payload_sha256"],) if self.digest is None else None
            self.digest=params["payload_sha256"] if self.digest is None else self.digest
        elif "SELECT payload_sha256" in sql:
            self.row=(self.digest,)
        else:
            self.row=None

    def fetchone(self):
        return self.row

    def __enter__(self): return self
    def __exit__(self,*args): return False


class FakeConnection:
    def __init__(self):
        self.cur=FakeCursor()

    def cursor(self): return self.cur
    def transaction(self): return self
    def __enter__(self): return self
    def __exit__(self,*args): return False


def test_store_is_bounded_idempotent_without_any_actionable_tables():
    conn=FakeConnection()
    with patch.object(v.persistence,"persistence_configured",return_value=True), \
         patch.object(v.persistence,"_connect",return_value=conn):
        first=v.store_frozen({"snapshots":[snapshot()]})
        again=v.store_frozen({"snapshots":[snapshot()]})
    assert first["new_research_rows"]==1
    assert again["new_research_rows"]==0 and again["identical_existing_rows"]==1
    assert first["provider_requests_added"]==first["market_prices_added"]==0
    assert first["canonical_clv_rows_added"]==first["canonical_deep_dives_added"]==0
    assert first["production_promotion_allowed"] is False
    assert any("statement_timeout" in sql for sql,_ in conn.cur.queries)
    assert not any("INSERT INTO soccer_refresh_events" in sql for sql,_ in conn.cur.queries)
    assert not any("UPDATE soccer_" in sql for sql,_ in conn.cur.queries)


def test_immutable_conflict_rejects_modified_source():
    conn=FakeConnection()
    original=snapshot()
    changed=deepcopy(original)
    changed["raw_sport_projection"]["raw_btts_yes_prob"]=0.52
    with patch.object(v.persistence,"persistence_configured",return_value=True), \
         patch.object(v.persistence,"_connect",return_value=conn):
        v.store_frozen({"snapshots":[original]})
        with pytest.raises(RuntimeError,match="IMMUTABILITY_CONFLICT"):
            v.store_frozen({"snapshots":[changed]})


def test_owner_preview_bridge_requires_immutable_asof_sample_and_no_market():
    from mcp_gateway import soccer_p0_persisted_view_v1 as preview
    row = snapshot()
    row["raw_sport_projection"]["sample"] = {"minimum_split_sample": 5}
    now = datetime(2026, 11, 12, 18, 30, tzinfo=timezone.utc)
    got = preview.asof_isolated_preview(row, now=now)
    assert got is not None
    assert got["fixture_id"] == 42
    assert got["min_split_sample"] == 5
    assert got["raw_probabilities"]["raw_home_win_prob"] == 0.52
    assert got["model_validation"] == "NOT_VERIFIED_OOS"
    assert got["bet_eligible"] is False
    assert got["commercial_display_allowed"] is False
    assert got["market_queried"] is False

    weak = deepcopy(row)
    weak["raw_sport_projection"]["sample"]["minimum_split_sample"] = 4
    assert preview.asof_isolated_preview(weak, now=now) is None
    assert preview.asof_isolated_preview(row, now=datetime(2026, 11, 12, 21, tzinfo=timezone.utc)) is None
    assert preview.asof_isolated_preview(row, now=datetime(2026, 11, 12, 17, tzinfo=timezone.utc)) is None
    altered = deepcopy(row)
    altered["provider_odds_requests_added"] = 1
    assert preview.asof_isolated_preview(altered, now=now) is None


def test_owner_preview_bridge_reads_only_matching_fixtures_with_no_provider_calls(monkeypatch):
    from mcp_gateway import soccer_p0_persisted_view_v1 as preview
    row = snapshot()
    row["raw_sport_projection"]["sample"] = {"minimum_split_sample": 7}
    now = datetime(2026, 11, 12, 18, 30, tzinfo=timezone.utc)

    class Cursor:
        def __init__(self):
            self.queries = []
            self.result = []
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, sql, params=None):
            self.queries.append((sql, params))
            self.result = ([(42, row)] if "soccer_isolated_sport_frozen_research_v1" in sql else [])
        def fetchall(self): return self.result

    class Connection:
        def __init__(self): self.cur = Cursor()
        def cursor(self): return self.cur
        def __enter__(self): return self
        def __exit__(self, *a): return False

    conn = Connection()
    monkeypatch.setattr(preview.persistence, "_connect", lambda: conn)
    result = preview.read_previews([42, 42, -1, "42"], now=now)
    assert result["status"] == "PREMATCH_LIMITED_RESEARCH"
    assert result["forecast_count"] == 1
    assert result["provider_requests_added"] == 0
    assert result["bet_eligible"] is False
    assert result["source_redisplay_rights_verified"] is False
    assert len(conn.cur.queries) == 2
    sql, params = conn.cur.queries[-1]
    assert "f.home_team_id = s.home_team_id" in sql
    assert "f.away_team_id = s.away_team_id" in sql
    assert "f.kickoff = s.kickoff" in sql
    assert params[0] == [42]

    no_slates = preview.read_previews([], now=now)
    assert no_slates["status"] == "NO_SLATE_FIXTURES"
