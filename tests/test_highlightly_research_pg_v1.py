from __future__ import annotations

from datetime import datetime, timedelta, timezone
import pytest
from mcp_gateway import highlightly_research_pg_v1 as p

NOW=datetime(2026,11,11,5,5,tzinfo=timezone.utc)
KICKOFF=(NOW+timedelta(hours=3)).isoformat()
HISTORY=(NOW-timedelta(days=1)).isoformat()


def _row(**extra):
    row={
        "fixture_id":1581906,
        "kickoff_utc":KICKOFF,
        "captured_at_utc":(NOW-timedelta(minutes=2)).isoformat(),
        "identity_status":"IDENTITY_EXACT_VERIFIED",
        "evidence_status":"PREMATCH_RESEARCH_LIMITED",
        "families":["FORM"],
        "source_updated_at_verified":False,
        "is_model_input":False,
        "bet_eligible":False,
        "commercial_display_allowed":False,
    }
    row.update(extra)
    return row


def _envelope(rows):
    return {
        "schema":p.SCHEMA, "source":p.SOURCE,
        "report_generated_at_utc":NOW.isoformat(),
        "source_rights":"NOT_VERIFIED",
        "model_input_promotion_allowed":False,
        "bet_eligible":False,"odds_queried":False,
        "canonical_database_writes":0,
        "requests_executed":6,"quota_remaining":88,
        "rows":rows,
    }


def test_valid_prematch_research_only_has_no_model_or_bet_promotion():
    validated=p.validate_batch(_envelope([_row()]),now=NOW)
    assert len(validated)==1
    assert validated[0]["fixture_id"]==1581906
    assert validated[0]["evidence_status"]=="PREMATCH_RESEARCH_LIMITED"
    assert validated[0]["families"]=='["FORM"]'
    assert len(validated[0]["fingerprint"])==64


@pytest.mark.parametrize("mutate",[
    {"evidence_status":"PREMATCH_RESEARCH_LIMITED","kickoff_utc":HISTORY},
    {"source_updated_at_verified":True},
    {"is_model_input":True},
    {"commercial_display_allowed":True},
    {"bet_eligible":True},
    {"families":["FORM","FORM"]},
    {"families":["FAKE_XG"]},
    {"identity_status":"LEAGUE_NOT_VERIFIED"},
    {"captured_at_utc":(NOW-timedelta(hours=2)).isoformat()},
])
def test_prematch_research_rejects_ineligible_identity_timing_and_flags(mutate):
    with pytest.raises(ValueError):
        p.validate_batch(_envelope([_row(**mutate)]),now=NOW)


def test_historical_identity_can_be_recorded_but_not_as_pregame_stats():
    v=p.validate_batch(_envelope([_row(
        kickoff_utc=HISTORY,
        evidence_status="HISTORY_NOT_PREGAME",
        families=[],
    )]),now=NOW)
    assert v[0]["evidence_status"]=="HISTORY_NOT_PREGAME"
    assert v[0]["families"]=="[]"
    with pytest.raises(ValueError):
        p.validate_batch(_envelope([_row(
            kickoff_utc=HISTORY,
            evidence_status="HISTORY_NOT_PREGAME",
            families=["FORM"],
        )]),now=NOW)


def test_no_match_status_must_match_identity():
    v=p.validate_batch(_envelope([_row(
        identity_status="NO_MATCH_ON_FETCHED_PAGES",
        evidence_status="NO_MATCH",families=[],
    )]),now=NOW)
    assert v[0]["evidence_status"]=="NO_MATCH"
    with pytest.raises(ValueError):
        p.validate_batch(_envelope([_row(
            identity_status="IDENTITY_EXACT_VERIFIED",
            evidence_status="NO_MATCH",families=[],
        )]),now=NOW)


def test_batch_rejects_missing_source_or_duplicate_or_low_quota():
    with pytest.raises(ValueError):
        p.validate_batch(_envelope([_row(),_row()]),now=NOW)
    bad=_envelope([_row()]);bad["quota_remaining"]=65
    with pytest.raises(ValueError):
        p.validate_batch(bad,now=NOW)
    bad=_envelope([_row()]);bad["odds_queried"]=True
    with pytest.raises(ValueError):
        p.validate_batch(bad,now=NOW)


def test_read_api_does_not_contact_db_with_no_canonical_fixture_ids(monkeypatch):
    def never():
        raise AssertionError("No DB call expected")
    monkeypatch.setattr(p.persistence,"_connect",never)
    assert p.read_for_fixture_ids([])=={}
    assert p.read_for_fixture_ids([-12,0,"123"])=={}
