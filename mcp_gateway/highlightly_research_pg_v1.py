"""Private Highlightly sport-only source coverage bridge for OWNER research Radar.

Only source presence & quality flags are stored. No vendor response JSON,
numbers, identifiers for player data, bookmaker prices, or recommendations.
Rows are scoped to a trusted main-workflow OIDC capture, never model input.
"""
from __future__ import annotations
import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any
from mcp_gateway import persistence

SOURCE = "HIGHLIGHTLY_BASIC_RESEARCH_ONLY"
SCHEMA = "SOCCER_HIGHLIGHTLY_PRIVATE_RESEARCH_v1"
MAX_ROWS = 16
MAX_BODY_BYTES = 28_000
MAX_AGE = timedelta(minutes=25)
ALLOWED_IDENTITIES = {
    "IDENTITY_EXACT_VERIFIED", "LEAGUE_NOT_VERIFIED",
    "NO_MATCH_ON_FETCHED_PAGES", "SEASON_NOT_VERIFIED",
    "COUNTRY_NOT_VERIFIED", "KICKOFF_NOT_VERIFIED",
    "TEAM_ID_INVALID", "HIGHLIGHTLY_ID_INVALID",
    "UNVERIFIED_SHAPE", "AMBIGUOUS_PROVIDER_FIXTURE",
}
ALLOWED_EVIDENCE = {
    "PREMATCH_RESEARCH_LIMITED", "HISTORY_NOT_PREGAME",
    "IDENTITY_EXACT_NO_FEATURES", "IDENTITY_PENDING", "NO_MATCH",
}
FAMILIES = {"GOALS", "FORM", "CORNERS", "CARDS", "SHOTS", "SOT", "POSSESSION", "PASSES", "XG"}
CREATE = """
CREATE TABLE IF NOT EXISTS soccer_highlightly_research_coverage_v1 (
  fixture_id BIGINT NOT NULL CHECK (fixture_id > 0),
  source_captured_at TIMESTAMPTZ NOT NULL,
  kickoff TIMESTAMPTZ NOT NULL,
  identity_status TEXT NOT NULL,
  evidence_status TEXT NOT NULL,
  families JSONB NOT NULL,
  fingerprint CHAR(64) NOT NULL,
  source TEXT NOT NULL DEFAULT 'HIGHLIGHTLY_BASIC_RESEARCH_ONLY'
     CHECK (source='HIGHLIGHTLY_BASIC_RESEARCH_ONLY'),
  source_updated_at_verified BOOLEAN NOT NULL DEFAULT FALSE
     CHECK (source_updated_at_verified=FALSE),
  is_model_input BOOLEAN NOT NULL DEFAULT FALSE CHECK (is_model_input=FALSE),
  bet_eligible BOOLEAN NOT NULL DEFAULT FALSE CHECK (bet_eligible=FALSE),
  commercial_display_allowed BOOLEAN NOT NULL DEFAULT FALSE
     CHECK (commercial_display_allowed=FALSE),
  imported_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY(fixture_id, source_captured_at),
  CHECK (evidence_status <> 'PREMATCH_RESEARCH_LIMITED'
         OR source_captured_at < kickoff - INTERVAL '5 minutes')
)
"""


def _dt(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        d = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return d.astimezone(timezone.utc) if d.tzinfo else None


def validate_batch(payload: Any, *, now: datetime | None = None) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or set(payload) != {
        "schema", "source", "report_generated_at_utc", "source_rights",
        "model_input_promotion_allowed", "bet_eligible", "odds_queried",
        "canonical_database_writes", "requests_executed", "quota_remaining", "rows",
    }:
        raise ValueError("INVALID_HIGHLIGHTLY_RESEARCH_ENVELOPE")
    if payload.get("schema") != SCHEMA or payload.get("source") != SOURCE:
        raise ValueError("UNVERIFIED_HIGHLIGHTLY_RESEARCH_SOURCE")
    if (payload.get("source_rights") != "NOT_VERIFIED"
            or payload.get("model_input_promotion_allowed") is not False
            or payload.get("bet_eligible") is not False
            or payload.get("odds_queried") is not False
            or payload.get("canonical_database_writes") != 0):
        raise ValueError("RESEARCH_COMMERCIAL_FIREWALL_FAILED")
    calls = payload.get("requests_executed")
    remaining = payload.get("quota_remaining")
    if type(calls) is not int or not 1 <= calls <= 10:
        raise ValueError("HIGHLIGHTLY_BUDGET_NOT_VERIFIED")
    if type(remaining) is not int or remaining < 70 or remaining > 100:
        raise ValueError("HIGHLIGHTLY_DAILY_QUOTA_FLOOR_FAILED")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_ROWS:
        raise ValueError("HIGHLIGHTLY_ROW_COUNT_INVALID")
    if now is None:
        now = datetime.now(timezone.utc)
    created = _dt(payload.get("report_generated_at_utc"))
    if not created or not now-timedelta(minutes=25) <= created <= now+timedelta(seconds=5):
        raise ValueError("HIGHLIGHTLY_REPORT_NOT_RECENT")
    if len(json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()) > MAX_BODY_BYTES:
        raise ValueError("HIGHLIGHTLY_RESEARCH_PAYLOAD_OVERSIZE")
    normalized = []
    seen=set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "fixture_id", "kickoff_utc", "captured_at_utc",
            "identity_status", "evidence_status", "families",
            "source_updated_at_verified", "is_model_input",
            "bet_eligible", "commercial_display_allowed",
        }:
            raise ValueError("HIGHLIGHTLY_ROW_SCHEMA_INVALID")
        fid = row["fixture_id"]
        if type(fid) is not int or not 0 < fid < 10**12 or fid in seen:
            raise ValueError("HIGHLIGHTLY_FIXTURE_ID_INVALID_OR_DUPLICATE")
        seen.add(fid)
        at, kickoff = _dt(row["captured_at_utc"]), _dt(row["kickoff_utc"])
        if not at or not kickoff or not now-MAX_AGE <= at <= now+timedelta(seconds=5):
            raise ValueError("HIGHLIGHTLY_SOURCE_TIME_INVALID")
        ident, evidence = row["identity_status"], row["evidence_status"]
        if ident not in ALLOWED_IDENTITIES or evidence not in ALLOWED_EVIDENCE:
            raise ValueError("HIGHLIGHTLY_STATUS_NOT_WHITELISTED")
        fam = row["families"]
        if (not isinstance(fam, list) or any(v not in FAMILIES or type(v) is not str for v in fam)
                or len(fam) != len(set(fam)) or len(fam)>len(FAMILIES)):
            raise ValueError("UNVERIFIED_HIGHLIGHTLY_FAMILIES")
        if any(row.get(k) is not False for k in (
            "source_updated_at_verified", "is_model_input",
            "bet_eligible", "commercial_display_allowed"
        )):
            raise ValueError("HIGHLIGHTLY_PRICE_OR_MODEL_PROMOTION_FAILED")
        if evidence == "PREMATCH_RESEARCH_LIMITED":
            if ident != "IDENTITY_EXACT_VERIFIED" or not fam or at >= kickoff-timedelta(minutes=5) or now >= kickoff-timedelta(minutes=5):
                raise ValueError("PREMATCH_EVIDENCE_LOOKAHEAD_OR_IDENTITY_FAILED")
        elif fam:
            raise ValueError("NON_PROSPECTIVE_HIGHLIGHTLY_FAMILIES_FORBIDDEN")
        if evidence == "HISTORY_NOT_PREGAME" and at < kickoff:
            raise ValueError("HIGHLIGHTLY_HISTORICAL_STATUS_BEFORE_KICKOFF")
        if evidence == "IDENTITY_EXACT_NO_FEATURES" and (ident != "IDENTITY_EXACT_VERIFIED" or at >= kickoff):
            raise ValueError("HIGHLIGHTLY_NO_FEATURES_STATUS_INVALID")
        if evidence == "NO_MATCH" and ident != "NO_MATCH_ON_FETCHED_PAGES":
            raise ValueError("NO_MATCH_IDENTITY_MISMATCH")
        if evidence == "IDENTITY_PENDING" and ident in ("NO_MATCH_ON_FETCHED_PAGES", "IDENTITY_EXACT_VERIFIED"):
            raise ValueError("IDENTITY_PENDING_MISMATCH")
        encoded = json.dumps(row,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
        normalized.append({
            "fixture_id":fid,"source_captured_at":at,"kickoff":kickoff,
            "identity_status":ident,"evidence_status":evidence,
            "families":json.dumps(sorted(fam)),
            "fingerprint":hashlib.sha256(encoded).hexdigest(),
        })
    return normalized


def store_batch(payload: Any) -> dict[str, Any]:
    rows=validate_batch(payload)
    inserted=0
    with persistence._connect() as conn:
        with conn.cursor() as cur:
            cur.execute(CREATE)
            for row in rows:
                cur.execute("""
                    SELECT fingerprint FROM soccer_highlightly_research_coverage_v1
                    WHERE fixture_id=%s AND source_captured_at=%s
                """,(row["fixture_id"],row["source_captured_at"]))
                found=cur.fetchone()
                if found is not None:
                    if str(found[0]) != row["fingerprint"]:
                        raise RuntimeError("HIGHLIGHTLY_SOURCE_IMMUTABILITY_CONFLICT")
                    continue
                cur.execute("""
                    INSERT INTO soccer_highlightly_research_coverage_v1
                    (fixture_id,source_captured_at,kickoff,identity_status,
                     evidence_status,families,fingerprint)
                    VALUES (%s,%s,%s,%s,%s,%s::jsonb,%s)
                """,(
                    row["fixture_id"],row["source_captured_at"],
                    row["kickoff"],row["identity_status"],
                    row["evidence_status"],row["families"],row["fingerprint"],
                ))
                inserted+=1
        conn.commit()
    return {"status":"HIGHLIGHTLY_RESEARCH_COVERAGE_PERSISTED",
            "records_received":len(rows),"rows_inserted":inserted,
            "bet_eligible":False,"canonical_model_changed":False,
            "commercial_display_allowed":False}


def read_for_fixture_ids(fixture_ids: list[int]) -> dict[str, dict[str, Any]]:
    ids=sorted({x for x in fixture_ids if type(x) is int and 0 < x < 10**12})[:250]
    if not ids:return {}
    # Most recent record for each actual eligible fixture. Avoid stale data.
    with persistence._connect() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT ON (fixture_id)
                    fixture_id, source_captured_at, kickoff, identity_status,
                    evidence_status, families
                FROM soccer_highlightly_research_coverage_v1
                WHERE fixture_id=ANY(%s)
                  AND source_captured_at >= NOW() - INTERVAL '48 hours'
                ORDER BY fixture_id, source_captured_at DESC
            """,(ids,))
            rows=cur.fetchall()
    result={}
    for fid,at,kickoff,ident,evidence,families in rows:
        result[str(fid)]={
            "source":"HIGHLIGHTLY_BASIC_RESEARCH_ONLY",
            "identity_status":ident,
            "evidence_status":evidence,
            "families_research":families if isinstance(families,list) else [],
            "captured_at":at.isoformat(),
            "kickoff_at":kickoff.isoformat(),
            "source_updated_at_verified":False,
            "model_input_allowed":False,
            "bet_eligible":False,
            "commercial_display_allowed":False,
        }
    return result
