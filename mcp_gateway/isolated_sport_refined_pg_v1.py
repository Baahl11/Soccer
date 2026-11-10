"""Private, append-only research intake for prospectively captured refined SPORT ONLY RAW.

Distinct from the 45-minute season-split baseline table. One GitHub main-only
OIDC workflow submits a narrowly bounded, source-hashed research projection
while the match is still at least 5 minutes away. Server clock is authoritative.
No upstream raw team payload, commercial event, price, or BET is persisted.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import re
from typing import Any

from mcp_gateway import persistence

MAX_ROWS = 12
MAX_BODY_BYTES = 250_000
MAX_AGE = timedelta(minutes=20)
MIN_KICKOFF_LEAD = timedelta(minutes=5)
SOURCE = "LIVE_CANONICAL_V5_REFINED_SPORT_ONLY_OIDC"
STATUS = "CANONICAL_V5_SPORT_ONLY_RESEARCH_LIMITED"
MODEL = "GOAL_RATE_POISSON_LIMITED_v0.1"
PROBABILITIES = (
    "raw_home_win_prob", "raw_draw_prob", "raw_away_win_prob",
    "raw_btts_yes_prob", "raw_over_2_5_prob",
)
ALLOWED_STAGES = {"EARLY_RESEARCH", "T-90", "T-60", "T-40", "T-20", "T-10"}
_CREATE = """
CREATE TABLE IF NOT EXISTS soccer_isolated_sport_refined_research_v1 (
    fixture_id BIGINT NOT NULL CHECK (fixture_id > 0),
    source_analysis_finished_at TIMESTAMPTZ NOT NULL,
    source_analysis_started_at TIMESTAMPTZ NOT NULL,
    kickoff TIMESTAMPTZ NOT NULL,
    home_team_id BIGINT NOT NULL CHECK (home_team_id > 0),
    away_team_id BIGINT NOT NULL CHECK (away_team_id > 0),
    source TEXT NOT NULL CHECK (source = 'LIVE_CANONICAL_V5_REFINED_SPORT_ONLY_OIDC'),
    projection_model TEXT NOT NULL CHECK (projection_model = 'GOAL_RATE_POISSON_LIMITED_v0.1'),
    model_git_sha TEXT NOT NULL,
    sporting_inputs_sha256 TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    research_forecast JSONB NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    classification TEXT NOT NULL DEFAULT 'RESEARCH_ONLY'
        CHECK (classification = 'RESEARCH_ONLY'),
    bet_eligible BOOLEAN NOT NULL DEFAULT FALSE CHECK (bet_eligible = FALSE),
    production_promotion_allowed BOOLEAN NOT NULL DEFAULT FALSE
        CHECK (production_promotion_allowed = FALSE),
    PRIMARY KEY (fixture_id, source_analysis_finished_at),
    CHECK (source_analysis_started_at <= source_analysis_finished_at),
    CHECK (source_analysis_finished_at <= kickoff - INTERVAL '5 minutes'),
    CHECK (home_team_id <> away_team_id)
)
"""


def _time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    return parsed.astimezone(timezone.utc) if parsed.tzinfo else None


def _int_id(value: Any) -> bool:
    return type(value) is int and 0 < value < 10**12


def _p(value: Any) -> bool:
    return type(value) in (float, int) and math.isfinite(value) and 0 <= value <= 1


def _hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False, ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def validated_row(row: Any, *, now: datetime, runtime_sha: str) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError("RESEARCH_SNAPSHOT_NOT_OBJECT")
    if row.get("status") != STATUS or row.get("schema_version") != "1.0.0":
        raise ValueError("UNVERIFIED_REFINED_SPORT_SOURCE")
    if row.get("source_runtime_commit") != runtime_sha:
        raise ValueError("SOURCE_RUNTIME_SHA_MISMATCH")
    if any(row.get(k) is not False for k in (
        "full_deep_dive_verified", "availability_verified_for_bet",
        "market_used_to_generate_raw_probability", "market_price_verified",
        "canonical_clv_eligible", "bet_eligible", "production_promotion_allowed",
        "retroactive_forecast_created",
    )):
        raise ValueError("SPORT_ONLY_NON_COMMERCIAL_GATES_FAILED")
    if row.get("source_stage") not in ALLOWED_STAGES:
        raise ValueError("UNVERIFIED_SPORT_STAGE")
    if not all(_int_id(row.get(k)) for k in
               ("fixture_id", "league_id", "home_team_id", "away_team_id")):
        raise ValueError("INVALID_SPORT_IDENTITY")
    if row["home_team_id"] == row["away_team_id"]:
        raise ValueError("SAME_TEAM_IN_BOTH_SLOTS")
    started = _time(row.get("sport_analysis_started_at_utc"))
    finished = _time(row.get("sport_analysis_finished_at_utc"))
    kickoff = _time(row.get("kickoff_utc"))
    if not started or not finished or not kickoff:
        raise ValueError("SOURCE_TIME_NOT_VERIFIED")
    if not (now - MAX_AGE <= started <= finished <= now + timedelta(seconds=5)):
        raise ValueError("SOURCE_NOT_RECENT_AT_INGEST")
    if not (now < kickoff - MIN_KICKOFF_LEAD and
            finished <= kickoff - MIN_KICKOFF_LEAD):
        raise ValueError("NOT_PROSPECTIVE_BEFORE_KICKOFF")
    source_hash = row.get("sporting_inputs_sha256")
    if not isinstance(source_hash, str) or not re.fullmatch(r"[a-f0-9]{64}", source_hash):
        raise ValueError("MODEL_INPUT_FINGERPRINT_MISSING")
    if row.get("projection_model") != MODEL:
        raise ValueError("MODEL_SCOPE_NOT_WHITELISTED")
    if row.get("model_version") != "SOCCER EDGE ENGINE v1.0":
        raise ValueError("SPORT_MODEL_VERSION_MISMATCH")
    availability = row.get("availability")
    confidence = availability.get("availability_confidence") if isinstance(availability, dict) else None
    if not _p(confidence) or confidence < 0.5:
        raise ValueError("AVAILABILITY_CONFIDENCE_NOT_GROUNDED")
    raw = row.get("raw_sport_projection")
    if (not isinstance(raw, dict) or raw.get("status") != "MODELED_LIMITED"
            or raw.get("projection_model") != MODEL
            or any(not _p(raw.get(k)) for k in PROBABILITIES)):
        raise ValueError("RAW_SPORT_PROBABILITY_NOT_VERIFIED")
    if abs(sum(raw[k] for k in PROBABILITIES[:3]) - 1) > 0.002:
        raise ValueError("ONE_X_TWO_MASS_INVALID")
    if row.get("sporting_inputs") is not None:
        # Explicitly refuse source team payloads, even though the source runner
        # holds them transiently for local fingerprinting.
        raise ValueError("UPSTREAM_PROVIDER_INPUTS_NOT_ACCEPTED")
    if len(json.dumps(row, ensure_ascii=False, allow_nan=False).encode()) > 32_000:
        raise ValueError("UNBOUNDED_SPORT_RESEARCH_SNAPSHOT")
    minimal = {
        "fixture_id": row["fixture_id"], "league_id": row["league_id"],
        "home_team_id": row["home_team_id"], "away_team_id": row["away_team_id"],
        "kickoff_utc": kickoff.isoformat(),
        "sport_analysis_started_at_utc": started.isoformat(),
        "sport_analysis_finished_at_utc": finished.isoformat(),
        "source_runtime_commit": runtime_sha,
        "model_version": row["model_version"],
        "projection_model": MODEL,
        "raw_sport_projection": raw,
        "sporting_inputs_sha256": source_hash,
        "availability_confidence": confidence,
        "source_stage": row["source_stage"],
        "source": SOURCE, "classification": "RESEARCH_ONLY",
        "bet_eligible": False, "production_promotion_allowed": False,
        "oos_validated": False, "clv_validated": False,
    }
    return {
        "fixture_id": row["fixture_id"],
        "source_analysis_finished_at": finished,
        "source_analysis_started_at": started,
        "kickoff": kickoff,
        "home_team_id": row["home_team_id"],
        "away_team_id": row["away_team_id"],
        "source": SOURCE,
        "projection_model": MODEL,
        "model_git_sha": runtime_sha,
        "sporting_inputs_sha256": source_hash,
        "payload_sha256": _hash(minimal),
        "research_forecast": json.dumps(minimal, sort_keys=True, ensure_ascii=False),
    }


def validate_batch(payload: Any, *, now: datetime | None = None) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or set(payload) != {"snapshots", "transport_proof"}:
        raise ValueError("UNEXPECTED_REFINED_RESEARCH_PAYLOAD")
    if now is None:
        now = datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("SERVER_TIME_MUST_BE_AWARE")
    proof = payload["transport_proof"]
    if not isinstance(proof, dict):
        raise ValueError("TRANSPORT_PROOF_NOT_PRESENT")
    runtime_sha = proof.get("runtime_sha")
    calls = proof.get("actual_provider_calls")
    verified = proof.get("quota_headers_verified")
    if (not isinstance(runtime_sha, str) or not re.fullmatch(r"[a-f0-9]{40}", runtime_sha)
            or type(calls) is not int or not 1 <= calls <= 40
            or type(verified) is not int or verified < calls
            or type(proof.get("last_daily_remaining")) is not int
            or proof["last_daily_remaining"] < 1000
            or type(proof.get("transport_budget_denied")) is not int
            or proof["transport_budget_denied"] != 0
            or proof.get("markets_queried") != 0
            or proof.get("price_resolver_invocations") != 0
            or proof.get("db_writes") != 0
            or proof.get("clv_rows_added") != 0
            or proof.get("bet_eligible") is not False
            or proof.get("production_promotion_allowed") is not False
            or proof.get("full_deep_dives_verified") != 0):
        raise ValueError("PROSPECTIVE_QUOTA_OR_COMMERCIAL_GATE_FAILED")
    snapshots = payload["snapshots"]
    if not isinstance(snapshots, list) or not 1 <= len(snapshots) <= MAX_ROWS:
        raise ValueError("INVALID_REFINEMENT_BATCH")
    records = [validated_row(row, now=now, runtime_sha=runtime_sha) for row in snapshots]
    keys = [(r["fixture_id"], r["source_analysis_finished_at"]) for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError("DUPLICATE_RESEARCH_SOURCE")
    if len(json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()) > MAX_BODY_BYTES:
        raise ValueError("RESEARCH_BODY_TOO_LARGE")
    return records


def store_frozen(payload: Any) -> dict[str, Any]:
    # Validation executes *again* immediately before the database transaction,
    # using server current time to reject delayed ingestion after kickoff.
    rows = validate_batch(payload)
    if not persistence.persistence_configured():
        raise RuntimeError("RESEARCH_POSTGRES_NOT_CONFIGURED")
    inserted, existing = 0, 0
    with persistence._connect() as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute("SET LOCAL statement_timeout = '2500ms'")
                cur.execute("SET LOCAL lock_timeout = '1000ms'")
                cur.execute(_CREATE)
                for row in rows:
                    cur.execute("""
                    INSERT INTO soccer_isolated_sport_refined_research_v1
                    (fixture_id, source_analysis_finished_at, source_analysis_started_at,
                     kickoff, home_team_id, away_team_id, source, projection_model,
                     model_git_sha, sporting_inputs_sha256, payload_sha256, research_forecast)
                    VALUES
                    (%(fixture_id)s, %(source_analysis_finished_at)s,
                     %(source_analysis_started_at)s, %(kickoff)s, %(home_team_id)s,
                     %(away_team_id)s, %(source)s, %(projection_model)s, %(model_git_sha)s,
                     %(sporting_inputs_sha256)s, %(payload_sha256)s,
                     %(research_forecast)s::jsonb)
                    ON CONFLICT (fixture_id, source_analysis_finished_at) DO NOTHING
                    RETURNING payload_sha256
                    """, row)
                    if cur.fetchone():
                        inserted += 1
                    else:
                        cur.execute("""
                        SELECT payload_sha256 FROM soccer_isolated_sport_refined_research_v1
                        WHERE fixture_id = %s AND source_analysis_finished_at = %s
                        LIMIT 1
                        """, (row["fixture_id"], row["source_analysis_finished_at"]))
                        found = cur.fetchone()
                        if not found or found[0] != row["payload_sha256"]:
                            raise RuntimeError("REFINED_RESEARCH_IMMUTABILITY_CONFLICT")
                        existing += 1
    return {
        "status": "AUTHENTIC_REFINED_SPORT_RESEARCH_IMPORTED",
        "snapshots_validated": len(rows),
        "new_research_rows": inserted,
        "identical_existing_rows": existing,
        "database": "POSTGRES_ISOLATED_SPORT_REFINED_RESEARCH_ONLY",
        "source": SOURCE,
        "canonical_clv_rows_added": 0,
        "canonical_deep_dives_added": 0,
        "market_prices_added": 0,
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
        "bet_eligible": False,
    }
