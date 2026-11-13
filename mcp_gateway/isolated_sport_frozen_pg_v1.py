"""Immutable OIDC-only intake for authentic isolated SPORT FIRST research models.

Stored in a separate Postgres namespace from canonical actionable signal events:
a research Poisson baseline is never a Tier A/B deep dive, CLV anchor or BET.
No provider requests, no fixture/result edits, no model recalculation.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import re
from typing import Any
from mcp_gateway import persistence

MAX_ROWS = 16
MAX_BODY_BYTES = 250_000
SOURCE = "LIVE_API_FOOTBALL_OFFICIAL_PREGAME_SPORT_INPUTS"
STATUS = "FROZEN_RESEARCH_PREMATCH_SPORT_PROJECTION"
MODEL = "GOAL_RATE_POISSON_LIMITED_v0.1"
PROBABILITIES = (
    "raw_home_win_prob", "raw_draw_prob", "raw_away_win_prob",
    "raw_btts_yes_prob", "raw_over_2_5_prob",
)
_CREATE = """
CREATE TABLE IF NOT EXISTS soccer_isolated_sport_frozen_research_v1 (
    fixture_id BIGINT NOT NULL CHECK (fixture_id > 0),
    source_generated_at TIMESTAMPTZ NOT NULL,
    kickoff TIMESTAMPTZ NOT NULL,
    home_team_id BIGINT NOT NULL CHECK (home_team_id > 0),
    away_team_id BIGINT NOT NULL CHECK (away_team_id > 0),
    source TEXT NOT NULL CHECK (source = 'LIVE_API_FOOTBALL_OFFICIAL_PREGAME_SPORT_INPUTS'),
    projection_model TEXT NOT NULL CHECK (projection_model = 'GOAL_RATE_POISSON_LIMITED_v0.1'),
    model_git_sha TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    raw_projection JSONB NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    classification TEXT NOT NULL DEFAULT 'RESEARCH_ONLY' CHECK (classification = 'RESEARCH_ONLY'),
    bet_eligible BOOLEAN NOT NULL DEFAULT FALSE CHECK (bet_eligible = FALSE),
    production_promotion_allowed BOOLEAN NOT NULL DEFAULT FALSE
        CHECK (production_promotion_allowed = FALSE),
    PRIMARY KEY (fixture_id, source_generated_at),
    CHECK (source_generated_at <= kickoff - INTERVAL '45 minutes'),
    CHECK (home_team_id <> away_team_id)
)
"""


def _time(x: Any) -> datetime | None:
    if not isinstance(x, str):
        return None
    try:
        d = datetime.fromisoformat(x.replace("Z", "+00:00"))
    except ValueError:
        return None
    return d.astimezone(timezone.utc) if d.tzinfo else None


def _positive_int(x: Any) -> bool:
    return type(x) is int and 0 < x < 10**12


def _prob(x: Any) -> bool:
    return type(x) in (float, int) and math.isfinite(x) and 0 <= x <= 1


def validated_row(row: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(row, dict) or row.get("status") != STATUS or row.get("source") != SOURCE:
        raise ValueError("UNVERIFIED_OR_RECONSTRUCTED_SPORT_SOURCE")
    if (row.get("schema_version") != "1.0.0"
            or row.get("raw_projection_before_market") is not True
            or row.get("provider_observed_before_kickoff") is not True
            or row.get("historical_predictions_recomputed") is not False
            or row.get("provider_odds_requests_added") != 0
            or row.get("bet_eligible") is not False
            or row.get("production_promotion_allowed") is not False
            or row.get("oos_validated") is not False
            or row.get("clv_validated") is not False):
        raise ValueError("SPORT_FIRST_RESEARCH_ONLY_GATES_FAILED")
    fixture = row.get("fixture")
    if not isinstance(fixture, dict) or fixture.get("source_status") != "NS":
        raise ValueError("OFFICIAL_PREGAME_FIXTURE_NOT_VERIFIED")
    fid = row.get("fixture_id")
    if (not _positive_int(fid) or fixture.get("fixture_id") != fid
            or not all(_positive_int(fixture.get(k)) for k in
                       ("league_id", "home_team_id", "away_team_id"))
            or fixture["home_team_id"] == fixture["away_team_id"]):
        raise ValueError("FIXTURE_OR_TEAM_IDENTITY_MISMATCH")
    captured, kickoff = _time(row.get("captured_at_utc")), _time(row.get("kickoff"))
    generated = _time(row.get("generated_at_utc"))
    nested_kickoff = _time(fixture.get("kickoff"))
    if not captured or not kickoff or not generated or not nested_kickoff:
        raise ValueError("SOURCE_TIMESTAMP_MISSING")
    if generated != captured or nested_kickoff != kickoff or captured > kickoff - timedelta(minutes=45):
        raise ValueError("PREGAME_SOURCE_TIME_NOT_VERIFIED")
    model_sha = row.get("canonical_runtime_model_sha")
    if not isinstance(model_sha, str) or not re.fullmatch(r"[a-fA-F0-9]{40}", model_sha):
        raise ValueError("MODEL_GIT_COMMIT_NOT_VERIFIED")
    file_sha = row.get("model_file_sha256")
    if not isinstance(file_sha, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", file_sha):
        raise ValueError("MODEL_FILE_SHA_NOT_VERIFIED")
    if row.get("projection_model") != MODEL:
        raise ValueError("MODEL_VERSION_NOT_WHITELISTED")
    inputs = row.get("model_inputs_missing")
    if not isinstance(inputs, list) or not {"CONFIRMED_XI", "INJURIES"}.issubset(set(inputs)):
        raise ValueError("AVAILABILITY_NOT_EXPRESSED")
    sporting_source = row.get("sporting_source")
    if not isinstance(sporting_source, dict) or sporting_source.get("availability_confidence") != 0.50:
        raise ValueError("AVAILABILITY_CONFIDENCE_NOT_LIMITED")
    raw = row.get("raw_sport_projection")
    if (not isinstance(raw, dict) or raw.get("status") != "MODELED_LIMITED"
            or raw.get("projection_model") != MODEL or
            any(not _prob(raw.get(k)) for k in PROBABILITIES)):
        raise ValueError("MODEL_PROBABILITIES_INVALID")
    if abs(sum(raw[k] for k in PROBABILITIES[:3]) - 1) > 0.001:
        raise ValueError("ONE_X_TWO_MASS_NOT_VERIFIED")
    raw_bytes = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    if len(raw_bytes) > 65_000:
        raise ValueError("INDIVIDUAL_MODEL_SOURCE_UNBOUNDED")
    digest = hashlib.sha256(raw_bytes).hexdigest()
    return {
        "fixture_id": fid,
        "source_generated_at": captured, "kickoff": kickoff,
        "home_team_id": fixture["home_team_id"],
        "away_team_id": fixture["away_team_id"],
        "source": SOURCE, "projection_model": MODEL,
        "model_git_sha": model_sha, "payload_sha256": digest,
        "raw_projection": json.dumps(row, sort_keys=True, ensure_ascii=False),
    }


def validate_batch(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or set(payload) != {"snapshots"}:
        raise ValueError("UNEXPECTED_BRIDGE_INPUT")
    rows = payload["snapshots"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_ROWS:
        raise ValueError("INVALID_BATCH_SIZE")
    parsed = [validated_row(r) for r in rows]
    keys = [(r["fixture_id"], r["source_generated_at"]) for r in parsed]
    if len(keys) != len(set(keys)):
        raise ValueError("DUPLICATE_BATCH_SOURCE")
    return parsed


def store_frozen(payload: Any) -> dict[str, Any]:
    rows = validate_batch(payload)
    if not persistence.persistence_configured():
        raise RuntimeError("RESEARCH_POSTGRES_NOT_CONFIGURED")
    inserted, already = 0, 0
    with persistence._connect() as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute("SET LOCAL statement_timeout = '2500ms'")
                cur.execute("SET LOCAL lock_timeout = '1000ms'")
                cur.execute(_CREATE)
                for r in rows:
                    cur.execute("""
                        INSERT INTO soccer_isolated_sport_frozen_research_v1
                        (fixture_id, source_generated_at, kickoff, home_team_id, away_team_id,
                         source, projection_model, model_git_sha, payload_sha256, raw_projection)
                        VALUES (%(fixture_id)s, %(source_generated_at)s, %(kickoff)s,
                                %(home_team_id)s, %(away_team_id)s, %(source)s,
                                %(projection_model)s, %(model_git_sha)s,
                                %(payload_sha256)s, %(raw_projection)s::jsonb)
                        ON CONFLICT (fixture_id, source_generated_at) DO NOTHING
                        RETURNING payload_sha256
                    """, r)
                    if cur.fetchone():
                        inserted += 1
                    else:
                        cur.execute("""
                            SELECT payload_sha256 FROM soccer_isolated_sport_frozen_research_v1
                            WHERE fixture_id = %s AND source_generated_at = %s
                            LIMIT 1
                        """, (r["fixture_id"], r["source_generated_at"]))
                        seen = cur.fetchone()
                        if seen is None or seen[0] != r["payload_sha256"]:
                            raise RuntimeError("FROZEN_SPORT_MODEL_IMMUTABILITY_CONFLICT")
                        already += 1
    return {
        "status": "AUTHENTIC_PREGAME_SPORT_RESEARCH_IMPORTED",
        "snapshots_validated": len(rows),
        "new_research_rows": inserted,
        "identical_existing_rows": already,
        "source": SOURCE,
        "database": "POSTGRES_ISOLATED_SPORT_FROZEN_RESEARCH",
        "canonical_model_promotion": False,
        "canonical_clv_rows_added": 0,
        "canonical_deep_dives_added": 0,
        "market_prices_added": 0,
        "provider_requests_added": 0,
        "retroactive_models_created": False,
        "production_promotion_allowed": False,
        "bet_eligible": False,
    }


def read_for_owner_match(fixture: dict[str, Any]) -> dict[str, Any] | None:
    """Only invoked after OWNER/admin entitlement. No schema writes or provider calls."""
    fid, home, away, league = (fixture.get(k) for k in
                             ("fixture_id", "home_team_id", "away_team_id", "league_id"))
    start = _time(fixture.get("kickoff"))
    if not all(_positive_int(x) for x in (fid, home, away, league)) or not start:
        return None
    if not persistence.persistence_configured():
        return None
    with persistence._connect() as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute("SET TRANSACTION READ ONLY")
                cur.execute("SET LOCAL statement_timeout = '2000ms'")
                cur.execute("""
                    SELECT raw_projection, payload_sha256, source_generated_at
                    FROM soccer_isolated_sport_frozen_research_v1
                    WHERE fixture_id = %s AND home_team_id = %s AND away_team_id = %s
                      AND kickoff = %s AND source_generated_at <= now()
                      AND classification = 'RESEARCH_ONLY' AND bet_eligible = FALSE
                      AND production_promotion_allowed = FALSE
                    ORDER BY source_generated_at ASC LIMIT 1
                """, (fid, home, away, start))
                stored = cur.fetchone()
    if not stored:
        return None
    row, digest, stamp = stored
    try:
        verified = validated_row(row)
    except (ValueError, TypeError, OverflowError):
        return None
    if (verified["payload_sha256"] != digest or verified["source_generated_at"] != stamp
            or verified["fixture_id"] != fid or verified["home_team_id"] != home
            or verified["away_team_id"] != away or verified["kickoff"] != start
            or row["fixture"]["league_id"] != league or stamp > datetime.now(timezone.utc)):
        return None
    season = row["fixture"].get("season")
    if season is not None and fixture.get("season") is not None and season != fixture["season"]:
        return None
    raw = row["raw_sport_projection"]
    rates = {k: raw[k] for k in ("raw_home_goal_rate", "raw_away_goal_rate")
             if type(raw.get(k)) in (int, float) and 0.15 <= raw[k] <= 4.5}
    sample = raw.get("sample")
    sample = sample.get("minimum_split_sample") if isinstance(sample, dict) else None
    return {
        "fixture_id": fid, "kickoff_utc": start.isoformat(), "captured_at_utc": stamp.isoformat(),
        "source": SOURCE, "storage": "POSTGRES_ISOLATED_SPORT_FROZEN_RESEARCH",
        "model_version": MODEL, "model_git_sha": verified["model_git_sha"],
        "min_split_sample": sample if type(sample) is int and sample > 0 else None,
        "raw_probabilities": {k: raw[k] for k in PROBABILITIES},
        "raw_goal_rates": rates, "market_queried": False, "bet_eligible": False,
        "commercial_display_allowed": False, "oos_validated": False,
    }
