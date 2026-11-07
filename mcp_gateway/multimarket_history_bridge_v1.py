"""Private, immutable SPORT-FIRST historical profile transport to App V2.

One authenticated GitHub main-branch scanner source, no price feeds. Existing
soccer_fixtures identity and kickoff must match before persistence.
A historical pattern is never a calibrated probability, value signal or BET.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
import json
from typing import Any

from mcp_gateway import persistence

VERSION = "SOCCER_FE11_MULTIMARKET_PROFILE_BRIDGE_v1.0"
SOURCE = "SPORTS_EDGE_SCANNER_V15_HISTORICAL_CONTEXT"
MODEL_VERSION = "SOCCER_SPORT_FIRST_MULTIMARKET_PROFILE_v1.0"
MAX_REPORTS = 3
MAX_BODY_BYTES = 400_000
MAX_OBSERVATIONS_PER_SIDE = 20

INSERT_SQL = """
INSERT INTO soccer_multimarket_history_profiles (
  fixture_id, source_generated_at, kickoff, home_team_id, away_team_id,
  source, model_version, home_profile, away_profile, home_pattern, away_pattern
)
SELECT f.fixture_id, %s::timestamptz, f.kickoff,
       f.home_team_id, f.away_team_id, %s, %s,
       %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb
FROM soccer_fixtures f
WHERE f.fixture_id = %s AND f.home_team_id = %s
  AND f.away_team_id = %s AND f.kickoff = %s::timestamptz
  AND f.status = 'NS' AND f.kickoff > now()
ON CONFLICT (fixture_id, source_generated_at) DO NOTHING
RETURNING fixture_id
"""

READ_SQL = """
SELECT p.fixture_id, p.source_generated_at, p.kickoff, p.model_version,
       p.home_team_id, p.away_team_id,
       p.home_profile, p.away_profile, p.home_pattern, p.away_pattern,
       p.ingested_at
FROM soccer_multimarket_history_profiles p
JOIN soccer_fixtures f ON f.fixture_id = p.fixture_id
WHERE p.fixture_id = %s AND f.kickoff = p.kickoff
  AND f.home_team_id = p.home_team_id AND f.away_team_id = p.away_team_id
  AND f.status = 'NS' AND f.kickoff > now()
ORDER BY p.source_generated_at DESC
LIMIT 1
"""


def _utc(v: Any) -> datetime | None:
    if isinstance(v, str):
        try:
            v = datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            return None
    if isinstance(v, datetime) and v.tzinfo is not None:
        return v.astimezone(timezone.utc)
    return None


def _id(v: Any) -> int | None:
    return v if type(v) is int and 0 < v < 10**12 else None


def _safe_profile(profile: Any, team_id: int, kickoff: datetime) -> dict[str, Any] | None:
    if not isinstance(profile, dict):
        return None
    if (
        profile.get("version") != MODEL_VERSION
        or profile.get("evidence_scope") != "PAST_VERIFIED_FT_ONLY"
        or profile.get("sports_first_no_market_inputs") is not True
        or profile.get("descriptive_only") is not True
        or profile.get("calibrated_prediction") is not False
        or profile.get("bet_eligible") is not False
        or _id(profile.get("team_id")) != team_id
        or _utc(profile.get("target_kickoff")) != kickoff
        or not isinstance(profile.get("splits"), dict)
    ):
        return None
    observations = profile.get("observations")
    if not isinstance(observations, list) or len(observations) > MAX_OBSERVATIONS_PER_SIDE:
        return None
    seen: set[int] = set()
    for obs in observations:
        if not isinstance(obs, dict):
            return None
        fid = _id(obs.get("fixture_id"))
        when = _utc(obs.get("kickoff"))
        if (
            fid is None or fid in seen
            or when is None or when >= kickoff
            or obs.get("venue") not in ("home", "away")
        ):
            return None
        seen.add(fid)
        for field in ("goals_for","goals_against","ft_goals_total",
                      "first_half_goals_for","first_half_goals_against",
                      "first_half_goals_total","second_half_goals_for",
                      "second_half_goals_against","second_half_goals_total",
                      "team_corners_ft","opponent_corners_ft","total_corners_ft"):
            x = obs.get(field)
            if x is not None and (type(x) is not int or x < 0 or x > 80):
                return None
        if obs.get("goals_for") is None or obs.get("goals_against") is None:
            return None
        if obs.get("ft_goals_total") != obs["goals_for"] + obs["goals_against"]:
            return None
        if (obs.get("first_half_goals_total") is not None
            and obs.get("second_half_goals_total") is not None
            and obs["first_half_goals_total"] + obs["second_half_goals_total"] != obs["ft_goals_total"]):
            return None
        if (obs.get("total_corners_ft") is not None
            and (obs.get("team_corners_ft") is None
                 or obs.get("opponent_corners_ft") is None
                 or obs["total_corners_ft"] != obs["team_corners_ft"]+obs["opponent_corners_ft"])):
            return None
    for key in ("all","home","away"):
        if not isinstance(profile["splits"].get(key), dict):
            return None
    # No bookmaker price, odds, EV, or derived betting decision can be imported.
    forbidden = {"price","odds","bet","stake","ev","market_shrunk",
                 "fair_market_probability","bet_eligible"}
    if any(k in profile for k in forbidden - {"bet_eligible"}):
        return None
    # Keep bounded original research provenance. The reader only serves
    # verified summary/observations with explicit limitations, never a BET.
    return profile


def _safe_pattern(pattern: Any) -> dict[str, Any] | None:
    if not isinstance(pattern, dict):
        return None
    for variant in ("ASIAN","THREE_WAY"):
        x = pattern.get(variant)
        if (
            not isinstance(x, dict)
            or x.get("version") != MODEL_VERSION
            or x.get("eligible_as_joint_model_probability") is not False
            or x.get("bet_eligible") is not False
            or x.get("price_decimal") is not None
            or x.get("correlated_sgp_quote") is not None
            or not isinstance(x.get("complete_historical_rows"), int)
            or x["complete_historical_rows"] < 0
            or x["complete_historical_rows"] > MAX_OBSERVATIONS_PER_SIDE
        ):
            return None
        legs = x.get("legs")
        if not isinstance(legs,list) or len(legs) != 3:
            return None
        if legs[0] != {"type":"FT_GOALS_OVER","line":1.5}:
            return None
        if legs[1] != {"type":"FT_CORNERS_OVER","line":6.5}:
            return None
        if legs[2] != {"type":"TEAM_HANDICAP","line":1,"settlement":variant}:
            return None
    return pattern


def validate_import(doc: Any, *, now: datetime | None = None) -> list[dict[str, Any]]:
    now = now or datetime.now(timezone.utc)
    if not isinstance(doc,dict) or doc.get("source") != SOURCE:
        raise ValueError("UNTRUSTED_SOURCE")
    generated = _utc(doc.get("generated_at_utc"))
    if generated is None or not now-timedelta(hours=48) <= generated <= now+timedelta(minutes=5):
        raise ValueError("SOURCE_TIMESTAMP_OUTSIDE_WINDOW")
    reports = doc.get("reports")
    if not isinstance(reports,list) or len(reports) > MAX_REPORTS:
        raise ValueError("INVALID_REPORTS_COUNT")
    used: set[int] = set()
    accepted = []
    for report in reports:
        if not isinstance(report,dict):
            raise ValueError("REPORT_NOT_OBJECT")
        fid = _id(report.get("fixture_id"))
        h, a = _id(report.get("home_team_id")), _id(report.get("away_team_id"))
        kickoff = _utc(report.get("kickoff"))
        if (
            fid is None or fid in used or h is None or a is None or h == a
            or kickoff is None or kickoff <= generated or kickoff <= now
            or kickoff > now+timedelta(days=3)
        ):
            raise ValueError("FIXTURE_IDENTITY_OR_CUTOFF_NOT_VERIFIED")
        used.add(fid)
        home = _safe_profile(report.get("home_profile"),h,kickoff)
        away = _safe_profile(report.get("away_profile"),a,kickoff)
        hp = _safe_pattern(report.get("home_pattern"))
        ap = _safe_pattern(report.get("away_pattern"))
        if any(x is None for x in (home,away,hp,ap)):
            raise ValueError("SPORT_FIRST_PROFILE_OR_HANDICAP_INVALID")
        accepted.append({
            "fixture_id":fid,"home_id":h,"away_id":a,"kickoff":kickoff,
            "generated":generated,"home":home,"away":away,
            "home_pattern":hp,"away_pattern":ap,
        })
    return accepted


def ingest(doc: Any) -> dict[str, Any]:
    items = validate_import(doc)
    persistence.ensure_schema()
    saved, identities = 0, []
    not_inserted: dict[str, int] = {}
    with persistence._connect() as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                for row in items:
                    cur.execute(INSERT_SQL, (
                        row["generated"],SOURCE,MODEL_VERSION,
                        json.dumps(row["home"],separators=(",",":")),
                        json.dumps(row["away"],separators=(",",":")),
                        json.dumps(row["home_pattern"],separators=(",",":")),
                        json.dumps(row["away_pattern"],separators=(",",":")),
                        row["fixture_id"],row["home_id"],row["away_id"],row["kickoff"],
                    ))
                    if cur.fetchone() is not None:
                        saved += 1
                        identities.append(row["fixture_id"])
                    else:
                        cur.execute(
                            "SELECT status,kickoff,home_team_id,away_team_id "
                            "FROM soccer_fixtures WHERE fixture_id=%s",
                            (row["fixture_id"],)
                        )
                        reg = cur.fetchone()
                        if reg is None:
                            reason="REGISTRY_FIXTURE_NOT_FOUND"
                        elif reg[2] != row["home_id"] or reg[3] != row["away_id"]:
                            reason="REGISTRY_TEAM_ID_CONFLICT"
                        elif _utc(reg[1]) != row["kickoff"]:
                            reason="REGISTRY_KICKOFF_MISMATCH"
                        elif reg[0] != "NS":
                            reason="REGISTRY_STATUS_NOT_NS"
                        elif row["kickoff"] <= datetime.now(timezone.utc):
                            reason="FIXTURE_ALREADY_STARTED"
                        else:
                            reason="ALREADY_IMPORTED_OR_DUPLICATE"
                        not_inserted[reason]=not_inserted.get(reason,0)+1
    return {
        "status":"SPORT_FIRST_HISTORY_IMPORTED",
        "source_version":VERSION,"validated_reports":len(items),
        "newly_persisted":saved,"persisted_fixture_ids":identities,
        "not_persisted_by_registry_reason":not_inserted,
        "bet_eligible":False,"probabilities_calibrated":False,
        "provider_requests_added":0,
    }


def read_for_match(fixture_id: Any) -> dict[str, Any]:
    fid = _id(fixture_id)
    if fid is None:
        return {"status":"NOT_VERIFIED","reason":"FIXTURE_ID_INVALID","bet_eligible":False}
    if not persistence.persistence_configured():
        return {"status":"NOT_VERIFIED","reason":"DB_UNAVAILABLE","bet_eligible":False}
    persistence.ensure_schema()
    with persistence._connect() as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute("SET TRANSACTION READ ONLY")
                cur.execute(READ_SQL,(fid,))
                row = cur.fetchone()
    if not row:
        return {"status":"NOT_VERIFIED","reason":"NO_VERIFIED_HISTORICAL_PROFILE_FOR_FIXTURE",
                "bet_eligible":False}
    keys = ("fixture_id","source_generated_at","kickoff","model_version",
            "home_team_id","away_team_id","home_profile","away_profile",
            "home_pattern","away_pattern","ingested_at")
    data = dict(zip(keys,row))
    now = datetime.now(timezone.utc)
    age = now - _utc(data["source_generated_at"])
    fresh = timedelta(0) <= age <= timedelta(hours=36)
    return {
        "status":"HISTORICAL_DESCRIPTIVE_ONLY" if fresh else "STALE_HISTORICAL_RESEARCH",
        "source":SOURCE,"model_version":MODEL_VERSION,
        "source_generated_at_utc":_utc(data["source_generated_at"]).isoformat(),
        "ingested_at_utc":_utc(data["ingested_at"]).isoformat(),
        "fixture_id":fid,"kickoff":_utc(data["kickoff"]).isoformat(),
        "home":data["home_profile"],"away":data["away_profile"],
        "home_pattern":data["home_pattern"],"away_pattern":data["away_pattern"],
        "sport_first":True,"history_fresh":fresh,
        "observations_not_forecasts":True,
        "calibrated_joint_probability":None,
        "bookmaker_combined_quote":None,
        "estimated_ev":None,
        "classification":"WATCH","bet_eligible":False,
        "blockers":["NO_CALIBRATED_JOINT_PROBABILITY",
                    "NO_VERIFIED_BOOKMAKER_BUILDER_TICKET",
                    "XI_AND_MATERIAL_AVAILABILITY_NOT_VERIFIED",
                    "DESCRIPTIVE_ONLY_NOT_OPPONENT_ADJUSTED"],
    }



def inventory_readiness() -> dict[str, Any]:
    """OIDC-guarded, aggregate-only verification of actual FE11 DB persistence."""
    if not persistence.persistence_configured():
        return {"status":"NOT_VERIFIED","reason":"DB_UNAVAILABLE"}
    persistence.ensure_schema()
    with persistence._connect() as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute("SET TRANSACTION READ ONLY")
                cur.execute("""
                    SELECT COUNT(*)::integer,
                           COUNT(DISTINCT p.fixture_id)::integer,
                           COUNT(DISTINCT p.fixture_id) FILTER (
                             WHERE p.kickoff > now()
                               AND f.status = 'NS'
                               AND f.kickoff = p.kickoff
                               AND f.home_team_id = p.home_team_id
                               AND f.away_team_id = p.away_team_id
                               AND p.source_generated_at >= now() - interval '36 hours'
                           )::integer,
                           MAX(p.source_generated_at)
                    FROM soccer_multimarket_history_profiles p
                    JOIN soccer_fixtures f ON f.fixture_id = p.fixture_id
                """)
                row=cur.fetchone()
    total, fixtures, current, latest = row or (0,0,0,None)
    return {
        "status":"VERIFIED_READ",
        "research_rows_persisted":int(total),
        "unique_fixtures":int(fixtures),
        "current_pregame_profiles":int(current),
        "last_source_generated_at_utc":_utc(latest).isoformat() if _utc(latest) else None,
        "production_bets_authorized":False,
        "provider_requests_added":0,
    }
