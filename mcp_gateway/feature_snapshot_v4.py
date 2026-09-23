from __future__ import annotations

from typing import Any

SCHEMA_VERSION = "4.0.0"


def _feature(
    value: Any,
    *,
    source: str,
    captured_at: str,
    sample_n: int | None = None,
    freshness: str | None = None,
    confidence: float | None = None,
    missing_reason: str | None = None,
) -> dict[str, Any]:
    if value is None and not missing_reason:
        missing_reason = "NOT_AVAILABLE_IN_CURRENT_RUNTIME"
    return {
        "value": value,
        "source": source,
        "captured_at": captured_at,
        "sample_n": sample_n,
        "freshness": freshness,
        "confidence": confidence,
        "missing_reason": missing_reason,
    }


def _lineup_team(lineups: Any, team_id: Any) -> dict[str, Any]:
    if not isinstance(lineups, dict):
        return {}
    for team in lineups.get("teams") or []:
        if isinstance(team, dict) and str(team.get("team_id")) == str(team_id):
            return team
    return {}


def build(tick: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    captured_at = str(tick.get("generated_at_utc") or "")
    fx = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
    raw = event.get("raw_projection") if isinstance(event.get("raw_projection"), dict) else {}
    coverage = event.get("coverage") if isinstance(event.get("coverage"), dict) else {}
    lineups = event.get("lineups") if isinstance(event.get("lineups"), dict) else {}
    injuries = event.get("injuries") if isinstance(event.get("injuries"), list) else None
    sample = raw.get("sample") if isinstance(raw.get("sample"), dict) else {}

    home_id = fx.get("home_team_id")
    away_id = fx.get("away_team_id")
    home_lineup = _lineup_team(lineups, home_id)
    away_lineup = _lineup_team(lineups, away_id)
    sample_min = sample.get("minimum_split_sample")
    try:
        sample_min_int = int(sample_min) if sample_min is not None else None
    except (TypeError, ValueError):
        sample_min_int = None

    features: dict[str, dict[str, Any]] = {
        # Team-performance engineered inputs currently available in the runtime.
        "team_performance.home_goal_rate_blend": _feature(
            raw.get("raw_home_goal_rate"),
            source="SOCCER_EDGE_VERIFIED_GOAL_RATE_BASELINE",
            captured_at=captured_at,
            sample_n=sample_min_int,
            freshness="CURRENT_TICK",
            confidence=0.7 if raw.get("raw_home_goal_rate") is not None else None,
            missing_reason=None if raw.get("raw_home_goal_rate") is not None else "INSUFFICIENT_VERIFIED_GOAL_RATE_INPUTS",
        ),
        "team_performance.away_goal_rate_blend": _feature(
            raw.get("raw_away_goal_rate"),
            source="SOCCER_EDGE_VERIFIED_GOAL_RATE_BASELINE",
            captured_at=captured_at,
            sample_n=sample_min_int,
            freshness="CURRENT_TICK",
            confidence=0.7 if raw.get("raw_away_goal_rate") is not None else None,
            missing_reason=None if raw.get("raw_away_goal_rate") is not None else "INSUFFICIENT_VERIFIED_GOAL_RATE_INPUTS",
        ),
        "team_performance.total_goal_rate_blend": _feature(
            raw.get("raw_total_goals"),
            source="SOCCER_EDGE_VERIFIED_GOAL_RATE_BASELINE",
            captured_at=captured_at,
            sample_n=sample_min_int,
            freshness="CURRENT_TICK",
            confidence=0.7 if raw.get("raw_total_goals") is not None else None,
            missing_reason=None if raw.get("raw_total_goals") is not None else "INSUFFICIENT_VERIFIED_GOAL_RATE_INPUTS",
        ),
        # Availability / XI.
        "availability.both_xi_confirmed": _feature(
            lineups.get("both_xi_confirmed") if lineups else None,
            source="API_FOOTBALL_LINEUPS",
            captured_at=captured_at,
            freshness="CURRENT_OR_CACHED_LINEUP_WINDOW",
            confidence=event.get("availability_confidence"),
            missing_reason=None if lineups else "LINEUPS_NOT_VERIFIED_AT_THIS_STAGE",
        ),
        "availability.both_goalkeepers_confirmed": _feature(
            lineups.get("both_goalkeepers_confirmed") if lineups else None,
            source="API_FOOTBALL_LINEUPS",
            captured_at=captured_at,
            freshness="CURRENT_OR_CACHED_LINEUP_WINDOW",
            confidence=event.get("availability_confidence"),
            missing_reason=None if lineups else "GOALKEEPERS_NOT_VERIFIED_AT_THIS_STAGE",
        ),
        "availability.home_formation": _feature(
            home_lineup.get("formation") if home_lineup else None,
            source="API_FOOTBALL_LINEUPS",
            captured_at=captured_at,
            freshness="CURRENT_OR_CACHED_LINEUP_WINDOW",
            confidence=event.get("availability_confidence"),
            missing_reason=None if home_lineup.get("formation") else "HOME_FORMATION_NOT_VERIFIED",
        ),
        "availability.away_formation": _feature(
            away_lineup.get("formation") if away_lineup else None,
            source="API_FOOTBALL_LINEUPS",
            captured_at=captured_at,
            freshness="CURRENT_OR_CACHED_LINEUP_WINDOW",
            confidence=event.get("availability_confidence"),
            missing_reason=None if away_lineup.get("formation") else "AWAY_FORMATION_NOT_VERIFIED",
        ),
        "availability.injury_report_count": _feature(
            len(injuries) if injuries is not None else None,
            source="API_FOOTBALL_INJURIES",
            captured_at=captured_at,
            freshness="CURRENT_OR_2H_CACHE",
            confidence=event.get("availability_confidence"),
            missing_reason=None if injuries is not None else "INJURY_REPORT_NOT_VERIFIED_AT_THIS_STAGE",
        ),
        # Fixture/context data.
        "context.league_id": _feature(
            fx.get("league_id"),
            source="API_FOOTBALL_FIXTURE",
            captured_at=captured_at,
            freshness="CURRENT_TICK",
            missing_reason=None if fx.get("league_id") is not None else "LEAGUE_ID_MISSING",
        ),
        "context.season": _feature(
            fx.get("season"),
            source="API_FOOTBALL_FIXTURE",
            captured_at=captured_at,
            freshness="CURRENT_TICK",
            missing_reason=None if fx.get("season") is not None else "SEASON_MISSING",
        ),
        "context.venue": _feature(
            fx.get("venue"),
            source="API_FOOTBALL_FIXTURE",
            captured_at=captured_at,
            freshness="CURRENT_TICK",
            missing_reason=None if fx.get("venue") else "VENUE_NOT_VERIFIED",
        ),
        "context.city": _feature(
            fx.get("city"),
            source="API_FOOTBALL_FIXTURE",
            captured_at=captured_at,
            freshness="CURRENT_TICK",
            missing_reason=None if fx.get("city") else "CITY_NOT_VERIFIED",
        ),
        "context.data_tier": _feature(
            coverage.get("data_tier"),
            source="API_FOOTBALL_COVERAGE",
            captured_at=captured_at,
            freshness="CURRENT_OR_CACHED_COVERAGE",
            missing_reason=None if coverage.get("data_tier") else "DATA_TIER_NOT_AVAILABLE",
        ),
        # Advanced families are explicit missing values until verified sources exist.
        "xg.xgf": _feature(None, source="NO_VERIFIED_RUNTIME_SOURCE", captured_at=captured_at, missing_reason="XG_NOT_VERIFIED"),
        "xg.xga": _feature(None, source="NO_VERIFIED_RUNTIME_SOURCE", captured_at=captured_at, missing_reason="XG_NOT_VERIFIED"),
        "xg.npxg": _feature(None, source="NO_VERIFIED_RUNTIME_SOURCE", captured_at=captured_at, missing_reason="NPXG_NOT_VERIFIED"),
        "xg.npxga": _feature(None, source="NO_VERIFIED_RUNTIME_SOURCE", captured_at=captured_at, missing_reason="NPXGA_NOT_VERIFIED"),
        "territory.ppda": _feature(None, source="NO_VERIFIED_RUNTIME_SOURCE", captured_at=captured_at, missing_reason="PPDA_NOT_VERIFIED"),
        "territory.field_tilt": _feature(None, source="NO_VERIFIED_RUNTIME_SOURCE", captured_at=captured_at, missing_reason="FIELD_TILT_NOT_VERIFIED"),
        "territory.box_entries": _feature(None, source="NO_VERIFIED_RUNTIME_SOURCE", captured_at=captured_at, missing_reason="BOX_ENTRIES_NOT_VERIFIED"),
        "territory.final_third_entries": _feature(None, source="NO_VERIFIED_RUNTIME_SOURCE", captured_at=captured_at, missing_reason="FINAL_THIRD_ENTRIES_NOT_VERIFIED"),
        "context.referee": _feature(None, source="NO_VERIFIED_RUNTIME_SOURCE", captured_at=captured_at, missing_reason="REFEREE_NOT_CAPTURED_IN_FEATURE_SNAPSHOT_YET"),
        "context.weather": _feature(None, source="NO_VERIFIED_RUNTIME_SOURCE", captured_at=captured_at, missing_reason="WEATHER_NOT_CAPTURED_IN_FEATURE_SNAPSHOT_YET"),
        "context.rest_days_home": _feature(None, source="NO_VERIFIED_RUNTIME_SOURCE", captured_at=captured_at, missing_reason="REST_DAYS_NOT_CAPTURED_YET"),
        "context.rest_days_away": _feature(None, source="NO_VERIFIED_RUNTIME_SOURCE", captured_at=captured_at, missing_reason="REST_DAYS_NOT_CAPTURED_YET"),
    }

    missing_count = sum(1 for row in features.values() if row.get("value") is None)
    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": fx.get("fixture_id"),
        "captured_at": captured_at,
        "stage": event.get("stage"),
        "model_version": event.get("model_version") or tick.get("model_version"),
        "data_tier": coverage.get("data_tier"),
        "sport_first": True,
        "market_fields_included": False,
        "feature_count": len(features),
        "missing_feature_count": missing_count,
        "features": features,
    }


def validate(snapshot: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if snapshot.get("schema_version") != SCHEMA_VERSION:
        errors.append("SCHEMA_VERSION_MISMATCH")
    if not snapshot.get("fixture_id"):
        errors.append("FIXTURE_ID_REQUIRED")
    if not snapshot.get("captured_at"):
        errors.append("CAPTURED_AT_REQUIRED")
    features = snapshot.get("features")
    if not isinstance(features, dict) or not features:
        errors.append("FEATURES_REQUIRED")
        return errors
    required = {"value", "source", "captured_at", "sample_n", "freshness", "confidence", "missing_reason"}
    for key, row in features.items():
        if not isinstance(row, dict):
            errors.append(f"{key}:INVALID_ENVELOPE")
            continue
        missing_keys = required - set(row)
        if missing_keys:
            errors.append(f"{key}:MISSING_ENVELOPE_FIELDS")
        if row.get("value") is None and not row.get("missing_reason"):
            errors.append(f"{key}:MISSING_REASON_REQUIRED")
    if snapshot.get("market_fields_included") is not False:
        errors.append("MARKET_FIELDS_MUST_BE_EXCLUDED")
    return errors
