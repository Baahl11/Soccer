from __future__ import annotations

from typing import Any

SCHEMA_VERSION = "1.0.0"
_ACCEPTED_STAT_TYPES = {"big_chances", "big_chances_created"}


def _num(value: Any) -> float | None:
    try:
        value = float(value)
        return value if value >= 0 else None
    except (TypeError, ValueError):
        return None


def _norm(value: Any) -> str:
    return "_".join(str(value or "").strip().lower().replace("-", " ").split())


def _fixture(event: dict[str, Any]) -> dict[str, Any]:
    row = event.get("fixture")
    return row if isinstance(row, dict) else {}


def _extract(match_stats: Any, team_id: Any) -> tuple[float | None, str | None]:
    if not isinstance(match_stats, list):
        return None, None
    for row in match_stats:
        if not isinstance(row, dict):
            continue
        team = row.get("team") if isinstance(row.get("team"), dict) else {}
        if str(team.get("id")) != str(team_id):
            continue
        for stat in row.get("statistics") or []:
            if not isinstance(stat, dict):
                continue
            stat_type = _norm(stat.get("type"))
            if stat_type not in _ACCEPTED_STAT_TYPES:
                continue
            value = _num(stat.get("value"))
            if value is not None:
                return value, stat_type
    return None, None


def build_postgame_observation(event: dict[str, Any]) -> dict[str, Any]:
    fixture = _fixture(event)
    home, home_type = _extract(event.get("match_stats"), fixture.get("home_team_id"))
    away, away_type = _extract(event.get("match_stats"), fixture.get("away_team_id"))
    verified = home is not None and away is not None and home_type == away_type
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "VERIFIED_API_FOOTBALL_BIG_CHANCES" if verified else "API_FOOTBALL_BIG_CHANCES_NOT_AVAILABLE",
        "api_fixture_id": fixture.get("fixture_id"),
        "kickoff": fixture.get("kickoff"),
        "home_team_id": fixture.get("home_team_id"),
        "home_team": fixture.get("home_team"),
        "away_team_id": fixture.get("away_team_id"),
        "away_team": fixture.get("away_team"),
        "home_big_chances": round(home, 6) if home is not None else None,
        "away_big_chances": round(away, 6) if away is not None else None,
        "source": "API-Football v3",
        "source_endpoint": "/fixtures/statistics",
        "source_stat_type": home_type if verified else None,
        "provider_requests_added": 0,
        "provider_dependent": True,
        "actionable": False,
        "decision_weight": 0.0,
    }


def build_pregame(event: dict[str, Any]) -> dict[str, Any]:
    fixture = _fixture(event)
    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": fixture.get("fixture_id"),
        "status": "BIG_CHANCES_HISTORY_NOT_YET_MODELED",
        "source_policy": "API_FOOTBALL_EXPLICIT_PROVIDER_FIELD_ONLY",
        "provider_dependent": True,
        "shots_or_xg_proxy_allowed": False,
        "actionable": False,
        "decision_weight": 0.0,
        "block_reason": "CONSISTENT_VERIFIED_HISTORICAL_BIG_CHANCES_SAMPLE_NOT_YET_ESTABLISHED",
        "policy": (
            "CAPTURE ONLY AN EXPLICIT API-FOOTBALL BIG CHANCES FIELD WHEN PRESENT. "
            "DO NOT RECONSTRUCT BIG CHANCES FROM SHOTS, SHOTS INSIDE BOX, xG OR GOALS."
        ),
    }


def attach(payload: dict[str, Any]) -> dict[str, int | str]:
    captured = unavailable = blocked = 0
    for event in payload.get("events") or []:
        if not isinstance(event, dict) or event.get("event_type") != "SOCCER_REFRESH":
            continue
        stage = event.get("stage")
        if stage == "POSTGAME":
            obs = build_postgame_observation(event)
            event["postgame_big_chances_observation"] = obs
            if obs.get("status") == "VERIFIED_API_FOOTBALL_BIG_CHANCES":
                captured += 1
            else:
                unavailable += 1
            continue
        if stage in {"HT", "CLOSE"}:
            continue
        intel = build_pregame(event)
        event["big_chances_intelligence"] = intel
        blocked += 1
        mi = event.get("match_intelligence")
        if isinstance(mi, dict) and isinstance(mi.get("areas"), dict):
            mi["areas"]["big_chances"] = intel
    return {
        "provider_policy": "API_FOOTBALL_EXPLICIT_FIELD_ONLY",
        "postgame_big_chances_captured": captured,
        "postgame_big_chances_unavailable": unavailable,
        "pregame_data_blocked_events": blocked,
        "provider_requests_added": 0,
    }
