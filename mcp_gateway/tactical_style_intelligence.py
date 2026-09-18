from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import httpx

SCHEMA_VERSION = "1.0.0"
TREND_URL = (
    "https://raw.githubusercontent.com/Baahl11/Soccer/"
    "soccer-edge-state/soccer_edge_state/analysis/trend_intelligence.json"
)
_CACHE: dict[str, Any] = {}
_CACHE_AT: datetime | None = None
TTL_SECONDS = 1800


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load() -> dict[str, Any]:
    global _CACHE, _CACHE_AT
    now = datetime.now(timezone.utc)
    if _CACHE_AT and (now - _CACHE_AT).total_seconds() < TTL_SECONDS:
        return _CACHE
    try:
        r = httpx.get(TREND_URL, timeout=4.0, follow_redirects=True)
        data = r.json() if r.status_code == 200 else {}
        _CACHE = data if isinstance(data, dict) else {}
    except Exception:
        _CACHE = {}
    _CACHE_AT = now
    return _CACHE


def _team_row(data: dict[str, Any], team_id: Any) -> dict[str, Any] | None:
    for row in data.get("teams") or []:
        if isinstance(row, dict) and str(row.get("team_id")) == str(team_id):
            return row
    return None


def _window(team: dict[str, Any] | None, venue: str) -> dict[str, Any]:
    if not isinstance(team, dict):
        return {}
    venue_row = ((team.get("venue_windows") or {}).get(f"{venue}_last_10") or {})
    overall = ((team.get("windows") or {}).get("last_10") or {})
    return venue_row if int(venue_row.get("n") or 0) >= 5 else overall


def _formation(event: dict[str, Any], team_id: Any) -> str | None:
    lineups = event.get("lineups") if isinstance(event.get("lineups"), dict) else {}
    for row in lineups.get("teams") or []:
        if isinstance(row, dict) and str(row.get("team_id")) == str(team_id):
            value = str(row.get("formation") or "").strip()
            return value or None
    return None


def _formation_family(value: str | None) -> str:
    if not value:
        return "NOT_VERIFIED"
    parts = []
    for p in value.split("-"):
        try:
            parts.append(int(p))
        except ValueError:
            return "FORM_NOT_NORMALIZED"
    if not parts:
        return "FORM_NOT_NORMALIZED"
    defenders = parts[0]
    if defenders >= 5:
        return "BACK_FIVE_BASE"
    if defenders == 4:
        return "BACK_FOUR_BASE"
    if defenders == 3:
        return "BACK_THREE_BASE"
    return "OTHER_FORMATION_BASE"


def _band(value: float | None, low: float, high: float, low_label: str, mid_label: str, high_label: str) -> str:
    if value is None:
        return "NOT_VERIFIED"
    if value <= low:
        return low_label
    if value >= high:
        return high_label
    return mid_label


def _profile(team: dict[str, Any] | None, venue: str, formation: str | None) -> dict[str, Any]:
    w = _window(team, venue)
    n = int(w.get("n") or 0)
    possession = _num(w.get("avg_possession"))
    shots = _num(w.get("avg_team_shots"))
    sot = _num(w.get("avg_team_sot"))
    corners = _num(w.get("avg_team_corners"))
    yellows = _num(w.get("avg_yellow_cards"))
    return {
        "sample_n": n,
        "formation": formation,
        "formation_family": _formation_family(formation),
        "avg_possession": possession,
        "possession_tendency": _band(possession, 45.0, 55.0, "LOW_POSSESSION", "BALANCED_POSSESSION", "HIGH_POSSESSION"),
        "avg_shots": shots,
        "shot_volume_tendency": _band(shots, 9.0, 14.0, "LOW_SHOT_VOLUME", "MID_SHOT_VOLUME", "HIGH_SHOT_VOLUME"),
        "avg_sot": sot,
        "sot_volume_tendency": _band(sot, 3.0, 5.0, "LOW_SOT_VOLUME", "MID_SOT_VOLUME", "HIGH_SOT_VOLUME"),
        "avg_corners": corners,
        "corner_volume_tendency": _band(corners, 3.5, 6.0, "LOW_CORNER_VOLUME", "MID_CORNER_VOLUME", "HIGH_CORNER_VOLUME"),
        "avg_yellow_cards": yellows,
        "discipline_tendency": _band(yellows, 1.5, 3.0, "LOW_CARD_VOLUME", "MID_CARD_VOLUME", "HIGH_CARD_VOLUME"),
        "press_intensity": "NOT_VERIFIED_PPDA_REQUIRED",
        "block_height": "NOT_VERIFIED_SPATIAL_DATA_REQUIRED",
        "transition_speed": "NOT_VERIFIED_EVENT_SEQUENCE_DATA_REQUIRED",
        "width_style": "NOT_VERIFIED_CORNERS_ARE_NOT_WIDTH",
    }


def build(event: dict[str, Any]) -> dict[str, Any]:
    fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
    data = _load()
    home = _team_row(data, fixture.get("home_team_id"))
    away = _team_row(data, fixture.get("away_team_id"))
    hp = _profile(home, "home", _formation(event, fixture.get("home_team_id")))
    ap = _profile(away, "away", _formation(event, fixture.get("away_team_id")))
    usable = hp["sample_n"] >= 5 and ap["sample_n"] >= 5
    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": fixture.get("fixture_id"),
        "status": "LIVE_RESEARCH_TACTICAL_STYLE_CONTEXT" if usable else "TACTICAL_STYLE_HISTORY_INSUFFICIENT",
        "source_generated_at_utc": data.get("generated_at_utc"),
        "home": hp,
        "away": ap,
        "verified_descriptors": [
            "formation_family",
            "possession_tendency",
            "shot_volume_tendency",
            "sot_volume_tendency",
            "corner_volume_tendency",
            "discipline_tendency",
        ],
        "blocked_descriptors": [
            "press_intensity",
            "block_height",
            "transition_speed",
            "width_style",
        ],
        "actionable": False,
        "decision_weight": 0.0,
        "production_status": "CONTEXT_ONLY_NOT_ACTIONABLE",
        "promotion_gate": {
            "minimum_oos_feature_lift_sample": 500,
            "requires_versioned_residual_test": True,
            "requires_no_degradation_in_brier_logloss": True,
        },
        "policy": (
            "DESCRIBE ONLY OBSERVED FORMATION AND PERSISTED MATCH-STAT TENDENCIES. "
            "DO NOT INFER PRESS/BLOCK/TRANSITION/WIDTH WITHOUT PPDA/SPATIAL/EVENT-SEQUENCE EVIDENCE."
        ),
    }


def attach(payload: dict[str, Any]) -> dict[str, int]:
    attached = insufficient = 0
    for event in payload.get("events") or []:
        if not isinstance(event, dict) or event.get("event_type") != "SOCCER_REFRESH":
            continue
        if event.get("stage") in {"POSTGAME", "HT", "CLOSE"}:
            continue
        intel = build(event)
        event["tactical_style_intelligence"] = intel
        if intel.get("status") == "LIVE_RESEARCH_TACTICAL_STYLE_CONTEXT":
            attached += 1
        else:
            insufficient += 1
        mi = event.get("match_intelligence")
        if isinstance(mi, dict) and isinstance(mi.get("areas"), dict):
            mi["areas"]["tactical_style"] = intel
    return {
        "context_events": attached,
        "history_insufficient_events": insufficient,
        "provider_requests_added": 0,
    }
