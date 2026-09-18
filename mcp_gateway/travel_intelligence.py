from __future__ import annotations
from typing import Any

SCHEMA_VERSION = "1.0.0"

def build(event: dict[str, Any]) -> dict[str, Any]:
    fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
    venue = fixture.get("venue")
    city = fixture.get("city")
    country = fixture.get("country")
    destination_verified = bool(venue or city or country)
    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": fixture.get("fixture_id"),
        "status": "LIVE_TRAVEL_SOURCE_GUARD",
        "destination": {"venue": venue, "city": city, "country": country},
        "destination_verified": destination_verified,
        "away_team": fixture.get("away_team"),
        "travel_origin": "NOT_VERIFIED",
        "distance_km": None,
        "timezone_shift_hours": None,
        "destination_altitude_m": None,
        "logistics_burden": "NOT_MODELED",
        "actionable": False,
        "decision_weight": 0.0,
        "block_reasons": [
            "VERIFIED_AWAY_TEAM_TRAVEL_ORIGIN_COORDINATES_REQUIRED",
            "VERIFIED_VENUE_COORDINATES_REQUIRED",
            "TIMEZONE_AND_ALTITUDE_SOURCE_REQUIRED",
        ],
        "policy": (
            "DO NOT ESTIMATE DISTANCE FROM CITY/COUNTRY LABELS ALONE; "
            "DO NOT ASSUME TEAM BASE, TRAVEL ORIGIN, ALTITUDE OR TIMEZONE SHIFT."
        ),
    }

def attach(payload: dict[str, Any]) -> dict[str, int]:
    attached=0
    for event in payload.get("events") or []:
        if not isinstance(event,dict) or event.get("event_type")!="SOCCER_REFRESH" or event.get("stage") in {"POSTGAME","HT","CLOSE"}:
            continue
        intel=build(event); event["travel_intelligence"]=intel; attached+=1
        mi=event.get("match_intelligence")
        if isinstance(mi,dict) and isinstance(mi.get("areas"),dict): mi["areas"]["travel"]=intel
    return {"guard_events":attached,"provider_requests_added":0}
