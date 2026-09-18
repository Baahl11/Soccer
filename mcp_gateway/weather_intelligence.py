from __future__ import annotations
from typing import Any

SCHEMA_VERSION = "1.0.0"

def build(event: dict[str, Any]) -> dict[str, Any]:
    fixture=event.get("fixture") if isinstance(event.get("fixture"),dict) else {}
    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": fixture.get("fixture_id"),
        "status": "WEATHER_SOURCE_NOT_INTEGRATED",
        "venue": fixture.get("venue"),
        "city": fixture.get("city"),
        "country": fixture.get("country"),
        "kickoff": fixture.get("kickoff"),
        "temperature_c": None,
        "wind_kph": None,
        "precipitation": None,
        "humidity_pct": None,
        "severe_weather": "NOT_VERIFIED",
        "materiality": "NOT_MODELED",
        "actionable": False,
        "decision_weight": 0.0,
        "block_reasons": [
            "APPROVED_WEATHER_PROVIDER_NOT_INTEGRATED",
            "VENUE_COORDINATES_OR_PROVIDER_LOCATION_MATCH_REQUIRED",
            "KICKOFF_ALIGNED_FORECAST_REQUIRED",
        ],
        "policy": "NO WEATHER VALUE MAY BE INVENTED OR INFERRED FROM CITY/SEASON/COUNTRY.",
    }

def attach(payload: dict[str, Any]) -> dict[str,int]:
    attached=0
    for event in payload.get("events") or []:
        if not isinstance(event,dict) or event.get("event_type")!="SOCCER_REFRESH" or event.get("stage") in {"POSTGAME","HT","CLOSE"}:
            continue
        intel=build(event); event["weather_intelligence"]=intel; attached+=1
        mi=event.get("match_intelligence")
        if isinstance(mi,dict) and isinstance(mi.get("areas"),dict): mi["areas"]["weather"]=intel
    return {"guard_events":attached,"provider_requests_added":0}
