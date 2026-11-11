"""Daily SPORT-FIRST P0 screening for independently sourced football goal splits.

Pre-game, cheap, provider-budgeted, no odds, no injuries/lineups claims, no BET.
Persistent stage de-duplication is coordinated by caller's SQLite cache.
"""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import Any

VERSION="SOCCER_P0_SPORT_SCREEN_v1.0"
STAGE="SPORT_PREVIEW"
MAX_PER_TICK=12
MAX_PROVIDER_CALLS_FOR_SCREEN=24
MIN_CALLS_RESERVED_FOR_REST_OF_TICK=8
MINIMUM_PREGAME_SECONDS=15 * 60


def eligible(fixture: Any, now: datetime) -> bool:
    if not isinstance(fixture,dict): return False
    if type(fixture.get("fixture_id")) is not int or fixture["fixture_id"]<=0:return False
    if any(type(fixture.get(k)) is not int or fixture[k]<=0 for k in (
        "league_id","season","home_team_id","away_team_id")):return False
    if fixture["home_team_id"]==fixture["away_team_id"]:return False
    if fixture.get("status") in {"FT","AET","PEN","PST","SUSP","CANC","ABD","AWD","WO","LIVE","1H","HT","2H","ET","BT","P","INT"}:
        return False
    try:
        kickoff=datetime.fromisoformat(str(fixture["kickoff"]).replace("Z","+00:00"))
        if kickoff.tzinfo is None:return False
    except (KeyError,TypeError,ValueError):return False
    return (kickoff.astimezone(timezone.utc)-now).total_seconds()>=MINIMUM_PREGAME_SECONDS


def preview_event(
    fx: dict[str,Any],
    now: datetime,
    sporting: dict[str,Any],
    raw: dict[str,Any],
    shortlist: dict[str,Any],
) -> dict[str,Any]:
    status=raw.get("status")
    forecast=status=="MODELED_LIMITED"
    sample=raw.get("sample") if isinstance(raw.get("sample"),dict) else {}
    minimum=sample.get("minimum_split_sample")
    if type(minimum) is not int or minimum<5:
        forecast=False
    return {
        "event_type":"SOCCER_REFRESH","stage":STAGE,
        "fixture":fx,
        "source":"API_FOOTBALL_TEAM_SEASON_SPLITS",
        "sporting":sporting,
        "raw_projection":raw,
        "sporting_shortlist":shortlist,
        "raw_sport_screen_status":"PREMATCH_LIMITED_PROJECTION" if forecast else "SPORT_INPUTS_INSUFFICIENT",
        "raw_sport_show_as_forecast":forecast,
        "source_observed_at_utc":now.isoformat(),
        "source_updated_at_utc":"NOT_VERIFIED",
        "model_version":raw.get("model_version"),
        "forecast_limitations":["GOAL_RATES_NOT_XG","OPPONENT_ADJUSTMENT_NOT_VERIFIED",
             "OOS_CALIBRATION_NOT_VERIFIED","LINEUP_AVAILABILITY_NOT_VERIFIED"],
        "classification":"WATCH",
        "stake_units":0.0,
        "bet_eligible":False,
        "quote_requested":False,
        "market":"NOT_REQUESTED_SPORT_FIRST",
        "availability_confidence":None,
        "source_license_redisplay_verified":False,
        "historical_backfill_allowed":False,
        "notes":[
            "Pregame SPORT-only goal baseline built from real team season splits.",
            "No market odds were accessed. Not xG, not independently OOS-calibrated.",
            "Missing or split sample <5 is not a customer betting forecast.",
        ],
    }


def asof_research_projection(event: Any, *, now: datetime) -> dict[str,Any] | None:
    """Extract only safe prospective owner research projection from a persisted event."""
    if not isinstance(event,dict) or event.get("stage")!=STAGE:
        return None
    if event.get("quote_requested") is not False or event.get("bet_eligible") is not False:
        return None
    if event.get("raw_sport_show_as_forecast") is not True:
        return None
    fx=event.get("fixture") if isinstance(event.get("fixture"),dict) else {}
    raw=event.get("raw_projection") if isinstance(event.get("raw_projection"),dict) else {}
    if raw.get("status")!="MODELED_LIMITED" or type(fx.get("fixture_id")) is not int:
        return None
    try:
        kickoff=datetime.fromisoformat(str(fx["kickoff"]).replace("Z","+00:00"))
        captured=datetime.fromisoformat(str(event["source_observed_at_utc"]).replace("Z","+00:00"))
        kickoff=kickoff.astimezone(timezone.utc)
        captured=captured.astimezone(timezone.utc)
    except (KeyError,TypeError,ValueError,AttributeError):
        return None
    if not captured<=now or captured>=kickoff-timedelta(minutes=5) or kickoff<=now:
        return None
    sample=raw.get("sample") if isinstance(raw.get("sample"),dict) else {}
    min_n=sample.get("minimum_split_sample")
    if type(min_n) is not int or min_n<5:
        return None
    fields=(
        "raw_home_win_prob","raw_draw_prob","raw_away_win_prob",
        "raw_btts_yes_prob","raw_over_1_5_prob","raw_over_2_5_prob","raw_over_3_5_prob",
        "raw_home_goal_rate","raw_away_goal_rate",
    )
    probs={}
    for name in fields:
        value=raw.get(name)
        if type(value) not in (int,float):return None
        if name.endswith("_prob") and not 0<=value<=1:return None
        if name.endswith("_rate") and not 0<=value<=10:return None
        probs[name]=round(float(value),6)
    if abs(sum(probs[k] for k in ("raw_home_win_prob","raw_draw_prob","raw_away_win_prob"))-1)>0.003:
        return None
    return {
        "fixture_id":fx["fixture_id"],
        "kickoff_utc":kickoff.isoformat(),
        "captured_at_utc":captured.isoformat(),
        "sport_source":"API_FOOTBALL_TEAM_SEASON_SPLITS",
        "model_version":raw.get("model_version"),
        "projection_type":"LIMITED_GOAL_RATE_BASELINE_NOT_XG",
        "min_split_sample":min_n,
        "raw_probabilities":probs,
        "model_validation":"NOT_VERIFIED_OOS",
        "source_updated_at":"NOT_VERIFIED",
        "market_queried":False,
        "bet_eligible":False,
        "direct_executable_quote_verified":False,
        "commercial_display_allowed":False,
    }
