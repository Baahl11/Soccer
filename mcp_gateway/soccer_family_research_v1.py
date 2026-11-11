"""Independent soccer SPORT-FIRST family forecasts; no betting/market access.

These are *limited* statistical baselines, not fitted/calibrated models.
Missing source features are never imputed as zero, and results from later than
kickoff cannot be used as prior features. Frozen raw probabilities are
immutable inputs to later market analysis; this module never sees odds.
"""
from __future__ import annotations
from datetime import datetime, timezone, timedelta
import math
from typing import Any

VERSION = "SOCCER_FAMILY_RESEARCH_v1.0"
MARKETS = (
    "1X2", "FT_TOTALS", "BTTS", "TEAM_TOTALS", "1H_1X2",
    "1H_TOTALS", "2H_1X2", "2H_TOTALS", "FT_CORNERS",
    "TEAM_CORNERS", "FT_CARDS", "PLAYER_SHOTS",
)
STATUSES = {"MODELED_LIMITED", "NOT_MODELED_DATA_MISSING", "NOT_VERIFIED_ASOF"}
MIN_SAMPLE = 5


def _time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return value.astimezone(timezone.utc) if value.tzinfo else None


def _rate(x: Any, maximum: float = 30.0) -> float | None:
    if type(x) not in (float, int) or not math.isfinite(x) or x < 0 or x > maximum:
        return None
    return float(x)


def _poisson(lam: float, maximum: int = 60) -> list[float]:
    # Stable recurrence, tail folded into last bucket. All lines below 30.
    if not 0 <= lam <= 30:
        raise ValueError("INVALID_RATE")
    probs = [math.exp(-lam)]
    for i in range(1, maximum):
        probs.append(probs[-1] * lam / i)
    probs.append(max(0.0, 1.0 - sum(probs)))
    total = sum(probs)
    return [x / total for x in probs]


def _over(lam: float, line: float) -> float:
    if line not in (0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5):
        raise ValueError("UNSUPPORTED_BINARY_LINE")
    return sum(_poisson(lam)[int(line + 0.5):])


def _win_probs(h: float, a: float) -> dict[str, float]:
    hp, ap = _poisson(h), _poisson(a)
    win = sum(x * sum(ap[:i]) for i, x in enumerate(hp))
    draw = sum(x * ap[i] for i, x in enumerate(hp))
    away = max(0.0, 1 - win - draw)
    total = win + draw + away
    return {"home": win / total, "draw": draw / total, "away": away / total}


def _result(status: str, reason: str, **kwargs: Any) -> dict[str, Any]:
    return {
        "status": status,
        "reason_code": reason,
        "projection_version": VERSION,
        "classification": "RESEARCH_ONLY",
        "bet_eligible": False,
        "oos_calibrated": False,
        "commercial_market_release": False,
        **kwargs,
    }


def _binary_family(lam: float, lines: tuple[float, ...], *, descriptor: str) -> dict[str, Any]:
    selections = {}
    for line in lines:
        p = _over(lam, line)
        selections[f"over_{line:g}"] = round(p, 8)
        selections[f"under_{line:g}"] = round(1.0 - p, 8)
    return _result("MODELED_LIMITED", "INDEPENDENT_POISSON_BASELINE_NOT_CALIBRATED",
                   expected_count=round(lam, 6), selection_probabilities=selections,
                   raw_basis=descriptor)


def _pair(node: Any, minimum_n: int = MIN_SAMPLE) -> tuple[float, float] | None:
    if not isinstance(node, dict) or type(node.get("min_sample")) is not int:
        return None
    if node["min_sample"] < minimum_n:
        return None
    left, right = _rate(node.get("home_rate")), _rate(node.get("away_rate"))
    return (left, right) if left is not None and right is not None else None


def _team_rates(node: Any) -> tuple[float, float] | None:
    if not isinstance(node, dict) or type(node.get("min_sample")) is not int or node["min_sample"] < MIN_SAMPLE:
        return None
    for key in ("home_for", "home_against", "away_for", "away_against"):
        if _rate(node.get(key)) is None:
            return None
    return ((_rate(node["home_for"]) + _rate(node["away_against"])) / 2,
            (_rate(node["away_for"]) + _rate(node["home_against"])) / 2)


def build_family_forecasts(
    fixture: dict[str, Any],
    source: dict[str, Any],
    *,
    frozen_at_utc: str,
) -> dict[str, Any]:
    """Freeze independent research predictions with every family's blockers.

    source has explicit *_observed_at_utc and source_updated_at_utc; no guess.
    A source updated after freeze or a freeze after kickoff is rejected.
    """
    if type(fixture.get("fixture_id")) is not int or fixture["fixture_id"] <= 0:
        raise ValueError("CANONICAL_FIXTURE_ID_REQUIRED")
    kickoff = _time(fixture.get("kickoff_utc"))
    freeze = _time(frozen_at_utc)
    if kickoff is None or freeze is None or freeze >= kickoff - timedelta(minutes=5):
        raise ValueError("SPORT_FORECAST_NOT_PROSPECTIVE")
    if "odds" in source or "market" in source or "price" in source:
        raise ValueError("SPORT_PROJECTION_MUST_NOT_SEE_MARKET")
    captured = _time(source.get("observed_at_utc"))
    updated = _time(source.get("source_updated_at_utc"))
    valid_source = bool(captured and updated and
                        updated <= captured <= freeze)
    result = {family: _result(
        "NOT_MODELED_DATA_MISSING", "INDEPENDENT_SPORT_INPUTS_NOT_VERIFIED"
    ) for family in MARKETS}
    if not valid_source:
        result = {family: _result("NOT_VERIFIED_ASOF", "SPORT_SOURCE_TIMESTAMPS_NOT_VERIFIED")
                  for family in MARKETS}
    else:
        goals = _pair(source.get("goal_rates"), minimum_n=MIN_SAMPLE)
        if goals:
            h, a = goals
            w = _win_probs(h, a)
            result["1X2"] = _result(
                "MODELED_LIMITED", "GOAL_RATE_POISSON_NOT_CALIBRATED",
                selection_probabilities={k: round(v, 8) for k, v in w.items()},
                expected_home_goals=h, expected_away_goals=a)
            total = _binary_family(h+a, (1.5, 2.5, 3.5), descriptor="GOAL_RATE_TOTAL")
            result["FT_TOTALS"] = total
            yes = (1.0-math.exp(-h)) * (1.0-math.exp(-a))
            result["BTTS"] = _result(
                "MODELED_LIMITED", "INDEPENDENT_GOAL_RATE_NOT_XG",
                selection_probabilities={"yes":round(yes, 8), "no":round(1-yes, 8)})
            result["TEAM_TOTALS"] = _result(
                "MODELED_LIMITED", "INDEPENDENT_TEAM_GOAL_RATE",
                selections_by_team={
                    "home": _binary_family(h,(0.5,1.5,2.5),descriptor="HOME_GOAL_RATE")["selection_probabilities"],
                    "away": _binary_family(a,(0.5,1.5,2.5),descriptor="AWAY_GOAL_RATE")["selection_probabilities"],
                })
        halves = source.get("period_goal_rates")
        if isinstance(halves,dict):
            for period, side_family, total_family in (
                ("1H","1H_1X2","1H_TOTALS"), ("2H","2H_1X2","2H_TOTALS"),
            ):
                rates=_pair(halves.get(period))
                if rates:
                    hh,aa=rates
                    result[side_family]=_result(
                        "MODELED_LIMITED","PERIOD_SPECIFIC_GOAL_RATES_NOT_CALIBRATED",
                        selection_probabilities={k:round(v,8) for k,v in _win_probs(hh,aa).items()})
                    result[total_family]=_binary_family(
                        hh+aa,(0.5,1.5,2.5),descriptor=f"{period}_GOAL_RATE")
        corners=_team_rates(source.get("corner_rates"))
        if corners:
            hc,ac=corners
            result["FT_CORNERS"]=_binary_family(
                hc+ac,(6.5,8.5,9.5,10.5,11.5,12.5),descriptor="CORNER_COUNT_BASELINE_NOT_TERRITORY_MODEL")
            result["TEAM_CORNERS"]=_result(
                "MODELED_LIMITED","CORNER_TEAM_COUNT_BASELINE_NOT_CALIBRATED",
                selections_by_team={
                    "home":_binary_family(hc,(2.5,3.5,4.5,5.5,6.5),descriptor="HOME_CORNERS")["selection_probabilities"],
                    "away":_binary_family(ac,(2.5,3.5,4.5,5.5,6.5),descriptor="AWAY_CORNERS")["selection_probabilities"]})
        cards=_team_rates(source.get("card_rates"))
        if cards and source.get("referee_cards_verified") is True:
            result["FT_CARDS"]=_binary_family(
                sum(cards),(2.5,3.5,4.5,5.5,6.5),descriptor="CARD_COUNT_REFEREE_VERIFIED_LIMITED")
        player=source.get("player_shots")
        if isinstance(player,dict):
            shots=_rate(player.get("shots_per_90"),10)
            minutes=_rate(player.get("expected_minutes"),90)
            if (type(player.get("player_id")) is int and player["player_id"]>0 and
                player.get("xi_confirmed") is True and
                type(player.get("sample_matches")) is int and player["sample_matches"]>=8 and
                shots is not None and minutes is not None and minutes > 0):
                result["PLAYER_SHOTS"]=_binary_family(
                    shots*minutes/90,(0.5,1.5,2.5,3.5),descriptor="CONFIRMED_XI_SHOT_RATE_LIMITED")
                result["PLAYER_SHOTS"]["player_id"]=player["player_id"]
                result["PLAYER_SHOTS"]["estimated_minutes"]=minutes
    return {
        "fixture_id":fixture["fixture_id"], "kickoff_utc":kickoff.isoformat(),
        "frozen_at_utc":freeze.isoformat(), "source_observed_at_utc":source.get("observed_at_utc"),
        "source_updated_at_utc":source.get("source_updated_at_utc"),
        "model_version":VERSION, "source_quality":"VERIFIED_TIMING_ONLY" if valid_source else "NOT_VERIFIED_ASOF",
        "sport_first":True, "market_queried":False, "bet_eligible":False,
        "families":result,
    }
