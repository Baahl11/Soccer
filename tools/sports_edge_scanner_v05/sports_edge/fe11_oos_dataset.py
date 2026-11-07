"""FE-11 Phase 2: strict, as-of, training-only Soccer multi-market rows.

Research dataset; NOT an estimate of joint probabilities, a bookmaker quote,
market value, or a permission to bet. No provider calls or odds inputs.
All source-available times must be independently traceable. A late backfill
cannot silently become a feature that was known before kickoff.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from statistics import mean
from typing import Any

from .multi_market_sport_profile import build_multimarket_sport_profile

VERSION = "SOCCER_FE11_OOS_DATASET_v0.1"
CORNER_SOURCE = "API_FOOTBALL_MATCH_TOTALS_TWO_TEAMS"
FORBIDDEN = frozenset({
    "odds", "price", "bookmaker", "market_price", "bet", "stake",
    "breakeven_probability", "market_probability", "estimated_ev",
})


def _utc(v: Any) -> datetime | None:
    if not isinstance(v, str):
        return None
    try:
        dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc) if dt.tzinfo is not None else None
    except ValueError:
        return None


def _id(v: Any) -> int | None:
    return v if type(v) is int and 0 < v < 10**12 else None


def _score_pair(v: Any) -> tuple[int, int] | None:
    if not isinstance(v, dict):
        return None
    a, b = v.get("home"), v.get("away")
    return (a, b) if all(type(z) is int and 0 <= z <= 40 for z in (a, b)) else None


def _verified_game(game: Any, research_cutoff: datetime) -> dict[str, Any] | None:
    if not isinstance(game, dict):
        return None
    if set(game) & FORBIDDEN:
        raise ValueError("MARKET_DATA_CONTAMINATES_SPORT_FIRST")
    fixture_id, home, away = (_id(game.get(x)) for x in
                              ("fixture_id", "home_team_id", "away_team_id"))
    kickoff, known = _utc(game.get("kickoff")), _utc(game.get("source_available_at_utc"))
    goals = _score_pair(game.get("goals"))
    if (fixture_id is None or home is None or away is None or home == away
            or kickoff is None or known is None or known > research_cutoff
            or kickoff >= research_cutoff or known < kickoff + timedelta(minutes=90)
            or game.get("status") != "FT" or goals is None):
        return None
    half = _score_pair(game.get("first_half_goals"))
    second = _score_pair(game.get("second_half_goals"))
    if half is not None:
        if any(half[i] > goals[i] for i in (0, 1)):
            return None
        expected = (goals[0] - half[0], goals[1] - half[1])
        if second is not None and second != expected:
            return None
        second = expected
    elif second is not None:
        # One undocumented interval must not be used to infer another.
        return None
    return {
        "fixture_id": fixture_id, "home_team_id": home, "away_team_id": away,
        "kickoff": kickoff.isoformat(), "source_available_at_utc": known.isoformat(),
        "status": "FT", "goals": {"home": goals[0], "away": goals[1]},
        "first_half_goals": ({"home": half[0], "away": half[1]} if half else None),
        "second_half_goals": ({"home": second[0], "away": second[1]} if second else None),
        "league_id": _id(game.get("league_id")),
        "season": game.get("season") if type(game.get("season")) is int else None,
    }


def _verified_corners(raw: Any, kickoff: datetime,
                      cutoff: datetime) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    if set(raw) & FORBIDDEN:
        raise ValueError("MARKET_DATA_CONTAMINATES_SPORT_FIRST")
    known = _utc(raw.get("source_available_at_utc"))
    pair = _score_pair(raw)
    if (raw.get("period") != "FT_90" or raw.get("source") != CORNER_SOURCE
            or pair is None or known is None
            or not kickoff + timedelta(minutes=90) <= known <= cutoff):
        return None
    return {"home": pair[0], "away": pair[1],
            "period": "FT_90", "source": CORNER_SOURCE,
            "source_available_at_utc": known.isoformat()}


def _outcome(goal_ok: bool, corner_ok: bool, margin_plus_one: int,
             settlement: str) -> str:
    """Component intersection; an unrelated losing leg overrides a handicap push.

    A surviving Asian push is not treated as a winning builder, because
    sportsbook builder repricing/void rules have not been authenticated.
    """
    if not goal_ok or not corner_ok or margin_plus_one < 0:
        return "LOSS"
    if margin_plus_one > 0:
        return "WIN"
    return "PUSH_REQUIRES_BOOKMAKER_RULE" if settlement == "ASIAN" else "LOSS"


def _past_features(games: list[dict[str, Any]], corners: dict[int, dict[str, Any]],
                   team_id: int, venue: str, target_kickoff: str,
                   max_history: int) -> tuple[dict[str, Any], dict[str, Any]]:
    kick = _utc(target_kickoff)
    past = [g for g in games if team_id in (g["home_team_id"], g["away_team_id"])
            and _utc(g["source_available_at_utc"]) < kick
            and _utc(g["kickoff"]) < kick]
    past.sort(key=lambda g: (g["kickoff"], g["fixture_id"]), reverse=True)
    past = past[:max_history]
    available = {g["fixture_id"]: corners[g["fixture_id"]]
                 for g in past if g["fixture_id"] in corners
                 and _utc(corners[g["fixture_id"]]["source_available_at_utc"]) < kick}
    profile = build_multimarket_sport_profile(
        past, team_id=team_id, target_kickoff=target_kickoff,
        corners_by_fixture=available)
    all_s, venue_s = profile["splits"]["all"], profile["splits"][venue]
    vf = venue_s["features"]
    return ({
        "prior_ft_all": all_s["matches_with_valid_ft"],
        "prior_ft_venue": venue_s["matches_with_valid_ft"],
        "prior_corners_venue": vf["total_corners_ft"]["n"],
        "prior_period_venue": vf["first_half_goals_total"]["n"],
        "goals_for_ft": vf["goals_for"]["mean"],
        "goals_against_ft": vf["goals_against"]["mean"],
        "goals_1h_total": vf["first_half_goals_total"]["mean"],
        "goals_2h_total": vf["second_half_goals_total"]["mean"],
        "corners_for_ft": vf["team_corners_ft"]["mean"],
        "corners_against_ft": vf["opponent_corners_ft"]["mean"],
        "second_half_goal_share_scored": venue_s["period_goals_share_scored_second_half"],
    }, profile)


def build_fe11_oos_dataset(
    fixtures: list[dict[str, Any]],
    corners_by_fixture: dict[int, dict[str, Any]],
    *,
    research_cutoff_utc: str,
    max_history: int = 20,
    min_prior_ft: int = 8,
    min_prior_venue: int = 4,
    min_prior_corners: int = 4,
    goals_over: float = 1.5,
    corners_over: float = 6.5,
) -> dict[str, Any]:
    """Build immutable labels + provably prior sport features for chronological OOS.

    Fixture metadata must include *source_available_at_utc* (when final
    information became available, not the time it was backfilled).
    Corners require independently verified availability timestamps and both
    sides' FT_90 data; missing is never zero. Return NO calibrated probabilities.
    """
    cutoff = _utc(research_cutoff_utc)
    if (cutoff is None or not isinstance(fixtures, list)
            or not isinstance(corners_by_fixture, dict)
            or type(max_history) is not int or not 8 <= max_history <= 100
            or any(type(x) is not int or x < 1 for x in
                   (min_prior_ft, min_prior_venue, min_prior_corners))
            or not (0 < min_prior_ft <= max_history)
            or not (0 < min_prior_venue <= max_history)
            or not (0 < min_prior_corners <= max_history)
            or goals_over < 0 or corners_over < 0
            or goals_over % 1 != .5 or corners_over % 1 != .5):
        raise ValueError("FE11_DATASET_CONFIGURATION_UNVERIFIED")
    verified, ids = [], set()
    excluded = {"invalid_or_unavailable_fixture": 0, "duplicate_fixture": 0,
                "missing_or_unavailable_corners": 0,
                "insufficient_asof_history": 0}
    for raw in fixtures:
        g = _verified_game(raw, cutoff)
        if g is None:
            excluded["invalid_or_unavailable_fixture"] += 1
            continue
        if g["fixture_id"] in ids:
            excluded["duplicate_fixture"] += 1
            continue
        ids.add(g["fixture_id"])
        verified.append(g)
    verified.sort(key=lambda g: (g["kickoff"], g["fixture_id"]))
    corners: dict[int, dict[str, Any]] = {}
    for g in verified:
        fc = _verified_corners(corners_by_fixture.get(g["fixture_id"]),
                               _utc(g["kickoff"]), cutoff)
        if fc is not None:
            corners[g["fixture_id"]] = fc

    output = []
    for g in verified:
        fid = g["fixture_id"]
        if fid not in corners:
            excluded["missing_or_unavailable_corners"] += 1
            continue
        home, _ = _past_features(verified, corners, g["home_team_id"],
                                 "home", g["kickoff"], max_history)
        away, _ = _past_features(verified, corners, g["away_team_id"],
                                 "away", g["kickoff"], max_history)
        if any(s["prior_ft_all"] < min_prior_ft
               or s["prior_ft_venue"] < min_prior_venue
               or s["prior_corners_venue"] < min_prior_corners
               or s["prior_period_venue"] < min_prior_venue
               for s in (home, away)):
            excluded["insufficient_asof_history"] += 1
            continue
        goals = g["goals"]["home"] + g["goals"]["away"]
        cs = corners[fid]
        total_corners = cs["home"] + cs["away"]
        goal_hit, corner_hit = goals > goals_over, total_corners > corners_over
        margin = g["goals"]["home"] - g["goals"]["away"]
        halftime = g["first_half_goals"]
        second = g["second_half_goals"]
        labels = {
            "ft_goals": goals, "ft_corners": total_corners,
            "first_half_goals": sum(halftime.values()) if halftime else None,
            "second_half_goals": sum(second.values()) if second else None,
            "home_margin": margin,
            "goals_over_hit": goal_hit, "corners_over_hit": corner_hit,
            "home_three_way_plus_one_joint": _outcome(goal_hit, corner_hit, margin + 1, "THREE_WAY"),
            "away_three_way_plus_one_joint": _outcome(goal_hit, corner_hit, 1 - margin, "THREE_WAY"),
            "home_asian_plus_one_joint": _outcome(goal_hit, corner_hit, margin + 1, "ASIAN"),
            "away_asian_plus_one_joint": _outcome(goal_hit, corner_hit, 1 - margin, "ASIAN"),
            "home_asian_plus_one_leg": ("WIN" if margin + 1 > 0 else "PUSH" if margin + 1 == 0 else "LOSS"),
            "away_asian_plus_one_leg": ("WIN" if 1 - margin > 0 else "PUSH" if 1 - margin == 0 else "LOSS"),
        }
        output.append({
            "fixture_id": fid, "league_id": g["league_id"], "season": g["season"],
            "kickoff_utc": g["kickoff"], "home_team_id": g["home_team_id"],
            "away_team_id": g["away_team_id"],
            "target_source_available_at_utc": g["source_available_at_utc"],
            "target_corners_available_at_utc": cs["source_available_at_utc"],
            "features": {**{"home_" + k: v for k, v in home.items()},
                         **{"away_" + k: v for k, v in away.items()}},
            "labels": labels,
        })
    return {
        "version": VERSION,
        "research_cutoff_utc": cutoff.isoformat(),
        "lines": {"goals_over": goals_over, "corners_over": corners_over,
                  "handicap": 1},
        "source_provenance_required": True,
        "market_inputs_used": False, "calibrated_joint_probability": None,
        "bet_eligible": False, "classification": "RESEARCH_ONLY",
        "verified_ft_fixtures": len(verified), "eligible_rows": len(output),
        "excluded": excluded, "rows": output,
    }


def chronological_holdout(dataset: dict[str, Any], *, cutoff_utc: str,
                          embargo_hours: int = 72) -> dict[str, Any]:
    """Separate rows by kickoff; training labels must be published pre-cutoff.

    Never split a shared kickoff across train and test. This is a fold
    manifest, NOT trained/validated model output.
    """
    cutoff = _utc(cutoff_utc)
    if cutoff is None or type(embargo_hours) is not int or not 0 <= embargo_hours <= 720:
        raise ValueError("OOS_CUTOFF_INVALID")
    if dataset.get("version") != VERSION or dataset.get("market_inputs_used") is not False:
        raise ValueError("UNTRUSTED_DATASET")
    gap = cutoff - timedelta(hours=embargo_hours)
    rows = dataset.get("rows") or []
    train = [r["fixture_id"] for r in rows
             if _utc(r["kickoff_utc"]) < gap
             and _utc(r["target_source_available_at_utc"]) < gap
             and _utc(r["target_corners_available_at_utc"]) < gap]
    test = [r["fixture_id"] for r in rows if _utc(r["kickoff_utc"]) >= cutoff]
    return {
        "version": VERSION, "cutoff_utc": cutoff.isoformat(),
        "embargo_hours": embargo_hours,
        "training_fixture_ids": train, "holdout_fixture_ids": test,
        "n_train": len(train), "n_holdout": len(test),
        "calibration_status": "NOT_TRAINED",
        "bet_eligible": False,
    }
