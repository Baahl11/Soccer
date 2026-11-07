"""SPORT-FIRST descriptive inputs for correlated multi-market soccer builders.

Research-only: no sportsbook, no market prices, no future games, no betting
recommendations, no calibrated joint probability, no hypothetical SGP price.

Uses documented 90-minute FT and halftime scores; corners require both clubs'
same-fixture validated full-time statistical records. Missing is never zero.
"""
from __future__ import annotations

from datetime import datetime, timezone
from statistics import mean
from typing import Any

VERSION = "SOCCER_SPORT_FIRST_MULTIMARKET_PROFILE_v1.0"
MIN_DESCRIPTIVE_SAMPLE = 8


def _utc(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else None
    if not isinstance(value, str):
        return None
    try:
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return value.astimezone(timezone.utc) if value.tzinfo else None
    except ValueError:
        return None


def _count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if 0 <= value <= 40 else None


def _scores(period: Any) -> tuple[int, int] | None:
    if not isinstance(period, dict):
        return None
    h, a = _count(period.get("home")), _count(period.get("away"))
    return (h, a) if h is not None and a is not None else None


def _observed(vals: list[int], cutoff: float | None = None) -> dict[str, Any]:
    n = len(vals)
    return {
        "n": n,
        "mean": round(mean(vals), 3) if n else None,
        "over_line_frequency_descriptive": (
            round(sum(v > cutoff for v in vals) / n, 4)
            if n and cutoff is not None else None
        ),
        "line": cutoff,
        "model_probability": None,
        "minimum_descriptive_sample_met": n >= MIN_DESCRIPTIVE_SAMPLE,
    }


def build_multimarket_sport_profile(
    historical_fixtures: list[dict[str, Any]],
    *,
    team_id: int,
    target_kickoff: str,
    corners_by_fixture: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Summarize past verified FT observations, split by team venue/period.

    corners_by_fixture example:
        {fixture_id: {"home": 4, "away": 5, "period": "FT_90"}}
    A fixture with incomplete or uncertain-period stats contributes **no**
    corner datapoint, but can still contribute independently valid goals.
    """
    target = _utc(target_kickoff)
    if target is None or type(team_id) is not int or team_id <= 0:
        raise ValueError("TARGET_KICKOFF_OR_TEAM_ID_NOT_VERIFIED")
    corners_by_fixture = corners_by_fixture or {}
    observations: list[dict[str, Any]] = []
    used: set[int] = set()
    excluded = {"not_ft": 0, "missing_score": 0, "future_or_invalid": 0,
                "team_identity": 0, "duplicate": 0}
    for game in historical_fixtures:
        if not isinstance(game, dict):
            excluded["future_or_invalid"] += 1
            continue
        fid = game.get("fixture_id")
        kickoff = _utc(game.get("kickoff"))
        if type(fid) is not int or fid <= 0 or kickoff is None or kickoff >= target:
            excluded["future_or_invalid"] += 1
            continue
        if fid in used:
            excluded["duplicate"] += 1
            continue
        if game.get("status") != "FT":
            excluded["not_ft"] += 1
            continue
        home_id, away_id = game.get("home_team_id"), game.get("away_team_id")
        if (type(home_id) is not int or type(away_id) is not int
                or home_id == away_id or team_id not in (home_id, away_id)):
            excluded["team_identity"] += 1
            continue
        ft = _scores(game.get("goals"))
        if ft is None:
            excluded["missing_score"] += 1
            continue
        side = "home" if home_id == team_id else "away"
        idx = 0 if side == "home" else 1
        ht = _scores(game.get("first_half_goals"))
        sh = _scores(game.get("second_half_goals"))
        if ht is not None and (ht[0] > ft[0] or ht[1] > ft[1]):
            ht = None
        if ht is not None:
            derived = (ft[0] - ht[0], ft[1] - ht[1])
            if sh is not None and sh != derived:
                # The two supplied period scores disagree; quarantine both.
                ht, sh = None, None
            else:
                sh = derived
        else:
            sh = None
        cs = corners_by_fixture.get(fid)
        total_corners = None
        team_corners = None
        opponent_corners = None
        if isinstance(cs, dict) and cs.get("period") == "FT_90":
            ch, ca = _count(cs.get("home")), _count(cs.get("away"))
            if ch is not None and ca is not None:
                total_corners = ch + ca
                team_corners, opponent_corners = (ch, ca) if idx == 0 else (ca, ch)
        used.add(fid)
        observations.append({
            "fixture_id": fid,
            "kickoff": kickoff.isoformat(),
            "venue": side,
            "goals_for": ft[idx],
            "goals_against": ft[1-idx],
            "ft_goals_total": sum(ft),
            "first_half_goals_for": ht[idx] if ht is not None else None,
            "first_half_goals_against": ht[1-idx] if ht is not None else None,
            "first_half_goals_total": sum(ht) if ht is not None else None,
            "second_half_goals_for": sh[idx] if sh is not None else None,
            "second_half_goals_against": sh[1-idx] if sh is not None else None,
            "second_half_goals_total": sum(sh) if sh is not None else None,
            "team_corners_ft": team_corners,
            "opponent_corners_ft": opponent_corners,
            "total_corners_ft": total_corners,
        })
    observations.sort(key=lambda x: x["kickoff"], reverse=True)

    splits: dict[str, Any] = {}
    for venue in ("all", "home", "away"):
        subset = [r for r in observations if venue == "all" or r["venue"] == venue]
        fields = {}
        for key in (
            "goals_for", "goals_against", "ft_goals_total",
            "first_half_goals_for", "first_half_goals_against",
            "first_half_goals_total", "second_half_goals_for",
            "second_half_goals_against", "second_half_goals_total",
            "team_corners_ft", "opponent_corners_ft", "total_corners_ft",
        ):
            vals = [r[key] for r in subset if r[key] is not None]
            threshold = {
                "ft_goals_total": 1.5,
                "first_half_goals_total": 1.5,
                "second_half_goals_total": 1.5,
                "total_corners_ft": 6.5,
            }.get(key)
            fields[key] = _observed(vals, threshold)
        h = [r for r in subset if r["first_half_goals_for"] is not None
             and r["second_half_goals_for"] is not None]
        first_total = sum(r["first_half_goals_for"] for r in h)
        second_total = sum(r["second_half_goals_for"] for r in h)
        both = first_total + second_total
        splits[venue] = {
            "matches_with_valid_ft": len(subset),
            "period_goals_share_scored_second_half": (
                round(second_total / both, 4) if both else None),
            "second_half_goal_share_sample_fixtures": len(h),
            "features": fields,
        }

    return {
        "version": VERSION,
        "team_id": team_id,
        "target_kickoff": target.isoformat(),
        "evidence_scope": "PAST_VERIFIED_FT_ONLY",
        "sports_first_no_market_inputs": True,
        "descriptive_only": True,
        "calibrated_prediction": False,
        "bet_eligible": False,
        "excluded": excluded,
        "splits": splits,
        "observations": observations,
        "limitations": [
            "DESCRIPTIVE_NOT_OPPONENT_ADJUSTED",
            "HALF_SCORING_TRENDS_NOT_A_PROSPECTIVE_PROBABILITY",
            "FULL_TIME_CORNERS_REQUIRE_BOTH_TEAM_STATS_FT90",
            "SGP_MULTIMARKET_JOINT_PROBABILITY_NOT_CALIBRATED",
            "HANDICAP_SETTLEMENT_TYPE_MUST_BE_VERIFIED",
        ],
    }


def describe_three_leg_historical_pattern(
    profile: dict[str, Any],
    *,
    goals_over: float,
    corners_over: float,
    team_handicap: int = 1,
    handicap_settlement: str,
    venue: str = "all",
) -> dict[str, Any]:
    """Historical intersection of three legs, with explicit handicap rules.

    These are observed prior-game frequencies **not future joint predictions**.
    On Asian +1, a one-goal loss pushes that leg; another losing leg still
    defeats the joint condition. Otherwise builder push terms remain unknown.
    Three-way (+1) has a handicap draw outcome and does NOT push.
    Unknown settlement => refuse pattern evaluation.
    """
    if (handicap_settlement not in ("ASIAN", "THREE_WAY")
            or venue not in ("all", "home", "away")
            or goals_over < 0 or corners_over < 0
            or goals_over % 1 != .5 or corners_over % 1 != .5
            or type(team_handicap) is not int):
        raise ValueError("HANDICAP_RULE_OR_EXACT_LINE_NOT_VERIFIED")
    if profile.get("sports_first_no_market_inputs") is not True:
        raise ValueError("SPORT_FIRST_PROFILE_NOT_VERIFIED")
    rows = [r for r in profile.get("observations", [])
            if venue == "all" or r.get("venue") == venue]
    complete = [r for r in rows if r.get("total_corners_ft") is not None
                and type(r.get("ft_goals_total")) is int
                and type(r.get("goals_for")) is int
                and type(r.get("goals_against")) is int]
    hits, misses, pushes, asian_leg_pushes = 0, 0, 0, 0
    for r in complete:
        gd_after_handicap = r["goals_for"] - r["goals_against"] + team_handicap
        goal_hit = r["ft_goals_total"] > goals_over
        corner_hit = r["total_corners_ft"] > corners_over
        if gd_after_handicap == 0 and handicap_settlement == "ASIAN":
            # A push on *one leg* does not rescue an already-losing builder.
            # Only surviving legs require a sportsbook-specific push/void rule.
            asian_leg_pushes += 1
            if not goal_hit or not corner_hit:
                misses += 1
            else:
                pushes += 1
            continue
        handicap_hit = gd_after_handicap > 0
        if goal_hit and corner_hit and handicap_hit:
            hits += 1
        else:
            misses += 1
    valid = hits + misses
    return {
        "version": VERSION,
        "venue": venue,
        "legs": [
            {"type": "FT_GOALS_OVER", "line": goals_over},
            {"type": "FT_CORNERS_OVER", "line": corners_over},
            {"type": "TEAM_HANDICAP", "line": team_handicap,
             "settlement": handicap_settlement},
        ],
        "complete_historical_rows": len(complete),
        "excluded_missing_corners_or_results": len(rows) - len(complete),
        "resolved_without_push": valid,
        "observed_joint_hits": hits,
        "observed_joint_misses": misses,
        "asian_handicap_push_records": pushes,
        "asian_handicap_leg_push_records": asian_leg_pushes,
        "settlement_semantics_patch": "FE11_1.0.1_LOSING_LEG_OVERRIDES_PUSH",
        "observed_joint_hit_frequency": round(hits / valid, 4) if valid else None,
        "eligible_as_joint_model_probability": False,
        "price_decimal": None,
        "correlated_sgp_quote": None,
        "bet_eligible": False,
        "settlement_warning": (
            "ASIAN_PUSH_REQUIRES_BOOKMAKER_BET_BUILDER_SETTLEMENT_RULE"
            if pushes else None),
        "limitations": [
            "DESCRIPTIVE_SHARED_FIXTURE_SAMPLE_ONLY",
            "NOT_OPPONENT_ADJUSTED_OR_OOS_CALIBRATED",
            "NO_BOOKMAKER_TICKET_QUOTE_OR_JOINT_EV",
        ],
    }
