from __future__ import annotations

from typing import Any

from mcp_gateway import soccer_model as legacy

MODEL_VERSION = "SOCCER EDGE ENGINE v1.7"
PROJECTION_MODEL = "LEAGUE_RELATIVE_STRENGTH_POISSON_v0.2"
PRIOR_GAMES = 5.0
MIN_SPLIT_GAMES = 3


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _nested(d: Any, *keys: str) -> Any:
    cur = d
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _rate(stats: dict[str, Any], side: str, which: str) -> float | None:
    return _num(_nested(stats, "goals", which, "average", side))


def _played(stats: dict[str, Any], side: str) -> int:
    try:
        return int(_nested(stats, "fixtures", "played", side) or 0)
    except (TypeError, ValueError):
        return 0


def _recent_rates(matches: list[dict[str, Any]], team_id: int | None) -> tuple[float | None, float | None, int]:
    return legacy._recent_rates(matches, team_id)


def _blend(primary: float | None, recent: float | None, recent_n: int) -> float | None:
    return legacy._blend(primary, recent, recent_n)


def _clamp(value: float, lo: float = 0.15, hi: float = 4.5) -> float:
    return max(lo, min(hi, value))


def _league_baselines(home_stats: dict[str, Any], away_stats: dict[str, Any]) -> tuple[float | None, float | None, str]:
    # API-Football team statistics do not expose a league-wide scoring table.
    # Estimate the league environment only from complementary venue rates that
    # are already verified for the two teams; do not fabricate a universal prior.
    h_for = _rate(home_stats, "home", "for")
    h_against = _rate(home_stats, "home", "against")
    a_for = _rate(away_stats, "away", "for")
    a_against = _rate(away_stats, "away", "against")
    home_components = [x for x in (h_for, a_against) if x is not None and x > 0]
    away_components = [x for x in (a_for, h_against) if x is not None and x > 0]
    if len(home_components) != 2 or len(away_components) != 2:
        return None, None, "MISSING_COMPLEMENTARY_VENUE_RATES"
    return (
        sum(home_components) / len(home_components),
        sum(away_components) / len(away_components),
        "PAIR_COMPLEMENTARY_VENUE_BASELINE",
    )


def build_raw_projection(
    fx: dict[str, Any], sporting: dict[str, Any], availability_confidence: float | None = None
) -> dict[str, Any]:
    home_stats = sporting.get("home_stats") or {}
    away_stats = sporting.get("away_stats") or {}
    home_recent = sporting.get("home_recent") or []
    away_recent = sporting.get("away_recent") or []

    h_for = _rate(home_stats, "home", "for")
    h_against = _rate(home_stats, "home", "against")
    a_for = _rate(away_stats, "away", "for")
    a_against = _rate(away_stats, "away", "against")
    values = {
        "home_home_scoring_rate": h_for,
        "home_home_concede_rate": h_against,
        "away_away_scoring_rate": a_for,
        "away_away_concede_rate": a_against,
    }
    missing = [k for k, v in values.items() if v is None]
    home_played = _played(home_stats, "home")
    away_played = _played(away_stats, "away")
    if missing or min(home_played, away_played) < MIN_SPLIT_GAMES:
        raw = legacy.build_raw_projection(fx, sporting, availability_confidence)
        if isinstance(raw, dict):
            raw = dict(raw)
            raw["relative_strength_status"] = "INSUFFICIENT_VERIFIED_SPLIT_SAMPLE"
            raw["relative_strength_missing"] = missing
            raw["relative_strength_min_split_games"] = MIN_SPLIT_GAMES
        return raw

    league_home, league_away, baseline_source = _league_baselines(home_stats, away_stats)
    if not league_home or not league_away:
        return legacy.build_raw_projection(fx, sporting, availability_confidence)

    hrf, hra, hrn = _recent_rates(home_recent, fx.get("home_team_id"))
    arf, ara, arn = _recent_rates(away_recent, fx.get("away_team_id"))
    h_attack_rate = _blend(h_for, hrf, hrn) or h_for
    h_concede_rate = _blend(h_against, hra, hrn) or h_against
    a_attack_rate = _blend(a_for, arf, arn) or a_for
    a_concede_rate = _blend(a_against, ara, arn) or a_against

    # Shrink venue rates toward the verified pair baseline. This changes the
    # legacy arithmetic mean into multiplicative attack x opponent-defense
    # strengths while keeping total scale grounded in observed venue scoring.
    h_attack_rate = (home_played * h_attack_rate + PRIOR_GAMES * league_home) / (home_played + PRIOR_GAMES)
    h_concede_rate = (home_played * h_concede_rate + PRIOR_GAMES * league_away) / (home_played + PRIOR_GAMES)
    a_attack_rate = (away_played * a_attack_rate + PRIOR_GAMES * league_away) / (away_played + PRIOR_GAMES)
    a_concede_rate = (away_played * a_concede_rate + PRIOR_GAMES * league_home) / (away_played + PRIOR_GAMES)

    h_attack_strength = h_attack_rate / league_home
    h_def_weakness = h_concede_rate / league_away
    a_attack_strength = a_attack_rate / league_away
    a_def_weakness = a_concede_rate / league_home
    home_lam = _clamp(league_home * h_attack_strength * a_def_weakness)
    away_lam = _clamp(league_away * a_attack_strength * h_def_weakness)

    probs = legacy._probabilities(home_lam, away_lam)
    scores = legacy._quality_scores(home_lam, away_lam, probs, availability_confidence)
    total = home_lam + away_lam
    weaker = min(home_lam, away_lam)
    scoring_path = (
        "TWO-WAY OPEN GAME" if total >= 2.85 and weaker >= 1.05 else
        "FAVORITE-CARRY OVER" if total >= 2.85 and max(home_lam, away_lam) >= 2.0 else
        "MUTUAL SUPPRESSION UNDER" if total <= 2.05 and weaker <= 0.75 else
        "MIXED / NO STRONG SCORING PATH"
    )

    return {
        "status": "MODELED_LIMITED",
        "model_version": MODEL_VERSION,
        "projection_model": PROJECTION_MODEL,
        "model_limitations": [
            "Relative-strength fallback uses verified API-Football venue goal rates and recent final goals only.",
            "Pair complementary venue rates are a temporary scoring-environment baseline, not a true league-wide table.",
            "1X2 remains research-only; this challenger cannot upgrade BET/LEAN.",
            "xG/npxG/PPDA/field tilt are not fabricated or inferred from goals.",
        ],
        "advanced_metrics": "NOT VERIFIED",
        "relative_strength_status": "RESEARCH_CHALLENGER_ACTIVE",
        "relative_strength_baseline_source": baseline_source,
        "sample": {
            "home_home_played": home_played,
            "away_away_played": away_played,
            "minimum_split_sample": min(home_played, away_played),
            "shrinkage_prior_games": PRIOR_GAMES,
        },
        "strengths": {
            "home_attack": round(h_attack_strength, 4),
            "home_defensive_weakness": round(h_def_weakness, 4),
            "away_attack": round(a_attack_strength, 4),
            "away_defensive_weakness": round(a_def_weakness, 4),
            "baseline_home_goals": round(league_home, 4),
            "baseline_away_goals": round(league_away, 4),
        },
        "raw_home_goal_rate": round(home_lam, 4),
        "raw_away_goal_rate": round(away_lam, 4),
        "raw_total_goals": round(total, 4),
        "raw_home_xg": "NOT VERIFIED",
        "raw_away_xg": "NOT VERIFIED",
        "raw_home_win_prob": round(probs["home_win"], 6),
        "raw_draw_prob": round(probs["draw"], 6),
        "raw_away_win_prob": round(probs["away_win"], 6),
        "raw_btts_yes_prob": round(probs["btts_yes"], 6),
        "raw_over_1_5_prob": round(probs["over_1_5"], 6),
        "raw_over_2_5_prob": round(probs["over_2_5"], 6),
        "raw_over_3_5_prob": round(probs["over_3_5"], 6),
        "top_scorelines": probs["top_scores"],
        "scoring_path": scoring_path,
        "screen_scores": scores,
        "_total_dist": probs["total_dist"],
    }


def public_raw_projection(raw: dict[str, Any]) -> dict[str, Any]:
    return legacy.public_raw_projection(raw)
