from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone as dt_timezone
from typing import Any

import httpx

from mcp_gateway import automation as base

SCHEMA_VERSION = "1.0.0"
PLAYER_REGISTRY_URL = "https://raw.githubusercontent.com/Baahl11/Soccer/soccer-edge-state/soccer_edge_state/analysis/player_trend_model_registry.json"
TEAM_TRENDS_URL = "https://raw.githubusercontent.com/Baahl11/Soccer/soccer-edge-state/soccer_edge_state/analysis/trend_intelligence.json"
CACHE_TTL = timedelta(hours=6)
LINES = (0.5, 1.5, 2.5, 3.5, 4.5, 5.5)
OPPONENT_PRIOR_MATCHES = 12.0
OPPONENT_FACTOR_CLIP = (0.75, 1.25)


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_json(cache_namespace: str, url: str, expected_status: str) -> dict[str, Any] | None:
    now = datetime.now(dt_timezone.utc)
    cached = base._cache_get(cache_namespace, "latest", CACHE_TTL, now)
    if isinstance(cached, dict) and cached.get("status") == expected_status:
        return cached
    try:
        response = httpx.get(url, timeout=5.0, follow_redirects=True)
        if response.status_code != 200:
            return None
        payload = response.json()
    except Exception:
        return None
    if not isinstance(payload, dict) or payload.get("status") != expected_status:
        return None
    base._cache_set(cache_namespace, "latest", payload, now)
    return payload


def load_player_registry() -> dict[str, Any] | None:
    return _load_json(
        "shots_player_trend_registry",
        PLAYER_REGISTRY_URL,
        "RESEARCH_PLAYER_TREND_MODEL_REGISTRY",
    )


def load_team_trends() -> dict[str, Any] | None:
    return _load_json(
        "shots_team_trends",
        TEAM_TRENDS_URL,
        "RESEARCH_ONLY_TREND_INTELLIGENCE",
    )


def _team_index(report: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(report, dict):
        return out
    for row in report.get("teams") or []:
        if isinstance(row, dict) and row.get("team_id") is not None:
            out[str(row["team_id"])] = row
    return out


def _opponent_modifier(
    opponent_team_id: Any,
    team_report: dict[str, Any] | None,
    team_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    global_context = team_report.get("global_context") if isinstance(team_report, dict) and isinstance(team_report.get("global_context"), dict) else {}
    global_shots = _num(global_context.get("avg_team_shots"))
    opp = team_index.get(str(opponent_team_id)) if opponent_team_id is not None else None
    windows = opp.get("windows") if isinstance(opp, dict) and isinstance(opp.get("windows"), dict) else {}
    l10 = windows.get("last_10") if isinstance(windows.get("last_10"), dict) else {}
    n = int(l10.get("n") or 0)
    allowed = _num(l10.get("avg_opponent_shots"))

    if global_shots is None or global_shots <= 0 or allowed is None or n < 5:
        return {
            "factor": 1.0,
            "status": "NEUTRAL_INSUFFICIENT_OPPONENT_SHOT_HISTORY",
            "opponent_last10_n": n,
            "opponent_avg_shots_allowed": allowed,
            "global_avg_team_shots": global_shots,
        }

    raw = allowed / global_shots
    shrunk = (n * raw + OPPONENT_PRIOR_MATCHES) / (n + OPPONENT_PRIOR_MATCHES)
    factor = max(OPPONENT_FACTOR_CLIP[0], min(OPPONENT_FACTOR_CLIP[1], shrunk))
    return {
        "factor": round(factor, 6),
        "status": "SHRUNK_OPPONENT_SHOT_SUPPRESSION",
        "opponent_last10_n": n,
        "opponent_avg_shots_allowed": allowed,
        "global_avg_team_shots": global_shots,
        "raw_factor": round(raw, 6),
        "prior_matches_at_neutral": OPPONENT_PRIOR_MATCHES,
        "clip": list(OPPONENT_FACTOR_CLIP),
    }


def _nb_pmf(k: int, alpha: float, beta_minutes: float, future_exposure_minutes: float) -> float:
    # Gamma(alpha, rate=beta_minutes) posterior over event rate per minute
    # integrated over a Poisson future count gives a Negative Binomial predictive.
    if k < 0 or alpha <= 0 or beta_minutes <= 0 or future_exposure_minutes < 0:
        return 0.0
    if future_exposure_minutes == 0:
        return 1.0 if k == 0 else 0.0
    log_coeff = math.lgamma(k + alpha) - math.lgamma(alpha) - math.lgamma(k + 1)
    p_prior = beta_minutes / (beta_minutes + future_exposure_minutes)
    p_future = future_exposure_minutes / (beta_minutes + future_exposure_minutes)
    return math.exp(log_coeff + alpha * math.log(p_prior) + k * math.log(p_future))


def _prob_at_least(threshold_count: int, alpha: float, beta: float, exposure: float) -> float:
    if threshold_count <= 0:
        return 1.0
    cdf = sum(_nb_pmf(k, alpha, beta, exposure) for k in range(threshold_count))
    return max(0.0, min(1.0, 1.0 - cdf))


def _line_table(alpha: float, beta: float, exposure: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in LINES:
        need = int(math.floor(line)) + 1
        over = _prob_at_least(need, alpha, beta, exposure)
        under = 1.0 - over
        rows.append({
            "line": line,
            "over_requires_shots": need,
            "p_over": round(over, 6),
            "p_under": round(under, 6),
            "fair_decimal_over_no_vig": round(1.0 / over, 4) if over > 0 else None,
            "fair_decimal_under_no_vig": round(1.0 / under, 4) if under > 0 else None,
            "market_price_attached": False,
            "ev_computed": False,
        })
    return rows


def _formation_context(event: dict[str, Any]) -> dict[str, Any]:
    intel = event.get("formation_live_intelligence") if isinstance(event.get("formation_live_intelligence"), dict) else {}
    return {
        "status": intel.get("status"),
        "home_formation": intel.get("home_formation"),
        "away_formation": intel.get("away_formation"),
        "matchup": intel.get("matchup"),
        "sample_band": intel.get("sample_band"),
        "numeric_modifier_applied": False,
        "factor": 1.0,
        "reason": "NO_PLAYER_SHOTS_SPECIFIC_FORMATION_RESIDUAL_OOS_MODEL",
    }


def build(
    event: dict[str, Any],
    player_registry: dict[str, Any] | None,
    team_report: dict[str, Any] | None,
) -> dict[str, Any]:
    fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
    lineups = event.get("lineups") if isinstance(event.get("lineups"), dict) else {}
    if not lineups.get("both_xi_confirmed"):
        return {
            "schema_version": SCHEMA_VERSION,
            "fixture_id": fixture.get("fixture_id"),
            "status": "NOT_VERIFIED",
            "reason": "BOTH_STARTING_XI_NOT_CONFIRMED",
            "actionable": False,
            "decision_weight": 0.0,
        }

    profiles = player_registry.get("profiles") if isinstance(player_registry, dict) and isinstance(player_registry.get("profiles"), dict) else {}
    team_idx = _team_index(team_report)
    home_id = fixture.get("home_team_id")
    away_id = fixture.get("away_team_id")
    formation = _formation_context(event)
    players: list[dict[str, Any]] = []
    modeled = 0

    for team in lineups.get("teams") or []:
        if not isinstance(team, dict):
            continue
        team_id = team.get("team_id")
        opponent_id = away_id if str(team_id) == str(home_id) else home_id
        opp_mod = _opponent_modifier(opponent_id, team_report, team_idx)

        for starter in team.get("starters") or []:
            if not isinstance(starter, dict):
                continue
            pid = starter.get("id")
            profile = profiles.get(str(pid)) if pid is not None and isinstance(profiles.get(str(pid)), dict) else None
            role = profile.get("role_model") if isinstance(profile, dict) and isinstance(profile.get("role_model"), dict) else {}
            windows = profile.get("windows") if isinstance(profile, dict) and isinstance(profile.get("windows"), dict) else {}
            l20 = windows.get("last_20") if isinstance(windows.get("last_20"), dict) else {}
            metrics = l20.get("metrics") if isinstance(l20.get("metrics"), dict) else {}
            shots = metrics.get("shots") if isinstance(metrics.get("shots"), dict) else {}
            alpha = _num(shots.get("posterior_gamma_shape"))
            beta = _num(shots.get("posterior_gamma_rate_minutes"))
            expected_minutes = _num(role.get("expected_minutes_if_confirmed_starter"))
            sample_band = profile.get("sample_band") if profile else "NONE"

            if alpha is None or beta is None or expected_minutes is None or expected_minutes <= 0:
                players.append({
                    "team_id": team_id,
                    "team": team.get("team"),
                    "opponent_team_id": opponent_id,
                    "player_id": pid,
                    "player": starter.get("name"),
                    "position": starter.get("pos"),
                    "confirmed_starter": True,
                    "status": "PROFILE_NOT_MODELABLE",
                    "sample_band": sample_band,
                    "actionable": False,
                    "decision_weight": 0.0,
                })
                continue

            opponent_factor = _num(opp_mod.get("factor")) or 1.0
            formation_factor = 1.0
            effective_exposure = expected_minutes * opponent_factor * formation_factor
            mean_count = alpha / beta * effective_exposure
            lines = _line_table(alpha, beta, effective_exposure)
            modeled += 1
            players.append({
                "team_id": team_id,
                "team": team.get("team"),
                "opponent_team_id": opponent_id,
                "player_id": pid,
                "player": starter.get("name"),
                "position": starter.get("pos"),
                "confirmed_starter": True,
                "status": "LIVE_RESEARCH_SHOTS_DISTRIBUTION",
                "sample_band": sample_band,
                "expected_minutes_if_confirmed_starter": round(expected_minutes, 3),
                "base_posterior_mean_shots_per90": shots.get("posterior_mean_per90"),
                "opponent_adjustment": opp_mod,
                "formation_adjustment": formation,
                "effective_future_exposure_minutes": round(effective_exposure, 3),
                "predictive_distribution": "GAMMA_POISSON_NEGATIVE_BINOMIAL",
                "expected_shots": round(mean_count, 6),
                "lines": lines,
                "observed_sportsbook_line": None,
                "observed_sportsbook_price": None,
                "edge_vs_market": None,
                "actionable": False,
                "decision_weight": 0.0,
            })

    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": fixture.get("fixture_id"),
        "status": "LIVE_RESEARCH_SHOTS_PROPS",
        "model": "CONFIRMED_STARTER_MINUTES_X_SHRUNK_SHOTS_RATE_X_OPPONENT_SUPPRESSION__NB_PREDICTIVE_v0.1",
        "players": players,
        "modeled_players": modeled,
        "confirmed_starters_seen": len(players),
        "formation_context_used": bool(formation.get("home_formation") and formation.get("away_formation")),
        "formation_numeric_modifier_applied": False,
        "market_prices_attached": False,
        "actionable": False,
        "decision_weight": 0.0,
        "production_status": "LIVE_RESEARCH_NOT_ACTIONABLE",
        "calibration_gate": {
            "minimum_oos_player_games_for_review": 500,
            "minimum_oos_player_games_for_market_review": 1000,
            "minimum_oos_player_games_for_actionable_review": 2000,
            "requires": [
                "walk-forward Brier/log-loss by line and probability bucket",
                "count calibration and dispersion review by position/competition",
                "confirmed starter and minutes calibration",
                "observed sportsbook player-shot line and price",
                "true CLV before any production promotion",
                "shots-specific formation residual OOS evidence before non-neutral formation modifier",
            ],
        },
        "policy": "EXACT HALF-LINE PROBABILITIES ARE FAIR RESEARCH ESTIMATES ONLY; NO MARKET PRICE OR EV MEANS NO BET/LEAN/GALAXY LEG; FORMATION FACTOR REMAINS 1.0 UNTIL SHOTS-SPECIFIC OOS LIFT",
    }


def attach(payload: dict[str, Any]) -> dict[str, int | bool]:
    player_registry = load_player_registry()
    team_report = load_team_trends()
    modeled_events = modeled_players = 0

    for event in payload.get("events") or []:
        if not isinstance(event, dict) or event.get("event_type") != "SOCCER_REFRESH" or event.get("stage") in {"POSTGAME", "HT", "CLOSE"}:
            continue
        intel = build(event, player_registry, team_report)
        event["player_shots_intelligence"] = intel
        if intel.get("status") == "LIVE_RESEARCH_SHOTS_PROPS":
            modeled_events += 1
            modeled_players += int(intel.get("modeled_players") or 0)
        mi = event.get("match_intelligence")
        if isinstance(mi, dict) and isinstance(mi.get("areas"), dict):
            mi["areas"]["player_shots"] = intel

    return {
        "player_registry_loaded": bool(player_registry),
        "team_trends_loaded": bool(team_report),
        "modeled_events": modeled_events,
        "modeled_players": modeled_players,
        "provider_requests_added": 0,
    }
