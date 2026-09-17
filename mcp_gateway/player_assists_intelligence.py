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
ENV_PRIOR_MATCHES = 12.0
ENV_FACTOR_CLIP = (0.75, 1.25)


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
        "assists_player_trend_registry",
        PLAYER_REGISTRY_URL,
        "RESEARCH_PLAYER_TREND_MODEL_REGISTRY",
    )


def load_team_trends() -> dict[str, Any] | None:
    return _load_json(
        "assists_team_trends",
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


def _shrink_factor(raw: float | None, n: int) -> float:
    if raw is None or n < 5:
        return 1.0
    shrunk = (n * raw + ENV_PRIOR_MATCHES) / (n + ENV_PRIOR_MATCHES)
    return max(ENV_FACTOR_CLIP[0], min(ENV_FACTOR_CLIP[1], shrunk))


def _scoring_environment(
    team_id: Any,
    opponent_team_id: Any,
    report: dict[str, Any] | None,
    index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    global_context = report.get("global_context") if isinstance(report, dict) and isinstance(report.get("global_context"), dict) else {}
    global_goals = _num(global_context.get("avg_team_goals"))
    own = index.get(str(team_id)) if team_id is not None else None
    opp = index.get(str(opponent_team_id)) if opponent_team_id is not None else None
    own_l10 = ((own.get("windows") or {}).get("last_10") or {}) if isinstance(own, dict) else {}
    opp_l10 = ((opp.get("windows") or {}).get("last_10") or {}) if isinstance(opp, dict) else {}
    own_n = int(own_l10.get("n") or 0)
    opp_n = int(opp_l10.get("n") or 0)
    own_gf = _num(own_l10.get("avg_goals_for"))
    opp_ga = _num(opp_l10.get("avg_goals_against"))

    if global_goals is None or global_goals <= 0:
        return {
            "factor": 1.0,
            "status": "NEUTRAL_GLOBAL_GOAL_BASELINE_UNAVAILABLE",
            "global_avg_team_goals": global_goals,
        }

    attack_raw = own_gf / global_goals if own_gf is not None else None
    defense_raw = opp_ga / global_goals if opp_ga is not None else None
    attack = _shrink_factor(attack_raw, own_n)
    defense = _shrink_factor(defense_raw, opp_n)
    combined = math.sqrt(max(0.0, attack * defense))
    combined = max(ENV_FACTOR_CLIP[0], min(ENV_FACTOR_CLIP[1], combined))
    status = "SHRUNK_TEAM_SCORING_ENVIRONMENT" if own_n >= 5 or opp_n >= 5 else "NEUTRAL_INSUFFICIENT_TEAM_HISTORY"
    return {
        "factor": round(combined, 6),
        "status": status,
        "global_avg_team_goals": global_goals,
        "team_last10_n": own_n,
        "team_avg_goals_for": own_gf,
        "opponent_last10_n": opp_n,
        "opponent_avg_goals_against": opp_ga,
        "attack_factor": round(attack, 6),
        "opponent_defense_factor": round(defense, 6),
        "prior_matches_at_neutral": ENV_PRIOR_MATCHES,
        "clip": list(ENV_FACTOR_CLIP),
    }


def _nb_pmf(k: int, alpha: float, beta_minutes: float, future_exposure_minutes: float) -> float:
    if k < 0 or alpha <= 0 or beta_minutes <= 0 or future_exposure_minutes < 0:
        return 0.0
    if future_exposure_minutes == 0:
        return 1.0 if k == 0 else 0.0
    log_coeff = math.lgamma(k + alpha) - math.lgamma(alpha) - math.lgamma(k + 1)
    p_prior = beta_minutes / (beta_minutes + future_exposure_minutes)
    p_future = future_exposure_minutes / (beta_minutes + future_exposure_minutes)
    return math.exp(log_coeff + alpha * math.log(p_prior) + k * math.log(p_future))


def _prob_at_least(threshold_count: int, alpha: float, beta: float, exposure: float) -> float:
    cdf = sum(_nb_pmf(k, alpha, beta, exposure) for k in range(threshold_count))
    return max(0.0, min(1.0, 1.0 - cdf))


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
    rows: list[dict[str, Any]] = []
    modeled = 0

    for team in lineups.get("teams") or []:
        if not isinstance(team, dict):
            continue
        team_id = team.get("team_id")
        opponent_id = away_id if str(team_id) == str(home_id) else home_id
        env = _scoring_environment(team_id, opponent_id, team_report, team_idx)

        for starter in team.get("starters") or []:
            if not isinstance(starter, dict):
                continue
            pid = starter.get("id")
            profile = profiles.get(str(pid)) if pid is not None and isinstance(profiles.get(str(pid)), dict) else None
            role = profile.get("role_model") if isinstance(profile, dict) and isinstance(profile.get("role_model"), dict) else {}
            windows = profile.get("windows") if isinstance(profile, dict) and isinstance(profile.get("windows"), dict) else {}
            l20 = windows.get("last_20") if isinstance(windows.get("last_20"), dict) else {}
            metrics = l20.get("metrics") if isinstance(l20.get("metrics"), dict) else {}
            assists = metrics.get("assists") if isinstance(metrics.get("assists"), dict) else {}
            alpha = _num(assists.get("posterior_gamma_shape"))
            beta = _num(assists.get("posterior_gamma_rate_minutes"))
            expected_minutes = _num(role.get("expected_minutes_if_confirmed_starter"))
            sample_band = profile.get("sample_band") if profile else "NONE"

            if alpha is None or beta is None or expected_minutes is None or expected_minutes <= 0:
                rows.append({
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

            env_factor = _num(env.get("factor")) or 1.0
            effective_exposure = expected_minutes * env_factor
            p_one = _prob_at_least(1, alpha, beta, effective_exposure)
            p_two = _prob_at_least(2, alpha, beta, effective_exposure)
            expected_assists = alpha / beta * effective_exposure
            modeled += 1
            rows.append({
                "team_id": team_id,
                "team": team.get("team"),
                "opponent_team_id": opponent_id,
                "player_id": pid,
                "player": starter.get("name"),
                "position": starter.get("pos"),
                "confirmed_starter": True,
                "status": "LIVE_RESEARCH_ASSIST_DISTRIBUTION",
                "sample_band": sample_band,
                "expected_minutes_if_confirmed_starter": round(expected_minutes, 3),
                "base_posterior_mean_assists_per90": assists.get("posterior_mean_per90"),
                "scoring_environment": env,
                "chance_creation_context": {
                    "xA_available": False,
                    "key_pass_quality_model_available": False,
                    "numeric_modifier_applied": False,
                    "factor": 1.0,
                    "reason": "NO_XA_OR_CALIBRATED_KEY_PASS_TO_ASSIST_MODEL",
                },
                "predictive_distribution": "GAMMA_POISSON_NEGATIVE_BINOMIAL",
                "effective_future_exposure_minutes": round(effective_exposure, 3),
                "expected_assists": round(expected_assists, 6),
                "p_1plus_assist": round(p_one, 6),
                "p_2plus_assists": round(p_two, 6),
                "fair_decimal_1plus_assist_no_vig": round(1.0 / p_one, 4) if p_one > 0 else None,
                "observed_sportsbook_assist_price": None,
                "edge_vs_market": None,
                "actionable": False,
                "decision_weight": 0.0,
            })

    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": fixture.get("fixture_id"),
        "status": "LIVE_RESEARCH_ASSISTS",
        "model": "CONFIRMED_STARTER_MINUTES_X_SHRUNK_ASSIST_RATE_X_SCORING_ENVIRONMENT__NB_PREDICTIVE_v0.1",
        "players": rows,
        "modeled_players": modeled,
        "confirmed_starters_seen": len(rows),
        "xA_numeric_modifier_applied": False,
        "market_prices_attached": False,
        "actionable": False,
        "decision_weight": 0.0,
        "production_status": "LIVE_RESEARCH_NOT_ACTIONABLE",
        "calibration_gate": {
            "minimum_oos_player_games_for_review": 1000,
            "minimum_oos_player_games_for_market_review": 2000,
            "minimum_oos_player_games_for_actionable_review": 4000,
            "requires": [
                "walk-forward Brier/log-loss for 1+ assist",
                "assist-count calibration by position/competition",
                "confirmed starter/minutes calibration",
                "xA or calibrated chance-quality evidence before non-neutral chance-creation effect",
                "observed sportsbook assist price and true CLV",
            ],
        },
        "policy": "ASSIST PROBABILITY IS SPORT-FIRST RESEARCH ONLY; TEAM SCORING ENVIRONMENT IS SHRUNK; NO XA OR VERIFIED ASSIST PRICE=NO EV PICK",
    }


def attach(payload: dict[str, Any]) -> dict[str, int | bool]:
    player_registry = load_player_registry()
    team_report = load_team_trends()
    modeled_events = modeled_players = 0

    for event in payload.get("events") or []:
        if not isinstance(event, dict) or event.get("event_type") != "SOCCER_REFRESH" or event.get("stage") in {"POSTGAME", "HT", "CLOSE"}:
            continue
        intel = build(event, player_registry, team_report)
        event["player_assists_intelligence"] = intel
        if intel.get("status") == "LIVE_RESEARCH_ASSISTS":
            modeled_events += 1
            modeled_players += int(intel.get("modeled_players") or 0)
        mi = event.get("match_intelligence")
        if isinstance(mi, dict) and isinstance(mi.get("areas"), dict):
            mi["areas"]["player_assists"] = intel

    return {
        "player_registry_loaded": bool(player_registry),
        "team_trends_loaded": bool(team_report),
        "modeled_events": modeled_events,
        "modeled_players": modeled_players,
        "provider_requests_added": 0,
    }
