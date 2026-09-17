from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone as dt_timezone
from typing import Any

import httpx

from mcp_gateway import automation as base

SCHEMA_VERSION = "1.0.0"
PLAYER_REGISTRY_URL = "https://raw.githubusercontent.com/Baahl11/Soccer/soccer-edge-state/soccer_edge_state/analysis/player_trend_model_registry.json"
TEAM_REGISTRY_URL = "https://raw.githubusercontent.com/Baahl11/Soccer/soccer-edge-state/soccer_edge_state/analysis/team_shot_suppression_registry.json"
CACHE_TTL = timedelta(hours=6)
THRESHOLDS = (1, 2, 3, 4)


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load(url: str, cache_name: str, expected_status: str) -> dict[str, Any] | None:
    now = datetime.now(dt_timezone.utc)
    cached = base._cache_get(cache_name, "latest", CACHE_TTL, now)
    if isinstance(cached, dict) and cached.get("schema_version"):
        return cached
    try:
        response = httpx.get(url, timeout=8.0, follow_redirects=True)
        if response.status_code != 200:
            return None
        payload = response.json()
    except Exception:
        return None
    if not isinstance(payload, dict) or payload.get("status") != expected_status:
        return None
    base._cache_set(cache_name, "latest", payload, now)
    return payload


def load_player_registry() -> dict[str, Any] | None:
    return _load(PLAYER_REGISTRY_URL, "player_trend_model_registry", "RESEARCH_PLAYER_TREND_MODEL_REGISTRY")


def load_team_registry() -> dict[str, Any] | None:
    return _load(TEAM_REGISTRY_URL, "team_shot_suppression_registry", "RESEARCH_TEAM_SHOT_SUPPRESSION_REGISTRY")


def _nb_tail(shape: float, rate_minutes: float, exposure_minutes: float, minimum_count: int, multiplier: float) -> float | None:
    if shape <= 0 or rate_minutes <= 0 or exposure_minutes <= 0 or minimum_count <= 0 or multiplier <= 0:
        return None
    effective_exposure = exposure_minutes * multiplier
    p = rate_minutes / (rate_minutes + effective_exposure)
    q = 1.0 - p
    # Gamma-Poisson posterior predictive:
    # P(K=k) = Gamma(k+r)/(Gamma(r)k!) * p^r * q^k
    pk = math.exp(shape * math.log(p))
    cdf = pk
    for k in range(1, minimum_count):
        pk = pk * ((k - 1 + shape) / k) * q
        cdf += pk
    return max(0.0, min(1.0, 1.0 - cdf))


def _opponent_factor(team_registry: dict[str, Any] | None, opponent_id: Any) -> tuple[float, dict[str, Any]]:
    teams = team_registry.get("teams") if isinstance(team_registry, dict) and isinstance(team_registry.get("teams"), dict) else {}
    row = teams.get(str(opponent_id)) if opponent_id is not None and isinstance(teams.get(str(opponent_id)), dict) else None
    if not row:
        return 1.0, {"status": "NO_OPPONENT_PROFILE", "multiplier": 1.0}
    raw = _num(row.get("opponent_shot_factor"))
    eligible = row.get("factor_status") == "ELIGIBLE"
    if raw is None or not eligible:
        return 1.0, {
            "status": "LOW_SAMPLE_CONTEXT_ONLY",
            "matches": row.get("matches"),
            "raw_registry_factor": raw,
            "multiplier": 1.0,
        }
    return raw, {
        "status": "APPLIED_RESEARCH_ONLY",
        "matches": row.get("matches"),
        "shrunk_shots_allowed_per_match": row.get("shrunk_shots_allowed_per_match"),
        "multiplier": raw,
    }


def build(event: dict[str, Any], players: dict[str, Any] | None, teams: dict[str, Any] | None) -> dict[str, Any]:
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

    profiles = players.get("profiles") if isinstance(players, dict) and isinstance(players.get("profiles"), dict) else {}
    home_id = fixture.get("home_team_id")
    away_id = fixture.get("away_team_id")
    team_formations = {
        str(team.get("team_id")): team.get("formation")
        for team in (lineups.get("teams") or [])
        if isinstance(team, dict) and team.get("team_id") is not None
    }

    rows: list[dict[str, Any]] = []
    matched = probabilistic = 0
    for team in lineups.get("teams") or []:
        if not isinstance(team, dict):
            continue
        team_id = team.get("team_id")
        opponent_id = away_id if str(team_id) == str(home_id) else home_id
        opp_factor, opp_context = _opponent_factor(teams, opponent_id)
        for starter in team.get("starters") or []:
            if not isinstance(starter, dict):
                continue
            pid = starter.get("id")
            profile = profiles.get(str(pid)) if pid is not None and isinstance(profiles.get(str(pid)), dict) else None
            if profile:
                matched += 1
            role = profile.get("role_model") if isinstance(profile, dict) and isinstance(profile.get("role_model"), dict) else {}
            expected_minutes = _num(role.get("expected_minutes_if_confirmed_starter"))
            windows = profile.get("windows") if isinstance(profile, dict) and isinstance(profile.get("windows"), dict) else {}
            l20 = windows.get("last_20") if isinstance(windows.get("last_20"), dict) else {}
            metrics = l20.get("metrics") if isinstance(l20.get("metrics"), dict) else {}
            shots = metrics.get("shots") if isinstance(metrics.get("shots"), dict) else {}
            shape = _num(shots.get("posterior_gamma_shape"))
            rate_minutes = _num(shots.get("posterior_gamma_rate_minutes"))
            probs: dict[str, float | None] = {}
            if shape is not None and rate_minutes is not None and expected_minutes is not None and expected_minutes > 0:
                for n in THRESHOLDS:
                    value = _nb_tail(shape, rate_minutes, expected_minutes, n, opp_factor)
                    probs[f"p_{n}plus_shots"] = round(value, 6) if value is not None else None
                probabilistic += 1
            else:
                for n in THRESHOLDS:
                    probs[f"p_{n}plus_shots"] = None

            rows.append({
                "team_id": team_id,
                "team": team.get("team"),
                "opponent_id": opponent_id,
                "player_id": pid,
                "player": starter.get("name"),
                "position": starter.get("pos"),
                "confirmed_starter": True,
                "expected_minutes": expected_minutes,
                "sample_band": profile.get("sample_band") if profile else "NONE",
                "base_shots_rate_per90": shots.get("posterior_mean_per90") if shots else None,
                "opponent_shot_suppression": opp_context,
                "formation_context": {
                    "own_formation": team_formations.get(str(team_id)),
                    "opponent_formation": team_formations.get(str(opponent_id)),
                    "multiplier": 1.0,
                    "status": "CONTEXT_ONLY_UNCALIBRATED",
                },
                "probabilities": probs,
                "distribution": "GAMMA_POISSON_POSTERIOR_PREDICTIVE",
                "market_line_mapping": {
                    "over_0_5_shots": probs.get("p_1plus_shots"),
                    "over_1_5_shots": probs.get("p_2plus_shots"),
                    "over_2_5_shots": probs.get("p_3plus_shots"),
                    "over_3_5_shots": probs.get("p_4plus_shots"),
                },
                "market_comparison_status": "BLOCKED_NO_VERIFIED_PLAYER_PROP_PRICE",
                "actionable": False,
                "decision_weight": 0.0,
            })

    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": fixture.get("fixture_id"),
        "status": "LIVE_RESEARCH_PROBABILITIES",
        "model": "PLAYER_SHOTS_GAMMA_POISSON_OPPONENT_SUPPRESSION_v0.1",
        "players": rows,
        "confirmed_starter_count": len(rows),
        "matched_player_profiles": matched,
        "players_with_shot_probabilities": probabilistic,
        "actionable": False,
        "decision_weight": 0.0,
        "production_status": "RESEARCH_ONLY_NO_PLAYER_PROP_ODDS",
        "market_odds_required_for_ev": True,
        "calibration_gate": {
            "minimum_oos_player_games_for_review": 500,
            "minimum_oos_player_games_for_actionable_review": 1500,
            "requires": [
                "walk-forward threshold Brier/log-loss by line",
                "dispersion review vs Poisson and negative-binomial alternatives",
                "opponent-factor lift vs no-opponent baseline",
                "formation effect remains zero until independent OOS lift",
                "verified sportsbook player-shot line and price for EV/CLV",
            ],
        },
        "policy": "EXACT PLAYER SHOT THRESHOLD PROBABILITIES ARE SPORT-FIRST RESEARCH OUTPUTS; NO ODDS=NO EV PICK; FORMATION MULTIPLIER REMAINS 1.0 UNTIL OOS VALIDATED",
    }


def attach(payload: dict[str, Any]) -> dict[str, int | bool]:
    player_registry = load_player_registry()
    team_registry = load_team_registry()
    modeled_events = starters = matched = probabilistic = 0
    for event in payload.get("events") or []:
        if not isinstance(event, dict) or event.get("event_type") != "SOCCER_REFRESH" or event.get("stage") in {"POSTGAME", "HT", "CLOSE"}:
            continue
        intel = build(event, player_registry, team_registry)
        event["player_shots_prop_intelligence"] = intel
        if intel.get("status") == "LIVE_RESEARCH_PROBABILITIES":
            modeled_events += 1
        starters += int(intel.get("confirmed_starter_count") or 0)
        matched += int(intel.get("matched_player_profiles") or 0)
        probabilistic += int(intel.get("players_with_shot_probabilities") or 0)
        mi = event.get("match_intelligence")
        if isinstance(mi, dict) and isinstance(mi.get("areas"), dict):
            mi["areas"]["player_shots"] = intel
    return {
        "player_registry_loaded": bool(player_registry),
        "team_shot_registry_loaded": bool(team_registry),
        "modeled_events": modeled_events,
        "confirmed_starters_seen": starters,
        "matched_player_profiles": matched,
        "players_with_shot_probabilities": probabilistic,
        "provider_requests_added": 0,
    }
