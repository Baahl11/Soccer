from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone as dt_timezone
from typing import Any

import httpx

from mcp_gateway import automation as base

SCHEMA_VERSION = "1.0.0"
PLAYER_REGISTRY_URL = "https://raw.githubusercontent.com/Baahl11/Soccer/soccer-edge-state/soccer_edge_state/analysis/player_trend_model_registry.json"
CARDS_REGISTRY_URL = "https://raw.githubusercontent.com/Baahl11/Soccer/soccer-edge-state/soccer_edge_state/analysis/cards_rate_registry.json"
CACHE_TTL = timedelta(hours=6)
LINES = (0.5, 1.5)
TEAM_FACTOR_CLIP = (0.75, 1.25)
MIN_TEAM_MATCHES = 5


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
        "cards_player_trend_registry",
        PLAYER_REGISTRY_URL,
        "RESEARCH_PLAYER_TREND_MODEL_REGISTRY",
    )


def load_cards_registry() -> dict[str, Any] | None:
    return _load_json(
        "player_cards_rate_registry",
        CARDS_REGISTRY_URL,
        "RESEARCH_YELLOW_CARD_RATE_REGISTRY",
    )


def _mean(row: dict[str, Any] | None, field: str) -> float | None:
    if not isinstance(row, dict):
        return None
    n = _num(row.get("n"))
    total = _num(row.get(field))
    if n is None or n <= 0 or total is None:
        return None
    return total / n


def _team_discipline_environment(
    team_id: Any,
    is_home: bool,
    registry: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(registry, dict):
        return {
            "factor": 1.0,
            "status": "NEUTRAL_CARDS_REGISTRY_UNAVAILABLE",
            "numeric_modifier_applied": False,
        }

    global_row = registry.get("global") if isinstance(registry.get("global"), dict) else {}
    collection_key = "home_teams" if is_home else "away_teams"
    field = "home_yellow" if is_home else "away_yellow"
    collection = registry.get(collection_key) if isinstance(registry.get(collection_key), dict) else {}
    team_row = collection.get(str(team_id)) if team_id is not None and isinstance(collection.get(str(team_id)), dict) else {}
    global_rate = _mean(global_row, field)
    team_rate = _mean(team_row, field)
    n = int(_num(team_row.get("n")) or 0) if isinstance(team_row, dict) else 0
    prior_n = _num(registry.get("team_ratio_pseudo_n")) or 8.0

    if global_rate is None or global_rate <= 0 or team_rate is None or n < MIN_TEAM_MATCHES:
        return {
            "factor": 1.0,
            "status": "NEUTRAL_INSUFFICIENT_TEAM_CARD_HISTORY",
            "numeric_modifier_applied": False,
            "team_venue_sample_n": n,
            "team_yellow_per_match": round(team_rate, 6) if team_rate is not None else None,
            "global_yellow_per_team_match": round(global_rate, 6) if global_rate is not None else None,
        }

    raw = team_rate / global_rate
    shrunk = (n * raw + prior_n) / (n + prior_n)
    factor = max(TEAM_FACTOR_CLIP[0], min(TEAM_FACTOR_CLIP[1], shrunk))
    return {
        "factor": round(factor, 6),
        "status": "SHRUNK_TEAM_DISCIPLINE_ENVIRONMENT",
        "numeric_modifier_applied": True,
        "team_venue_sample_n": n,
        "team_yellow_per_match": round(team_rate, 6),
        "global_yellow_per_team_match": round(global_rate, 6),
        "raw_factor": round(raw, 6),
        "prior_matches_at_neutral": prior_n,
        "clip": list(TEAM_FACTOR_CLIP),
    }


def _referee_context(event: dict[str, Any]) -> dict[str, Any]:
    fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
    referee = fixture.get("referee")
    return {
        "referee": referee,
        "verified": bool(referee),
        "numeric_modifier_applied": False,
        "factor": 1.0,
        "reason": "REFEREE_EFFECT_REQUIRES_PLAYER_CARD_SPECIFIC_OOS_CALIBRATION",
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
            "over_requires_yellow_cards": need,
            "p_over": round(over, 6),
            "p_under": round(under, 6),
            "fair_decimal_over_no_vig": round(1.0 / over, 4) if over > 0 else None,
            "fair_decimal_under_no_vig": round(1.0 / under, 4) if under > 0 else None,
            "market_price_attached": False,
            "ev_computed": False,
        })
    return rows


def build(
    event: dict[str, Any],
    player_registry: dict[str, Any] | None,
    cards_registry: dict[str, Any] | None,
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
    home_id = fixture.get("home_team_id")
    referee = _referee_context(event)
    players: list[dict[str, Any]] = []
    modeled = 0

    for team in lineups.get("teams") or []:
        if not isinstance(team, dict):
            continue
        team_id = team.get("team_id")
        is_home = str(team_id) == str(home_id)
        discipline = _team_discipline_environment(team_id, is_home, cards_registry)

        for starter in team.get("starters") or []:
            if not isinstance(starter, dict):
                continue
            pid = starter.get("id")
            profile = profiles.get(str(pid)) if pid is not None and isinstance(profiles.get(str(pid)), dict) else None
            role = profile.get("role_model") if isinstance(profile, dict) and isinstance(profile.get("role_model"), dict) else {}
            windows = profile.get("windows") if isinstance(profile, dict) and isinstance(profile.get("windows"), dict) else {}
            l20 = windows.get("last_20") if isinstance(windows.get("last_20"), dict) else {}
            metrics = l20.get("metrics") if isinstance(l20.get("metrics"), dict) else {}
            cards = metrics.get("yellow_cards") if isinstance(metrics.get("yellow_cards"), dict) else {}
            alpha = _num(cards.get("posterior_gamma_shape"))
            beta = _num(cards.get("posterior_gamma_rate_minutes"))
            expected_minutes = _num(role.get("expected_minutes_if_confirmed_starter"))
            sample_band = profile.get("sample_band") if profile else "NONE"

            if alpha is None or beta is None or expected_minutes is None or expected_minutes <= 0:
                players.append({
                    "team_id": team_id,
                    "team": team.get("team"),
                    "player_id": pid,
                    "player": starter.get("name"),
                    "position": starter.get("pos"),
                    "confirmed_starter": True,
                    "status": "PROFILE_NOT_MODELABLE",
                    "reason": "FINALIZED_PLAYER_YELLOW_CARD_RATE_NOT_AVAILABLE",
                    "sample_band": sample_band,
                    "actionable": False,
                    "decision_weight": 0.0,
                })
                continue

            team_factor = _num(discipline.get("factor")) or 1.0
            effective_exposure = expected_minutes * team_factor
            expected_yellow = alpha / beta * effective_exposure
            lines = _line_table(alpha, beta, effective_exposure)
            p_carded = _prob_at_least(1, alpha, beta, effective_exposure)
            p_two_plus = _prob_at_least(2, alpha, beta, effective_exposure)
            modeled += 1
            players.append({
                "team_id": team_id,
                "team": team.get("team"),
                "player_id": pid,
                "player": starter.get("name"),
                "position": starter.get("pos"),
                "confirmed_starter": True,
                "status": "LIVE_RESEARCH_PLAYER_YELLOW_CARD_DISTRIBUTION",
                "sample_band": sample_band,
                "expected_minutes_if_confirmed_starter": round(expected_minutes, 3),
                "base_posterior_mean_yellow_cards_per90": cards.get("posterior_mean_per90"),
                "team_discipline_environment": discipline,
                "referee_context": referee,
                "effective_future_exposure_minutes": round(effective_exposure, 3),
                "predictive_distribution": "GAMMA_POISSON_NEGATIVE_BINOMIAL",
                "expected_yellow_cards": round(expected_yellow, 6),
                "p_player_booked_yellow": round(p_carded, 6),
                "p_2plus_yellow_cards": round(p_two_plus, 6),
                "fair_decimal_player_booked_no_vig": round(1.0 / p_carded, 4) if p_carded > 0 else None,
                "lines": lines,
                "observed_sportsbook_player_card_price": None,
                "bookmaker_card_scoring_rule_mapped": False,
                "edge_vs_market": None,
                "actionable": False,
                "decision_weight": 0.0,
            })

    status = "LIVE_RESEARCH_PLAYER_CARDS" if modeled > 0 else "DATA_BLOCKED_PLAYER_CARDS"
    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": fixture.get("fixture_id"),
        "status": status,
        "model": "CONFIRMED_STARTER_MINUTES_X_SHRUNK_YELLOW_CARD_RATE_X_TEAM_DISCIPLINE__NB_PREDICTIVE_v0.1",
        "target": "PLAYER_YELLOW_CARD_COUNT",
        "players": players,
        "modeled_players": modeled,
        "confirmed_starters_seen": len(players),
        "red_cards_modeled": False,
        "referee_numeric_modifier_applied": False,
        "market_prices_attached": False,
        "actionable": False,
        "decision_weight": 0.0,
        "production_status": "LIVE_RESEARCH_NOT_ACTIONABLE" if modeled > 0 else "DORMANT_DATA_BLOCKED",
        "calibration_gate": {
            "minimum_oos_player_games_for_review": 1000,
            "minimum_oos_player_games_for_market_review": 2000,
            "minimum_oos_player_games_for_actionable_review": 4000,
            "requires": [
                "walk-forward Brier/log-loss for player booked (yellow) outcome",
                "yellow-card count calibration by position and competition",
                "confirmed starter/minutes calibration",
                "player-card-specific referee residual OOS evidence before any referee numeric modifier",
                "observed sportsbook player-card price plus explicit bookmaker card-scoring rule",
                "true CLV before any production promotion",
            ],
        },
        "policy": "YELLOW-CARD PLAYER PROP RESEARCH ONLY; RED CARDS REMAIN SEPARATE; NO BOOKMAKER CARD-POINT EQUIVALENCE IS ASSUMED; NO VERIFIED PRICE/RULE/EV MEANS NO BET/LEAN/GALAXY LEG",
    }


def attach(payload: dict[str, Any]) -> dict[str, int | bool]:
    player_registry = load_player_registry()
    cards_registry = load_cards_registry()
    modeled_events = modeled_players = blocked_events = 0

    for event in payload.get("events") or []:
        if not isinstance(event, dict) or event.get("event_type") != "SOCCER_REFRESH" or event.get("stage") in {"POSTGAME", "HT", "CLOSE"}:
            continue
        intel = build(event, player_registry, cards_registry)
        event["player_cards_intelligence"] = intel
        if intel.get("status") == "LIVE_RESEARCH_PLAYER_CARDS":
            modeled_events += 1
            modeled_players += int(intel.get("modeled_players") or 0)
        elif intel.get("status") == "DATA_BLOCKED_PLAYER_CARDS":
            blocked_events += 1
        mi = event.get("match_intelligence")
        if isinstance(mi, dict) and isinstance(mi.get("areas"), dict):
            mi["areas"]["player_cards"] = intel

    return {
        "player_registry_loaded": bool(player_registry),
        "cards_registry_loaded": bool(cards_registry),
        "modeled_events": modeled_events,
        "blocked_events": blocked_events,
        "modeled_players": modeled_players,
        "provider_requests_added": 0,
    }
