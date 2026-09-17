from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone as dt_timezone
from typing import Any

import httpx

from mcp_gateway import automation as base

SCHEMA_VERSION = "1.0.0"
GK_REGISTRY_URL = "https://raw.githubusercontent.com/Baahl11/Soccer/soccer-edge-state/soccer_edge_state/analysis/goalkeeper_profiles.json"
TEAM_TRENDS_URL = "https://raw.githubusercontent.com/Baahl11/Soccer/soccer-edge-state/soccer_edge_state/analysis/trend_intelligence.json"
CACHE_TTL = timedelta(hours=6)
TEAM_PRIOR_MATCHES = 10.0
SAVE_PROXY_PRIOR_SOT = 30.0
SOT_FACTOR_CLIP = (0.65, 1.35)
LINES = (1.5, 2.5, 3.5, 4.5, 5.5)


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


def load_gk_registry() -> dict[str, Any] | None:
    return _load_json(
        "gk_saves_profile_registry",
        GK_REGISTRY_URL,
        "RESEARCH_GOALKEEPER_PROFILE_REGISTRY",
    )


def load_team_trends() -> dict[str, Any] | None:
    return _load_json(
        "gk_saves_team_trends",
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


def _shrink_value(raw: float | None, n: int, global_value: float) -> float:
    if raw is None or n <= 0:
        return global_value
    return (n * raw + TEAM_PRIOR_MATCHES * global_value) / (n + TEAM_PRIOR_MATCHES)


def _project_sot_faced(
    gk_team_id: Any,
    opponent_team_id: Any,
    team_report: dict[str, Any] | None,
    team_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    global_context = team_report.get("global_context") if isinstance(team_report, dict) and isinstance(team_report.get("global_context"), dict) else {}
    global_sot = _num(global_context.get("avg_team_sot"))
    if global_sot is None or global_sot <= 0:
        return {"status": "GLOBAL_SOT_BASELINE_UNAVAILABLE", "projected_sot_faced": None}

    attack = team_index.get(str(opponent_team_id)) if opponent_team_id is not None else None
    defense = team_index.get(str(gk_team_id)) if gk_team_id is not None else None
    attack_l10 = ((attack.get("windows") or {}).get("last_10") or {}) if isinstance(attack, dict) else {}
    defense_l10 = ((defense.get("windows") or {}).get("last_10") or {}) if isinstance(defense, dict) else {}
    attack_n = int(attack_l10.get("n") or 0)
    defense_n = int(defense_l10.get("n") or 0)
    attack_sot = _num(attack_l10.get("avg_team_sot"))
    defense_sot_allowed = _num(defense_l10.get("avg_opponent_sot"))
    attack_shrunk = _shrink_value(attack_sot, attack_n, global_sot)
    defense_shrunk = _shrink_value(defense_sot_allowed, defense_n, global_sot)
    projected = math.sqrt(max(0.0, attack_shrunk * defense_shrunk))
    raw_factor = projected / global_sot
    clipped_factor = max(SOT_FACTOR_CLIP[0], min(SOT_FACTOR_CLIP[1], raw_factor))
    projected_clipped = global_sot * clipped_factor
    return {
        "status": "SHRUNK_OPPONENT_SOT_PROJECTION",
        "global_avg_team_sot": global_sot,
        "opponent_last10_n": attack_n,
        "opponent_avg_team_sot": attack_sot,
        "gk_team_last10_n": defense_n,
        "gk_team_avg_sot_allowed": defense_sot_allowed,
        "attack_sot_shrunk": round(attack_shrunk, 6),
        "defense_sot_allowed_shrunk": round(defense_shrunk, 6),
        "raw_factor_vs_global": round(raw_factor, 6),
        "factor_clip": list(SOT_FACTOR_CLIP),
        "projected_sot_faced": round(projected_clipped, 6),
    }


def _global_save_prior(registry: dict[str, Any] | None) -> dict[str, Any]:
    profiles = registry.get("goalkeepers") if isinstance(registry, dict) and isinstance(registry.get("goalkeepers"), dict) else {}
    saves = conceded = 0.0
    gks = 0
    for profile in profiles.values():
        if not isinstance(profile, dict):
            continue
        windows = profile.get("windows") if isinstance(profile.get("windows"), dict) else {}
        l20 = windows.get("last_20") if isinstance(windows.get("last_20"), dict) else {}
        s = _num(l20.get("save_result_proxy_saves"))
        c = _num(l20.get("save_result_proxy_goals_conceded"))
        if s is None or c is None or s + c <= 0:
            continue
        saves += s
        conceded += c
        gks += 1
    denom = saves + conceded
    return {
        "goalkeepers_contributing": gks,
        "saves": round(saves, 3),
        "goals_conceded": round(conceded, 3),
        "sot_proxy": round(denom, 3),
        "save_probability_proxy": (saves / denom) if denom > 0 else None,
    }


def _keeper_save_probability(profile: dict[str, Any] | None, global_prior: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(profile, dict):
        return {"status": "GK_PROFILE_NOT_AVAILABLE", "save_probability": None}
    windows = profile.get("windows") if isinstance(profile.get("windows"), dict) else {}
    l20 = windows.get("last_20") if isinstance(windows.get("last_20"), dict) else {}
    saves = _num(l20.get("save_result_proxy_saves"))
    conceded = _num(l20.get("save_result_proxy_goals_conceded"))
    global_p = _num(global_prior.get("save_probability_proxy"))
    if saves is None or conceded is None or saves + conceded <= 0 or global_p is None:
        return {
            "status": "GK_SAVE_COUNTS_NOT_AVAILABLE",
            "save_probability": None,
            "sample_band": profile.get("sample_band"),
        }

    observed_sot_proxy = saves + conceded
    posterior = (saves + SAVE_PROXY_PRIOR_SOT * global_p) / (observed_sot_proxy + SAVE_PROXY_PRIOR_SOT)
    return {
        "status": "SHRUNK_SAVE_RESULT_PROXY",
        "sample_band": profile.get("sample_band"),
        "observed_saves": round(saves, 3),
        "observed_goals_conceded": round(conceded, 3),
        "observed_sot_proxy": round(observed_sot_proxy, 3),
        "raw_save_result_proxy": round(saves / observed_sot_proxy, 6),
        "global_save_result_proxy": round(global_p, 6),
        "prior_sot_proxy": SAVE_PROXY_PRIOR_SOT,
        "save_probability": round(posterior, 6),
        "definition_warning": "saves/(saves+goals_conceded) proxy; NOT PSxG and not shot-quality adjusted",
    }


def _poisson_cdf(k: int, lam: float) -> float:
    if lam < 0:
        return 0.0
    term = math.exp(-lam)
    total = term
    for i in range(1, k + 1):
        term *= lam / i
        total += term
    return max(0.0, min(1.0, total))


def _line_table(lam: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in LINES:
        need = int(math.floor(line)) + 1
        over = 1.0 - _poisson_cdf(need - 1, lam)
        under = 1.0 - over
        rows.append({
            "line": line,
            "over_requires_saves": need,
            "p_over": round(over, 6),
            "p_under": round(under, 6),
            "fair_decimal_over_no_vig": round(1.0 / over, 4) if over > 0 else None,
            "fair_decimal_under_no_vig": round(1.0 / under, 4) if under > 0 else None,
            "market_price_attached": False,
            "ev_computed": False,
        })
    return rows


def _confirmed_goalkeepers(event: dict[str, Any]) -> list[dict[str, Any]]:
    lineups = event.get("lineups") if isinstance(event.get("lineups"), dict) else {}
    rows: list[dict[str, Any]] = []
    for team in lineups.get("teams") or []:
        if not isinstance(team, dict):
            continue
        for gk in team.get("goalkeepers") or []:
            if isinstance(gk, dict):
                rows.append({
                    "team_id": team.get("team_id"),
                    "team": team.get("team"),
                    "player_id": gk.get("id"),
                    "player": gk.get("name"),
                })
    return rows


def build(event: dict[str, Any], registry: dict[str, Any] | None, team_report: dict[str, Any] | None) -> dict[str, Any]:
    fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
    lineups = event.get("lineups") if isinstance(event.get("lineups"), dict) else {}
    if not lineups.get("both_goalkeepers_confirmed"):
        return {
            "schema_version": SCHEMA_VERSION,
            "fixture_id": fixture.get("fixture_id"),
            "status": "NOT_VERIFIED",
            "reason": "BOTH_STARTING_GOALKEEPERS_NOT_CONFIRMED",
            "actionable": False,
            "decision_weight": 0.0,
        }

    profiles = registry.get("goalkeepers") if isinstance(registry, dict) and isinstance(registry.get("goalkeepers"), dict) else {}
    team_idx = _team_index(team_report)
    global_prior = _global_save_prior(registry)
    home_id = fixture.get("home_team_id")
    away_id = fixture.get("away_team_id")
    rows: list[dict[str, Any]] = []
    modeled = 0

    for gk in _confirmed_goalkeepers(event):
        team_id = gk.get("team_id")
        opponent_id = away_id if str(team_id) == str(home_id) else home_id
        profile = profiles.get(str(gk.get("player_id"))) if gk.get("player_id") is not None and isinstance(profiles.get(str(gk.get("player_id"))), dict) else None
        save = _keeper_save_probability(profile, global_prior)
        sot = _project_sot_faced(team_id, opponent_id, team_report, team_idx)
        p_save = _num(save.get("save_probability"))
        projected_sot = _num(sot.get("projected_sot_faced"))

        if p_save is None or projected_sot is None:
            rows.append({
                **gk,
                "opponent_team_id": opponent_id,
                "status": "PROFILE_NOT_MODELABLE",
                "save_profile": save,
                "opponent_sot_projection": sot,
                "actionable": False,
                "decision_weight": 0.0,
            })
            continue

        lam = max(0.0, projected_sot * p_save)
        modeled += 1
        rows.append({
            **gk,
            "opponent_team_id": opponent_id,
            "status": "LIVE_RESEARCH_GK_SAVES_DISTRIBUTION",
            "save_profile": save,
            "opponent_sot_projection": sot,
            "expected_minutes": 90.0,
            "goalkeeper_substitution_risk_modeled": False,
            "predictive_distribution": "POISSON_SAVES_FROM_PROJECTED_SOT_X_SHRUNK_SAVE_PROXY",
            "expected_saves": round(lam, 6),
            "lines": _line_table(lam),
            "observed_sportsbook_save_line": None,
            "observed_sportsbook_price": None,
            "edge_vs_market": None,
            "actionable": False,
            "decision_weight": 0.0,
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": fixture.get("fixture_id"),
        "status": "LIVE_RESEARCH_GK_SAVES",
        "model": "PROJECTED_OPPONENT_SOT_X_SHRUNK_GK_SAVE_RESULT_PROXY__POISSON_v0.1",
        "goalkeepers": rows,
        "modeled_goalkeepers": modeled,
        "global_save_proxy_prior": global_prior,
        "psxg_available": False,
        "shot_quality_adjusted": False,
        "market_prices_attached": False,
        "actionable": False,
        "decision_weight": 0.0,
        "production_status": "LIVE_RESEARCH_NOT_ACTIONABLE",
        "calibration_gate": {
            "minimum_oos_gk_games_for_review": 300,
            "minimum_oos_gk_games_for_market_review": 750,
            "minimum_oos_gk_games_for_actionable_review": 1500,
            "requires": [
                "walk-forward Brier/log-loss by save line",
                "save-count calibration and overdispersion review",
                "opponent SOT projection calibration",
                "verified goalkeeper starter at prediction time",
                "observed sportsbook save line and price",
                "true CLV before any production promotion",
                "PSxG/shot-quality data before claiming goalkeeper quality impact",
            ],
        },
        "policy": "GK SAVES ARE MODELED AS PROJECTED SOT FACED X SHRUNK SAVE RESULT PROXY; PROXY IS NOT PSxG; NO VERIFIED SAVE PRICE=NO EV PICK",
    }


def attach(payload: dict[str, Any]) -> dict[str, int | bool]:
    registry = load_gk_registry()
    team_report = load_team_trends()
    modeled_events = modeled_goalkeepers = 0
    for event in payload.get("events") or []:
        if not isinstance(event, dict) or event.get("event_type") != "SOCCER_REFRESH" or event.get("stage") in {"POSTGAME", "HT", "CLOSE"}:
            continue
        intel = build(event, registry, team_report)
        event["gk_saves_intelligence"] = intel
        if intel.get("status") == "LIVE_RESEARCH_GK_SAVES":
            modeled_events += 1
            modeled_goalkeepers += int(intel.get("modeled_goalkeepers") or 0)
        mi = event.get("match_intelligence")
        if isinstance(mi, dict) and isinstance(mi.get("areas"), dict):
            mi["areas"]["gk_saves"] = intel
    return {
        "goalkeeper_registry_loaded": bool(registry),
        "team_trends_loaded": bool(team_report),
        "modeled_events": modeled_events,
        "modeled_goalkeepers": modeled_goalkeepers,
        "provider_requests_added": 0,
    }
