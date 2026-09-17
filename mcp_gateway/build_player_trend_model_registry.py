from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any

PRIOR_EXPOSURE_MINUTES = 450.0
MIN_POSITION_EXPOSURE_MINUTES = 1800.0
METRICS = {
    "shots": "shots_total",
    "shots_on_target": "sot_total",
    "goals": "goals_total",
    "assists": "assists_total",
    "yellow_cards": "yellow_cards_total",
}


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _position(value: Any) -> str:
    text = str(value or "UNKNOWN").strip().upper()
    return text or "UNKNOWN"


def _load(path: str) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _window(player: dict[str, Any], name: str) -> dict[str, Any]:
    windows = player.get("windows") if isinstance(player.get("windows"), dict) else {}
    block = windows.get(name) if isinstance(windows.get(name), dict) else {}
    return block


def _metric_total(block: dict[str, Any], metric: str) -> float | None:
    field = METRICS[metric]
    value = _num(block.get(field))
    if value is not None:
        return max(0.0, value)
    legacy = {"goals": "goals", "assists": "assists"}.get(metric)
    if legacy:
        value = _num(block.get(legacy))
        return max(0.0, value) if value is not None else None
    return None


def _collect_priors(players: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    global_totals = {metric: 0.0 for metric in METRICS}
    global_exposure = {metric: 0.0 for metric in METRICS}
    position_totals: dict[str, dict[str, float]] = defaultdict(lambda: {metric: 0.0 for metric in METRICS})
    position_exposure: dict[str, dict[str, float]] = defaultdict(lambda: {metric: 0.0 for metric in METRICS})

    for player in players:
        block = _window(player, "last_20")
        minutes = _num(block.get("total_minutes"))
        if minutes is None or minutes <= 0:
            continue
        pos = _position(player.get("last_known_position"))
        for metric in METRICS:
            total = _metric_total(block, metric)
            if total is None:
                continue
            global_totals[metric] += total
            global_exposure[metric] += minutes
            position_totals[pos][metric] += total
            position_exposure[pos][metric] += minutes

    global_prior: dict[str, Any] = {}
    for metric in METRICS:
        exposure = global_exposure[metric]
        total = global_totals[metric]
        global_prior[metric] = {
            "events": round(total, 6),
            "exposure_minutes": round(exposure, 3),
            "rate_per90": round(total * 90.0 / exposure, 6) if exposure > 0 else None,
        }

    by_position: dict[str, dict[str, Any]] = {}
    for pos in position_totals:
        by_position[pos] = {}
        for metric in METRICS:
            exposure = position_exposure[pos][metric]
            total = position_totals[pos][metric]
            by_position[pos][metric] = {
                "events": round(total, 6),
                "exposure_minutes": round(exposure, 3),
                "rate_per90": round(total * 90.0 / exposure, 6) if exposure > 0 else None,
            }
    return global_prior, by_position


def _posterior(
    count: float | None,
    exposure_minutes: float | None,
    global_prior: dict[str, Any],
    position_prior: dict[str, Any] | None,
) -> dict[str, Any]:
    if count is None or exposure_minutes is None or exposure_minutes <= 0:
        return {"status": "INSUFFICIENT_EXPOSURE", "posterior_mean_per90": None}

    chosen = position_prior if (
        isinstance(position_prior, dict)
        and (_num(position_prior.get("exposure_minutes")) or 0.0) >= MIN_POSITION_EXPOSURE_MINUTES
        and _num(position_prior.get("rate_per90")) is not None
    ) else global_prior
    prior_scope = "POSITION" if chosen is position_prior else "GLOBAL"
    prior_per90 = _num(chosen.get("rate_per90")) if isinstance(chosen, dict) else None
    if prior_per90 is None:
        return {"status": "NO_PRIOR_AVAILABLE", "posterior_mean_per90": None}

    prior_rate_per_minute = prior_per90 / 90.0
    prior_pseudo_count = max(0.25, prior_rate_per_minute * PRIOR_EXPOSURE_MINUTES)
    shape = max(0.0, count) + prior_pseudo_count
    rate_minutes = exposure_minutes + PRIOR_EXPOSURE_MINUTES
    posterior_per90 = shape / rate_minutes * 90.0
    raw_per90 = max(0.0, count) / exposure_minutes * 90.0
    return {
        "status": "MODELED",
        "model": "GAMMA_POISSON_RATE_SHRINKAGE_v0.1",
        "observed_events": round(max(0.0, count), 6),
        "observed_exposure_minutes": round(exposure_minutes, 3),
        "raw_rate_per90": round(raw_per90, 6),
        "prior_scope": prior_scope,
        "prior_rate_per90": round(prior_per90, 6),
        "prior_exposure_minutes": PRIOR_EXPOSURE_MINUTES,
        "posterior_gamma_shape": round(shape, 6),
        "posterior_gamma_rate_minutes": round(rate_minutes, 3),
        "posterior_mean_per90": round(posterior_per90, 6),
        "line_probability_created": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build research-only probabilistic player trend rate profiles.")
    parser.add_argument("--player-trends", default="soccer_edge_state/analysis/player_trends.json")
    parser.add_argument("--player-roles", default="soccer_edge_state/analysis/player_role_registry.json")
    parser.add_argument("--output", default="soccer_edge_state/analysis/player_trend_model_registry.json")
    args = parser.parse_args()

    trends = _load(args.player_trends)
    roles = _load(args.player_roles)
    players = [row for row in (trends.get("players") or []) if isinstance(row, dict)]
    role_profiles = roles.get("profiles") if isinstance(roles.get("profiles"), dict) else {}
    global_prior, position_priors = _collect_priors(players)

    profiles: dict[str, dict[str, Any]] = {}
    modeled_metric_windows = 0
    for player in players:
        pid = player.get("player_id")
        if pid is None:
            continue
        pos = _position(player.get("last_known_position"))
        role = role_profiles.get(str(pid)) if isinstance(role_profiles.get(str(pid)), dict) else {}
        windows_out: dict[str, Any] = {}
        for name in ("last_5", "last_10", "last_20"):
            block = _window(player, name)
            minutes = _num(block.get("total_minutes"))
            metrics: dict[str, Any] = {}
            for metric in METRICS:
                total = _metric_total(block, metric)
                modeled = _posterior(
                    total,
                    minutes,
                    global_prior.get(metric) if isinstance(global_prior.get(metric), dict) else {},
                    (position_priors.get(pos) or {}).get(metric) if isinstance(position_priors.get(pos), dict) else None,
                )
                if modeled.get("status") == "MODELED":
                    modeled_metric_windows += 1
                metrics[metric] = modeled
            windows_out[name] = {
                "captured_matches": int(block.get("n") or 0),
                "played_matches": int(block.get("played_n") or 0),
                "total_minutes": minutes,
                "metrics": metrics,
            }

        l20 = windows_out["last_20"]
        expected_minutes = _num(role.get("expected_minutes_if_confirmed_starter"))
        expected_counts: dict[str, float | None] = {}
        for metric, model in (l20.get("metrics") or {}).items():
            rate = _num(model.get("posterior_mean_per90")) if isinstance(model, dict) else None
            expected_counts[metric] = (
                round(rate * expected_minutes / 90.0, 6)
                if rate is not None and expected_minutes is not None and expected_minutes > 0
                else None
            )

        exposure20 = _num(l20.get("total_minutes")) or 0.0
        profiles[str(pid)] = {
            "player_id": pid,
            "name": player.get("name"),
            "position": player.get("last_known_position"),
            "team_ids": player.get("team_ids") or [],
            "matches_captured": int(player.get("matches_captured") or 0),
            "postgame_matches_captured": int(player.get("postgame_matches_captured") or 0),
            "sample_band": "HIGH" if exposure20 >= 900 else "MEDIUM" if exposure20 >= 360 else "LOW",
            "role_model": {
                "historical_p_start_smoothed": role.get("p_start_smoothed"),
                "p_60plus_smoothed": role.get("p_60plus_smoothed"),
                "expected_minutes_if_confirmed_starter": role.get("expected_minutes_if_confirmed_starter"),
                "role_sample_band": role.get("sample_band"),
            },
            "windows": windows_out,
            "expected_counts_if_confirmed_starter": expected_counts,
            "actionable": False,
            "decision_weight": 0.0,
        }

    report = {
        "schema_version": "1.1.0",
        "status": "RESEARCH_PLAYER_TREND_MODEL_REGISTRY",
        "source_player_trends_schema": trends.get("schema_version"),
        "source_player_role_registry_schema": roles.get("schema_version"),
        "model": "ROLE_MINUTES_PLUS_GAMMA_POISSON_COUNT_RATES_v0.1",
        "prior_exposure_minutes": PRIOR_EXPOSURE_MINUTES,
        "minimum_position_prior_exposure_minutes": MIN_POSITION_EXPOSURE_MINUTES,
        "global_priors": global_prior,
        "position_priors": position_priors,
        "profiles": profiles,
        "profile_count": len(profiles),
        "modeled_metric_windows": modeled_metric_windows,
        "decision_weight": 0.0,
        "actionable": False,
        "calibration_gate": {
            "minimum_oos_player_games_for_rate_review": 500,
            "minimum_oos_player_games_before_prop_dependency": 1500,
            "requires": [
                "walk-forward calibration by competition and position",
                "confirmed XI at live prediction time",
                "minutes MAE review",
                "count-rate Poisson dispersion review",
                "no sportsbook threshold conversion before dedicated prop modules",
                "bookmaker card-scoring rules mapped explicitly before any player-card promotion",
            ],
        },
        "policy": [
            "Only persisted player trend history and role/minutes registry inputs are used.",
            "Count rates are exposure-normalized by observed minutes and shrunk toward sufficiently sampled position priors, otherwise global priors.",
            "Expected counts for a confirmed starter use shrunk rate times role-model expected minutes; they are research expectations, not betting probabilities.",
            "No exact shots/SOT/goals/assists/yellow-card line probability is created here; dedicated prop modules remain separate checkpoints.",
            "Player-card rate modeling uses yellow cards only; red cards remain separate until sportsbook-specific card rules are explicitly mapped.",
            "No player trend output can change classification, tier, stake, canonical probabilities, or bet eligibility.",
        ],
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps({
        "status": report["status"],
        "profile_count": report["profile_count"],
        "modeled_metric_windows": modeled_metric_windows,
        "source_player_trends_schema": report["source_player_trends_schema"],
    }, indent=2))


if __name__ == "__main__":
    main()
