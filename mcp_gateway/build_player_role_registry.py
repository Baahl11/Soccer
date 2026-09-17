from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any

PRIOR_STRENGTH_RATE = 8.0
PRIOR_STRENGTH_MINUTES = 6.0


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _aggregate_prior(players: list[dict[str, Any]]) -> dict[str, Any]:
    start_hits = start_n = m60_hits = m60_n = 0
    starter_minutes_weighted = starter_minutes_n = 0.0
    for player in players:
        block = (((player.get("windows") or {}).get("last_20") or {}).get("role") or {})
        sn = int(block.get("substitute_flag_observed_n") or 0)
        sh = int(block.get("starts") or 0)
        m60 = block.get("minutes_60plus") if isinstance(block.get("minutes_60plus"), dict) else {}
        mn = int(m60.get("n") or 0); mh = int(m60.get("hits") or 0)
        sam = _num(block.get("starter_avg_minutes"))
        start_n += sn; start_hits += sh; m60_n += mn; m60_hits += mh
        if sam is not None and sh > 0:
            starter_minutes_weighted += sam * sh
            starter_minutes_n += sh
    return {
        "start_n": start_n,
        "start_rate": start_hits / start_n if start_n else None,
        "minutes_60_n": m60_n,
        "minutes_60_rate": m60_hits / m60_n if m60_n else None,
        "starter_minutes_n": int(starter_minutes_n),
        "starter_avg_minutes": starter_minutes_weighted / starter_minutes_n if starter_minutes_n else None,
    }


def _position_key(value: Any) -> str:
    text = str(value or "UNKNOWN").strip().upper()
    return text or "UNKNOWN"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build shrinkage-based player role/minutes research probabilities.")
    parser.add_argument("--player-trends", default="soccer_edge_state/analysis/player_trends.json")
    parser.add_argument("--output", default="soccer_edge_state/analysis/player_role_registry.json")
    args = parser.parse_args()
    try:
        with open(args.player_trends, encoding="utf-8") as fh:
            source = json.load(fh)
    except (OSError, json.JSONDecodeError):
        source = {}
    players = [p for p in (source.get("players") or []) if isinstance(p, dict)]
    global_prior = _aggregate_prior(players)
    by_position: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for player in players:
        by_position[_position_key(player.get("last_known_position"))].append(player)
    position_priors = {key: _aggregate_prior(group) for key, group in by_position.items()}

    profiles: dict[str, dict[str, Any]] = {}
    for player in players:
        pid = player.get("player_id")
        if pid is None:
            continue
        position = _position_key(player.get("last_known_position"))
        pos_prior = position_priors.get(position) or {}
        # Position prior is used only when it has enough observations; otherwise global.
        prior = pos_prior if int(pos_prior.get("minutes_60_n") or 0) >= 30 else global_prior
        block = (((player.get("windows") or {}).get("last_20") or {}).get("role") or {})
        start_n = int(block.get("substitute_flag_observed_n") or 0)
        start_hits = int(block.get("starts") or 0)
        m60 = block.get("minutes_60plus") if isinstance(block.get("minutes_60plus"), dict) else {}
        m60_n = int(m60.get("n") or 0); m60_hits = int(m60.get("hits") or 0)
        starter_avg = _num(block.get("starter_avg_minutes"))
        global_start = _num(prior.get("start_rate"))
        global_m60 = _num(prior.get("minutes_60_rate"))
        global_starter_minutes = _num(prior.get("starter_avg_minutes"))
        p_start = (
            (start_hits + PRIOR_STRENGTH_RATE * global_start) / (start_n + PRIOR_STRENGTH_RATE)
            if global_start is not None else None
        )
        p60 = (
            (m60_hits + PRIOR_STRENGTH_RATE * global_m60) / (m60_n + PRIOR_STRENGTH_RATE)
            if global_m60 is not None else None
        )
        expected_starter_minutes = None
        if global_starter_minutes is not None:
            if starter_avg is None or start_hits <= 0:
                expected_starter_minutes = global_starter_minutes
            else:
                expected_starter_minutes = (
                    starter_avg * start_hits + global_starter_minutes * PRIOR_STRENGTH_MINUTES
                ) / (start_hits + PRIOR_STRENGTH_MINUTES)
        profiles[str(pid)] = {
            "player_id": pid,
            "name": player.get("name"),
            "position": player.get("last_known_position"),
            "team_ids": player.get("team_ids") or [],
            "matches_captured": int(player.get("matches_captured") or 0),
            "postgame_matches_captured": int(player.get("postgame_matches_captured") or 0),
            "last20_start_observed_n": start_n,
            "last20_starts": start_hits,
            "p_start_smoothed": round(p_start, 6) if p_start is not None else None,
            "last20_minutes_60_n": m60_n,
            "last20_minutes_60_hits": m60_hits,
            "p_60plus_smoothed": round(p60, 6) if p60 is not None else None,
            "expected_minutes_if_confirmed_starter": round(expected_starter_minutes, 3) if expected_starter_minutes is not None else None,
            "prior_scope": position if prior is pos_prior else "GLOBAL",
            "sample_band": "HIGH" if m60_n >= 15 else "MEDIUM" if m60_n >= 6 else "LOW",
        }

    report = {
        "schema_version": "1.0.0",
        "status": "RESEARCH_PLAYER_ROLE_REGISTRY",
        "source_player_trends_schema": source.get("schema_version"),
        "global_prior": global_prior,
        "position_priors": position_priors,
        "prior_strength_rate": PRIOR_STRENGTH_RATE,
        "prior_strength_minutes": PRIOR_STRENGTH_MINUTES,
        "profiles": profiles,
        "profile_count": len(profiles),
        "policy": [
            "Player-specific start and 60+ rates are shrunk toward a sufficiently sampled position prior, otherwise global prior.",
            "Confirmed live XI overrides pregame start probability; historical p_start remains diagnostic only.",
            "Expected starter minutes are descriptive research output and cannot alter canonical bets without OOS calibration.",
            "No missing player or replacement quality is inferred.",
        ],
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps({"status": report["status"], "profile_count": len(profiles), "global_prior": global_prior}, indent=2))


if __name__ == "__main__":
    main()
