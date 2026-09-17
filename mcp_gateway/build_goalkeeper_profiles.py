from __future__ import annotations

import argparse
import json
import os
from typing import Any


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build descriptive GK profiles from persisted player-trend history.")
    parser.add_argument("--player-trends", default="soccer_edge_state/analysis/player_trends.json")
    parser.add_argument("--output", default="soccer_edge_state/analysis/goalkeeper_profiles.json")
    args = parser.parse_args()

    try:
        with open(args.player_trends, encoding="utf-8") as fh:
            source = json.load(fh)
    except (OSError, json.JSONDecodeError):
        source = {}

    profiles: dict[str, dict[str, Any]] = {}
    for player in source.get("players") or []:
        if not isinstance(player, dict) or player.get("player_id") is None:
            continue
        position = str(player.get("last_known_position") or "").upper()
        windows = player.get("windows") if isinstance(player.get("windows"), dict) else {}
        last20 = windows.get("last_20") if isinstance(windows.get("last_20"), dict) else {}
        gk20 = last20.get("goalkeeper") if isinstance(last20.get("goalkeeper"), dict) else {}
        gk_matches = int(gk20.get("gk_matches") or 0)
        is_gk = position in {"G", "GK", "GOALKEEPER"} or gk_matches > 0
        if not is_gk:
            continue

        window_out: dict[str, Any] = {}
        for key in ("last_5", "last_10", "last_20"):
            block = windows.get(key) if isinstance(windows.get(key), dict) else {}
            gk = block.get("goalkeeper") if isinstance(block.get("goalkeeper"), dict) else {}
            proxy = gk.get("save_result_proxy") if isinstance(gk.get("save_result_proxy"), dict) else {}
            window_out[key] = {
                "gk_matches": int(gk.get("gk_matches") or 0),
                "avg_minutes": _num(block.get("avg_minutes")),
                "avg_rating": _num(block.get("avg_rating")),
                "avg_saves": _num(gk.get("avg_saves")),
                "avg_goals_conceded": _num(gk.get("avg_goals_conceded")),
                "save_result_proxy_n": int(proxy.get("n") or 0),
                "save_result_proxy_saves": _num(proxy.get("saves")),
                "save_result_proxy_goals_conceded": _num(proxy.get("goals_conceded")),
                "save_result_proxy_sot_proxy": (
                    round((_num(proxy.get("saves")) or 0.0) + (_num(proxy.get("goals_conceded")) or 0.0), 3)
                    if _num(proxy.get("saves")) is not None and _num(proxy.get("goals_conceded")) is not None
                    else None
                ),
                "save_result_proxy": _num(proxy.get("save_result_proxy")),
            }

        sample = window_out["last_20"]["gk_matches"]
        profiles[str(player["player_id"])] = {
            "player_id": player["player_id"],
            "name": player.get("name"),
            "last_known_position": player.get("last_known_position"),
            "team_ids": player.get("team_ids") or [],
            "matches_captured": int(player.get("matches_captured") or 0),
            "postgame_matches_captured": int(player.get("postgame_matches_captured") or 0),
            "sample_band": "HIGH" if sample >= 15 else "MEDIUM" if sample >= 6 else "LOW",
            "windows": window_out,
            "impact_model_available": False,
            "profile_scope": "DESCRIPTIVE_SAVES_AND_CONCEDED_ONLY",
        }

    report = {
        "schema_version": "1.1.0",
        "status": "RESEARCH_GOALKEEPER_PROFILE_REGISTRY",
        "source_player_trends_schema": source.get("schema_version"),
        "goalkeepers": profiles,
        "goalkeeper_count": len(profiles),
        "goalkeepers_with_5plus_matches": sum((p["windows"]["last_20"]["gk_matches"] >= 5) for p in profiles.values()),
        "goalkeepers_with_10plus_matches": sum((p["windows"]["last_20"]["gk_matches"] >= 10) for p in profiles.values()),
        "policy": [
            "Profiles are descriptive only and have zero decision weight.",
            "save_result_proxy = saves/(saves+goals_conceded); it is not PSxG, not shot-quality adjusted and not a verified complete SOT-faced model.",
            "No canonical goal lambda is adjusted from this registry.",
            "A true goalkeeper impact model requires sufficient finalized GK samples plus an OOS feature-lift study; PSxG/xG quality remains external data.",
        ],
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps({k: report[k] for k in ("status", "goalkeeper_count", "goalkeepers_with_5plus_matches", "goalkeepers_with_10plus_matches")}, indent=2))


if __name__ == "__main__":
    main()
