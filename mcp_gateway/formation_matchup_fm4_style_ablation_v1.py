from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from mcp_gateway import formation_matchup_engine_v1 as fm1
from mcp_gateway import formation_matchup_fm3_oos_v1 as fm3

MODEL_VERSION = "FORMATION_MATCHUP_FM4_STYLE_ABLATION_V1.0.0"
SCHEMA_VERSION = "1.0.0"
MIN_TEAM_STYLE_N = 3
MIN_STYLE_TRAIN_N = 30
RIDGE_ALPHA = 8.0
MAX_STYLE_RESIDUAL_ABS = {
    "SHOTS": 3.0,
    "SOT": 1.25,
    "GOALS": 0.75,
}

STYLE_FIELDS = (
    "possession",
    "shot_accuracy",
    "box_share",
    "blocked_share",
    "fouls",
    "yellow_cards",
)


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _dt(value: Any) -> datetime:
    text = str(value or "").strip().replace("Z", "+00:00")
    try:
        out = datetime.fromisoformat(text)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    if out.tzinfo is None:
        out = out.replace(tzinfo=timezone.utc)
    return out.astimezone(timezone.utc)


def _formation_geometry(value: Any) -> dict[str, float] | None:
    text = str(value or "").strip()
    if not re.fullmatch(r"\d(?:-\d){2,4}", text):
        return None
    parts = [int(piece) for piece in text.split("-")]
    if any(piece < 0 or piece > 6 for piece in parts):
        return None
    return {
        "back_line": float(parts[0]),
        "front_line": float(parts[-1]),
        "outfield_lines": float(len(parts)),
        "central_layers": float(sum(parts[1:-1])) if len(parts) > 2 else 0.0,
    }


def _history_events(
    history_dir: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fixtures = fm1.formation_v2._enhanced_load_history(history_dir)
    tactical_events: list[dict[str, Any]] = []
    personnel_events: list[dict[str, Any]] = []
    for rec in fixtures.values():
        kickoff = _dt(rec.get("kickoff_local"))
        if kickoff == datetime.min.replace(tzinfo=timezone.utc):
            continue
        home_id = int(rec.get("home_team_id") or 0)
        away_id = int(rec.get("away_team_id") or 0)
        if not home_id or not away_id:
            continue

        home_shots, away_shots, _ = fm1.metric_triplet(rec, "total_shots")
        home_sot, away_sot, _ = fm1.metric_triplet(rec, "shots_on_goal")
        home_inside, away_inside, _ = fm1.metric_triplet(rec, "shots_inside_box")
        home_blocked, away_blocked, _ = fm1.metric_triplet(rec, "blocked_shots")
        home_possession, away_possession, _ = fm1.metric_triplet(rec, "possession")
        home_fouls, away_fouls, _ = fm1.metric_triplet(rec, "fouls")
        home_yellow, away_yellow, _ = fm1.metric_triplet(rec, "yellow_cards")

        def obs(shots, sot, inside, blocked, possession, fouls, yellow):
            return {
                "possession": possession,
                "shot_accuracy": (
                    sot / shots if sot is not None and shots is not None and shots > 0 else None
                ),
                "box_share": (
                    inside / shots if inside is not None and shots is not None and shots > 0 else None
                ),
                "blocked_share": (
                    blocked / shots if blocked is not None and shots is not None and shots > 0 else None
                ),
                "fouls": fouls,
                "yellow_cards": yellow,
            }

        tactical_events.append(
            {
                "fixture_id": rec.get("fixture_id"),
                "kickoff": kickoff,
                "home_team_id": home_id,
                "away_team_id": away_id,
                "home_observation": obs(
                    home_shots,
                    home_sot,
                    home_inside,
                    home_blocked,
                    home_possession,
                    home_fouls,
                    home_yellow,
                ),
                "away_observation": obs(
                    away_shots,
                    away_sot,
                    away_inside,
                    away_blocked,
                    away_possession,
                    away_fouls,
                    away_yellow,
                ),
            }
        )

        lineup_detail = rec.get("lineup_detail")
        if isinstance(lineup_detail, dict) and lineup_detail.get("both_xi_confirmed") is True:
            detail_teams = []
            for team in lineup_detail.get("teams") or []:
                if not isinstance(team, dict) or team.get("team_id") is None:
                    continue
                starters = [
                    {
                        "id": player.get("id"),
                        "name": player.get("name"),
                        "pos": player.get("pos"),
                        "grid": player.get("grid"),
                    }
                    for player in (team.get("starters") or [])
                    if isinstance(player, dict) and player.get("id") is not None
                ]
                if len(starters) < 11:
                    continue
                detail_teams.append(
                    {
                        "team_id": int(team.get("team_id")),
                        "team": team.get("team"),
                        "formation": team.get("formation"),
                        "coach_id": team.get("coach_id"),
                        "coach": team.get("coach"),
                        "starters": starters,
                    }
                )
            if len(detail_teams) == 2:
                personnel_events.append(
                    {
                        "fixture_id": rec.get("fixture_id"),
                        "kickoff": kickoff,
                        "captured_at": rec.get("lineup_detail_at"),
                        "stage": rec.get("lineup_detail_stage"),
                        "teams": detail_teams,
                    }
                )

    tactical_events.sort(key=lambda row: (row["kickoff"], int(row.get("fixture_id") or 0)))
    personnel_events.sort(key=lambda row: (row["kickoff"], int(row.get("fixture_id") or 0)))
    return tactical_events, personnel_events


def _tactical_history_events(history_dir: str) -> list[dict[str, Any]]:
    return _history_events(history_dir)[0]


def _team_from_personnel_event(
    event: dict[str, Any] | None,
    team_id: int,
) -> dict[str, Any] | None:
    if not isinstance(event, dict) or not team_id:
        return None
    for team in event.get("teams") or []:
        if not isinstance(team, dict):
            continue
        try:
            candidate_id = int(team.get("team_id") or 0)
        except (TypeError, ValueError):
            continue
        if candidate_id == team_id:
            return team
    return None


def _starter_map(team: dict[str, Any] | None) -> dict[int, dict[str, Any]]:
    if not isinstance(team, dict):
        return {}
    out: dict[int, dict[str, Any]] = {}
    for player in team.get("starters") or []:
        if not isinstance(player, dict) or player.get("id") is None:
            continue
        try:
            pid = int(player.get("id"))
        except (TypeError, ValueError):
            continue
        out[pid] = {
            "pos": player.get("pos"),
            "grid": player.get("grid"),
        }
    return out


def _personnel_features(
    current: dict[str, Any] | None,
    prior: list[dict[str, Any]],
) -> dict[str, Any] | None:
    current_map = _starter_map(current)
    if len(current_map) < 11:
        return None

    prior = [row for row in prior if isinstance(row, dict)]
    previous = prior[-1] if prior else None
    previous_map = _starter_map(previous)
    overlap_ids = set(current_map) & set(previous_map)
    denominator = float(len(current_map))

    comparable_pos = [
        pid
        for pid in overlap_ids
        if current_map[pid].get("pos") not in (None, "")
        and previous_map[pid].get("pos") not in (None, "")
    ]
    comparable_grid = [
        pid
        for pid in overlap_ids
        if current_map[pid].get("grid") not in (None, "")
        and previous_map[pid].get("grid") not in (None, "")
    ]

    current_coach = current.get("coach_id") if isinstance(current, dict) else None
    previous_coach = previous.get("coach_id") if isinstance(previous, dict) else None
    coach_same_previous = (
        bool(current_coach == previous_coach)
        if current_coach is not None and previous_coach is not None
        else None
    )
    coach_consecutive_prior_matches = 0
    if current_coach is not None:
        for row in reversed(prior):
            prior_coach = row.get("coach_id")
            if prior_coach is None or prior_coach != current_coach:
                break
            coach_consecutive_prior_matches += 1

    last3 = prior[-3:]
    core_return_rate = None
    if len(last3) == 3:
        appearances = {
            pid: sum(int(pid in _starter_map(prev)) for prev in last3)
            for pid in current_map
        }
        core_return_rate = sum(int(count >= 2) for count in appearances.values()) / denominator

    return {
        "current_starter_count": len(current_map),
        "prior_confirmed_xi_count": len(prior),
        "previous_xi_overlap_count": len(overlap_ids) if previous is not None else None,
        "previous_xi_overlap_rate": (
            round(len(overlap_ids) / denominator, 6)
            if previous is not None
            else None
        ),
        "new_starter_count_vs_previous": (
            len(current_map) - len(overlap_ids)
            if previous is not None
            else None
        ),
        "last3_core_return_rate": (
            round(core_return_rate, 6) if core_return_rate is not None else None
        ),
        "position_continuity_rate": (
            round(
                sum(
                    int(current_map[pid].get("pos") == previous_map[pid].get("pos"))
                    for pid in comparable_pos
                )
                / len(comparable_pos),
                6,
            )
            if comparable_pos
            else None
        ),
        "position_comparable_starters": len(comparable_pos),
        "grid_continuity_rate": (
            round(
                sum(
                    int(current_map[pid].get("grid") == previous_map[pid].get("grid"))
                    for pid in comparable_grid
                )
                / len(comparable_grid),
                6,
            )
            if comparable_grid
            else None
        ),
        "grid_comparable_starters": len(comparable_grid),
        "coach_id": current_coach,
        "previous_coach_id": previous_coach,
        "coach_same_as_previous": coach_same_previous,
        "coach_consecutive_prior_matches": coach_consecutive_prior_matches,
    }


def _append_style_observation(
    team_style: dict[int, dict[str, list[float]]],
    team_id: int,
    observation: dict[str, Any],
) -> None:
    if not team_id:
        return
    for field in STYLE_FIELDS:
        value = _num(observation.get(field))
        if value is not None:
            team_style[team_id][field].append(value)


def _team_observation(row: dict[str, Any], role: str) -> dict[str, float | None]:
    prefix = "home" if role == "home" else "away"
    shots = _num(row.get(f"{prefix}_shots"))
    sot = _num(row.get(f"{prefix}_sot"))
    inside = _num(row.get(f"{prefix}_shots_inside_box"))
    blocked = _num(row.get(f"{prefix}_blocked_shots"))
    return {
        "possession": _num(row.get(f"{prefix}_possession")),
        "shot_accuracy": (
            sot / shots if sot is not None and shots is not None and shots > 0 else None
        ),
        "box_share": (
            inside / shots if inside is not None and shots is not None and shots > 0 else None
        ),
        "blocked_share": (
            blocked / shots if blocked is not None and shots is not None and shots > 0 else None
        ),
        "fouls": _num(row.get(f"{prefix}_fouls")),
        "yellow_cards": _num(row.get(f"{prefix}_yellow_cards")),
    }


def _profile(history: dict[str, list[float]]) -> tuple[dict[str, float], int] | tuple[None, int]:
    counts = [len(history.get(field, [])) for field in STYLE_FIELDS]
    if not counts:
        return None, 0
    n = min(counts)
    if n < MIN_TEAM_STYLE_N:
        return None, n
    profile = {
        field: sum(history[field]) / len(history[field])
        for field in STYLE_FIELDS
    }
    return profile, n


def _style_features(
    home_profile: dict[str, float] | None,
    away_profile: dict[str, float] | None,
    home_formation: Any,
    away_formation: Any,
) -> list[float] | None:
    if home_profile is None or away_profile is None:
        return None
    hg = _formation_geometry(home_formation)
    ag = _formation_geometry(away_formation)
    if hg is None or ag is None:
        return None

    # Continuous contrasts only. Every style value comes from prior verified matches;
    # formation geometry comes from the current verified pre-kickoff formation.
    return [
        home_profile["possession"] - away_profile["possession"],
        home_profile["shot_accuracy"] - away_profile["shot_accuracy"],
        home_profile["box_share"] - away_profile["box_share"],
        home_profile["blocked_share"] - away_profile["blocked_share"],
        home_profile["fouls"] - away_profile["fouls"],
        home_profile["yellow_cards"] - away_profile["yellow_cards"],
        hg["back_line"] - ag["back_line"],
        hg["front_line"] - ag["front_line"],
        hg["outfield_lines"] - ag["outfield_lines"],
        hg["central_layers"] - ag["central_layers"],
    ]


def _transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [list(col) for col in zip(*matrix)] if matrix else []


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float] | None:
    n = len(vector)
    if n == 0 or len(matrix) != n:
        return None
    aug = [list(matrix[i]) + [float(vector[i])] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(aug[row][col]))
        if abs(aug[pivot][col]) < 1e-10:
            return None
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        denom = aug[col][col]
        aug[col] = [value / denom for value in aug[col]]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if abs(factor) < 1e-12:
                continue
            aug[row] = [
                aug[row][j] - factor * aug[col][j]
                for j in range(n + 1)
            ]
    return [aug[i][-1] for i in range(n)]


def _standardize_training(
    records: list[tuple[list[float], float]],
    current: list[float],
) -> tuple[list[list[float]], list[float], list[float]] | None:
    if len(records) < MIN_STYLE_TRAIN_N:
        return None
    width = len(current)
    if any(len(features) != width for features, _ in records):
        return None
    columns = [[features[j] for features, _ in records] for j in range(width)]
    means = [sum(col) / len(col) for col in columns]
    stds = []
    for col, mean in zip(columns, means):
        variance = sum((value - mean) ** 2 for value in col) / max(len(col) - 1, 1)
        stds.append(math.sqrt(variance) if variance > 1e-12 else 1.0)

    x = [
        [1.0] + [(features[j] - means[j]) / stds[j] for j in range(width)]
        for features, _ in records
    ]
    y = [target for _, target in records]
    cur = [1.0] + [(current[j] - means[j]) / stds[j] for j in range(width)]
    return x, y, cur


def _ridge_predict(
    records: list[tuple[list[float], float]],
    current: list[float],
) -> float | None:
    prepared = _standardize_training(records, current)
    if prepared is None:
        return None
    x, y, cur = prepared
    xt = _transpose(x)
    width = len(xt)
    gram = [[0.0 for _ in range(width)] for _ in range(width)]
    rhs = [0.0 for _ in range(width)]
    for i in range(width):
        for j in range(width):
            gram[i][j] = sum(xt[i][k] * x[k][j] for k in range(len(x)))
        rhs[i] = sum(xt[i][k] * y[k] for k in range(len(x)))
    # Do not penalize intercept.
    for i in range(1, width):
        gram[i][i] += RIDGE_ALPHA
    beta = _solve(gram, rhs)
    if beta is None:
        return None
    return sum(beta[i] * cur[i] for i in range(width))


def _metric(rows: list[dict[str, Any]], prediction_key: str) -> dict[str, Any]:
    errors: list[float] = []
    signed: list[float] = []
    for row in rows:
        actual = _num(row.get("actual"))
        pred = _num(row.get(prediction_key))
        if actual is None or pred is None:
            continue
        delta = pred - actual
        signed.append(delta)
        errors.append(abs(delta))
    return {
        "n": len(errors),
        "mae": round(sum(errors) / len(errors), 6) if errors else None,
        "rmse": (
            round(math.sqrt(sum(value * value for value in signed) / len(signed)), 6)
            if signed else None
        ),
        "mean_error": round(sum(signed) / len(signed), 6) if signed else None,
    }


def _delta(base: float | None, challenger: float | None) -> float | None:
    if base is None or challenger is None:
        return None
    return round(challenger - base, 6)


def _improves(base: float | None, challenger: float | None) -> bool:
    return base is not None and challenger is not None and challenger < base


def build_report(
    source: dict[str, Any],
    history_dir: str | None = None,
) -> dict[str, Any]:
    rows = [
        dict(row)
        for row in (source.get("rows") or [])
        if isinstance(row, dict)
    ]
    rows.sort(key=lambda row: (_dt(row.get("kickoff_local")), int(row.get("fixture_id") or 0)))

    team_style: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: {field: [] for field in STYLE_FIELDS}
    )
    if history_dir:
        tactical_events, personnel_events = _history_events(history_dir)
    else:
        tactical_events, personnel_events = [], []
    tactical_index = 0
    personnel_index = 0
    personnel_by_fixture = {
        int(event.get("fixture_id") or 0): event
        for event in personnel_events
        if int(event.get("fixture_id") or 0)
    }
    team_lineup_history: dict[int, list[dict[str, Any]]] = defaultdict(list)
    tactical_team_ids = {
        int(event["home_team_id"]) for event in tactical_events
    } | {
        int(event["away_team_id"]) for event in tactical_events
    }

    target_state: dict[str, dict[str, Any]] = {}
    for target, (home_key, away_key) in fm3.TARGETS.items():
        target_state[target] = {
            "global_home": [],
            "global_away": [],
            "league_home": defaultdict(list),
            "league_away": defaultdict(list),
            "home_attack": defaultdict(list),
            "home_concede": defaultdict(list),
            "away_attack": defaultdict(list),
            "away_concede": defaultdict(list),
            "training_home": [],
            "training_away": [],
            "eval_home": [],
            "eval_away": [],
            "eval_total": [],
            "eligible_fixture_ids": set(),
        }

    style_eligible_rows = 0
    missing_style_profile_rows = 0
    geometry_missing_rows = 0
    source_team_ids = {
        int(row.get("home_team_id") or 0)
        for row in rows
        if int(row.get("home_team_id") or 0)
    } | {
        int(row.get("away_team_id") or 0)
        for row in rows
        if int(row.get("away_team_id") or 0)
    }
    source_team_overlap = source_team_ids & tactical_team_ids
    prior_density = {
        "rows_with_home_any_prior_style": 0,
        "rows_with_away_any_prior_style": 0,
        "rows_with_both_any_prior_style": 0,
        "rows_with_home_min_field_n_ge_1": 0,
        "rows_with_away_min_field_n_ge_1": 0,
        "rows_with_both_min_field_n_ge_1": 0,
        "rows_with_home_min_field_n_ge_2": 0,
        "rows_with_away_min_field_n_ge_2": 0,
        "rows_with_both_min_field_n_ge_2": 0,
        "rows_with_home_min_field_n_ge_3": 0,
        "rows_with_away_min_field_n_ge_3": 0,
        "rows_with_both_min_field_n_ge_3": 0,
    }
    profile_failure_reasons: dict[str, int] = defaultdict(int)
    field_prior_coverage = {
        field: {
            "home_rows_n_ge_1": 0,
            "away_rows_n_ge_1": 0,
            "both_rows_n_ge_1": 0,
            "home_rows_n_ge_3": 0,
            "away_rows_n_ge_3": 0,
            "both_rows_n_ge_3": 0,
        }
        for field in STYLE_FIELDS
    }
    evaluation_rows: list[dict[str, Any]] = []
    personnel_rows: list[dict[str, Any]] = []
    personnel_coverage = {
        "source_rows": len(rows),
        "current_both_xi_confirmed_rows": 0,
        "rows_with_home_prior_confirmed_xi": 0,
        "rows_with_away_prior_confirmed_xi": 0,
        "rows_with_both_prior_confirmed_xi": 0,
        "rows_with_home_previous_coach_comparable": 0,
        "rows_with_away_previous_coach_comparable": 0,
        "rows_with_both_previous_coach_comparable": 0,
        "rows_with_home_last3_core_return_rate": 0,
        "rows_with_away_last3_core_return_rate": 0,
        "rows_with_both_last3_core_return_rate": 0,
    }

    for row in rows:
        row_kickoff = _dt(row.get("kickoff_local"))
        if history_dir:
            while (
                personnel_index < len(personnel_events)
                and personnel_events[personnel_index]["kickoff"] < row_kickoff
            ):
                personnel_event = personnel_events[personnel_index]
                for team in personnel_event.get("teams") or []:
                    if not isinstance(team, dict) or team.get("team_id") is None:
                        continue
                    history_row = dict(team)
                    history_row["fixture_id"] = personnel_event.get("fixture_id")
                    history_row["kickoff"] = personnel_event.get("kickoff")
                    history_row["captured_at"] = personnel_event.get("captured_at")
                    history_row["stage"] = personnel_event.get("stage")
                    team_lineup_history[int(team.get("team_id"))].append(history_row)
                personnel_index += 1

            while (
                tactical_index < len(tactical_events)
                and tactical_events[tactical_index]["kickoff"] < row_kickoff
            ):
                event = tactical_events[tactical_index]
                _append_style_observation(
                    team_style,
                    int(event["home_team_id"]),
                    event["home_observation"],
                )
                _append_style_observation(
                    team_style,
                    int(event["away_team_id"]),
                    event["away_observation"],
                )
                tactical_index += 1

        home_id = int(row.get("home_team_id") or 0)
        away_id = int(row.get("away_team_id") or 0)

        fixture_id = int(row.get("fixture_id") or 0)
        current_personnel_event = personnel_by_fixture.get(fixture_id)
        current_home_lineup = _team_from_personnel_event(
            current_personnel_event, home_id
        )
        current_away_lineup = _team_from_personnel_event(
            current_personnel_event, away_id
        )
        home_personnel = _personnel_features(
            current_home_lineup, team_lineup_history[home_id]
        )
        away_personnel = _personnel_features(
            current_away_lineup, team_lineup_history[away_id]
        )
        if home_personnel is not None and away_personnel is not None:
            personnel_coverage["current_both_xi_confirmed_rows"] += 1
            home_prior = int(home_personnel.get("prior_confirmed_xi_count") or 0)
            away_prior = int(away_personnel.get("prior_confirmed_xi_count") or 0)
            personnel_coverage["rows_with_home_prior_confirmed_xi"] += int(home_prior > 0)
            personnel_coverage["rows_with_away_prior_confirmed_xi"] += int(away_prior > 0)
            personnel_coverage["rows_with_both_prior_confirmed_xi"] += int(
                home_prior > 0 and away_prior > 0
            )
            home_coach_comparable = (
                home_personnel.get("coach_same_as_previous") is not None
            )
            away_coach_comparable = (
                away_personnel.get("coach_same_as_previous") is not None
            )
            personnel_coverage["rows_with_home_previous_coach_comparable"] += int(
                home_coach_comparable
            )
            personnel_coverage["rows_with_away_previous_coach_comparable"] += int(
                away_coach_comparable
            )
            personnel_coverage["rows_with_both_previous_coach_comparable"] += int(
                home_coach_comparable and away_coach_comparable
            )
            home_last3 = home_personnel.get("last3_core_return_rate") is not None
            away_last3 = away_personnel.get("last3_core_return_rate") is not None
            personnel_coverage["rows_with_home_last3_core_return_rate"] += int(
                home_last3
            )
            personnel_coverage["rows_with_away_last3_core_return_rate"] += int(
                away_last3
            )
            personnel_coverage["rows_with_both_last3_core_return_rate"] += int(
                home_last3 and away_last3
            )

            home_overlap = _num(home_personnel.get("previous_xi_overlap_rate"))
            away_overlap = _num(away_personnel.get("previous_xi_overlap_rate"))
            mean_overlap = (
                round((home_overlap + away_overlap) / 2.0, 6)
                if home_overlap is not None and away_overlap is not None
                else None
            )
            overlap_gap = (
                round(home_overlap - away_overlap, 6)
                if home_overlap is not None and away_overlap is not None
                else None
            )
            both_coach_same = (
                bool(
                    home_personnel.get("coach_same_as_previous")
                    and away_personnel.get("coach_same_as_previous")
                )
                if home_coach_comparable and away_coach_comparable
                else None
            )
            personnel_rows.append(
                {
                    "fixture_id": fixture_id,
                    "kickoff_local": row.get("kickoff_local"),
                    "home_team_id": home_id,
                    "away_team_id": away_id,
                    "captured_at": (
                        current_personnel_event.get("captured_at")
                        if isinstance(current_personnel_event, dict)
                        else None
                    ),
                    "stage": (
                        current_personnel_event.get("stage")
                        if isinstance(current_personnel_event, dict)
                        else None
                    ),
                    "home": home_personnel,
                    "away": away_personnel,
                    "mean_previous_xi_overlap_rate": mean_overlap,
                    "xi_continuity_gap_home_minus_away": overlap_gap,
                    "both_coach_same_as_previous": both_coach_same,
                }
            )

        home_history = team_style[home_id]
        away_history = team_style[away_id]
        home_counts = {field: len(home_history.get(field, [])) for field in STYLE_FIELDS}
        away_counts = {field: len(away_history.get(field, [])) for field in STYLE_FIELDS}
        home_any = max(home_counts.values(), default=0) > 0
        away_any = max(away_counts.values(), default=0) > 0
        home_min = min(home_counts.values(), default=0)
        away_min = min(away_counts.values(), default=0)

        prior_density["rows_with_home_any_prior_style"] += int(home_any)
        prior_density["rows_with_away_any_prior_style"] += int(away_any)
        prior_density["rows_with_both_any_prior_style"] += int(home_any and away_any)
        for threshold in (1, 2, 3):
            prior_density[f"rows_with_home_min_field_n_ge_{threshold}"] += int(home_min >= threshold)
            prior_density[f"rows_with_away_min_field_n_ge_{threshold}"] += int(away_min >= threshold)
            prior_density[f"rows_with_both_min_field_n_ge_{threshold}"] += int(
                home_min >= threshold and away_min >= threshold
            )

        for field in STYLE_FIELDS:
            h = int(home_counts.get(field) or 0)
            a = int(away_counts.get(field) or 0)
            field_prior_coverage[field]["home_rows_n_ge_1"] += int(h >= 1)
            field_prior_coverage[field]["away_rows_n_ge_1"] += int(a >= 1)
            field_prior_coverage[field]["both_rows_n_ge_1"] += int(h >= 1 and a >= 1)
            field_prior_coverage[field]["home_rows_n_ge_3"] += int(h >= 3)
            field_prior_coverage[field]["away_rows_n_ge_3"] += int(a >= 3)
            field_prior_coverage[field]["both_rows_n_ge_3"] += int(h >= 3 and a >= 3)

        if not home_any:
            profile_failure_reasons["HOME_NO_PRIOR_STYLE_HISTORY"] += 1
        elif home_min < MIN_TEAM_STYLE_N:
            profile_failure_reasons["HOME_MIN_FIELD_LT_3"] += 1
        if not away_any:
            profile_failure_reasons["AWAY_NO_PRIOR_STYLE_HISTORY"] += 1
        elif away_min < MIN_TEAM_STYLE_N:
            profile_failure_reasons["AWAY_MIN_FIELD_LT_3"] += 1

        home_profile, home_style_n = _profile(home_history)
        away_profile, away_style_n = _profile(away_history)
        features = _style_features(
            home_profile,
            away_profile,
            row.get("home_formation"),
            row.get("away_formation"),
        )
        if home_profile is None or away_profile is None:
            missing_style_profile_rows += 1
        elif _formation_geometry(row.get("home_formation")) is None or _formation_geometry(row.get("away_formation")) is None:
            geometry_missing_rows += 1
        else:
            style_eligible_rows += 1

        league = str(row.get("league_id") or row.get("league") or "UNKNOWN")

        for target, (home_key, away_key) in fm3.TARGETS.items():
            state = target_state[target]
            actual_home = _num(row.get(home_key))
            actual_away = _num(row.get(away_key))
            if actual_home is None or actual_away is None:
                continue

            base_home = fm3._baseline_side(
                league_values=state["league_home"][league],
                global_values=state["global_home"],
                team_attack_values=state["home_attack"][home_id],
                opponent_concession_values=state["away_concede"][away_id],
            )
            base_away = fm3._baseline_side(
                league_values=state["league_away"][league],
                global_values=state["global_away"],
                team_attack_values=state["away_attack"][away_id],
                opponent_concession_values=state["home_concede"][home_id],
            )

            if (
                base_home is not None
                and base_away is not None
                and features is not None
            ):
                residual_home = _ridge_predict(state["training_home"], features)
                residual_away = _ridge_predict(state["training_away"], features)
                if residual_home is not None and residual_away is not None:
                    cap = MAX_STYLE_RESIDUAL_ABS[target]
                    residual_home = max(-cap, min(cap, residual_home))
                    residual_away = max(-cap, min(cap, residual_away))
                    style_home = max(0.0, base_home + residual_home)
                    style_away = max(0.0, base_away + residual_away)
                    actual_total = actual_home + actual_away
                    base_total = base_home + base_away
                    style_total = style_home + style_away
                    state["eligible_fixture_ids"].add(int(row.get("fixture_id") or 0))
                    state["eval_home"].append(
                        {"actual": actual_home, "baseline": base_home, "style": style_home}
                    )
                    state["eval_away"].append(
                        {"actual": actual_away, "baseline": base_away, "style": style_away}
                    )
                    state["eval_total"].append(
                        {"actual": actual_total, "baseline": base_total, "style": style_total}
                    )
                    evaluation_rows.append(
                        {
                            "fixture_id": row.get("fixture_id"),
                            "kickoff_local": row.get("kickoff_local"),
                            "target": target,
                            "home_style_history_n": home_style_n,
                            "away_style_history_n": away_style_n,
                            "training_rows_home": len(state["training_home"]),
                            "training_rows_away": len(state["training_away"]),
                            "baseline_home": round(base_home, 6),
                            "baseline_away": round(base_away, 6),
                            "style_residual_home": round(residual_home, 6),
                            "style_residual_away": round(residual_away, 6),
                            "style_home": round(style_home, 6),
                            "style_away": round(style_away, 6),
                            "actual_home": actual_home,
                            "actual_away": actual_away,
                        }
                    )

                # Training record uses only this match after its prediction-time
                # features/baseline were frozen, so future rows may learn from it.
                state["training_home"].append((features, actual_home - base_home))
                state["training_away"].append((features, actual_away - base_away))

            state["global_home"].append(actual_home)
            state["global_away"].append(actual_away)
            state["league_home"][league].append(actual_home)
            state["league_away"][league].append(actual_away)
            state["home_attack"][home_id].append(actual_home)
            state["home_concede"][home_id].append(actual_away)
            state["away_attack"][away_id].append(actual_away)
            state["away_concede"][away_id].append(actual_home)

        # Fallback for isolated unit tests without a broader tactical-history source.
        # Production/offline research passes history_dir and consumes only strictly
        # earlier tactical events above.
        if not history_dir:
            _append_style_observation(team_style, home_id, _team_observation(row, "home"))
            _append_style_observation(team_style, away_id, _team_observation(row, "away"))

    targets: dict[str, Any] = {}
    global_blockers: list[str] = []
    for target, state in target_state.items():
        base_home = _metric(state["eval_home"], "baseline")
        base_away = _metric(state["eval_away"], "baseline")
        base_total = _metric(state["eval_total"], "baseline")
        style_home = _metric(state["eval_home"], "style")
        style_away = _metric(state["eval_away"], "style")
        style_total = _metric(state["eval_total"], "style")
        n = len(state["eligible_fixture_ids"])
        blockers: list[str] = []
        if n < 100:
            blockers.append(f"STYLE_ABLATION_{n}_LT_100")
        if n == 0:
            blockers.append("STYLE_PROFILE_DENSITY_INSUFFICIENT_NO_ELIGIBLE_FIXTURES")
        else:
            for side, baseline_metric, challenger_metric in (
                ("HOME", base_home, style_home),
                ("AWAY", base_away, style_away),
                ("TOTAL", base_total, style_total),
            ):
                if not _improves(
                    _num(baseline_metric.get("mae")),
                    _num(challenger_metric.get("mae")),
                ):
                    blockers.append(f"{side}_STYLE_MAE_NOT_BETTER_THAN_BASELINE")
        targets[target] = {
            "status": "OOS_REVIEW_ELIGIBLE" if not blockers else "RESEARCH_HOLD",
            "eligible_fixtures": n,
            "minimum_review_fixtures": 100,
            "baseline": {
                "home": base_home,
                "away": base_away,
                "total": base_total,
            },
            "style_challenger": {
                "home": style_home,
                "away": style_away,
                "total": style_total,
            },
            "improvement": {
                "home_mae_delta": _delta(base_home.get("mae"), style_home.get("mae")),
                "away_mae_delta": _delta(base_away.get("mae"), style_away.get("mae")),
                "total_mae_delta": _delta(base_total.get("mae"), style_total.get("mae")),
                "home_mae_improves": _improves(base_home.get("mae"), style_home.get("mae")),
                "away_mae_improves": _improves(base_away.get("mae"), style_away.get("mae")),
                "total_mae_improves": _improves(base_total.get("mae"), style_total.get("mae")),
            },
            "blockers": blockers,
            "production_enabled": False,
            "decision_weight": 0.0,
        }
        global_blockers.extend(f"{target}:{blocker}" for blocker in blockers)

    personnel = {
        "status": "NOT_MATERIALIZED",
        "coach_continuity": "NOT_VERIFIED_IN_CANONICAL_PREGAME_FEATURES",
        "player_role_continuity": "NOT_VERIFIED_IN_CANONICAL_PREGAME_FEATURES",
        "starter_continuity_score": "NOT_MATERIALIZED",
        "reason": (
            "FM-4 does not infer player roles or coach continuity from names/formation. "
            "A versioned pre-kickoff personnel source is required first."
        ),
    }
    global_blockers.append("PERSONNEL_OVERLAY_NOT_MATERIALIZED")

    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": "RESEARCH_HOLD_FM4_STYLE_PERSONNEL_ABLATION",
        "source_model_version": source.get("model_version"),
        "source_fixtures": int(source.get("fixtures_with_verified_formation_pair_and_final") or len(rows)),
        "policy": (
            "SPORT_FIRST; PRIOR_MATCHES_ONLY_FOR_STYLE; CURRENT_MATCH_POSTGAME_STATS_NEVER_USED_AS_INPUT; "
            "CURRENT_FORMATION_GEOMETRY_VERIFIED_ONLY; NO_MARKET; NO_INFERRED_PLAYER_ROLES_OR_COACH_CONTINUITY"
        ),
        "style_profile": {
            "history_source": (
                "ALL_PRIOR_VERIFIED_TACTICAL_HISTORY"
                if history_dir
                else "FORMATION_ROWS_ONLY_FALLBACK"
            ),
            "tactical_history_fixtures_loaded": len(tactical_events),
            "tactical_history_unique_teams": len(tactical_team_ids),
            "source_unique_teams": len(source_team_ids),
            "source_teams_present_anywhere_in_tactical_history": len(source_team_overlap),
            "source_team_overlap_pct": (
                round(len(source_team_overlap) / len(source_team_ids) * 100.0, 2)
                if source_team_ids
                else 0.0
            ),
            "prior_density": prior_density,
            "profile_failure_reasons": dict(sorted(profile_failure_reasons.items())),
            "field_prior_coverage": field_prior_coverage,
            "fields": list(STYLE_FIELDS),
            "minimum_prior_team_style_matches": MIN_TEAM_STYLE_N,
            "minimum_prior_residual_training_rows": MIN_STYLE_TRAIN_N,
            "ridge_alpha": RIDGE_ALPHA,
            "style_eligible_source_rows": style_eligible_rows,
            "missing_prior_style_profile_rows": missing_style_profile_rows,
            "formation_geometry_missing_rows": geometry_missing_rows,
        },
        "formation_geometry": {
            "allowed_fields": [
                "back_line",
                "front_line",
                "outfield_lines",
                "central_layers",
            ],
            "role_inference_used": False,
        },
        "targets": targets,
        "personnel_overlay": personnel,
        "health": {
            "leakage_policy": "STRICT_PRIOR_ONLY_STYLE_HISTORY",
            "odds_consumed": False,
            "market_prices_consumed": False,
            "current_match_postgame_style_consumed": False,
            "inferred_player_roles_used": False,
            "coach_continuity_used": False,
            "production_enabled": False,
            "decision_weight": 0.0,
            "provider_requests_added": 0,
            "model_weights_changed": False,
            "canonical_bet_logic_changed": False,
            "blockers": sorted(set(global_blockers)),
        },
        "evaluation_rows": evaluation_rows[-1000:],
    }


def _load(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FM-4 prior-only style and personnel readiness ablation."
    )
    parser.add_argument("--formation-report", required=True)
    parser.add_argument("--history-dir")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = build_report(
        _load(args.formation_report),
        history_dir=args.history_dir,
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {
                "model_version": report["model_version"],
                "status": report["status"],
                "style_profile": report["style_profile"],
                "targets": report["targets"],
                "personnel_overlay": report["personnel_overlay"],
                "health": report["health"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
