from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
import math
from typing import Any, Iterable

from mcp_gateway import persistence as persistence_base

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_PLAYER_PROPS_OOS_V4_1.0.0"
SIGNAL_STAGES = ("T-40", "T-20", "T-10")
STAGE_PRIORITY = {"T-40": 1, "T-20": 2, "T-10": 3}

FAMILY_CONFIG = {
    "SHOTS": {
        "intel_key": "player_shots_intelligence",
        "rows_key": "players",
        "outcome_key": "shots",
        "expected_key": "expected_shots",
        "mode": "LINES",
        "minimum_player_games_for_review": 500,
    },
    "SOT": {
        "intel_key": "player_sot_intelligence",
        "rows_key": "players",
        "outcome_key": "shots_on_target",
        "expected_key": "expected_sot",
        "mode": "LINES",
        "minimum_player_games_for_review": 500,
    },
    "GOALSCORER_ANYTIME": {
        "intel_key": "player_goalscorer_intelligence",
        "rows_key": "players",
        "outcome_key": "goals",
        "expected_key": "expected_goals",
        "probability_key": "p_anytime_goal",
        "mode": "BINARY_1PLUS",
        "minimum_player_games_for_review": 1000,
    },
    "ASSISTS": {
        "intel_key": "player_assists_intelligence",
        "rows_key": "players",
        "outcome_key": "assists",
        "expected_key": "expected_assists",
        "probability_key": "p_1plus_assist",
        "mode": "BINARY_1PLUS",
        "minimum_player_games_for_review": 1000,
    },
    "PLAYER_CARDS": {
        "intel_key": "player_cards_intelligence",
        "rows_key": "players",
        "outcome_key": "yellow_cards",
        "expected_key": "expected_yellow_cards",
        "probability_key": "p_player_booked_yellow",
        "mode": "BINARY_1PLUS",
        "minimum_player_games_for_review": 1000,
    },
    "GK_SAVES": {
        "intel_key": "gk_saves_intelligence",
        "rows_key": "goalkeepers",
        "outcome_key": "saves",
        "expected_key": "expected_saves",
        "mode": "LINES",
        "minimum_player_games_for_review": 300,
    },
}


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        out = value
    elif isinstance(value, str) and value.strip():
        try:
            out = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if out.tzinfo is None:
        out = out.replace(tzinfo=timezone.utc)
    return out.astimezone(timezone.utc)


def _event_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload = row.get("event_payload")
    return payload if isinstance(payload, dict) else row


def choose_canonical_pregame_events(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[int, tuple[int, datetime, dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        event = _event_payload(row)
        fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
        fixture_id = row.get("fixture_id") or fixture.get("fixture_id")
        stage = str(row.get("stage") or event.get("stage") or "")
        generated_at = _dt(row.get("generated_at") or row.get("captured_at"))
        kickoff = _dt(row.get("kickoff") or fixture.get("kickoff"))
        if fixture_id is None or stage not in STAGE_PRIORITY or generated_at is None or kickoff is None:
            continue
        if generated_at >= kickoff:
            continue
        try:
            fid = int(fixture_id)
        except (TypeError, ValueError):
            continue
        candidate = (STAGE_PRIORITY[stage], generated_at, row)
        previous = latest.get(fid)
        if previous is None or candidate[:2] > previous[:2]:
            latest[fid] = candidate
    return [value[2] for _, value in sorted(latest.items())]


def _postgame_players(event: dict[str, Any]) -> dict[str, dict[str, Any]]:
    stats = event.get("postgame_player_stats")
    if not isinstance(stats, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for team in stats.get("teams") or []:
        if not isinstance(team, dict):
            continue
        team_id = team.get("team_id")
        for player in team.get("players") or []:
            if not isinstance(player, dict) or player.get("player_id") is None:
                continue
            row = dict(player)
            row["team_id"] = team_id
            row["team"] = team.get("team")
            out[str(player["player_id"])] = row
    return out


def _latest_postgame_by_fixture(rows: Iterable[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    latest: dict[int, tuple[datetime, dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        event = _event_payload(row)
        fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
        fixture_id = row.get("fixture_id") or fixture.get("fixture_id")
        generated_at = _dt(row.get("generated_at") or row.get("captured_at"))
        if fixture_id is None or generated_at is None:
            continue
        if not _postgame_players(event):
            continue
        try:
            fid = int(fixture_id)
        except (TypeError, ValueError):
            continue
        previous = latest.get(fid)
        if previous is None or generated_at > previous[0]:
            latest[fid] = (generated_at, row)
    return {fid: value[1] for fid, value in latest.items()}


def _binary_rows_for_prediction(
    *,
    family: str,
    player: dict[str, Any],
    actual: float,
    fixture_id: int,
    stage: str,
    generated_at: datetime,
) -> list[dict[str, Any]]:
    config = FAMILY_CONFIG[family]
    out: list[dict[str, Any]] = []
    if config["mode"] == "LINES":
        for line_row in player.get("lines") or []:
            if not isinstance(line_row, dict):
                continue
            line = _num(line_row.get("line"))
            probability = _num(line_row.get("p_over"))
            if line is None or probability is None or not 0.0 <= probability <= 1.0:
                continue
            out.append({
                "fixture_id": fixture_id,
                "stage": stage,
                "prediction_timestamp": generated_at.isoformat(),
                "market_family": family,
                "player_id": player.get("player_id"),
                "player_name": player.get("player") or player.get("name"),
                "team_id": player.get("team_id"),
                "line": line,
                "probability": probability,
                "outcome": 1 if actual > line else 0,
                "actual_count": actual,
            })
        return out

    probability = _num(player.get(config.get("probability_key")))
    if probability is None or not 0.0 <= probability <= 1.0:
        return out
    out.append({
        "fixture_id": fixture_id,
        "stage": stage,
        "prediction_timestamp": generated_at.isoformat(),
        "market_family": family,
        "player_id": player.get("player_id"),
        "player_name": player.get("player") or player.get("name"),
        "team_id": player.get("team_id"),
        "line": 0.5,
        "probability": probability,
        "outcome": 1 if actual >= 1.0 else 0,
        "actual_count": actual,
    })
    return out


def build_oos_rows(
    pregame_rows: Iterable[dict[str, Any]],
    postgame_rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    canonical = choose_canonical_pregame_events(pregame_rows)
    postgame = _latest_postgame_by_fixture(postgame_rows)
    rows: list[dict[str, Any]] = []

    for row in canonical:
        event = _event_payload(row)
        fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
        fixture_id = row.get("fixture_id") or fixture.get("fixture_id")
        stage = str(row.get("stage") or event.get("stage") or "")
        generated_at = _dt(row.get("generated_at") or row.get("captured_at"))
        if fixture_id is None or generated_at is None:
            continue
        fid = int(fixture_id)

        post_row = postgame.get(fid)
        if post_row is None:
            continue
        post_event = _event_payload(post_row)
        actual_players = _postgame_players(post_event)
        if not actual_players:
            continue

        for family, config in FAMILY_CONFIG.items():
            intel = event.get(config["intel_key"])
            if not isinstance(intel, dict):
                continue
            for player in intel.get(config["rows_key"]) or []:
                if not isinstance(player, dict) or player.get("player_id") is None:
                    continue
                if player.get("confirmed_starter") is not True and family != "GK_SAVES":
                    continue
                actual = actual_players.get(str(player["player_id"]))
                if not isinstance(actual, dict):
                    continue
                actual_count = _num(actual.get(config["outcome_key"]))
                if actual_count is None:
                    continue

                expected_count = _num(player.get(config["expected_key"]))
                expected_minutes = _num(
                    player.get("expected_minutes_if_confirmed_starter")
                    if player.get("expected_minutes_if_confirmed_starter") is not None
                    else player.get("expected_minutes")
                )
                actual_minutes = _num(actual.get("minutes"))

                rows.append({
                    "schema_version": SCHEMA_VERSION,
                    "fixture_id": fid,
                    "stage": stage,
                    "prediction_timestamp": generated_at.isoformat(),
                    "market_family": family,
                    "player_id": player.get("player_id"),
                    "player_name": player.get("player") or player.get("name"),
                    "team_id": player.get("team_id") or actual.get("team_id"),
                    "position": player.get("position") or actual.get("position"),
                    "expected_count": expected_count,
                    "actual_count": actual_count,
                    "expected_minutes": expected_minutes,
                    "actual_minutes": actual_minutes,
                    "count_error": (
                        actual_count - expected_count
                        if expected_count is not None else None
                    ),
                    "minutes_error": (
                        actual_minutes - expected_minutes
                        if actual_minutes is not None and expected_minutes is not None else None
                    ),
                    "binary_rows": _binary_rows_for_prediction(
                        family=family,
                        player=player,
                        actual=actual_count,
                        fixture_id=fid,
                        stage=stage,
                        generated_at=generated_at,
                    ),
                    "decision_weight": 0.0,
                    "production_promotion_allowed": False,
                })
    return rows


def _calibration_bins(binary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for row in binary_rows:
        p = _num(row.get("probability"))
        y = row.get("outcome")
        if p is None or y not in (0, 1):
            continue
        idx = min(9, max(0, int(p * 10.0)))
        buckets[idx].append((p, int(y)))

    out: list[dict[str, Any]] = []
    for idx in range(10):
        values = buckets.get(idx, [])
        if not values:
            continue
        mean_p = sum(p for p, _ in values) / len(values)
        observed = sum(y for _, y in values) / len(values)
        out.append({
            "bucket": f"{idx/10:.1f}-{(idx+1)/10:.1f}",
            "rows": len(values),
            "mean_probability": round(mean_p, 6),
            "observed_rate": round(observed, 6),
            "calibration_gap_pp": round((observed - mean_p) * 100.0, 6),
        })
    return out


def _family_metrics(family: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    binary_rows = [
        b
        for row in rows
        for b in (row.get("binary_rows") or [])
        if isinstance(b, dict)
    ]
    brier_values: list[float] = []
    logloss_values: list[float] = []
    for row in binary_rows:
        p = _num(row.get("probability"))
        y = row.get("outcome")
        if p is None or y not in (0, 1):
            continue
        p_clip = min(1.0 - 1e-12, max(1e-12, p))
        brier_values.append((p - int(y)) ** 2)
        logloss_values.append(-(int(y) * math.log(p_clip) + (1 - int(y)) * math.log(1 - p_clip)))

    count_errors = [_num(row.get("count_error")) for row in rows]
    count_errors = [value for value in count_errors if value is not None]
    minute_errors = [_num(row.get("minutes_error")) for row in rows]
    minute_errors = [value for value in minute_errors if value is not None]
    calibration_bins = _calibration_bins(binary_rows)

    total_binary = sum(bin_row["rows"] for bin_row in calibration_bins)
    ece = None
    if total_binary > 0:
        ece = sum(
            (bin_row["rows"] / total_binary)
            * abs(bin_row["observed_rate"] - bin_row["mean_probability"])
            for bin_row in calibration_bins
        )

    unique_fixtures = {int(row["fixture_id"]) for row in rows if row.get("fixture_id") is not None}
    unique_player_fixtures = {
        (int(row["fixture_id"]), str(row.get("player_id")))
        for row in rows
        if row.get("fixture_id") is not None and row.get("player_id") is not None
    }
    minimum = int(FAMILY_CONFIG[family]["minimum_player_games_for_review"])
    sample = len(unique_player_fixtures)

    return {
        "player_game_rows": len(rows),
        "binary_probability_rows": len(binary_rows),
        "unique_fixtures": len(unique_fixtures),
        "unique_player_fixtures": sample,
        "minimum_player_games_for_review": minimum,
        "sample_target_met": sample >= minimum,
        "oos_validation_complete": sample >= minimum,
        "brier_score": round(sum(brier_values) / len(brier_values), 8) if brier_values else None,
        "log_loss": round(sum(logloss_values) / len(logloss_values), 8) if logloss_values else None,
        "expected_count_mae": round(sum(abs(x) for x in count_errors) / len(count_errors), 8) if count_errors else None,
        "expected_count_rmse": round(math.sqrt(sum(x*x for x in count_errors) / len(count_errors)), 8) if count_errors else None,
        "expected_minutes_mae": round(sum(abs(x) for x in minute_errors) / len(minute_errors), 8) if minute_errors else None,
        "calibration_ece": round(ece, 8) if ece is not None else None,
        "calibration_bins": calibration_bins,
        "status": "OOS_REVIEW_READY" if sample >= minimum else "COLLECTING_OOS",
    }


def summarize_oos(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    items = [row for row in rows if isinstance(row, dict)]
    families: dict[str, Any] = {}
    for family in FAMILY_CONFIG:
        family_rows = [row for row in items if row.get("market_family") == family]
        families[family] = _family_metrics(family, family_rows)
    return {
        "families": families,
        "player_game_rows": len(items),
        "unique_fixtures": len({
            int(row["fixture_id"])
            for row in items
            if row.get("fixture_id") is not None
        }),
        "review_ready_family_count": sum(
            1 for value in families.values()
            if value.get("oos_validation_complete") is True
        ),
    }


def _load_pregame(conn, *, lookback_days: int, max_rows: int) -> list[dict[str, Any]]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT e.fixture_id, e.generated_at, e.stage, e.payload AS event_payload, f.kickoff
            FROM soccer_refresh_events e
            JOIN soccer_fixtures f ON f.fixture_id = e.fixture_id
            WHERE e.generated_at >= %s
              AND e.generated_at < f.kickoff
              AND e.stage IN ('T-40','T-20','T-10')
            ORDER BY e.fixture_id, e.generated_at
            LIMIT %s
            """,
            (cutoff, max_rows),
        )
        columns = [desc.name for desc in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def _load_postgame(conn, *, lookback_days: int, max_rows: int) -> list[dict[str, Any]]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT e.fixture_id, e.generated_at, e.stage, e.payload AS event_payload, f.kickoff
            FROM soccer_refresh_events e
            JOIN soccer_fixtures f ON f.fixture_id = e.fixture_id
            WHERE e.generated_at >= %s
              AND e.stage = 'POSTGAME'
              AND e.payload ? 'postgame_player_stats'
            ORDER BY e.fixture_id, e.generated_at
            LIMIT %s
            """,
            (cutoff, max_rows),
        )
        columns = [desc.name for desc in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def build_from_postgres(*, lookback_days: int = 180, max_rows: int = 50000) -> dict[str, Any]:
    lookback_days = max(1, min(int(lookback_days), 730))
    max_rows = max(100, min(int(max_rows), 200000))
    if not persistence_base.persistence_configured():
        return {
            "schema_version": SCHEMA_VERSION,
            "model_version": MODEL_VERSION,
            "status": "POSTGRES_NOT_CONFIGURED",
            "families": {},
            "rows": [],
            "provider_requests_added": 0,
            "production_promotion_allowed": False,
        }

    persistence_base.ensure_schema()
    with persistence_base._connect() as conn:
        pregame = _load_pregame(conn, lookback_days=lookback_days, max_rows=max_rows)
        postgame = _load_postgame(conn, lookback_days=lookback_days, max_rows=max_rows)

    rows = build_oos_rows(pregame, postgame)
    summary = summarize_oos(rows)
    all_ready = bool(summary["families"]) and all(
        family.get("oos_validation_complete") is True
        for family in summary["families"].values()
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": "PLAYER_PROP_OOS_REVIEW_READY" if all_ready else "COLLECTING_PLAYER_PROP_OOS",
        "lookback_days": lookback_days,
        "pregame_event_rows_loaded": len(pregame),
        "canonical_pregame_fixtures": len(choose_canonical_pregame_events(pregame)),
        "postgame_event_rows_loaded": len(postgame),
        **summary,
        "rows": rows,
        "provider_requests_added": 0,
        "decision_weight": 0.0,
        "production_promotion_allowed": False,
        "policy": (
            "ONE CANONICAL PREGAME SNAPSHOT PER FIXTURE (T-10 > T-20 > T-40); "
            "JOIN ONLY FINALIZED POSTGAME PLAYER STATS BY PLAYER_ID; "
            "BINARY CALIBRATION + COUNT ERROR + MINUTES ERROR REPORTED BY PROP FAMILY; "
            "NO PROVIDER CALLS; REVIEW TARGETS ARE SAMPLE GATES, NOT PRODUCTION PROMOTION"
        ),
    }
