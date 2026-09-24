from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from mcp_gateway import market_mismatch_v4, persistence

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_PROMOTION_SHADOW_POSTGRES_V4_1.0.0"
PREGAME_STAGES = {"EARLY_RESEARCH", "T-90", "T-60", "T-40", "T-30", "T-20", "T-10", "CLOSE"}
DIRECTIONAL_MIN = 20
REVIEW_MIN = 50


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _parse_dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not value:
        return None
    try:
        out = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return out if out.tzinfo is not None else None


def _semantic_version(value: Any) -> tuple[int, ...] | None:
    match = re.search(r"v([0-9]+(?:[.][0-9]+)*)", str(value or ""), re.I)
    if not match:
        return None
    try:
        return tuple(int(part) for part in match.group(1).split("."))
    except ValueError:
        return None


def _sample_status(n: int) -> str:
    if n >= REVIEW_MIN:
        return "SHADOW_REVIEW_READY"
    if n >= DIRECTIONAL_MIN:
        return "DIRECTIONAL_SHADOW"
    return "DATA_BLOCKED"


def _grade_1x2(selection: Any, home_team: Any, away_team: Any, home_goals: int, away_goals: int) -> str | None:
    sel = _norm(selection)
    if home_goals > away_goals:
        actual = "home"
    elif away_goals > home_goals:
        actual = "away"
    else:
        actual = "draw"

    if sel in {"home", "1"} or sel == _norm(home_team):
        picked = "home"
    elif sel in {"away", "2"} or sel == _norm(away_team):
        picked = "away"
    elif sel in {"draw", "x"}:
        picked = "draw"
    else:
        return None
    return "WIN" if picked == actual else "LOSS"


def _roi(outcome: str | None, price: Any) -> float | None:
    p = _num(price)
    if outcome == "WIN" and p is not None and p > 1.0:
        return p - 1.0
    if outcome == "LOSS":
        return -1.0
    return None


def _current_source_model_version(rows: Iterable[dict[str, Any]]) -> str | None:
    versions = sorted({
        str(row.get("source_model_version") or "")
        for row in rows
        if row.get("source_model_version")
    })
    if not versions:
        return None
    semantic = [
        (parsed, version)
        for version in versions
        if (parsed := _semantic_version(version)) is not None
    ]
    if semantic:
        return max(semantic, key=lambda item: item[0])[1]
    return versions[-1]


def _primary_1x2_for_run(group: list[dict[str, Any]]) -> dict[str, Any] | None:
    raw_rows = [row.get("match_table_row") for row in group if isinstance(row.get("match_table_row"), dict)]
    scan = market_mismatch_v4.find_mismatches(raw_rows, top_n=max(len(raw_rows), 1))
    for candidate in scan.get("primary_candidates") or []:
        if candidate.get("market_family") == "1X2" and candidate.get("rankable") is True:
            return candidate
    return None


def normalize_rows(raw_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        if not isinstance(row, dict):
            continue
        try:
            fixture_id = int(row.get("fixture_id"))
            home_goals = int(row.get("home_goals"))
            away_goals = int(row.get("away_goals"))
        except (TypeError, ValueError):
            continue
        generated_at = _parse_dt(row.get("generated_at"))
        kickoff = _parse_dt(row.get("kickoff"))
        if generated_at is None or kickoff is None or generated_at >= kickoff:
            continue
        stage = str(row.get("stage") or "").upper()
        if stage not in PREGAME_STAGES:
            continue
        if not isinstance(row.get("match_table_row"), dict):
            continue

        clean = dict(row)
        clean["fixture_id"] = fixture_id
        clean["home_goals"] = home_goals
        clean["away_goals"] = away_goals
        clean["generated_at"] = generated_at
        clean["kickoff"] = kickoff
        grouped[(fixture_id, generated_at.isoformat())].append(clean)

    per_run: list[dict[str, Any]] = []
    for (_, _), group in grouped.items():
        candidate = _primary_1x2_for_run(group)
        if candidate is None:
            continue
        base = group[0]
        outcome = _grade_1x2(
            candidate.get("selection"),
            base.get("home_team"),
            base.get("away_team"),
            int(base["home_goals"]),
            int(base["away_goals"]),
        )
        if outcome is None:
            continue
        per_run.append({
            "fixture_id": int(base["fixture_id"]),
            "generated_at": base["generated_at"].isoformat(),
            "kickoff": base["kickoff"].isoformat(),
            "stage": candidate.get("stage") or base.get("stage"),
            "league": base.get("league"),
            "home_team": base.get("home_team"),
            "away_team": base.get("away_team"),
            "home_goals": int(base["home_goals"]),
            "away_goals": int(base["away_goals"]),
            "source_model_version": base.get("source_model_version"),
            "automation_version": base.get("automation_version"),
            "market_family": "1X2",
            "market": candidate.get("market"),
            "selection": candidate.get("selection"),
            "decimal_price": _num(candidate.get("price")),
            "bookmaker": candidate.get("bookmaker"),
            "calibrated_probability": _num(candidate.get("calibrated_probability")),
            "market_fair_probability": _num(candidate.get("market_fair_probability")),
            "calibrated_edge_pp": _num(candidate.get("calibrated_edge_pp")),
            "mismatch_score": _num(candidate.get("mismatch_score")),
            "sport_confidence_score": _num(candidate.get("sport_confidence_score")),
            "data_quality_score": _num(candidate.get("data_quality_score")),
            "price_quality_score": _num(candidate.get("price_quality_score")),
            "uncertainty": _num(candidate.get("uncertainty")),
            "outcome": outcome,
            "roi_units": _roi(outcome, candidate.get("price")),
            "rankable": True,
            "signal_source": "PIPELINE_MATCH_TABLE_PHASE16_REPLAY",
        })

    current_version = _current_source_model_version(per_run)
    if current_version:
        per_run = [row for row in per_run if row.get("source_model_version") == current_version]

    latest: dict[int, tuple[datetime, dict[str, Any]]] = {}
    for row in per_run:
        generated_at = _parse_dt(row.get("generated_at"))
        if generated_at is None:
            continue
        fid = int(row["fixture_id"])
        prior = latest.get(fid)
        if prior is None or generated_at > prior[0]:
            latest[fid] = (generated_at, row)

    return [value[1] for _, value in sorted(latest.items())]


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    settled = [row for row in rows if row.get("outcome") in {"WIN", "LOSS"}]
    wins = sum(1 for row in settled if row.get("outcome") == "WIN")
    losses = sum(1 for row in settled if row.get("outcome") == "LOSS")
    roi_values = [float(row["roi_units"]) for row in settled if row.get("roi_units") is not None]
    n = len(settled)
    return {
        "rows": len(rows),
        "unique_fixtures": len({row["fixture_id"] for row in rows}),
        "settled": n,
        "win": wins,
        "loss": losses,
        "hit_rate": round(wins / n, 6) if n else None,
        "roi_units": round(sum(roi_values), 6) if roi_values else 0.0,
        "roi_per_settled_unit": round(sum(roi_values) / n, 6) if roi_values and n else None,
        "sample_status": _sample_status(n),
    }


def build_report_from_rows(raw_rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = normalize_rows(raw_rows)
    by_stage_raw: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stage_raw[str(row.get("stage") or "UNKNOWN").upper()].append(row)

    by_stage = {stage: _summary(group) for stage, group in sorted(by_stage_raw.items())}
    negative_directional_stages = sorted(
        stage for stage, value in by_stage.items()
        if int(value.get("settled") or 0) >= DIRECTIONAL_MIN
        and value.get("roi_per_settled_unit") is not None
        and float(value["roi_per_settled_unit"]) <= 0
    )
    current_version = _current_source_model_version(rows)
    overall = _summary(rows)

    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": "PROMOTION_SHADOW_POSTGRES_ACTIVE",
        "market_family": "1X2",
        "source_model_version": current_version,
        "promotion_evaluable": {
            **overall,
            "negative_directional_stages": negative_directional_stages,
            "evidence_policy": "LATEST_PREKICKOFF_PHASE16_PRIMARY_RANKABLE_1X2_PER_FIXTURE_CURRENT_MODEL_ONLY",
        },
        "by_stage": by_stage,
        "rows": rows,
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
        "runtime_logic_changed": False,
        "notes": [
            "Replays Phase16 market_mismatch_v4 against persisted match_table_rows; raw legacy best_market WATCH rows are not used.",
            "At most one primary rankable 1X2 candidate is retained per fixture/run, then only the latest pre-kickoff run per fixture is kept.",
            "Only the latest semantic source model version is retained to avoid mixing historical model regimes.",
            "Results are joined after kickoff from soccer_results and never feed candidate generation.",
        ],
    }


def _load_rows(conn, *, lookback_days: int, max_rows: int) -> list[dict[str, Any]]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, int(lookback_days)))
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                (mr.row ->> 'fixture_id')::BIGINT AS fixture_id,
                p.generated_at_utc AS generated_at,
                COALESCE(NULLIF(mr.row ->> 'stage', ''), e.stage) AS stage,
                mr.row AS match_table_row,
                f.kickoff,
                f.league,
                f.home_team,
                f.away_team,
                r.home_goals,
                r.away_goals,
                p.payload ->> 'model_version' AS source_model_version,
                p.payload ->> 'version' AS automation_version
            FROM soccer_pipeline_runs p
            CROSS JOIN LATERAL jsonb_array_elements(
                COALESCE(p.payload -> 'match_table_rows', '[]'::jsonb)
            ) AS mr(row)
            JOIN soccer_fixtures f
              ON f.fixture_id = (mr.row ->> 'fixture_id')::BIGINT
            JOIN soccer_results r
              ON r.fixture_id = f.fixture_id
            LEFT JOIN soccer_refresh_events e
              ON e.fixture_id = f.fixture_id
             AND e.generated_at = p.generated_at_utc
            WHERE p.generated_at_utc >= %s
              AND p.generated_at_utc < f.kickoff
              AND r.home_goals IS NOT NULL
              AND r.away_goals IS NOT NULL
              AND NULLIF(mr.row ->> 'market', '') IS NOT NULL
              AND NULLIF(mr.row ->> 'selection', '') IS NOT NULL
            ORDER BY p.generated_at_utc ASC
            LIMIT %s
            """,
            (cutoff, max(100, int(max_rows))),
        )
        columns = [desc.name for desc in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def build_from_postgres(*, lookback_days: int = 180, max_rows: int = 50000) -> dict[str, Any]:
    persistence.ensure_schema()
    with persistence._connect() as conn:
        raw_rows = _load_rows(conn, lookback_days=lookback_days, max_rows=max_rows)
    report = build_report_from_rows(raw_rows)
    report["raw_match_table_rows_loaded"] = len(raw_rows)
    report["lookback_days"] = int(lookback_days)
    return report
