from __future__ import annotations

import math
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from mcp_gateway import persistence

SCHEMA_VERSION = "1.3.0"
MODEL_VERSION = "SOCCER_PROMOTION_SHADOW_POSTGRES_V4_1.3.0"
PREGAME_STAGES = {"EARLY_RESEARCH", "T-90", "T-60", "T-40", "T-30", "T-20", "T-10", "CLOSE"}
SUPPORTED_FAMILIES = {"1X2", "FT_TOTALS", "BTTS"}
REQUIRED_EVIDENCE_REGIME = "PHASE16_DISCRIMINATION_GATED_V2"
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


def _source_regime(row: dict[str, Any]) -> str | None:
    return str(row.get("source_model_version") or row.get("automation_version") or "").strip() or None


def _current_source_regime(rows: Iterable[dict[str, Any]]) -> str | None:
    regimes = sorted({_source_regime(row) for row in rows if _source_regime(row)})
    if not regimes:
        return None
    semantic = [(parsed, regime) for regime in regimes if (parsed := _semantic_version(regime)) is not None]
    if semantic:
        return max(semantic, key=lambda item: item[0])[1]
    return regimes[-1]


def _sample_status(n: int) -> str:
    if n >= REVIEW_MIN:
        return "SHADOW_REVIEW_READY"
    if n >= DIRECTIONAL_MIN:
        return "DIRECTIONAL_SHADOW"
    return "DATA_BLOCKED"


def _grade_1x2(selection: Any, home_team: Any, away_team: Any, home_goals: int, away_goals: int) -> str | None:
    sel = _norm(selection)
    actual = "home" if home_goals > away_goals else "away" if away_goals > home_goals else "draw"
    if sel in {"home", "1"} or sel == _norm(home_team):
        picked = "home"
    elif sel in {"away", "2"} or sel == _norm(away_team):
        picked = "away"
    elif sel in {"draw", "x"}:
        picked = "draw"
    else:
        return None
    return "WIN" if picked == actual else "LOSS"


def _grade_totals(selection: Any, line: Any, home_goals: int, away_goals: int) -> str | None:
    side = _norm(selection)
    threshold = _num(line)
    if threshold is None or side not in {"over", "under"}:
        return None
    total = home_goals + away_goals
    if abs(total - threshold) < 1e-9:
        return "PUSH"
    if side == "over":
        return "WIN" if total > threshold else "LOSS"
    return "WIN" if total < threshold else "LOSS"


def _grade_btts(selection: Any, home_goals: int, away_goals: int) -> str | None:
    side = _norm(selection)
    if side not in {"yes", "no"}:
        return None
    happened = home_goals > 0 and away_goals > 0
    picked_yes = side == "yes"
    return "WIN" if happened == picked_yes else "LOSS"


def _grade_candidate(
    family: str,
    candidate: dict[str, Any],
    home_team: Any,
    away_team: Any,
    home_goals: Any,
    away_goals: Any,
) -> str:
    try:
        hg = int(home_goals)
        ag = int(away_goals)
    except (TypeError, ValueError):
        return "PENDING"

    if family == "1X2":
        return _grade_1x2(candidate.get("selection"), home_team, away_team, hg, ag) or "UNGRADABLE"
    if family == "FT_TOTALS":
        return _grade_totals(candidate.get("selection"), candidate.get("line"), hg, ag) or "UNGRADABLE"
    if family == "BTTS":
        return _grade_btts(candidate.get("selection"), hg, ag) or "UNGRADABLE"
    return "UNGRADABLE"


def _roi(outcome: str | None, price: Any) -> float | None:
    p = _num(price)
    if outcome == "WIN" and p is not None and p > 1.0:
        return p - 1.0
    if outcome == "LOSS":
        return -1.0
    if outcome == "PUSH":
        return 0.0
    return None


def normalize_rows(raw_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in raw_rows:
        if not isinstance(row, dict):
            continue
        candidate = row.get("phase16_candidate")
        if not isinstance(candidate, dict) or candidate.get("rankable") is not True:
            continue
        if candidate.get("promotion_shadow_eligible") is not True:
            continue
        family = str(candidate.get("market_family") or "").upper()
        if family not in SUPPORTED_FAMILIES:
            continue
        if str(candidate.get("evidence_regime") or "") != REQUIRED_EVIDENCE_REGIME:
            continue
        try:
            fixture_id = int(row.get("fixture_id"))
        except (TypeError, ValueError):
            continue

        generated_at = _parse_dt(row.get("generated_at"))
        kickoff = _parse_dt(row.get("kickoff"))
        if generated_at is None or kickoff is None or generated_at >= kickoff:
            continue
        stage = str(candidate.get("stage") or row.get("stage") or "").upper()
        if stage not in PREGAME_STAGES:
            continue

        outcome = _grade_candidate(
            family,
            candidate,
            row.get("home_team"),
            row.get("away_team"),
            row.get("home_goals"),
            row.get("away_goals"),
        )

        candidates.append({
            "fixture_id": fixture_id,
            "generated_at": generated_at.isoformat(),
            "kickoff": kickoff.isoformat(),
            "stage": stage,
            "league": row.get("league"),
            "home_team": row.get("home_team"),
            "away_team": row.get("away_team"),
            "home_goals": row.get("home_goals"),
            "away_goals": row.get("away_goals"),
            "source_model_version": row.get("source_model_version"),
            "automation_version": row.get("automation_version"),
            "source_regime": _source_regime(row),
            "market_family": family,
            "market": candidate.get("market"),
            "selection": candidate.get("selection"),
            "line": _num(candidate.get("line")),
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
            "settlement_status": "SETTLED" if outcome in {"WIN", "LOSS", "PUSH"} else outcome,
            "roi_units": _roi(outcome, candidate.get("price")),
            "rankable": True,
            "promotion_shadow_eligible": True,
            "phase16_calibration_source": candidate.get("phase16_calibration_source"),
            "phase16_calibration_policy": candidate.get("phase16_calibration_policy"),
            "signal_source": "PERSISTED_PHASE16_MARKET_MISMATCH",
        })

    current_regime = _current_source_regime(candidates)
    if current_regime:
        candidates = [row for row in candidates if row.get("source_regime") == current_regime]

    latest: dict[tuple[int, str], tuple[datetime, dict[str, Any]]] = {}
    for row in candidates:
        generated_at = _parse_dt(row.get("generated_at"))
        if generated_at is None:
            continue
        key = (int(row["fixture_id"]), str(row["market_family"]))
        prior = latest.get(key)
        if prior is None or generated_at > prior[0]:
            latest[key] = (generated_at, row)

    return [value[1] for _, value in sorted(latest.items(), key=lambda item: item[0])]


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    settled = [row for row in rows if row.get("outcome") in {"WIN", "LOSS", "PUSH"}]
    wins = sum(1 for row in settled if row.get("outcome") == "WIN")
    losses = sum(1 for row in settled if row.get("outcome") == "LOSS")
    pushes = sum(1 for row in settled if row.get("outcome") == "PUSH")
    pending = sum(1 for row in rows if row.get("outcome") == "PENDING")
    ungradable = sum(1 for row in rows if row.get("outcome") == "UNGRADABLE")
    roi_values = [float(row["roi_units"]) for row in settled if row.get("roi_units") is not None]
    n = len(settled)
    return {
        "rows": len(rows),
        "unique_fixtures": len({row["fixture_id"] for row in rows}),
        "settled": n,
        "pending": pending,
        "ungradable": ungradable,
        "win": wins,
        "loss": losses,
        "push": pushes,
        "hit_rate": round(wins / (wins + losses), 6) if (wins + losses) else None,
        "roi_units": round(sum(roi_values), 6) if roi_values else 0.0,
        "roi_per_settled_unit": round(sum(roi_values) / n, 6) if n else None,
        "sample_status": _sample_status(n),
    }


def _family_report(family: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
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
    overall = _summary(rows)
    return {
        "market_family": family,
        "promotion_evaluable": {
            **overall,
            "negative_directional_stages": negative_directional_stages,
            "evidence_policy": f"LATEST_PREKICKOFF_PERSISTED_PHASE16_PRIMARY_RANKABLE_{family}_PER_FIXTURE_LATEST_VERSIONED_REGIME",
        },
        "by_stage": by_stage,
    }


def build_report_from_rows(raw_rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    raw_list = [row for row in raw_rows if isinstance(row, dict)]
    supported_rankable_rows = [
        row for row in raw_list
        if isinstance(row.get("phase16_candidate"), dict)
        and row["phase16_candidate"].get("rankable") is True
        and str(row["phase16_candidate"].get("market_family") or "").upper() in SUPPORTED_FAMILIES
    ]
    excluded_without_current_calibration_policy = sum(
        1 for row in supported_rankable_rows
        if row["phase16_candidate"].get("promotion_shadow_eligible") is not True
    )
    rows = normalize_rows(raw_list)
    by_family_raw: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family_raw[str(row.get("market_family") or "UNKNOWN")].append(row)
    families = {family: _family_report(family, group) for family, group in sorted(by_family_raw.items())}
    for family in sorted(SUPPORTED_FAMILIES):
        families.setdefault(family, _family_report(family, []))

    aggregate = _summary(rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": "PROMOTION_SHADOW_POSTGRES_ACTIVE",
        "supported_market_families": sorted(SUPPORTED_FAMILIES),
        "required_evidence_regime": REQUIRED_EVIDENCE_REGIME,
        "source_regime": _current_source_regime(rows),
        "phase16_rankable_supported_rows_seen": len(supported_rankable_rows),
        "excluded_without_current_calibration_policy": excluded_without_current_calibration_policy,
        "promotion_evaluable": aggregate,
        "families": families,
        "rows": rows,
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
        "runtime_logic_changed": False,
        "notes": [
            "Reads persisted Phase16 market_mismatch_rows directly; legacy WATCH best_market and research-visibility rows are excluded.",
            "Only PHASE16_DISCRIMINATION_GATED_V2 candidates count as promotion-quality; earlier Phase16 candidates remain historical diagnostics and are excluded from promotion evidence.",
            "Candidates enter the ledger immediately as PENDING and settle automatically after soccer_results receives final goals.",
            "Only Phase16 candidates explicitly marked promotion_shadow_eligible under the current calibration policy are admitted; legacy candidates are excluded.",
            "At most the latest pre-kickoff primary rankable candidate per fixture and market family is retained.",
            "Supported settlement families are 1X2, FT_TOTALS and BTTS; PUSH is supported for integer totals.",
            "The latest available versioned source regime is used to avoid mixing old model regimes.",
            "Results are joined only for settlement and never feed candidate generation.",
        ],
    }


def _load_rows(conn, *, lookback_days: int, max_rows: int) -> list[dict[str, Any]]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, int(lookback_days)))
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                (mm.row ->> 'fixture_id')::BIGINT AS fixture_id,
                p.generated_at_utc AS generated_at,
                COALESCE(NULLIF(mm.row ->> 'stage', ''), e.stage) AS stage,
                mm.row AS phase16_candidate,
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
                COALESCE(p.payload -> 'market_mismatch_rows', '[]'::jsonb)
            ) AS mm(row)
            JOIN soccer_fixtures f
              ON f.fixture_id = (mm.row ->> 'fixture_id')::BIGINT
            LEFT JOIN soccer_results r
              ON r.fixture_id = f.fixture_id
            LEFT JOIN soccer_refresh_events e
              ON e.fixture_id = f.fixture_id
             AND e.generated_at = p.generated_at_utc
            WHERE p.generated_at_utc >= %s
              AND p.generated_at_utc < f.kickoff
              AND mm.row ->> 'market_family' IN ('1X2', 'FT_TOTALS', 'BTTS')
              AND COALESCE((mm.row ->> 'rankable')::boolean, false) = true
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
    report["raw_phase16_candidates_loaded"] = len(raw_rows)
    report["lookback_days"] = int(lookback_days)
    return report
