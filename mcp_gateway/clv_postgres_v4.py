from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import math
import re
from typing import Any

from mcp_gateway import market_mismatch_v4, persistence

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_TRUE_CLV_POSTGRES_V4_1.1.0"
SIGNAL_STAGES = ("T-40", "T-20", "T-10")
SIGNAL_CLASSES = ("BET", "LEAN", "WATCH")
MIN_TRUE_CLOSE_ROWS = 50


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def _line_from_selection(selection: Any) -> float | None:
    match = re.search(r"(?:over|under)\s*([0-9]+(?:\.[0-9]+)?)", str(selection or ""), re.I)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _same_line(a: float | None, b: float | None) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(a - b) < 1e-6


def _selection_side(selection: Any) -> str:
    text = _norm(selection)
    if text.startswith("over"):
        return "over"
    if text.startswith("under"):
        return "under"
    return text


def _selection_matches(value_selection: Any, target_selection: Any) -> bool:
    value_side = _selection_side(value_selection)
    target_side = _selection_side(target_selection)
    if target_side in {"over", "under"}:
        return value_side == target_side
    return _norm(value_selection) == _norm(target_selection)


def _value_line(value: dict[str, Any]) -> float | None:
    line = _num(value.get("line"))
    return line if line is not None else _line_from_selection(value.get("selection"))


def _value_price(value: dict[str, Any]) -> float | None:
    price = _num(value.get("decimal_price"))
    if price is None:
        price = _num(value.get("price"))
    return price


def _group_fair_probability(
    values: list[dict[str, Any]],
    selection: Any,
    line: float | None,
) -> tuple[float | None, float | None]:
    implied: list[tuple[dict[str, Any], float]] = []
    for value in values:
        if not isinstance(value, dict):
            continue
        if line is not None and not _same_line(_value_line(value), line):
            continue
        price = _value_price(value)
        if price is None or price <= 1.0:
            continue
        implied.append((value, 1.0 / price))
    total = sum(prob for _, prob in implied)
    if total <= 0:
        return None, None

    for value, prob in implied:
        if not _selection_matches(value.get("selection"), selection):
            continue
        return prob / total, _value_price(value)
    return None, None


def _closing_line_candidate(values: list[dict[str, Any]], selection: Any) -> tuple[float | None, float | None]:
    side = _selection_side(selection)
    candidates: list[tuple[float | None, float]] = []
    for value in values:
        if not isinstance(value, dict):
            continue
        if _selection_side(value.get("selection")) != side:
            continue
        price = _value_price(value)
        if price is None or price <= 1.0:
            continue
        candidates.append((_value_line(value), price))
    if len(candidates) != 1:
        return None, None
    return candidates[0]


def _family(market_candidate: dict[str, Any]) -> str | None:
    return market_mismatch_v4.canonical_market_family({
        "market_family": market_candidate.get("market_family") or market_candidate.get("family"),
        "market": market_candidate.get("market"),
        "selection": market_candidate.get("selection"),
    })


def _candidate_label(market_candidate: dict[str, Any]) -> str:
    raw_family = market_candidate.get("market_family") or market_candidate.get("family") or "(none)"
    market = market_candidate.get("market") or "(none)"
    return f"{raw_family} | {market}"


def _load_pipeline_market_signals(conn, *, lookback_days: int, max_rows: int) -> list[dict[str, Any]]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, int(lookback_days)))
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                (mr.row ->> 'fixture_id')::BIGINT AS fixture_id,
                p.generated_at_utc AS generated_at,
                COALESCE(NULLIF(mr.row ->> 'stage', ''), e.stage) AS stage,
                COALESCE(NULLIF(mr.row ->> 'classification', ''), e.classification) AS classification,
                mr.row AS market_candidate,
                e.payload AS event_payload,
                f.kickoff,
                f.league,
                f.home_team,
                f.away_team,
                p.payload ->> 'model_version' AS model_version,
                p.payload ->> 'version' AS automation_version,
                'PIPELINE_MATCH_TABLE'::TEXT AS signal_source
            FROM soccer_pipeline_runs p
            CROSS JOIN LATERAL jsonb_array_elements(
                COALESCE(p.payload -> 'match_table_rows', '[]'::jsonb)
            ) AS mr(row)
            JOIN soccer_fixtures f
              ON f.fixture_id = (mr.row ->> 'fixture_id')::BIGINT
            LEFT JOIN soccer_refresh_events e
              ON e.fixture_id = f.fixture_id
             AND e.generated_at = p.generated_at_utc
            WHERE p.generated_at_utc >= %s
              AND p.generated_at_utc < f.kickoff
              AND COALESCE(NULLIF(mr.row ->> 'stage', ''), e.stage) = ANY(%s)
              AND COALESCE(NULLIF(mr.row ->> 'classification', ''), e.classification) = ANY(%s)
              AND NULLIF(mr.row ->> 'market', '') IS NOT NULL
              AND NULLIF(mr.row ->> 'selection', '') IS NOT NULL
            ORDER BY p.generated_at_utc ASC
            LIMIT %s
            """,
            (cutoff, list(SIGNAL_STAGES), list(SIGNAL_CLASSES), max(1, int(max_rows))),
        )
        columns = [desc.name for desc in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def _load_legacy_signals(conn, *, lookback_days: int, max_rows: int) -> list[dict[str, Any]]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, int(lookback_days)))
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                e.fixture_id,
                e.generated_at,
                e.stage,
                e.classification,
                e.payload AS event_payload,
                f.kickoff,
                f.league,
                f.home_team,
                f.away_team,
                p.payload ->> 'model_version' AS model_version,
                p.payload ->> 'version' AS automation_version,
                'LEGACY_BEST_MARKET'::TEXT AS signal_source
            FROM soccer_refresh_events e
            JOIN soccer_fixtures f ON f.fixture_id = e.fixture_id
            LEFT JOIN soccer_pipeline_runs p ON p.generated_at_utc = e.generated_at
            WHERE e.generated_at >= %s
              AND e.stage = ANY(%s)
              AND e.classification = ANY(%s)
              AND e.generated_at < f.kickoff
            ORDER BY e.generated_at ASC
            LIMIT %s
            """,
            (cutoff, list(SIGNAL_STAGES), list(SIGNAL_CLASSES), max(1, int(max_rows))),
        )
        columns = [desc.name for desc in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def _load_market_snapshots(conn, fixture_ids: list[int], *, cutoff: datetime) -> list[dict[str, Any]]:
    if not fixture_ids:
        return []
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                m.fixture_id,
                m.captured_at,
                m.stage,
                m.bookmaker_id,
                m.bookmaker,
                m.market_id,
                m.market,
                m.values,
                m.provider_update,
                f.kickoff
            FROM soccer_market_snapshots m
            JOIN soccer_fixtures f ON f.fixture_id = m.fixture_id
            WHERE m.fixture_id = ANY(%s)
              AND m.captured_at >= %s
              AND m.captured_at < f.kickoff
            ORDER BY m.fixture_id, m.captured_at ASC
            """,
            (fixture_ids, cutoff),
        )
        columns = [desc.name for desc in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def _best_market_from_event(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    best = payload.get("best_market")
    return best if isinstance(best, dict) and best else None


def _legacy_to_signal(signal: dict[str, Any]) -> dict[str, Any] | None:
    best = _best_market_from_event(signal.get("event_payload"))
    if best is None:
        return None
    out = dict(signal)
    out["market_candidate"] = best
    return out


def _signal_identity(signal: dict[str, Any]) -> tuple[Any, ...]:
    candidate = signal.get("market_candidate") if isinstance(signal.get("market_candidate"), dict) else {}
    line = _num(candidate.get("line"))
    if line is None:
        line = _line_from_selection(candidate.get("selection"))
    generated_at = signal.get("generated_at")
    if hasattr(generated_at, "isoformat"):
        generated_at = generated_at.isoformat()
    return (
        signal.get("fixture_id"),
        generated_at,
        _norm(candidate.get("market")),
        _norm(candidate.get("selection")),
        line,
        _norm(candidate.get("bookmaker")),
    )


def _merge_signals(
    pipeline_signals: list[dict[str, Any]],
    legacy_signals: list[dict[str, Any]],
    *,
    max_rows: int,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for raw in pipeline_signals:
        if not isinstance(raw.get("market_candidate"), dict):
            continue
        key = _signal_identity(raw)
        if key in seen:
            continue
        seen.add(key)
        merged.append(raw)
        if len(merged) >= max_rows:
            return merged

    for raw in legacy_signals:
        signal = _legacy_to_signal(raw)
        if signal is None:
            continue
        key = _signal_identity(signal)
        if key in seen:
            continue
        seen.add(key)
        merged.append(signal)
        if len(merged) >= max_rows:
            break
    return merged


def _model_signal_from_event(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    direct = payload.get("model_signal")
    if direct:
        return str(direct)
    shortlist = payload.get("sporting_shortlist")
    if not isinstance(shortlist, dict):
        return None
    scores = [
        _num(shortlist.get("side_edge_score")),
        _num(shortlist.get("goal_environment_score")),
        _num(shortlist.get("two_way_scoring_score")),
    ]
    usable = [value for value in scores if value is not None]
    if not usable:
        return None
    score = max(usable)
    if score >= 85:
        return "VERY_STRONG"
    if score >= 75:
        return "STRONG"
    if score >= 60:
        return "MODERATE"
    return "WEAK"


def _model_signal_from_candidate(candidate: dict[str, Any], event_payload: Any) -> str | None:
    direct = candidate.get("model_signal")
    if direct:
        return str(direct)
    score = _num(candidate.get("model_signal_score"))
    if score is None:
        score = _num(candidate.get("sport_confidence_score"))
    if score is not None:
        if score >= 85:
            return "VERY_STRONG"
        if score >= 75:
            return "STRONG"
        if score >= 60:
            return "MODERATE"
        return "WEAK"
    return _model_signal_from_event(event_payload)


def build_from_postgres(*, lookback_days: int = 30, max_signals: int = 5000) -> dict[str, Any]:
    persistence.ensure_schema()
    with persistence._connect() as conn:
        pipeline_signals = _load_pipeline_market_signals(
            conn,
            lookback_days=lookback_days,
            max_rows=max_signals,
        )
        legacy_signals = _load_legacy_signals(
            conn,
            lookback_days=lookback_days,
            max_rows=max_signals,
        )
        signals = _merge_signals(
            pipeline_signals,
            legacy_signals,
            max_rows=max(1, int(max_signals)),
        )
        fixture_ids = sorted({int(row["fixture_id"]) for row in signals if row.get("fixture_id") is not None})
        cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, int(lookback_days)))
        snapshots = _load_market_snapshots(conn, fixture_ids, cutoff=cutoff)

    snapshots_by_fixture: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in snapshots:
        if row.get("fixture_id") is not None:
            snapshots_by_fixture[int(row["fixture_id"])].append(row)

    tracked: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    signal_source_counts: Counter[str] = Counter()
    skip_reason_market_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for signal in signals:
        candidate = signal.get("market_candidate")
        if not isinstance(candidate, dict):
            reasons["NO_MARKET_CANDIDATE"] += 1
            continue

        family = _family(candidate)
        if family is None:
            reasons["UNMAPPED_MARKET_FAMILY"] += 1
            skip_reason_market_counts["UNMAPPED_MARKET_FAMILY"][_candidate_label(candidate)] += 1
            continue

        entry_price = _num(candidate.get("decimal_price"))
        if entry_price is None:
            entry_price = _num(candidate.get("price"))
        if entry_price is None or entry_price <= 1.0:
            reasons["INVALID_ENTRY_PRICE"] += 1
            skip_reason_market_counts["INVALID_ENTRY_PRICE"][_candidate_label(candidate)] += 1
            continue

        entry_line = _num(candidate.get("line"))
        if entry_line is None:
            entry_line = _line_from_selection(candidate.get("selection"))

        entry_fair = _num(candidate.get("p_market_fair"))
        if entry_fair is None:
            entry_fair = _num(candidate.get("market_fair_probability"))
        if entry_fair is None:
            entry_fair = 1.0 / entry_price

        fixture_id = int(signal["fixture_id"])
        generated_at = signal.get("generated_at")
        kickoff = signal.get("kickoff")
        if generated_at is None or kickoff is None:
            reasons["MISSING_TIMESTAMPS"] += 1
            continue

        candidates = [
            snap
            for snap in snapshots_by_fixture.get(fixture_id, [])
            if snap.get("captured_at") is not None
            and generated_at <= snap["captured_at"] < kickoff
            and _norm(snap.get("market")) == _norm(candidate.get("market"))
        ]
        if not candidates:
            reasons["NO_PREKICKOFF_MARKET_SNAPSHOT"] += 1
            continue

        same_book = [
            snap for snap in candidates
            if _norm(snap.get("bookmaker")) == _norm(candidate.get("bookmaker"))
        ]
        pool = same_book or candidates
        close_at = max(snap["captured_at"] for snap in pool)
        close_groups = [snap for snap in pool if snap["captured_at"] == close_at]

        fair_values: list[float] = []
        price_values: list[float] = []
        close_line_values: list[float] = []
        exact_comparable = False

        for snap in close_groups:
            values = snap.get("values") if isinstance(snap.get("values"), list) else []
            fair, price = _group_fair_probability(values, candidate.get("selection"), entry_line)
            if fair is not None:
                fair_values.append(fair)
                exact_comparable = True
                if price is not None:
                    price_values.append(price)
                if entry_line is not None:
                    close_line_values.append(entry_line)
                continue
            close_line, close_price = _closing_line_candidate(values, candidate.get("selection"))
            if close_line is not None:
                close_line_values.append(close_line)
            if close_price is not None:
                price_values.append(close_price)

        if not price_values and not fair_values and not close_line_values:
            reasons["NO_SELECTION_MATCH_AT_CLOSE"] += 1
            continue

        closing_price = sorted(price_values)[len(price_values) // 2] if price_values else None
        closing_fair = sorted(fair_values)[len(fair_values) // 2] if fair_values else None
        closing_line = sorted(close_line_values)[len(close_line_values) // 2] if close_line_values else None

        probability_clv = (
            round((closing_fair - entry_fair) * 100.0, 6)
            if exact_comparable and closing_fair is not None and entry_fair is not None
            else None
        )
        price_clv = (
            round((entry_price / closing_price - 1.0) * 100.0, 6)
            if exact_comparable and closing_price is not None and closing_price > 1.0
            else None
        )
        line_movement = (
            round(closing_line - entry_line, 6)
            if closing_line is not None and entry_line is not None
            else None
        )

        source = str(signal.get("signal_source") or "UNKNOWN")
        family_counts[family] += 1
        signal_source_counts[source] += 1
        event_payload = signal.get("event_payload")
        tracked.append({
            "schema_version": SCHEMA_VERSION,
            "fixture_id": fixture_id,
            "league": signal.get("league"),
            "home_team": signal.get("home_team"),
            "away_team": signal.get("away_team"),
            "kickoff": kickoff.isoformat() if hasattr(kickoff, "isoformat") else str(kickoff),
            "stage": signal.get("stage"),
            "classification": signal.get("classification"),
            "market_family": family,
            "market": candidate.get("market"),
            "selection": candidate.get("selection"),
            "tier": candidate.get("tier") or (event_payload.get("tier") if isinstance(event_payload, dict) else None),
            "confidence": _model_signal_from_candidate(candidate, event_payload),
            "model_version": signal.get("model_version") or signal.get("automation_version"),
            "bookmaker": candidate.get("bookmaker"),
            "signal_source": source,
            "entry_timestamp": generated_at.isoformat() if hasattr(generated_at, "isoformat") else str(generated_at),
            "entry_line": entry_line,
            "line": entry_line,
            "entry_price": round(entry_price, 6),
            "signal_price": round(entry_price, 6),
            "entry_fair_probability": round(entry_fair, 8) if entry_fair is not None else None,
            "signal_fair_probability": round(entry_fair, 8) if entry_fair is not None else None,
            "closing_timestamp": close_at.isoformat() if hasattr(close_at, "isoformat") else str(close_at),
            "closing_line": closing_line,
            "closing_price": round(closing_price, 6) if closing_price is not None else None,
            "close_price": round(closing_price, 6) if closing_price is not None else None,
            "closing_fair_probability": round(closing_fair, 8) if closing_fair is not None else None,
            "close_fair_probability": round(closing_fair, 8) if closing_fair is not None else None,
            "probability_clv": probability_clv,
            "clv_probability_pp": probability_clv,
            "price_clv": price_clv,
            "clv_price_pct": price_clv,
            "line_movement": line_movement,
            "bookmaker_at_signal": candidate.get("bookmaker"),
            "is_true_closing_line": True,
            "closing_line_status": "POSTGRES_LATEST_PREKICKOFF_MARKET_SNAPSHOT",
            "probability_comparable_same_line": exact_comparable,
            "closing_source": "POSTGRES_LATEST_PREKICKOFF_MARKET_SNAPSHOT",
            "same_book_preferred": bool(same_book),
        })

    comparable = [row for row in tracked if row.get("probability_comparable_same_line")]
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": "ACTIVE_TRUE_CLV_SAMPLE" if len(comparable) >= MIN_TRUE_CLOSE_ROWS else "COLLECTING_TRUE_CLV",
        "lookback_days": int(lookback_days),
        "signal_rows_considered": len(signals),
        "pipeline_market_rows_loaded": len(pipeline_signals),
        "legacy_signal_rows_loaded": len(legacy_signals),
        "tracked_rows": len(tracked),
        "comparable_true_clv_rows": len(comparable),
        "minimum_true_close_rows": MIN_TRUE_CLOSE_ROWS,
        "family_counts": dict(sorted(family_counts.items())),
        "signal_source_counts": dict(sorted(signal_source_counts.items())),
        "skip_reasons": dict(sorted(reasons.items())),
        "skip_reason_market_counts": {
            reason: dict(counts.most_common())
            for reason, counts in sorted(skip_reason_market_counts.items())
        },
        "rows": tracked,
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
        "notes": [
            "Primary signal source is Postgres soccer_pipeline_runs.match_table_rows; legacy event best_market rows are fallback-only.",
            "Market closes come from Postgres soccer_market_snapshots; GitHub compact history is not required.",
            "Probability/price CLV is computed only when the exact same market side and line are comparable at close.",
            "Over/Under selections match by side plus explicit line, so 'Over' and 'Over 2.5' are equivalent only when line=2.5.",
            "Line movement may still be recorded when the selected side survives but the sportsbook line changes.",
            "Same-book close is preferred; otherwise the latest cross-book snapshot at the latest pre-kickoff timestamp is used.",
        ],
    }
