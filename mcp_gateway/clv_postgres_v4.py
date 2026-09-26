from __future__ import annotations

from collections import Counter, defaultdict
import gc
from datetime import datetime, timedelta, timezone
import math
import re
from typing import Any

from mcp_gateway import market_mismatch_v4, persistence

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_TRUE_CLV_POSTGRES_V4_1.1.10"
SIGNAL_STAGES = ("T-40", "T-20", "T-10")
TEAM_TOTALS_RESEARCH_STAGES = ("EARLY_RESEARCH", "T-90", "T-60", "T-40", "T-30", "T-20", "T-10", "CLOSE")
SIGNAL_CLASSES = ("BET", "LEAN", "WATCH")
MIN_TRUE_CLOSE_ROWS = 50
DERIVATIVE_MARKET_SOURCES = (
    ("TEAM_TOTALS", "team_totals_intelligence", "observed_exact_market_rows"),
    ("1H", "one_h_goals_intelligence", "observed_market_rows"),
    ("FT_CORNERS", "corners_intelligence", "observed_market_rows"),
    ("TEAM_CORNERS", "team_corners_intelligence", "observed_market_rows"),
)


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def _as_utc_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        out = value
    elif isinstance(value, str):
        text = value.strip().replace("Z", "+00:00")
        try:
            out = datetime.fromisoformat(text)
        except ValueError:
            return None
    else:
        return None
    if out.tzinfo is None:
        out = out.replace(tzinfo=timezone.utc)
    return out.astimezone(timezone.utc)


def _is_strictly_later_provider_quote(snapshot: dict[str, Any], generated_at: Any) -> bool:
    signal_at = _as_utc_datetime(generated_at)
    captured_at = _as_utc_datetime(snapshot.get("captured_at"))
    provider_update = _as_utc_datetime(snapshot.get("provider_update"))
    if signal_at is None or captured_at is None or provider_update is None:
        return False
    return captured_at > signal_at and provider_update > signal_at


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


def _build_ft_totals_clv_funnel(
    mapped_family_counts: Counter[str],
    priced_entry_family_counts: Counter[str],
    skip_reason_family_counts: dict[str, Counter[str]],
    family_counts: Counter[str],
    unpriced_research_placeholders_ignored: int,
) -> dict[str, Any]:
    return {
        "mapped_signal_rows": int(mapped_family_counts.get("FT_TOTALS", 0)),
        "unpriced_research_placeholders_ignored": int(unpriced_research_placeholders_ignored),
        "priced_entry_rows": int(priced_entry_family_counts.get("FT_TOTALS", 0)),
        "no_later_prekickoff_snapshot_rows": int(
            skip_reason_family_counts["NO_LATER_PREKICKOFF_MARKET_SNAPSHOT"].get("FT_TOTALS", 0)
        ),
        "later_snapshot_without_later_provider_update_rows": int(
            skip_reason_family_counts["NO_LATER_PROVIDER_UPDATE"].get("FT_TOTALS", 0)
        ),
        "selection_mismatch_at_close_rows": int(
            skip_reason_family_counts["NO_SELECTION_MATCH_AT_CLOSE"].get("FT_TOTALS", 0)
        ),
        "true_clv_rows": int(family_counts.get("FT_TOTALS", 0)),
        "provider_requests_added": 0,
        "policy": (
            "UNPRICED_FT_TOTALS_RESEARCH_PLACEHOLDERS_ARE_DIAGNOSTIC_ONLY; "
            "PRICED_ENTRIES_REQUIRE_STRICTLY_LATER_PREKICKOFF_PROVIDER_UPDATE_AND_SELECTION_MATCH"
        ),
    }


def _load_strict_team_totals_capture_fixture_ids(conn, *, lookback_days: int) -> set[int]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, int(lookback_days)))
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT e.fixture_id
            FROM soccer_refresh_events e
            JOIN soccer_fixtures f ON f.fixture_id = e.fixture_id
            WHERE e.fixture_id IS NOT NULL
              AND e.generated_at >= %s
              AND e.generated_at < f.kickoff
              AND COALESCE(
                    e.payload -> 'team_totals_diversity_capture' ->> 'qualifies',
                    'false'
                  ) = 'true'
            """,
            (cutoff,),
        )
        return {
            int(row[0])
            for row in cur.fetchall()
            if row and row[0] is not None
        }


def _build_team_totals_maturation_funnel(
    strict_capture_fixture_ids: set[int],
    modeled_signal_fixture_ids: set[int],
    true_clv_fixture_ids: set[int],
    modeled_signal_kickoffs: dict[int, datetime] | None = None,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    captures = set(strict_capture_fixture_ids)
    signals = set(modeled_signal_fixture_ids) & captures if captures else set(modeled_signal_fixture_ids)
    true_clv = set(true_clv_fixture_ids) & (captures | signals) if (captures or signals) else set(true_clv_fixture_ids)

    capture_without_signal = captures - signals
    signal_without_true_clv = signals - true_clv
    directional_target = 20
    modeled_signal_kickoffs = modeled_signal_kickoffs if isinstance(modeled_signal_kickoffs, dict) else {}
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    else:
        now = now.astimezone(timezone.utc)

    timing = {
        "already_kicked_off": 0,
        "within_55m": 0,
        "within_2h": 0,
        "within_6h": 0,
        "within_12h": 0,
        "within_24h": 0,
        "within_48h": 0,
        "beyond_48h": 0,
        "missing_kickoff": 0,
    }
    future_kickoffs: list[datetime] = []
    for fixture_id in signal_without_true_clv:
        kickoff = modeled_signal_kickoffs.get(int(fixture_id))
        if not isinstance(kickoff, datetime):
            timing["missing_kickoff"] += 1
            continue
        if kickoff.tzinfo is None:
            kickoff = kickoff.replace(tzinfo=timezone.utc)
        else:
            kickoff = kickoff.astimezone(timezone.utc)
        delta = kickoff - now
        if delta.total_seconds() <= 0:
            timing["already_kicked_off"] += 1
            continue
        future_kickoffs.append(kickoff)
        minutes = delta.total_seconds() / 60.0
        if minutes <= 55:
            timing["within_55m"] += 1
        if minutes <= 120:
            timing["within_2h"] += 1
        if minutes <= 360:
            timing["within_6h"] += 1
        if minutes <= 720:
            timing["within_12h"] += 1
        if minutes <= 1440:
            timing["within_24h"] += 1
        if minutes <= 2880:
            timing["within_48h"] += 1
        else:
            timing["beyond_48h"] += 1

    return {
        "strict_capture_unique_fixtures": len(captures),
        "modeled_signal_unique_fixtures": len(signals),
        "true_clv_unique_fixtures": len(true_clv),
        "capture_without_modeled_signal": len(capture_without_signal),
        "modeled_signal_without_later_real_close": len(signal_without_true_clv),
        "true_clv_directional_target": directional_target,
        "true_clv_unique_fixtures_remaining_to_directional": max(directional_target - len(true_clv), 0),
        "strict_capture_fixture_ids": sorted(captures),
        "modeled_signal_fixture_ids": sorted(signals),
        "true_clv_fixture_ids": sorted(true_clv),
        "capture_without_modeled_signal_fixture_ids": sorted(capture_without_signal),
        "modeled_signal_without_later_real_close_fixture_ids": sorted(signal_without_true_clv),
        "pending_timing": timing,
        "pending_future_fixtures": len(future_kickoffs),
        "next_pending_kickoff": min(future_kickoffs).isoformat() if future_kickoffs else None,
        "provider_requests_added": 0,
        "policy": (
            "STRICT_CAPTURE_MARKER -> DERIVATIVE_TEAM_TOTALS_MODELED_SIGNAL -> "
            "STRICTLY_LATER_PROVIDER_QUOTE -> COMPARABLE_TRUE_CLV"
        ),
    }


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
                jsonb_build_object(
                    'tier', e.payload -> 'tier',
                    'model_signal', e.payload -> 'model_signal',
                    'sporting_shortlist', e.payload -> 'sporting_shortlist'
                ) AS event_payload,
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
                jsonb_build_object(
                    'best_market', e.payload -> 'best_market',
                    'tier', e.payload -> 'tier',
                    'model_signal', e.payload -> 'model_signal',
                    'sporting_shortlist', e.payload -> 'sporting_shortlist'
                ) AS event_payload,
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
            ORDER BY e.generated_at DESC
            LIMIT %s
            """,
            (cutoff, list(SIGNAL_STAGES), list(SIGNAL_CLASSES), max(1, int(max_rows))),
        )
        columns = [desc.name for desc in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def _load_derivative_signals(conn, *, lookback_days: int, max_rows: int) -> list[dict[str, Any]]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, int(lookback_days)))
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                e.fixture_id,
                e.generated_at,
                e.stage,
                e.classification,
                jsonb_build_object(
                    'tier', e.payload -> 'tier',
                    'model_signal', e.payload -> 'model_signal',
                    'sporting_shortlist', e.payload -> 'sporting_shortlist'
                ) AS event_payload,
                f.kickoff,
                f.league,
                f.home_team,
                f.away_team,
                p.payload ->> 'model_version' AS model_version,
                p.payload ->> 'version' AS automation_version,
                d.market_candidate,
                d.signal_source
            FROM soccer_refresh_events e
            JOIN soccer_fixtures f ON f.fixture_id = e.fixture_id
            LEFT JOIN soccer_pipeline_runs p ON p.generated_at_utc = e.generated_at
            CROSS JOIN LATERAL (
                SELECT
                    jsonb_set(row_value, '{market_family}', to_jsonb('TEAM_TOTALS'::text), true) AS market_candidate,
                    'DERIVATIVE_INTELLIGENCE:team_totals_intelligence'::text AS signal_source
                FROM jsonb_array_elements(
                    COALESCE(e.payload -> 'team_totals_intelligence' -> 'observed_exact_market_rows', '[]'::jsonb)
                ) AS t(row_value)

                UNION ALL

                SELECT
                    jsonb_set(row_value, '{market_family}', to_jsonb('1H'::text), true),
                    'DERIVATIVE_INTELLIGENCE:one_h_goals_intelligence'::text
                FROM jsonb_array_elements(
                    COALESCE(e.payload -> 'one_h_goals_intelligence' -> 'observed_market_rows', '[]'::jsonb)
                ) AS h(row_value)

                UNION ALL

                SELECT
                    jsonb_set(row_value, '{market_family}', to_jsonb('FT_CORNERS'::text), true),
                    'DERIVATIVE_INTELLIGENCE:corners_intelligence'::text
                FROM jsonb_array_elements(
                    COALESCE(e.payload -> 'corners_intelligence' -> 'observed_market_rows', '[]'::jsonb)
                ) AS c(row_value)

                UNION ALL

                SELECT
                    jsonb_set(row_value, '{market_family}', to_jsonb('TEAM_CORNERS'::text), true),
                    'DERIVATIVE_INTELLIGENCE:team_corners_intelligence'::text
                FROM jsonb_array_elements(
                    COALESCE(e.payload -> 'team_corners_intelligence' -> 'observed_market_rows', '[]'::jsonb)
                ) AS tc(row_value)
            ) AS d
            WHERE e.generated_at >= %s
              AND (
                    e.stage = ANY(%s)
                    OR (
                        d.signal_source = 'DERIVATIVE_INTELLIGENCE:team_totals_intelligence'
                        AND e.stage = ANY(%s)
                    )
                  )
              AND e.generated_at < f.kickoff
            ORDER BY e.generated_at DESC
            LIMIT %s
            """,
            (
                cutoff,
                list(SIGNAL_STAGES),
                list(TEAM_TOTALS_RESEARCH_STAGES),
                max(1, int(max_rows)),
            ),
        )
        columns = [desc.name for desc in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def _derivative_signals_from_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compatibility helper for unit tests and legacy callers."""
    signals: list[dict[str, Any]] = []
    for event in events:
        payload = event.get("event_payload")
        if not isinstance(payload, dict):
            continue
        for family_hint, container_key, rows_key in DERIVATIVE_MARKET_SOURCES:
            container = payload.get(container_key)
            if not isinstance(container, dict):
                continue
            rows = container.get(rows_key)
            if not isinstance(rows, list):
                continue
            for market_row in rows:
                if not isinstance(market_row, dict):
                    continue
                candidate = dict(market_row)
                candidate.setdefault("market_family", family_hint)
                out = dict(event)
                out["market_candidate"] = candidate
                out["signal_source"] = f"DERIVATIVE_INTELLIGENCE:{container_key}"
                signals.append(out)
    return signals

def _load_market_snapshots(
    conn,
    fixture_ids: list[int],
    *,
    cutoff: datetime,
    market_names: list[str],
) -> list[dict[str, Any]]:
    if not fixture_ids or not market_names:
        return []
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT ON (m.fixture_id, m.market_id, m.bookmaker_id, m.provider_update)
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
              AND m.market = ANY(%s)
              AND m.captured_at >= %s
              AND m.captured_at < f.kickoff
            ORDER BY
                m.fixture_id,
                m.market_id,
                m.bookmaker_id,
                m.provider_update,
                m.captured_at DESC
            """,
            (fixture_ids, market_names, cutoff),
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
    derivative_signals: list[dict[str, Any]],
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

    for raw in derivative_signals:
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


def _is_period_team_total_signal(signal: dict[str, Any]) -> bool:
    if str(signal.get("signal_source") or "") != "DERIVATIVE_INTELLIGENCE:team_totals_intelligence":
        return False
    candidate = signal.get("market_candidate")
    if not isinstance(candidate, dict):
        return False
    return _family(candidate) in {"1H", "2H"}


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
        strict_team_totals_capture_fixture_ids = _load_strict_team_totals_capture_fixture_ids(
            conn,
            lookback_days=lookback_days,
        )
        pipeline_signals = _load_pipeline_market_signals(
            conn,
            lookback_days=lookback_days,
            max_rows=max_signals,
        )
        derivative_signals_raw = _load_derivative_signals(
            conn,
            lookback_days=lookback_days,
            max_rows=max_signals,
        )
        derivative_period_team_total_rows_excluded = sum(
            1 for signal in derivative_signals_raw if _is_period_team_total_signal(signal)
        )
        derivative_source_counts_raw = Counter(
            str(signal.get("signal_source") or "UNKNOWN")
            for signal in derivative_signals_raw
        )
        derivative_family_counts_raw = Counter(
            str(_family(signal.get("market_candidate") or {}) or "UNMAPPED")
            for signal in derivative_signals_raw
        )
        derivative_signals = [
            signal
            for signal in derivative_signals_raw
            if not _is_period_team_total_signal(signal)
        ]
        derivative_source_counts = Counter(
            str(signal.get("signal_source") or "UNKNOWN")
            for signal in derivative_signals
        )
        team_totals_modeled_signal_fixture_ids = {
            int(signal["fixture_id"])
            for signal in derivative_signals
            if str(signal.get("signal_source") or "") == "DERIVATIVE_INTELLIGENCE:team_totals_intelligence"
            and signal.get("fixture_id") is not None
        }
        team_totals_modeled_signal_kickoffs: dict[int, datetime] = {}
        for signal in derivative_signals:
            if str(signal.get("signal_source") or "") != "DERIVATIVE_INTELLIGENCE:team_totals_intelligence":
                continue
            fixture_id = signal.get("fixture_id")
            kickoff = signal.get("kickoff")
            if fixture_id is None or not isinstance(kickoff, datetime):
                continue
            team_totals_modeled_signal_kickoffs[int(fixture_id)] = kickoff
        derivative_family_counts = Counter(
            str(_family(signal.get("market_candidate") or {}) or "UNMAPPED")
            for signal in derivative_signals
        )
        legacy_signals = _load_legacy_signals(
            conn,
            lookback_days=lookback_days,
            max_rows=max_signals,
        )
        pipeline_market_rows_loaded = len(pipeline_signals)
        derivative_market_rows_loaded_raw = len(derivative_signals_raw)
        derivative_market_rows_loaded = len(derivative_signals)
        derivative_event_rows_loaded = len({
            (row.get("fixture_id"), row.get("generated_at"))
            for row in derivative_signals
        })
        legacy_signal_rows_loaded = len(legacy_signals)

        signals = _merge_signals(
            pipeline_signals,
            derivative_signals,
            legacy_signals,
            max_rows=max(1, int(max_signals)),
        )
        signal_rows_considered = len(signals)
        fixture_ids = sorted({int(row["fixture_id"]) for row in signals if row.get("fixture_id") is not None})
        cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, int(lookback_days)))

        # These source collections can each contain up to max_signals rows.
        # All information needed below is now represented by the bounded merged
        # signal list plus scalar/counter diagnostics.
        del pipeline_signals
        del derivative_signals_raw
        del derivative_signals
        del legacy_signals
        gc.collect()

    tracked: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    mapped_family_counts: Counter[str] = Counter()
    priced_entry_family_counts: Counter[str] = Counter()
    signal_source_counts: Counter[str] = Counter()
    skip_reason_market_counts: dict[str, Counter[str]] = defaultdict(Counter)
    skip_reason_family_counts: dict[str, Counter[str]] = defaultdict(Counter)
    ft_totals_unpriced_research_placeholders_ignored = 0

    signals_by_fixture: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for signal in signals:
        fixture_id = signal.get("fixture_id")
        if fixture_id is not None:
            signals_by_fixture[int(fixture_id)].append(signal)
    del signals
    gc.collect()

    snapshot_fixture_batch_size = 50
    for batch_start in range(0, len(fixture_ids), snapshot_fixture_batch_size):
        batch_fixture_ids = fixture_ids[batch_start:batch_start + snapshot_fixture_batch_size]
        batch_signals = [
            signal
            for fixture_id in batch_fixture_ids
            for signal in signals_by_fixture.get(fixture_id, [])
        ]
        batch_market_names = sorted({
            str(candidate.get("market"))
            for signal in batch_signals
            for candidate in [signal.get("market_candidate")]
            if isinstance(candidate, dict) and candidate.get("market")
        })

        with persistence._connect() as snapshot_conn:
            snapshots = _load_market_snapshots(
                snapshot_conn,
                batch_fixture_ids,
                cutoff=cutoff,
                market_names=batch_market_names,
            )

        snapshots_by_fixture: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for snapshot in snapshots:
            if snapshot.get("fixture_id") is not None:
                snapshots_by_fixture[int(snapshot["fixture_id"])].append(snapshot)
        for signal in batch_signals:
            candidate = signal.get("market_candidate")
            if not isinstance(candidate, dict):
                reasons["NO_MARKET_CANDIDATE"] += 1
                continue

            family = _family(candidate)
            if family is None:
                reasons["UNMAPPED_MARKET_FAMILY"] += 1
                skip_reason_market_counts["UNMAPPED_MARKET_FAMILY"][_candidate_label(candidate)] += 1
                skip_reason_family_counts["UNMAPPED_MARKET_FAMILY"]["UNMAPPED"] += 1
                continue

            mapped_family_counts[family] += 1
            entry_price = _num(candidate.get("decimal_price"))
            if entry_price is None:
                entry_price = _num(candidate.get("price"))
            if entry_price is None or entry_price <= 1.0:
                raw_family = str(
                    candidate.get("market_family")
                    or candidate.get("family")
                    or ""
                ).upper()
                if family == "FT_TOTALS" and raw_family == "FT_TOTALS_RESEARCH":
                    ft_totals_unpriced_research_placeholders_ignored += 1
                    continue
                reasons["INVALID_ENTRY_PRICE"] += 1
                skip_reason_market_counts["INVALID_ENTRY_PRICE"][_candidate_label(candidate)] += 1
                skip_reason_family_counts["INVALID_ENTRY_PRICE"][family] += 1
                continue

            priced_entry_family_counts[family] += 1
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
                skip_reason_market_counts["MISSING_TIMESTAMPS"][_candidate_label(candidate)] += 1
                skip_reason_family_counts["MISSING_TIMESTAMPS"][family] += 1
                continue

            later_market_snapshots = [
                snap
                for snap in snapshots_by_fixture.get(fixture_id, [])
                if snap.get("captured_at") is not None
                and generated_at < snap["captured_at"] < kickoff
                and _norm(snap.get("market")) == _norm(candidate.get("market"))
            ]
            if not later_market_snapshots:
                reasons["NO_LATER_PREKICKOFF_MARKET_SNAPSHOT"] += 1
                skip_reason_market_counts["NO_LATER_PREKICKOFF_MARKET_SNAPSHOT"][_candidate_label(candidate)] += 1
                skip_reason_family_counts["NO_LATER_PREKICKOFF_MARKET_SNAPSHOT"][family] += 1
                continue

            candidates = [
                snap
                for snap in later_market_snapshots
                if _is_strictly_later_provider_quote(snap, generated_at)
            ]
            if not candidates:
                reasons["NO_LATER_PROVIDER_UPDATE"] += 1
                skip_reason_market_counts["NO_LATER_PROVIDER_UPDATE"][_candidate_label(candidate)] += 1
                skip_reason_family_counts["NO_LATER_PROVIDER_UPDATE"][family] += 1
                continue

            same_book = [
                snap for snap in candidates
                if _norm(snap.get("bookmaker")) == _norm(candidate.get("bookmaker"))
            ]
            pool = same_book or candidates
            close_at = max(snap["captured_at"] for snap in pool)
            close_groups = [snap for snap in pool if snap["captured_at"] == close_at]
            close_provider_updates = [
                _as_utc_datetime(snap.get("provider_update"))
                for snap in close_groups
                if _as_utc_datetime(snap.get("provider_update")) is not None
            ]
            close_provider_update = max(close_provider_updates) if close_provider_updates else None

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
                skip_reason_market_counts["NO_SELECTION_MATCH_AT_CLOSE"][_candidate_label(candidate)] += 1
                skip_reason_family_counts["NO_SELECTION_MATCH_AT_CLOSE"][family] += 1
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
                "closing_provider_update": close_provider_update.isoformat() if close_provider_update is not None else None,
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
    team_totals_true_clv_fixture_ids = {
        int(row["fixture_id"])
        for row in comparable
        if row.get("fixture_id") is not None
        and str(row.get("market_family") or "").upper() in {"TEAM_TOTALS", "HOME_TT", "AWAY_TT"}
    }
    team_totals_maturation_funnel = _build_team_totals_maturation_funnel(
        strict_team_totals_capture_fixture_ids,
        team_totals_modeled_signal_fixture_ids,
        team_totals_true_clv_fixture_ids,
        team_totals_modeled_signal_kickoffs,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": "ACTIVE_TRUE_CLV_SAMPLE" if len(comparable) >= MIN_TRUE_CLOSE_ROWS else "COLLECTING_TRUE_CLV",
        "lookback_days": int(lookback_days),
        "signal_rows_considered": signal_rows_considered,
        "pipeline_market_rows_loaded": pipeline_market_rows_loaded,
        "derivative_event_rows_loaded": derivative_event_rows_loaded,
        "derivative_market_rows_loaded_raw": derivative_market_rows_loaded_raw,
        "derivative_period_team_total_rows_excluded": derivative_period_team_total_rows_excluded,
        "derivative_market_rows_loaded": derivative_market_rows_loaded,
        "derivative_source_counts_raw": dict(sorted(derivative_source_counts_raw.items())),
        "derivative_family_counts_raw": dict(sorted(derivative_family_counts_raw.items())),
        "derivative_source_counts": dict(sorted(derivative_source_counts.items())),
        "derivative_family_counts": dict(sorted(derivative_family_counts.items())),
        "legacy_signal_rows_loaded": legacy_signal_rows_loaded,
        "tracked_rows": len(tracked),
        "comparable_true_clv_rows": len(comparable),
        "minimum_true_close_rows": MIN_TRUE_CLOSE_ROWS,
        "snapshot_fixture_batch_size": snapshot_fixture_batch_size,
        "family_counts": dict(sorted(family_counts.items())),
        "signal_source_counts": dict(sorted(signal_source_counts.items())),
        "team_totals_maturation_funnel": team_totals_maturation_funnel,
        "skip_reasons": dict(sorted(reasons.items())),
        "skip_reason_market_counts": {
            reason: dict(counts.most_common())
            for reason, counts in sorted(skip_reason_market_counts.items())
        },
        "skip_reason_family_counts": {
            reason: dict(counts.most_common())
            for reason, counts in sorted(skip_reason_family_counts.items())
        },
        "mapped_family_counts": dict(sorted(mapped_family_counts.items())),
        "priced_entry_family_counts": dict(sorted(priced_entry_family_counts.items())),
        "ft_totals_maturation_funnel": _build_ft_totals_clv_funnel(
            mapped_family_counts,
            priced_entry_family_counts,
            skip_reason_family_counts,
            family_counts,
            ft_totals_unpriced_research_placeholders_ignored,
        ),
        "rows": tracked,
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
        "notes": [
            "Primary signal sources are Postgres match_table_rows plus persisted derivative intelligence observed-market rows; period-specific team totals are excluded from generic 1H/2H and FT team-total CLV until they have dedicated families; capped source reads prioritize recent pre-kickoff signals, close lookup reads only the latest pre-kickoff snapshot per bookmaker/market in bounded fixture batches, and legacy event best_market rows are fallback-only.",
            "Pipeline and legacy SQL select only CLV-required event metadata instead of duplicating full refresh payload JSON per signal; source collections are released before snapshot matching to stay within the runtime memory envelope without reducing the signal cap.",
            "Market closes come from Postgres soccer_market_snapshots; GitHub compact history is not required.",
            "Derivative source/family counts are reported before and after period-team-total exclusion so missing Team Totals can be localized to generation versus close matching.",
            "Team Totals may enter CLV collection from any explicitly pre-kickoff research stage, including EARLY_RESEARCH/T-90/T-60/T-30/CLOSE, while other derivative families retain the narrower T-40/T-20/T-10 stage policy.",
            "FT_TOTALS_RESEARCH rows without a real entry price are diagnostic placeholders, not CLV failures. Priced FT Totals retain strict later-snapshot and later-provider-update requirements.",
            "Probability/price CLV is computed only when the exact same market side and line are comparable at close.",
            "A true close must be a strictly later ingest AND carry a provider_update strictly later than the signal timestamp; cache replays and unchanged provider quotes do not count as new CLV evidence.",
            "Team Totals maturation is reported as a provider-free funnel from strict market capture to modeled derivative signal to later real close/true CLV, using unique fixture IDs so repeated selections cannot inflate progress.",
            "Over/Under selections match by side plus explicit line, so 'Over' and 'Over 2.5' are equivalent only when line=2.5.",
            "Line movement may still be recorded when the selected side survives but the sportsbook line changes.",
            "Same-book close is preferred; otherwise the latest cross-book snapshot at the latest pre-kickoff timestamp is used.",
        ],
    }
