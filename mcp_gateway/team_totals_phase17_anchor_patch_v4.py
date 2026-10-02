from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from mcp_gateway import clv_postgres_v4

MODEL_VERSION = "SOCCER_PHASE17_TEAM_TOTALS_OLDEST_EXACT_ANCHOR_PATCH_V4_1.1.0"
PATCH_STATUS = "RESEARCH_ONLY_BOUNDED_CAP_INDEPENDENT_TEAM_TOTALS_SIGNAL_SLICE"
RECENT_CLOSED_LOOKBACK_HOURS = 12
RECENT_FIXTURE_LIMIT = 250
RECENT_EVENT_LIMIT = 600

_ORIGINAL_LOADER = clv_postgres_v4._load_derivative_signals
_INSTALLED = False


def _signal_time(signal: dict[str, Any]) -> datetime:
    parsed = clv_postgres_v4._as_utc_datetime(signal.get("generated_at"))
    return parsed if parsed is not None else datetime.max.replace(tzinfo=timezone.utc)


def _team_total_exact_key(signal: dict[str, Any]) -> tuple[Any, ...] | None:
    if str(signal.get("signal_source") or "") != "DERIVATIVE_INTELLIGENCE:team_totals_intelligence":
        return None
    candidate = signal.get("market_candidate")
    if not isinstance(candidate, dict):
        return None
    fixture_id = signal.get("fixture_id")
    if fixture_id is None:
        return None
    line = clv_postgres_v4._num(candidate.get("line"))
    if line is None:
        line = clv_postgres_v4._line_from_selection(candidate.get("selection"))
    if line is None:
        return None
    market = clv_postgres_v4._norm(candidate.get("market"))
    side = clv_postgres_v4._selection_side(candidate.get("selection"))
    if not market or not side:
        return None
    return (int(fixture_id), market, side, float(line))


def collapse_oldest_exact_team_totals(signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse only Team Totals recycled timestamps to the oldest exact point-in-time signal."""
    oldest_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}

    for signal in signals:
        key = _team_total_exact_key(signal)
        if key is None:
            continue
        current = oldest_by_key.get(key)
        if current is None or _signal_time(signal) < _signal_time(current):
            oldest_by_key[key] = signal

    if not oldest_by_key:
        return list(signals)

    collapsed_team_totals = sorted(
        oldest_by_key.values(),
        key=lambda signal: (
            _signal_time(signal),
            int(signal.get("fixture_id") or 0),
            clv_postgres_v4._norm((signal.get("market_candidate") or {}).get("market")),
            clv_postgres_v4._selection_side((signal.get("market_candidate") or {}).get("selection")),
            float(
                clv_postgres_v4._num((signal.get("market_candidate") or {}).get("line"))
                or clv_postgres_v4._line_from_selection((signal.get("market_candidate") or {}).get("selection"))
                or 0.0
            ),
        ),
    )

    output: list[dict[str, Any]] = []
    inserted = False
    for signal in signals:
        if _team_total_exact_key(signal) is not None:
            if not inserted:
                output.extend(collapsed_team_totals)
                inserted = True
            continue
        output.append(signal)
    return output


def merge_bounded_recent_team_totals(
    raw_signals: list[dict[str, Any]],
    recent_team_totals: list[dict[str, Any]],
    *,
    max_rows: int,
) -> list[dict[str, Any]]:
    """Replace lower-priority raw rows with recent closed Team Totals without increasing the cap."""
    bounded_max = max(1, int(max_rows))
    collapsed_raw = collapse_oldest_exact_team_totals(raw_signals)
    collapsed_recent = collapse_oldest_exact_team_totals(recent_team_totals)
    prioritized = [signal for signal in collapsed_recent if _team_total_exact_key(signal) is not None]
    prioritized_keys = {
        key
        for signal in prioritized
        for key in [_team_total_exact_key(signal)]
        if key is not None
    }
    remainder = [
        signal
        for signal in collapsed_raw
        if _team_total_exact_key(signal) not in prioritized_keys
    ]
    return (prioritized + remainder)[:bounded_max]


def _load_recent_closed_team_totals(
    conn: Any,
    *,
    lookback_hours: int = RECENT_CLOSED_LOOKBACK_HOURS,
    fixture_limit: int = RECENT_FIXTURE_LIMIT,
    event_limit: int = RECENT_EVENT_LIMIT,
) -> list[dict[str, Any]]:
    """Load a small time-based closed-fixture slice without expanding JSON in Postgres."""
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=max(1, int(lookback_hours)))

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                fixture_id,
                league,
                home_team,
                away_team,
                kickoff
            FROM soccer_fixtures
            WHERE kickoff > %s
              AND kickoff <= %s
            ORDER BY kickoff DESC, fixture_id DESC
            LIMIT %s
            """,
            (cutoff, now, max(1, min(int(fixture_limit), 500))),
        )
        fixture_rows = cur.fetchall()

    fixture_meta: dict[int, dict[str, Any]] = {}
    for fixture_id, league, home_team, away_team, kickoff in fixture_rows:
        if fixture_id is None:
            continue
        fixture_meta[int(fixture_id)] = {
            "league": league,
            "home_team": home_team,
            "away_team": away_team,
            "kickoff": kickoff,
        }
    fixture_ids = sorted(fixture_meta)
    if not fixture_ids:
        return []

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                e.fixture_id,
                e.generated_at,
                e.stage,
                e.classification,
                CASE
                    WHEN jsonb_typeof(
                        e.payload -> 'team_totals_intelligence' -> 'observed_exact_market_rows'
                    ) = 'array'
                    THEN e.payload -> 'team_totals_intelligence' -> 'observed_exact_market_rows'
                    ELSE '[]'::jsonb
                END AS observed_exact_market_rows
            FROM soccer_refresh_events e
            JOIN soccer_fixtures f ON f.fixture_id = e.fixture_id
            WHERE e.fixture_id = ANY(%s)
              AND e.generated_at >= %s
              AND e.generated_at < f.kickoff
              AND e.stage = ANY(%s)
              AND e.payload -> 'team_totals_intelligence' IS NOT NULL
            ORDER BY e.generated_at ASC, e.fixture_id ASC
            LIMIT %s
            """,
            (
                fixture_ids,
                cutoff,
                list(clv_postgres_v4.TEAM_TOTALS_RESEARCH_STAGES),
                max(1, min(int(event_limit), 1000)),
            ),
        )
        refresh_rows = cur.fetchall()

    signals: list[dict[str, Any]] = []
    for fixture_id, generated_at, stage, classification, observed_rows in refresh_rows:
        if fixture_id is None:
            continue
        meta = fixture_meta.get(int(fixture_id)) or {}
        rows = observed_rows if isinstance(observed_rows, list) else []
        for candidate in rows:
            if not isinstance(candidate, dict):
                continue
            market = candidate.get("market")
            selection = candidate.get("selection")
            line = clv_postgres_v4._num(candidate.get("line"))
            if line is None:
                line = clv_postgres_v4._line_from_selection(selection)
            price = clv_postgres_v4._num(candidate.get("decimal_price"))
            if price is None:
                price = clv_postgres_v4._num(candidate.get("price"))
            if not market or not selection or line is None or price is None or price <= 1.0:
                continue
            normalized_candidate = dict(candidate)
            normalized_candidate["market_family"] = "TEAM_TOTALS"
            normalized_candidate["line"] = float(line)
            signals.append(
                {
                    "fixture_id": int(fixture_id),
                    "generated_at": generated_at,
                    "stage": stage,
                    "classification": classification,
                    "market_candidate": normalized_candidate,
                    "event_payload": {},
                    "kickoff": meta.get("kickoff"),
                    "league": meta.get("league"),
                    "home_team": meta.get("home_team"),
                    "away_team": meta.get("away_team"),
                    "model_version": None,
                    "automation_version": None,
                    "signal_source": "DERIVATIVE_INTELLIGENCE:team_totals_intelligence",
                    "v216_10_cap_independent_recent_slice": True,
                }
            )
    return collapse_oldest_exact_team_totals(signals)


def _patched_load_derivative_signals(
    conn: Any,
    *,
    lookback_days: int,
    max_rows: int,
) -> list[dict[str, Any]]:
    raw = _ORIGINAL_LOADER(conn, lookback_days=lookback_days, max_rows=max_rows)
    recent = _load_recent_closed_team_totals(conn)
    return merge_bounded_recent_team_totals(raw, recent, max_rows=max_rows)


def install() -> dict[str, Any]:
    global _INSTALLED
    if not _INSTALLED:
        clv_postgres_v4._load_derivative_signals = _patched_load_derivative_signals
        _INSTALLED = True
    return {
        "model_version": MODEL_VERSION,
        "status": PATCH_STATUS,
        "installed": _INSTALLED,
        "scope": ["HOME_TT", "AWAY_TT"],
        "recent_closed_lookback_hours": RECENT_CLOSED_LOOKBACK_HOURS,
        "recent_fixture_limit": RECENT_FIXTURE_LIMIT,
        "recent_event_limit": RECENT_EVENT_LIMIT,
        "provider_requests_added": 0,
        "global_signal_cap_changed": False,
        "raw_loader_max_rows_preserved": True,
        "strict_close_semantics_changed": False,
        "selection_logic_changed": False,
        "historical_rows_mutated": False,
        "decision_weight": 0.0,
        "production_promotion_allowed": False,
        "policy": (
            "INJECT_TIME_BASED_RECENT_CLOSED_TEAM_TOTALS_SIGNAL_SLICE_BEFORE_PHASE17_MERGE; "
            "COLLAPSE_RECYCLED_TIMESTAMPS_TO_OLDEST_EXACT_FIXTURE_MARKET_SIDE_LINE_SIGNAL; "
            "REPLACE_LOWER_PRIORITY_RAW_ROWS_TO_PRESERVE_THE_SAME_MAX_ROWS_CAP; "
            "STRICT_CLOSE_RULES_UNCHANGED"
        ),
    }
