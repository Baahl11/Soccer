from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import math
import re
from typing import Any, Iterable

from mcp_gateway import persistence as persistence_base
from mcp_gateway import research_derivative_postgres_audit as derivative_audit

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_PLAYER_PROPS_TRUE_CLV_V4_1.0.0"
SIGNAL_STAGES = {"T-40", "T-20", "T-10"}
MIN_TRUE_CLV_ROWS_PER_FAMILY = 50
MIN_TRUE_CLV_FIXTURES_PER_FAMILY = 20

FAMILY_CONFIG = {
    "SHOTS": {"intel_key": "player_shots_intelligence", "rows_key": "players", "mode": "LINES"},
    "SOT": {"intel_key": "player_sot_intelligence", "rows_key": "players", "mode": "LINES"},
    "GOALSCORER_ANYTIME": {"intel_key": "player_goalscorer_intelligence", "rows_key": "players", "mode": "ANYTIME"},
    "ASSISTS": {"intel_key": "player_assists_intelligence", "rows_key": "players", "mode": "ASSISTS"},
    "PLAYER_CARDS": {"intel_key": "player_cards_intelligence", "rows_key": "players", "mode": "CARDS"},
    "GK_SAVES": {"intel_key": "gk_saves_intelligence", "rows_key": "goalkeepers", "mode": "LINES"},
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


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def _price(value: dict[str, Any]) -> float | None:
    for key in ("decimal_price", "price", "odd"):
        parsed = _num(value.get(key))
        if parsed is not None and parsed > 1.0:
            return parsed
    return None


def _line(value: dict[str, Any]) -> float | None:
    for key in ("line", "parsed_line", "handicap"):
        parsed = _num(value.get(key))
        if parsed is not None:
            return parsed
    return derivative_audit.value_line(value)


def _side(selection: Any) -> str:
    text = _norm(selection)
    if re.search(r"\bover\b", text):
        return "OVER"
    if re.search(r"\bunder\b", text):
        return "UNDER"
    if re.search(r"\byes\b", text):
        return "YES"
    if re.search(r"\bno\b", text):
        return "NO"
    return "PLAYER_EVENT"


def _prob_from_model(player: dict[str, Any], *, mode: str, line: float | None, side: str) -> float | None:
    if mode == "LINES":
        if line is None or side not in {"OVER", "UNDER"}:
            return None
        for row in player.get("lines") or []:
            if not isinstance(row, dict):
                continue
            row_line = _num(row.get("line"))
            if row_line is None or abs(row_line - line) > 1e-6:
                continue
            return _num(row.get("p_over" if side == "OVER" else "p_under"))
        return None
    if mode == "ANYTIME":
        return _num(player.get("p_anytime_goal"))
    if mode == "ASSISTS":
        return _num(player.get("p_1plus_assist"))
    if mode == "CARDS":
        return _num(player.get("p_player_booked_yellow"))
    return None


def _event_market_rows(event: dict[str, Any], family: str) -> list[dict[str, Any]]:
    market = event.get("market") if isinstance(event.get("market"), dict) else {}
    out: list[dict[str, Any]] = []
    for row in market.get("research_cards_props_markets") or []:
        if not isinstance(row, dict):
            continue
        row_family = row.get("research_subfamily") or derivative_audit.classify_market(row.get("market"))
        if row_family == family:
            out.append(row)
    return out


def _align_market_values(event: dict[str, Any], market_row: dict[str, Any], family: str) -> list[dict[str, Any]]:
    lineup = event.get("lineups") if isinstance(event.get("lineups"), dict) else None
    values: list[dict[str, Any]] = []
    for value in market_row.get("values") or []:
        if not isinstance(value, dict):
            continue
        if value.get("xi_alignment_status") == "MATCHED_CONFIRMED_XI" and value.get("player_id") is not None:
            values.append(dict(value))
            continue
        aligned = derivative_audit.align_value_to_confirmed_xi(
            value,
            lineup_payload=lineup,
            family=family,
        )
        if aligned.get("xi_alignment_status") == "MATCHED_CONFIRMED_XI":
            values.append(aligned)
    return values


def _entry_market_fair(values: list[dict[str, Any]], target: dict[str, Any]) -> tuple[float | None, str]:
    target_pid = target.get("player_id")
    target_line = _line(target)
    target_side = _side(target.get("selection"))
    pairs: list[tuple[str, float]] = []
    for value in values:
        if str(value.get("player_id")) != str(target_pid):
            continue
        if target_line is not None:
            value_line = _line(value)
            if value_line is None or abs(value_line - target_line) > 1e-6:
                continue
        side = _side(value.get("selection"))
        price = _price(value)
        if price is None:
            continue
        pairs.append((side, 1.0 / price))

    sides = {side for side, _ in pairs}
    if target_side in {"OVER", "UNDER"} and {"OVER", "UNDER"} <= sides:
        total = sum(prob for side, prob in pairs if side in {"OVER", "UNDER"})
        for side, prob in pairs:
            if side == target_side and total > 0:
                return prob / total, "DEVIGGED_TWO_WAY"
    if target_side in {"YES", "NO"} and {"YES", "NO"} <= sides:
        total = sum(prob for side, prob in pairs if side in {"YES", "NO"})
        for side, prob in pairs:
            if side == target_side and total > 0:
                return prob / total, "DEVIGGED_TWO_WAY"
    return None, "ONE_WAY_OR_UNPAIRED"


def extract_shadow_signals(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []
    for row in events:
        if not isinstance(row, dict):
            continue
        event = row.get("event_payload") if isinstance(row.get("event_payload"), dict) else row
        stage = str(row.get("stage") or event.get("stage") or "")
        if stage not in SIGNAL_STAGES:
            continue
        fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
        fixture_id = row.get("fixture_id") or fixture.get("fixture_id")
        signal_at = _dt(row.get("generated_at") or row.get("captured_at") or event.get("generated_at_utc"))
        kickoff = _dt(row.get("kickoff") or fixture.get("kickoff"))
        if fixture_id is None or signal_at is None or kickoff is None or signal_at >= kickoff:
            continue

        for family, config in FAMILY_CONFIG.items():
            intel = event.get(config["intel_key"])
            if not isinstance(intel, dict):
                continue
            market_rows = _event_market_rows(event, family)
            if not market_rows:
                continue
            player_rows = [
                p for p in (intel.get(config["rows_key"]) or [])
                if isinstance(p, dict) and p.get("player_id") is not None and p.get("confirmed_starter") is True
            ]
            if not player_rows:
                continue
            players_by_id = {str(p["player_id"]): p for p in player_rows}

            for market_row in market_rows:
                values = _align_market_values(event, market_row, family)
                for value in values:
                    player = players_by_id.get(str(value.get("player_id")))
                    if player is None:
                        continue
                    price = _price(value)
                    if price is None:
                        continue
                    line = _line(value)
                    side = _side(value.get("selection"))
                    model_prob = _prob_from_model(player, mode=config["mode"], line=line, side=side)
                    if model_prob is None or not 0.0 < model_prob < 1.0:
                        continue
                    market_fair, fair_basis = _entry_market_fair(values, value)
                    signals.append({
                        "schema_version": SCHEMA_VERSION,
                        "fixture_id": int(fixture_id),
                        "kickoff": kickoff,
                        "signal_timestamp": signal_at,
                        "stage": stage,
                        "market_family": family,
                        "market": market_row.get("market"),
                        "market_id": market_row.get("market_id"),
                        "bookmaker_id": market_row.get("bookmaker_id"),
                        "bookmaker": market_row.get("bookmaker"),
                        "provider_update": _dt(market_row.get("provider_update")),
                        "player_id": value.get("player_id"),
                        "player_name": value.get("player_name") or player.get("player"),
                        "team_id": value.get("team_id") or player.get("team_id"),
                        "selection": value.get("selection"),
                        "side": side,
                        "line": line,
                        "entry_price": price,
                        "entry_raw_implied_probability": 1.0 / price,
                        "entry_market_fair_probability": market_fair,
                        "entry_market_fair_basis": fair_basis,
                        "model_probability": model_prob,
                        "expected_minutes": (
                            player.get("expected_minutes_if_confirmed_starter")
                            if player.get("expected_minutes_if_confirmed_starter") is not None
                            else player.get("expected_minutes")
                        ),
                        "signal_source": "PHASE15_XI_ALIGNED_SHADOW_MODEL",
                        "decision_weight": 0.0,
                        "production_promotion_allowed": False,
                    })
    return signals


def _snapshot_values(row: dict[str, Any], family: str) -> list[dict[str, Any]]:
    payload = row.get("confirmed_lineup_payload")
    values = row.get("values") if isinstance(row.get("values"), list) else []
    out: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, dict):
            continue
        if value.get("xi_alignment_status") == "MATCHED_CONFIRMED_XI" and value.get("player_id") is not None:
            out.append(dict(value))
            continue
        aligned = derivative_audit.align_value_to_confirmed_xi(
            value,
            lineup_payload=payload,
            family=family,
        )
        if aligned.get("xi_alignment_status") == "MATCHED_CONFIRMED_XI":
            out.append(aligned)
    return out


def _selection_close(values: list[dict[str, Any]], signal: dict[str, Any]) -> tuple[float | None, float | None, str]:
    target_pid = str(signal.get("player_id"))
    target_side = signal.get("side")
    target_line = signal.get("line")
    candidates: list[dict[str, Any]] = []
    for value in values:
        if str(value.get("player_id")) != target_pid:
            continue
        if _side(value.get("selection")) != target_side:
            continue
        value_line = _line(value)
        if target_line is None:
            if value_line is not None:
                continue
        elif value_line is None or abs(value_line - float(target_line)) > 1e-6:
            continue
        if _price(value) is not None:
            candidates.append(value)
    if len(candidates) != 1:
        return None, None, "SELECTION_NOT_UNIQUE"

    selected = candidates[0]
    price = _price(selected)
    fair, basis = _entry_market_fair(values, selected)
    return price, fair, basis


def pair_signals_to_closes(
    signals: Iterable[dict[str, Any]],
    snapshots: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_fixture: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for snapshot in snapshots:
        if not isinstance(snapshot, dict) or snapshot.get("fixture_id") is None:
            continue
        try:
            by_fixture[int(snapshot["fixture_id"])].append(snapshot)
        except (TypeError, ValueError):
            continue

    tracked: list[dict[str, Any]] = []
    skip = Counter()

    for signal in signals:
        fixture_id = int(signal["fixture_id"])
        signal_at = _dt(signal.get("signal_timestamp"))
        kickoff = _dt(signal.get("kickoff"))
        if signal_at is None or kickoff is None:
            skip["INVALID_SIGNAL_TIME"] += 1
            continue

        family = str(signal.get("market_family") or "")
        candidates: list[tuple[datetime, int, dict[str, Any], float, float | None, str]] = []
        for snapshot in by_fixture.get(fixture_id, []):
            snapshot_family = derivative_audit.classify_market(snapshot.get("market"))
            if snapshot_family != family:
                continue
            captured_at = _dt(snapshot.get("captured_at"))
            provider_update = _dt(snapshot.get("provider_update"))
            if captured_at is None or provider_update is None:
                continue
            if captured_at <= signal_at or captured_at >= kickoff:
                continue
            if provider_update <= signal_at or provider_update >= kickoff:
                continue

            values = _snapshot_values(snapshot, family)
            close_price, close_fair, close_basis = _selection_close(values, signal)
            if close_price is None:
                continue
            same_book = int(
                str(snapshot.get("bookmaker_id") or snapshot.get("bookmaker"))
                == str(signal.get("bookmaker_id") or signal.get("bookmaker"))
            )
            candidates.append((captured_at, same_book, snapshot, close_price, close_fair, close_basis))

        if not candidates:
            skip["NO_LATER_STRICT_PLAYER_PROP_CLOSE"] += 1
            continue

        same_book_candidates = [item for item in candidates if item[1] == 1]
        pool = same_book_candidates or candidates
        close_at, same_book, snapshot, close_price, close_fair, close_basis = max(
            pool, key=lambda item: item[0]
        )

        entry_price = float(signal["entry_price"])
        entry_fair = _num(signal.get("entry_market_fair_probability"))
        exact_devig_comparable = (
            entry_fair is not None
            and close_fair is not None
            and signal.get("entry_market_fair_basis") == "DEVIGGED_TWO_WAY"
            and close_basis == "DEVIGGED_TWO_WAY"
        )
        raw_close_implied = 1.0 / close_price
        raw_entry_implied = 1.0 / entry_price
        probability_clv = (
            round((close_fair - entry_fair) * 100.0, 6)
            if exact_devig_comparable else None
        )
        raw_implied_clv = round((raw_close_implied - raw_entry_implied) * 100.0, 6)
        price_clv = round((entry_price / close_price - 1.0) * 100.0, 6)

        tracked.append({
            **signal,
            "signal_timestamp": signal_at.isoformat(),
            "kickoff": kickoff.isoformat(),
            "closing_timestamp": close_at.isoformat(),
            "closing_provider_update": _dt(snapshot.get("provider_update")).isoformat(),
            "closing_bookmaker_id": snapshot.get("bookmaker_id"),
            "closing_bookmaker": snapshot.get("bookmaker"),
            "closing_price": round(close_price, 6),
            "closing_market_fair_probability": round(close_fair, 8) if close_fair is not None else None,
            "closing_market_fair_basis": close_basis,
            "clv_probability_pp": probability_clv,
            "raw_implied_probability_clv_pp": raw_implied_clv,
            "price_clv_pct": price_clv,
            "is_true_closing_line": True,
            "strict_later_provider_update": True,
            "probability_comparable_same_line": exact_devig_comparable,
            "price_comparable_same_instrument": True,
            "same_book_preferred": bool(same_book),
            "closing_line_status": (
                "TRUE_PREKICKOFF_PLAYER_PROP_CLOSE_DEVIGGED"
                if exact_devig_comparable
                else "TRUE_PREKICKOFF_PLAYER_PROP_CLOSE_PRICE_ONLY"
            ),
            "decision_weight": 0.0,
            "production_promotion_allowed": False,
        })

    return tracked, dict(skip)


def summarize_tracking(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    items = [row for row in rows if isinstance(row, dict)]
    families: dict[str, Any] = {}
    for family in FAMILY_CONFIG:
        family_rows = [row for row in items if row.get("market_family") == family]
        true_rows = [row for row in family_rows if row.get("is_true_closing_line") is True]
        devig_rows = [row for row in true_rows if row.get("probability_comparable_same_line") is True]
        fixtures = {int(row["fixture_id"]) for row in true_rows if row.get("fixture_id") is not None}
        player_fixtures = {
            (int(row["fixture_id"]), str(row.get("player_id")))
            for row in true_rows
            if row.get("fixture_id") is not None and row.get("player_id") is not None
        }
        families[family] = {
            "tracked_rows": len(family_rows),
            "true_clv_rows": len(true_rows),
            "devigged_probability_clv_rows": len(devig_rows),
            "price_only_true_clv_rows": len(true_rows) - len(devig_rows),
            "unique_fixtures": len(fixtures),
            "unique_player_fixtures": len(player_fixtures),
            "minimum_rows_for_review": MIN_TRUE_CLV_ROWS_PER_FAMILY,
            "minimum_unique_fixtures_for_review": MIN_TRUE_CLV_FIXTURES_PER_FAMILY,
            "row_target_met": len(true_rows) >= MIN_TRUE_CLV_ROWS_PER_FAMILY,
            "fixture_diversity_target_met": len(fixtures) >= MIN_TRUE_CLV_FIXTURES_PER_FAMILY,
            "review_ready": (
                len(true_rows) >= MIN_TRUE_CLV_ROWS_PER_FAMILY
                and len(fixtures) >= MIN_TRUE_CLV_FIXTURES_PER_FAMILY
            ),
        }
    return {
        "families": families,
        "true_clv_rows": sum(v["true_clv_rows"] for v in families.values()),
        "unique_fixtures": len({
            int(row["fixture_id"])
            for row in items
            if row.get("fixture_id") is not None and row.get("is_true_closing_line") is True
        }),
    }


def _load_events(conn, *, lookback_days: int, max_rows: int) -> list[dict[str, Any]]:
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
            ORDER BY e.generated_at DESC
            LIMIT %s
            """,
            (cutoff, max_rows),
        )
        columns = [desc.name for desc in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def _load_snapshots(conn, fixture_ids: list[int], *, lookback_days: int, max_rows: int) -> list[dict[str, Any]]:
    if not fixture_ids:
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                m.fixture_id, m.captured_at, m.stage, m.bookmaker_id, m.bookmaker,
                m.market_id, m.market, m.values, m.provider_update, f.kickoff,
                confirmed_lineup.payload AS confirmed_lineup_payload
            FROM soccer_market_snapshots m
            JOIN soccer_fixtures f ON f.fixture_id = m.fixture_id
            LEFT JOIN LATERAL (
                SELECT l.payload
                FROM soccer_lineup_snapshots l
                WHERE l.fixture_id = m.fixture_id
                  AND l.captured_at <= m.captured_at
                  AND l.both_xi_confirmed IS TRUE
                ORDER BY l.captured_at DESC
                LIMIT 1
            ) confirmed_lineup ON TRUE
            WHERE m.fixture_id = ANY(%s)
              AND m.captured_at >= %s
              AND m.captured_at < f.kickoff
              AND (
                LOWER(COALESCE(m.market,'')) LIKE '%%player%%'
                OR LOWER(COALESCE(m.market,'')) LIKE '%%scorer%%'
                OR LOWER(COALESCE(m.market,'')) LIKE '%%goalkeeper save%%'
                OR LOWER(COALESCE(m.market,'')) LIKE '%%keeper save%%'
              )
            ORDER BY m.fixture_id, m.captured_at
            LIMIT %s
            """,
            (fixture_ids, cutoff, max_rows),
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
            "provider_requests_added": 0,
            "production_promotion_allowed": False,
            "families": {},
            "rows": [],
        }

    persistence_base.ensure_schema()
    with persistence_base._connect() as conn:
        events = _load_events(conn, lookback_days=lookback_days, max_rows=max_rows)
        signals = extract_shadow_signals(events)
        fixture_ids = sorted({int(row["fixture_id"]) for row in signals})
        snapshots = _load_snapshots(
            conn,
            fixture_ids,
            lookback_days=lookback_days,
            max_rows=max_rows,
        )

    rows, skip_reasons = pair_signals_to_closes(signals, snapshots)
    summary = summarize_tracking(rows)
    all_ready = bool(summary["families"]) and all(
        item.get("review_ready") is True for item in summary["families"].values()
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": "PLAYER_PROP_TRUE_CLV_REVIEW_READY" if all_ready else "COLLECTING_PLAYER_PROP_TRUE_CLV",
        "lookback_days": lookback_days,
        "signal_rows": len(signals),
        "snapshot_rows": len(snapshots),
        "skip_reasons": skip_reasons,
        **summary,
        "rows": rows,
        "provider_requests_added": 0,
        "decision_weight": 0.0,
        "production_promotion_allowed": False,
        "policy": (
            "XI_ALIGNED_SHADOW_MODEL_SIGNAL -> EXACT PLAYER/FAMILY/LINE/SIDE PRICE -> "
            "STRICTLY_LATER PREKICKOFF PROVIDER UPDATE; SAME BOOK PREFERRED; "
            "ONE-WAY MARKETS TRACK PRICE CLV WITHOUT PRETENDING TO BE DEVIGGED; "
            "50 ROWS AND 20 UNIQUE FIXTURES PER FAMILY ARE RESEARCH REVIEW TARGETS ONLY; "
            "NO PRODUCTION PROMOTION"
        ),
    }
