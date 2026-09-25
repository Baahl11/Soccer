from __future__ import annotations

from collections import Counter, defaultdict
import gc
from datetime import datetime, timedelta, timezone
import math
import re
from typing import Any, Iterable

from mcp_gateway import persistence as persistence_base
from mcp_gateway import research_derivative_postgres_audit as derivative_audit

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_PLAYER_PROPS_TRUE_CLV_V4_1.2.1"
SIGNAL_STAGES = {"T-40", "T-30", "T-20", "T-10"}
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
    # API-Football "Player - N" means N+ events and is therefore an Over
    # threshold once converted to the equivalent N-0.5 line.
    if re.match(r"^.+?\s+-\s+\d+\s*$", str(selection or "").strip()):
        return "OVER"
    return "PLAYER_EVENT"


def _probability_reconciliation_reason(
    player: dict[str, Any],
    *,
    mode: str,
    line: float | None,
    side: str,
) -> tuple[str, dict[str, Any]]:
    status = str(player.get("status") or "UNKNOWN")
    detail: dict[str, Any] = {
        "player_status": status,
        "line": line,
        "side": side,
    }

    if mode == "LINES":
        if line is None:
            return "MARKET_LINE_MISSING", detail
        if side not in {"OVER", "UNDER"}:
            return "MARKET_SIDE_UNSUPPORTED", detail
        lines = [
            row for row in (player.get("lines") or [])
            if isinstance(row, dict)
        ]
        available_lines = [
            _num(row.get("line"))
            for row in lines
            if _num(row.get("line")) is not None
        ]
        detail["available_lines"] = available_lines
        if not lines:
            return "PLAYER_MODEL_LINES_MISSING", detail
        matched = next(
            (
                row for row in lines
                if _num(row.get("line")) is not None
                and abs(float(_num(row.get("line"))) - float(line)) <= 1e-6
            ),
            None,
        )
        if matched is None:
            return "MODEL_LINE_NOT_AVAILABLE", detail
        key = "p_over" if side == "OVER" else "p_under"
        probability = _num(matched.get(key))
        detail["probability_key"] = key
        detail["probability"] = probability
        if probability is None:
            return "MODEL_PROBABILITY_KEY_MISSING", detail
        if not 0.0 < probability < 1.0:
            return "MODEL_PROBABILITY_OUT_OF_RANGE", detail
        return "OK", detail

    key_by_mode = {
        "ANYTIME": "p_anytime_goal",
        "ASSISTS": "p_1plus_assist",
        "CARDS": "p_player_booked_yellow",
    }
    key = key_by_mode.get(mode)
    if key is None:
        return "UNKNOWN_MODEL_MODE", detail
    probability = _num(player.get(key))
    detail["probability_key"] = key
    detail["probability"] = probability
    if probability is None:
        return "MODEL_PROBABILITY_KEY_MISSING", detail
    if not 0.0 < probability < 1.0:
        return "MODEL_PROBABILITY_OUT_OF_RANGE", detail
    return "OK", detail


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


def extract_shadow_signals(
    events: Iterable[dict[str, Any]],
    diagnostics: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []
    diag = diagnostics if isinstance(diagnostics, dict) else {}
    families_diag = diag.setdefault("families", {})
    diag.setdefault("eligible_event_rows", 0)
    diag.setdefault("family_intelligence_hits", 0)
    diag.setdefault("family_market_overlap_hits", 0)
    diag.setdefault("modelable_player_rows", 0)
    diag.setdefault("aligned_market_values", 0)
    diag.setdefault("player_id_overlap_values", 0)
    diag.setdefault("priced_overlap_values", 0)
    diag.setdefault("model_probability_values", 0)
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

        diag["eligible_event_rows"] += 1
        for family, config in FAMILY_CONFIG.items():
            family_diag = families_diag.setdefault(family, {
                "intelligence_event_rows": 0,
                "market_overlap_event_rows": 0,
                "modelable_player_rows": 0,
                "market_rows": 0,
                "raw_market_values": 0,
                "aligned_market_values": 0,
                "player_id_overlap_values": 0,
                "priced_overlap_values": 0,
                "model_probability_values": 0,
                "signal_rows": 0,
                "probability_failure_reasons": {},
                "matched_player_statuses": {},
                "probability_failure_samples": [],
            })
            intel = event.get(config["intel_key"])
            if not isinstance(intel, dict):
                continue
            family_diag["intelligence_event_rows"] += 1
            diag["family_intelligence_hits"] += 1
            market_rows = _event_market_rows(event, family)
            if not market_rows:
                continue
            family_diag["market_overlap_event_rows"] += 1
            family_diag["market_rows"] += len(market_rows)
            family_diag["raw_market_values"] += sum(
                len(row.get("values") or [])
                for row in market_rows
                if isinstance(row, dict)
            )
            diag["family_market_overlap_hits"] += 1
            player_rows = [
                p for p in (intel.get(config["rows_key"]) or [])
                if isinstance(p, dict)
                and p.get("player_id") is not None
                and (
                    family == "GK_SAVES"
                    or p.get("confirmed_starter") is True
                )
            ]
            if not player_rows:
                continue
            family_diag["modelable_player_rows"] += len(player_rows)
            diag["modelable_player_rows"] += len(player_rows)
            players_by_id = {str(p["player_id"]): p for p in player_rows}

            for market_row in market_rows:
                values = _align_market_values(event, market_row, family)
                family_diag["aligned_market_values"] += len(values)
                diag["aligned_market_values"] += len(values)
                for value in values:
                    player = players_by_id.get(str(value.get("player_id")))
                    if player is None:
                        continue
                    family_diag["player_id_overlap_values"] += 1
                    diag["player_id_overlap_values"] += 1
                    price = _price(value)
                    if price is None:
                        continue
                    family_diag["priced_overlap_values"] += 1
                    diag["priced_overlap_values"] += 1
                    line = _line(value)
                    side = _side(value.get("selection"))
                    player_status = str(player.get("status") or "UNKNOWN")
                    status_counts = family_diag["matched_player_statuses"]
                    status_counts[player_status] = int(status_counts.get(player_status, 0)) + 1
                    reason, reason_detail = _probability_reconciliation_reason(
                        player,
                        mode=config["mode"],
                        line=line,
                        side=side,
                    )
                    if reason != "OK":
                        reason_counts = family_diag["probability_failure_reasons"]
                        reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
                        samples = family_diag["probability_failure_samples"]
                        if len(samples) < 12:
                            samples.append({
                                "fixture_id": int(fixture_id),
                                "stage": stage,
                                "player_id": value.get("player_id"),
                                "player_name": value.get("player_name") or player.get("player"),
                                "selection": value.get("selection"),
                                "market_line": line,
                                **reason_detail,
                                "reason": reason,
                            })
                        continue
                    model_prob = _prob_from_model(player, mode=config["mode"], line=line, side=side)
                    if model_prob is None or not 0.0 < model_prob < 1.0:
                        continue
                    family_diag["model_probability_values"] += 1
                    diag["model_probability_values"] += 1
                    market_fair, fair_basis = _entry_market_fair(values, value)
                    family_diag["signal_rows"] += 1
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
                        "provider_update": (
                            _dt(market_row.get("provider_update")).isoformat()
                            if _dt(market_row.get("provider_update")) is not None
                            else None
                        ),
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


def _line_key(value: float | None) -> float | None:
    return round(float(value), 6) if value is not None else None


def _build_snapshot_instrument_index(
    snapshots: Iterable[dict[str, Any]],
) -> dict[tuple[int, str, str, str, float | None], list[dict[str, Any]]]:
    index: dict[tuple[int, str, str, str, float | None], list[dict[str, Any]]] = defaultdict(list)
    for snapshot in snapshots:
        if not isinstance(snapshot, dict) or snapshot.get("fixture_id") is None:
            continue
        try:
            fixture_id = int(snapshot["fixture_id"])
        except (TypeError, ValueError):
            continue

        family = derivative_audit.classify_market(snapshot.get("market"))
        if family not in FAMILY_CONFIG:
            continue
        captured_at = _dt(snapshot.get("captured_at"))
        provider_update = _dt(snapshot.get("provider_update"))
        if captured_at is None or provider_update is None:
            continue

        values = _snapshot_values(snapshot, family)
        if not values:
            continue

        for value in values:
            player_id = value.get("player_id")
            price = _price(value)
            if player_id is None or price is None:
                continue
            side = _side(value.get("selection"))
            line = _line(value)
            fair, basis = _entry_market_fair(values, value)
            key = (
                fixture_id,
                family,
                str(player_id),
                side,
                _line_key(line),
            )
            index[key].append({
                "captured_at": captured_at,
                "provider_update": provider_update,
                "bookmaker_id": snapshot.get("bookmaker_id"),
                "bookmaker": snapshot.get("bookmaker"),
                "price": price,
                "fair_probability": fair,
                "fair_basis": basis,
            })

    for rows in index.values():
        rows.sort(key=lambda row: row["captured_at"])
    return index


def pair_signals_to_closes(
    signals: Iterable[dict[str, Any]],
    snapshots: Iterable[dict[str, Any]],
    diagnostics: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    instrument_index = _build_snapshot_instrument_index(snapshots)
    tracked: list[dict[str, Any]] = []
    skip = Counter()
    diag = diagnostics if isinstance(diagnostics, dict) else {}
    diag.setdefault("signals_checked", 0)
    diag.setdefault("exact_instrument_snapshot_rows", 0)
    diag.setdefault("failure_reasons", {})
    diag.setdefault("failure_samples", [])
    diag.setdefault("by_family", {})

    for signal in signals:
        fixture_id = int(signal["fixture_id"])
        signal_at = _dt(signal.get("signal_timestamp"))
        kickoff = _dt(signal.get("kickoff"))
        if signal_at is None or kickoff is None:
            skip["INVALID_SIGNAL_TIME"] += 1
            continue

        family = str(signal.get("market_family") or "")
        key = (
            fixture_id,
            family,
            str(signal.get("player_id")),
            str(signal.get("side") or ""),
            _line_key(_num(signal.get("line"))),
        )
        instrument_rows = instrument_index.get(key, [])
        diag["signals_checked"] += 1
        diag["exact_instrument_snapshot_rows"] += len(instrument_rows)
        family_diag = diag["by_family"].setdefault(family, {
            "signals_checked": 0,
            "exact_instrument_snapshot_rows": 0,
            "strict_close_rows": 0,
            "failure_reasons": {},
        })
        family_diag["signals_checked"] += 1
        family_diag["exact_instrument_snapshot_rows"] += len(instrument_rows)

        later_capture = [
            row for row in instrument_rows
            if row["captured_at"] > signal_at and row["captured_at"] < kickoff
        ]
        later_provider = [
            row for row in later_capture
            if row["provider_update"] > signal_at and row["provider_update"] < kickoff
        ]
        candidates = later_provider

        if not candidates:
            skip["NO_LATER_STRICT_PLAYER_PROP_CLOSE"] += 1
            if not instrument_rows:
                reason = "NO_EXACT_INSTRUMENT_SNAPSHOT"
            elif not later_capture:
                reason = "NO_LATER_CAPTURE_BEFORE_KICKOFF"
            else:
                provider_after_signal = [
                    row for row in later_capture
                    if row["provider_update"] > signal_at
                ]
                if not provider_after_signal:
                    reason = "LATER_CAPTURE_PROVIDER_UPDATE_NOT_NEWER"
                else:
                    reason = "LATER_PROVIDER_UPDATE_NOT_PREKICKOFF"
            diag["failure_reasons"][reason] = int(diag["failure_reasons"].get(reason, 0)) + 1
            family_diag["failure_reasons"][reason] = int(
                family_diag["failure_reasons"].get(reason, 0)
            ) + 1
            if len(diag["failure_samples"]) < 24:
                diag["failure_samples"].append({
                    "fixture_id": fixture_id,
                    "family": family,
                    "player_id": signal.get("player_id"),
                    "stage": signal.get("stage"),
                    "signal_timestamp": signal_at.isoformat(),
                    "kickoff": kickoff.isoformat(),
                    "line": signal.get("line"),
                    "side": signal.get("side"),
                    "instrument_snapshot_rows": len(instrument_rows),
                    "later_capture_rows": len(later_capture),
                    "later_provider_rows": len(later_provider),
                    "latest_captured_at": (
                        max(row["captured_at"] for row in instrument_rows).isoformat()
                        if instrument_rows else None
                    ),
                    "latest_provider_update": (
                        max(row["provider_update"] for row in instrument_rows).isoformat()
                        if instrument_rows else None
                    ),
                    "reason": reason,
                })
            continue

        signal_book = str(signal.get("bookmaker_id") or signal.get("bookmaker"))
        same_book_candidates = [
            row for row in candidates
            if str(row.get("bookmaker_id") or row.get("bookmaker")) == signal_book
        ]
        pool = same_book_candidates or candidates
        close = max(pool, key=lambda row: row["captured_at"])
        family_diag["strict_close_rows"] += 1

        close_at = close["captured_at"]
        close_price = float(close["price"])
        close_fair = _num(close.get("fair_probability"))
        close_basis = str(close.get("fair_basis") or "ONE_WAY_OR_UNPAIRED")
        same_book = bool(same_book_candidates)

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
            "closing_provider_update": close["provider_update"].isoformat(),
            "closing_bookmaker_id": close.get("bookmaker_id"),
            "closing_bookmaker": close.get("bookmaker"),
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
            "same_book_preferred": same_book,
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
            SELECT
                e.fixture_id,
                e.generated_at,
                e.stage,
                jsonb_build_object(
                    'fixture', e.payload->'fixture',
                    'lineups', e.payload->'lineups',
                    'market', e.payload->'market',
                    'player_shots_intelligence', e.payload->'player_shots_intelligence',
                    'player_sot_intelligence', e.payload->'player_sot_intelligence',
                    'player_goalscorer_intelligence', e.payload->'player_goalscorer_intelligence',
                    'player_assists_intelligence', e.payload->'player_assists_intelligence',
                    'player_cards_intelligence', e.payload->'player_cards_intelligence',
                    'gk_saves_intelligence', e.payload->'gk_saves_intelligence'
                ) AS event_payload,
                f.kickoff
            FROM soccer_refresh_events e
            JOIN soccer_fixtures f ON f.fixture_id = e.fixture_id
            WHERE e.generated_at >= %s
              AND e.generated_at < f.kickoff
              AND e.stage IN ('T-40','T-30','T-20','T-10')
              AND e.payload ? 'market'
              AND (
                    e.payload ? 'player_shots_intelligence'
                 OR e.payload ? 'player_sot_intelligence'
                 OR e.payload ? 'player_goalscorer_intelligence'
                 OR e.payload ? 'player_assists_intelligence'
                 OR e.payload ? 'player_cards_intelligence'
                 OR e.payload ? 'gk_saves_intelligence'
              )
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
        signal_diagnostics: dict[str, Any] = {}
        signals = extract_shadow_signals(events, signal_diagnostics)
        fixture_ids = sorted({int(row["fixture_id"]) for row in signals})
        event_rows_loaded = len(events)
        del events
        gc.collect()
        snapshots = _load_snapshots(
            conn,
            fixture_ids,
            lookback_days=lookback_days,
            max_rows=max_rows,
        )

    close_diagnostics: dict[str, Any] = {}
    rows, skip_reasons = pair_signals_to_closes(
        signals,
        snapshots,
        diagnostics=close_diagnostics,
    )
    summary = summarize_tracking(rows)
    all_ready = bool(summary["families"]) and all(
        item.get("review_ready") is True for item in summary["families"].values()
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": "PLAYER_PROP_TRUE_CLV_REVIEW_READY" if all_ready else "COLLECTING_PLAYER_PROP_TRUE_CLV",
        "lookback_days": lookback_days,
        "event_rows_loaded": event_rows_loaded,
        "signal_rows": len(signals),
        "signal_diagnostics": signal_diagnostics,
        "snapshot_rows": len(snapshots),
        "skip_reasons": skip_reasons,
        "close_diagnostics": close_diagnostics,
        **summary,
        "rows": rows,
        "provider_requests_added": 0,
        "decision_weight": 0.0,
        "production_promotion_allowed": False,
        "policy": (
            "XI_ALIGNED_SHADOW_MODEL_SIGNAL AT T-40/T-30/T-20/T-10 -> EXACT PLAYER/FAMILY/LINE/SIDE PRICE -> "
            "STRICTLY_LATER PREKICKOFF PROVIDER UPDATE; SAME BOOK PREFERRED; "
            "ONE-WAY MARKETS TRACK PRICE CLV WITHOUT PRETENDING TO BE DEVIGGED; "
            "50 ROWS AND 20 UNIQUE FIXTURES PER FAMILY ARE RESEARCH REVIEW TARGETS ONLY; "
            "NO PRODUCTION PROMOTION"
        ),
    }
