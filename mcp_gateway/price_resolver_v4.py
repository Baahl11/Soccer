from __future__ import annotations

import asyncio
import json
import math
import os
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from mcp_gateway import calibration_v4, one_x_two_multiclass_oos_v4, persistence

MODEL_VERSION = "SOCCER_PRICE_RESOLVER_V4_1.7.0"
API_BASE_URL = os.getenv("API_BASE_URL", "https://v3.football.api-sports.io").rstrip("/")
DEFAULT_MAX_API_CALLS = int(os.getenv("SOCCER_PRICE_RESOLVER_MAX_API_CALLS", "25"))
DEFAULT_TIMEOUT_SECONDS = float(os.getenv("SOCCER_PRICE_RESOLVER_TIMEOUT_SECONDS", "12"))

FRESHNESS_MINUTES = {
    "EARLY_RESEARCH": 180,
    "T-90": 60,
    "T-60": 45,
    "T-40": 30,
    "T-30": 20,
    "T-20": 15,
    "T-10": 10,
    "CLOSE": 5,
}

ELIGIBLE_STATUSES = {"WAIT_PRICE", "WAIT_FRESH_QUOTE", "STALE_QUOTE"}

CALIBRATION_STATE_TTL_SECONDS = int(os.getenv("SOCCER_PRICE_CALIBRATION_STATE_TTL_SECONDS", "900"))
CALIBRATION_STATE_URLS = {
    "binary": "https://raw.githubusercontent.com/Baahl11/Soccer/soccer-edge-state/soccer_edge_state/analysis/oos_stage_diagnostics_v4.json",
    "multiclass_1x2": "https://raw.githubusercontent.com/Baahl11/Soccer/soccer-edge-state/soccer_edge_state/analysis/one_x_two_multiclass_oos_v4.json",
}
_CALIBRATION_STATE_CACHE: dict[str, Any] | None = None
_CALIBRATION_STATE_FETCHED_AT: datetime | None = None



def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _parse_value(value: Any) -> tuple[str, float | None]:
    text = " ".join(str(value or "").strip().split())
    low = text.lower()
    for prefix in ("over ", "under "):
        if low.startswith(prefix):
            try:
                return text.split()[0].title(), float(text.split()[-1])
            except (ValueError, IndexError):
                return text.split()[0].title(), None
    return text, None


def _fair_probs(prices: list[float]) -> list[float | None]:
    implied = [(1.0 / p) if p and p > 1.0 else None for p in prices]
    total = sum(p for p in implied if p is not None)
    if total <= 0:
        return [None for _ in prices]
    return [(p / total) if p is not None else None for p in implied]


def _normalize_market_values(market_name: str, values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for item in values or []:
        if not isinstance(item, dict):
            continue
        raw_selection = item.get("selection")
        if raw_selection is None:
            raw_selection = item.get("value")
        selection, parsed_line = _parse_value(raw_selection)
        explicit_line = _num(item.get("line"))
        if explicit_line is None:
            explicit_line = _num(item.get("handicap"))
        line = explicit_line if explicit_line is not None else parsed_line
        price = _num(item.get("decimal_price"))
        if price is None:
            price = _num(item.get("odd"))
        if price is None:
            price = _num(item.get("price"))
        if price is None or price <= 1.0:
            continue
        parsed.append({
            "selection": selection,
            "line": line,
            "decimal_price": price,
            "fair_probability": _num(item.get("fair_probability")),
        })

    # Recalculate de-vig probabilities from the full observed mutually-exclusive
    # group whenever possible; this also upgrades legacy cached value/odd rows.
    market_low = _norm(market_name)
    if market_low in {"match winner", "both teams score", "both teams to score"}:
        fairs = _fair_probs([float(v["decimal_price"]) for v in parsed])
        for value, fair in zip(parsed, fairs):
            value["fair_probability"] = fair
    elif market_low in {"goals over/under", "over/under"}:
        by_line: dict[float, list[dict[str, Any]]] = defaultdict(list)
        for value in parsed:
            if value.get("line") is not None:
                by_line[float(value["line"])].append(value)
        for group in by_line.values():
            if len(group) < 2:
                continue
            fairs = _fair_probs([float(v["decimal_price"]) for v in group])
            for value, fair in zip(group, fairs):
                value["fair_probability"] = fair
    return parsed


def normalize_api_response(payload: dict[str, Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for response_row in payload.get("response") or []:
        fixture = response_row.get("fixture") or {}
        fixture_id = fixture.get("id")
        update = response_row.get("update")
        for bookmaker in response_row.get("bookmakers") or []:
            bookmaker_id = bookmaker.get("id")
            bookmaker_name = bookmaker.get("name")
            for bet in bookmaker.get("bets") or []:
                market_id = bet.get("id")
                market_name = str(bet.get("name") or "")
                parsed = _normalize_market_values(market_name, bet.get("values") or [])
                if parsed:
                    normalized.append({
                        "fixture_id": fixture_id,
                        "bookmaker_id": bookmaker_id,
                        "bookmaker": bookmaker_name,
                        "market_id": market_id,
                        "market": market_name,
                        "values": parsed,
                        "provider_update": update,
                        "source": "API_FOOTBALL_ODDS_V3",
                    })
    return normalized


def _market_kind(market: str) -> str | None:
    name = _norm(market)
    # Never let period/team derivative markets masquerade as full-time families.
    if any(token in name for token in ("first half", "1st half", "second half", "2nd half", "home team", "away team")):
        return None
    if name == "match winner":
        return "1X2"
    if name in {"both teams score", "both teams to score"}:
        return "BTTS"
    if name in {"goals over/under", "over/under"}:
        return "FT_TOTALS"
    return None


def _event_projection(event: dict[str, Any]) -> dict[str, Any]:
    raw = event.get("raw_projection")
    return raw if isinstance(raw, dict) else {}


def _desired_offer(row: dict[str, Any], event: dict[str, Any]) -> tuple[str | None, str | None, float | None, float | None]:
    family = str(row.get("market_family") or "").upper()
    selection = _norm(row.get("selection"))
    raw = _event_projection(event)

    if family in {"FT_TOTALS_RESEARCH", "FT_TOTALS", "TOTAL"}:
        side = "Over" if "over" in selection else "Under" if "under" in selection else None
        if side is None:
            return None, None, None, None
        p_over = _num(raw.get("raw_over_2_5_prob"))
        p_raw = p_over if side == "Over" else (1.0 - p_over if p_over is not None else None)
        return "FT_TOTALS", side, 2.5, p_raw

    if family in {"FT_BTTS_RESEARCH", "BTTS", "FT_BTTS"}:
        p_raw = _num(raw.get("raw_btts_yes_prob"))
        return "BTTS", "Yes", None, p_raw

    if family in {"FT_1X2_RESEARCH", "1X2", "FT_1X2"}:
        home = _num(raw.get("raw_home_win_prob"))
        draw = _num(raw.get("raw_draw_prob"))
        away = _num(raw.get("raw_away_win_prob"))
        exact = {"home": ("Home", home), "draw": ("Draw", draw), "x": ("Draw", draw), "away": ("Away", away)}
        if selection in exact and exact[selection][1] is not None:
            chosen, probability = exact[selection]
            return "1X2", chosen, None, probability
        candidates = [("Home", home), ("Draw", draw), ("Away", away)]
        usable = [(side, probability) for side, probability in candidates if probability is not None]
        if not usable:
            return None, None, None, None
        side, probability = max(usable, key=lambda item: float(item[1]))
        return "1X2", side, None, probability

    return None, None, None, None




async def _load_research_calibration_state(client: httpx.AsyncClient) -> tuple[dict[str, Any], str]:
    global _CALIBRATION_STATE_CACHE, _CALIBRATION_STATE_FETCHED_AT
    now = datetime.now(timezone.utc)
    if (
        isinstance(_CALIBRATION_STATE_CACHE, dict)
        and _CALIBRATION_STATE_FETCHED_AT is not None
        and (now - _CALIBRATION_STATE_FETCHED_AT).total_seconds() < CALIBRATION_STATE_TTL_SECONDS
    ):
        return _CALIBRATION_STATE_CACHE, "MEMORY_CACHE"

    try:
        responses = await asyncio.gather(
            client.get(CALIBRATION_STATE_URLS["binary"]),
            client.get(CALIBRATION_STATE_URLS["multiclass_1x2"]),
        )
        for response in responses:
            response.raise_for_status()
        binary = responses[0].json()
        multiclass = responses[1].json()
        if not isinstance(binary, dict) or not isinstance(multiclass, dict):
            raise ValueError("invalid calibration state payload")
        state = {"binary": binary, "multiclass_1x2": multiclass}
        _CALIBRATION_STATE_CACHE = state
        _CALIBRATION_STATE_FETCHED_AT = now
        return state, "SOCCER_EDGE_STATE_RAW"
    except Exception:
        return {}, "UNAVAILABLE"


def _binary_calibration_diagnostics(
    *,
    target: str,
    calibration_state: dict[str, Any],
    model_version: str | None,
) -> dict[str, Any]:
    report = calibration_state.get("binary") if isinstance(calibration_state.get("binary"), dict) else {}
    source_model_version = str(report.get("current_source_model_version") or "")
    version_matches = source_model_version == str(model_version or "")
    targets = report.get("current_model_deployment_calibrators")
    target_report = targets.get(target) if version_matches and isinstance(targets, dict) and isinstance(targets.get(target), dict) else {}
    discrimination = target_report.get("discrimination") if isinstance(target_report.get("discrimination"), dict) else {}
    calibrator = target_report.get("calibrator") if isinstance(target_report.get("calibrator"), dict) else {}
    lower_95 = _num(discrimination.get("auc_lower_95"))
    return {
        "target": target,
        "source_model_version": source_model_version or None,
        "requested_model_version": model_version,
        "source_model_version_matches": version_matches,
        "rows": int(target_report.get("rows") or 0),
        "positive_count": int(discrimination.get("positive_count") or 0),
        "negative_count": int(discrimination.get("negative_count") or 0),
        "auc": _num(discrimination.get("auc")),
        "auc_lower_95": lower_95,
        "auc_lower_95_gap_to_gate": round(lower_95 - 0.50, 8) if lower_95 is not None else None,
        "brier_delta": _num(target_report.get("brier_delta")),
        "log_loss_delta": _num(target_report.get("log_loss_delta")),
        "eligible_for_phase16_research": target_report.get("eligible_for_phase16_research") is True,
        "calibrator_status": calibrator.get("status"),
    }


def _binary_calibrated_probability(
    raw_probability: float | None,
    *,
    target: str,
    calibration_state: dict[str, Any],
    model_version: str | None,
) -> float | None:
    if raw_probability is None:
        return None
    report = calibration_state.get("binary") if isinstance(calibration_state.get("binary"), dict) else {}
    if str(report.get("current_source_model_version") or "") != str(model_version or ""):
        return None
    targets = report.get("current_model_deployment_calibrators")
    target_report = targets.get(target) if isinstance(targets, dict) and isinstance(targets.get(target), dict) else {}
    if target_report.get("eligible_for_phase16_research") is not True:
        return None
    calibrator = target_report.get("calibrator") if isinstance(target_report.get("calibrator"), dict) else {}
    if calibrator.get("status") != "RESEARCH_CALIBRATOR_FITTED":
        return None
    return calibration_v4.calibrate_probability(raw_probability, calibrator)




def _one_x_two_class_discrimination_diagnostics(
    *,
    calibration_state: dict[str, Any],
    model_version: str | None,
) -> dict[str, dict[str, Any]]:
    target_names = ("home_win", "draw", "away_win")
    report = calibration_state.get("binary") if isinstance(calibration_state.get("binary"), dict) else {}
    version_matches = str(report.get("current_source_model_version") or "") == str(model_version or "")
    targets = report.get("current_model_deployment_calibrators")
    targets = targets if isinstance(targets, dict) else {}

    out: dict[str, dict[str, Any]] = {}
    for target in target_names:
        target_report = targets.get(target) if version_matches and isinstance(targets.get(target), dict) else {}
        discrimination = (
            target_report.get("discrimination")
            if isinstance(target_report.get("discrimination"), dict)
            else {}
        )
        lower_95 = _num(discrimination.get("auc_lower_95"))
        calibrator = target_report.get("calibrator") if isinstance(target_report.get("calibrator"), dict) else {}
        out[target] = {
            "rows": int(target_report.get("rows") or 0),
            "positive_count": int(discrimination.get("positive_count") or 0),
            "negative_count": int(discrimination.get("negative_count") or 0),
            "auc": _num(discrimination.get("auc")),
            "auc_lower_95": lower_95,
            "auc_lower_95_gap_to_gate": round(lower_95 - 0.50, 8) if lower_95 is not None else None,
            "brier_delta": _num(target_report.get("brier_delta")),
            "log_loss_delta": _num(target_report.get("log_loss_delta")),
            "calibrator_status": calibrator.get("status"),
            "ready": target_report.get("eligible_for_phase16_research") is True,
            "source_model_version_matches": version_matches,
        }
    return out


def _one_x_two_class_discrimination_state(
    *,
    calibration_state: dict[str, Any],
    model_version: str | None,
) -> dict[str, bool]:
    diagnostics = _one_x_two_class_discrimination_diagnostics(
        calibration_state=calibration_state,
        model_version=model_version,
    )
    return {target: row.get("ready") is True for target, row in diagnostics.items()}


def _one_x_two_selection_discrimination_ready(
    selection: str,
    *,
    calibration_state: dict[str, Any],
    model_version: str | None,
) -> bool:
    target_map = {"Home": "home_win", "Draw": "draw", "Away": "away_win"}
    target = target_map.get(selection.title())
    if target is None:
        return False
    state = _one_x_two_class_discrimination_state(
        calibration_state=calibration_state,
        model_version=model_version,
    )
    return state.get(target) is True


def _multiclass_calibrated_probabilities(
    event: dict[str, Any],
    *,
    calibration_state: dict[str, Any],
    model_version: str | None,
) -> dict[str, float]:
    raw = _event_projection(event)
    probs = [
        _num(raw.get("raw_home_win_prob")),
        _num(raw.get("raw_draw_prob")),
        _num(raw.get("raw_away_win_prob")),
    ]
    if any(value is None or value < 0 for value in probs):
        return {}
    total = sum(float(value) for value in probs)
    if total <= 0:
        return {}
    normalized = tuple(float(value) / total for value in probs)

    report = calibration_state.get("multiclass_1x2") if isinstance(calibration_state.get("multiclass_1x2"), dict) else {}
    if str(report.get("source_model_version") or "") != str(model_version or ""):
        return {}
    deployment = report.get("research_deployment_calibrator")
    if not isinstance(deployment, dict) or deployment.get("status") != "RESEARCH_DEPLOYMENT_CALIBRATOR_FITTED":
        return {}
    temperature = _num(deployment.get("temperature"))
    if temperature is None or temperature <= 0:
        return {}
    scaled = one_x_two_multiclass_oos_v4.temperature_scale(normalized, temperature)
    return {"Home": scaled[0], "Draw": scaled[1], "Away": scaled[2]}


def _apply_phase16_calibration(
    row: dict[str, Any],
    event: dict[str, Any],
    *,
    family: str,
    selection: str,
    p_raw: float | None,
    calibration_state: dict[str, Any],
    model_version: str | None,
) -> float | None:
    calibrated: float | None = None
    source: str | None = None
    policy: str | None = None
    row["phase16_calibration_promotion_shadow_eligible"] = False

    if family == "BTTS":
        binary_diagnostics = _binary_calibration_diagnostics(
            target="btts",
            calibration_state=calibration_state,
            model_version=model_version,
        )
        row["phase16_binary_calibration_diagnostics"] = binary_diagnostics
        raw_yes = _num(_event_projection(event).get("raw_btts_yes_prob"))
        calibrated_yes = _binary_calibrated_probability(
            raw_yes,
            target="btts",
            calibration_state=calibration_state,
            model_version=model_version,
        )
        if calibrated_yes is not None:
            calibrated = calibrated_yes if _norm(selection) == "yes" else 1.0 - calibrated_yes
            source = "CURRENT_MODEL_OOS_PLATT:BTTS"
            policy = "BINARY_PLATT+BrierLogLossImprovement+AUC_L95_GT_0_50"
        elif binary_diagnostics.get("source_model_version_matches") is True and binary_diagnostics.get("calibrator_status") == "RESEARCH_CALIBRATOR_FITTED" and binary_diagnostics.get("eligible_for_phase16_research") is not True:
            row["phase16_calibration_status"] = "BINARY_DISCRIMINATION_NOT_READY"
            row["phase16_calibration_policy"] = "BINARY_PLATT+BrierLogLossImprovement+AUC_L95_GT_0_50"
            return None

    elif family == "FT_TOTALS":
        binary_diagnostics = _binary_calibration_diagnostics(
            target="over_2_5",
            calibration_state=calibration_state,
            model_version=model_version,
        )
        row["phase16_binary_calibration_diagnostics"] = binary_diagnostics
        raw_over = _num(_event_projection(event).get("raw_over_2_5_prob"))
        calibrated_over = _binary_calibrated_probability(
            raw_over,
            target="over_2_5",
            calibration_state=calibration_state,
            model_version=model_version,
        )
        if calibrated_over is not None:
            calibrated = calibrated_over if _norm(selection) == "over" else 1.0 - calibrated_over
            source = "CURRENT_MODEL_OOS_PLATT:OVER_2_5"
            policy = "BINARY_PLATT+BrierLogLossImprovement+AUC_L95_GT_0_50"
        elif binary_diagnostics.get("source_model_version_matches") is True and binary_diagnostics.get("calibrator_status") == "RESEARCH_CALIBRATOR_FITTED" and binary_diagnostics.get("eligible_for_phase16_research") is not True:
            row["phase16_calibration_status"] = "BINARY_DISCRIMINATION_NOT_READY"
            row["phase16_calibration_policy"] = "BINARY_PLATT+BrierLogLossImprovement+AUC_L95_GT_0_50"
            return None

    elif family == "1X2":
        class_diagnostics = _one_x_two_class_discrimination_diagnostics(
            calibration_state=calibration_state,
            model_version=model_version,
        )
        class_state = {target: item.get("ready") is True for target, item in class_diagnostics.items()}
        not_ready_classes = sorted(
            target.upper() for target, ready in class_state.items() if not ready
        )
        row["phase16_1x2_class_discrimination_ready"] = dict(class_state)
        row["phase16_1x2_class_discrimination_diagnostics"] = class_diagnostics
        row["phase16_1x2_family_discrimination_ready"] = not not_ready_classes
        row["phase16_1x2_not_ready_classes"] = not_ready_classes

        if not _one_x_two_selection_discrimination_ready(
            selection,
            calibration_state=calibration_state,
            model_version=model_version,
        ):
            row["phase16_calibration_status"] = "SELECTION_DISCRIMINATION_NOT_READY"
            row["phase16_calibration_policy"] = "MULTICLASS_TEMPERATURE+SELECTION_AUC_L95_GT_0_50"
            return None
        scaled = _multiclass_calibrated_probabilities(
            event,
            calibration_state=calibration_state,
            model_version=model_version,
        )
        calibrated = scaled.get(selection.title())
        if calibrated is not None:
            source = "CURRENT_MODEL_OOS_TEMPERATURE:1X2"
            policy = "MULTICLASS_TEMPERATURE+SELECTION_AUC_L95_GT_0_50"

    if calibrated is None:
        row["phase16_calibration_status"] = "CALIBRATION_NOT_AVAILABLE_FOR_CURRENT_MODEL"
        return None

    row["p_model_calibrated"] = round(float(calibrated), 8)
    row["phase16_calibration_status"] = "RESEARCH_CALIBRATION_APPLIED"
    row["phase16_calibration_source"] = source
    row["phase16_calibration_policy"] = policy
    row["phase16_calibration_promotion_shadow_eligible"] = True
    row["price_resolution_calibrated_probability_added"] = True
    return calibrated


def _selection_matches(value: dict[str, Any], desired_selection: str, desired_line: float | None) -> bool:
    selection = _norm(value.get("selection"))
    wanted = _norm(desired_selection)
    aliases = {
        "home": {"home", "1"},
        "away": {"away", "2"},
        "draw": {"draw", "x"},
        "yes": {"yes"},
        "no": {"no"},
        "over": {"over"},
        "under": {"under"},
    }
    if selection not in aliases.get(wanted, {wanted}):
        return False
    if desired_line is None:
        return True
    line = _num(value.get("line"))
    return line is not None and abs(line - desired_line) < 1e-9


def choose_reference_offer(
    markets: list[dict[str, Any]],
    *,
    family: str,
    selection: str,
    line: float | None,
) -> dict[str, Any] | None:
    offers: list[dict[str, Any]] = []
    for market in markets:
        if _market_kind(str(market.get("market") or "")) != family:
            continue
        for value in market.get("values") or []:
            if not _selection_matches(value, selection, line):
                continue
            price = _num(value.get("decimal_price"))
            fair = _num(value.get("fair_probability"))
            if price is None or price <= 1.0 or fair is None:
                continue
            offers.append({
                "fixture_id": market.get("fixture_id"),
                "bookmaker_id": market.get("bookmaker_id"),
                "bookmaker": market.get("bookmaker"),
                "market_id": market.get("market_id"),
                "market": market.get("market"),
                "selection": selection,
                "line": line,
                "decimal_price": price,
                "fair_probability": fair,
                "provider_update": market.get("provider_update"),
                "source": market.get("source") or "API_FOOTBALL_ODDS_V3",
            })
    if not offers:
        return None

    median_price = statistics.median(float(o["decimal_price"]) for o in offers)
    offers.sort(key=lambda o: (abs(float(o["decimal_price"]) - median_price), str(o.get("bookmaker") or "")))
    chosen = dict(offers[0])
    chosen["bookmaker_count"] = len(offers)
    chosen["reference_policy"] = "MEDIAN_PRICE_NEAREST_BOOKMAKER"
    return chosen


def _freshness_minutes(stage: Any) -> int:
    return int(FRESHNESS_MINUTES.get(str(stage or "").upper(), 30))


def _load_cached_markets(fixture_id: int, stage: Any) -> list[dict[str, Any]]:
    if not persistence.persistence_configured():
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=_freshness_minutes(stage))
    persistence.ensure_schema()
    with persistence._connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT fixture_id, bookmaker_id, bookmaker, market_id, market, values, provider_update
                FROM soccer_market_snapshots
                WHERE fixture_id = %s
                  AND captured_at >= %s
                ORDER BY captured_at DESC, snapshot_id DESC
                """,
                (fixture_id, cutoff),
            )
            rows = cur.fetchall()
            columns = [desc.name for desc in cur.description]
    seen: set[tuple[Any, ...]] = set()
    out: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(zip(columns, raw))
        provider_update = row.get("provider_update")
        if isinstance(provider_update, datetime):
            row["provider_update"] = provider_update.isoformat()
        row["values"] = _normalize_market_values(str(row.get("market") or ""), row.get("values") or [])
        key = (row.get("bookmaker_id"), row.get("market_id"), str(row.get("values")))
        if key in seen:
            continue
        seen.add(key)
        row["source"] = "POSTGRES_MARKET_SNAPSHOT_CACHE"
        out.append(row)
    return out


def _header_int(response: httpx.Response, name: str) -> int | None:
    value = response.headers.get(name)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _apply_quota_accounting(payload: dict[str, Any], calls: int, daily_remaining: int | None) -> None:
    payload["api_calls_this_tick"] = int(payload.get("api_calls_this_tick") or 0) + int(calls or 0)
    if daily_remaining is not None:
        current = payload.get("last_daily_remaining")
        try:
            current_int = int(current) if current is not None else None
        except (TypeError, ValueError):
            current_int = None
        payload["last_daily_remaining"] = daily_remaining if current_int is None else min(current_int, daily_remaining)
        quota = payload.get("quota")
        if isinstance(quota, dict):
            quota["daily_remaining"] = payload["last_daily_remaining"]


async def _fetch_fixture_odds(
    client: httpx.AsyncClient,
    fixture_id: int,
    *,
    api_key: str,
    remaining_calls: int,
) -> tuple[list[dict[str, Any]], int, str, int | None]:
    if remaining_calls <= 0:
        return [], 0, "PRICE_BUDGET_EXHAUSTED", None
    markets: list[dict[str, Any]] = []
    page = 1
    calls = 0
    daily_remaining: int | None = None
    while calls < remaining_calls:
        response = await client.get(
            f"{API_BASE_URL}/odds",
            params={"fixture": fixture_id, "page": page},
            headers={"x-apisports-key": api_key},
        )
        calls += 1
        response.raise_for_status()
        observed_remaining = _header_int(response, "x-ratelimit-requests-remaining")
        if observed_remaining is not None:
            daily_remaining = observed_remaining if daily_remaining is None else min(daily_remaining, observed_remaining)
        payload = response.json()
        markets.extend(normalize_api_response(payload))
        paging = payload.get("paging") or {}
        current = int(paging.get("current") or page)
        total = int(paging.get("total") or current)
        if current >= total:
            break
        page = current + 1
    return markets, calls, "PRICE_API_RESOLVED" if markets else "PRICE_API_NO_FIXTURE_OR_MARKET", daily_remaining


def _attach_market_to_event(event: dict[str, Any], markets: list[dict[str, Any]], source_status: str) -> None:
    if not markets:
        return
    event["market"] = {
        "source": "API_FOOTBALL_ODDS_V3" if source_status == "PRICE_API_RESOLVED" else "POSTGRES_MARKET_SNAPSHOT_CACHE",
        "resolution_status": source_status,
        "markets": markets,
    }


def _enrich_row(
    row: dict[str, Any],
    event: dict[str, Any],
    markets: list[dict[str, Any]],
    source_status: str,
    *,
    calibration_state: dict[str, Any] | None = None,
    model_version: str | None = None,
) -> str:
    family, selection, line, p_raw = _desired_offer(row, event)
    if family is None or selection is None:
        row["price_resolution_status"] = "PRICE_API_NO_EXACT_MARKET_MAPPING"
        return "PRICE_API_NO_EXACT_MARKET_MAPPING"

    offer = choose_reference_offer(markets, family=family, selection=selection, line=line)
    if offer is None:
        family_markets = [market for market in markets if _market_kind(str(market.get("market") or "")) == family]
        available_selection_rows = [
            value
            for market in family_markets
            for value in (market.get("values") or [])
            if _selection_matches(value, selection, None)
        ]
        available_lines = sorted({
            float(value["line"])
            for value in available_selection_rows
            if _num(value.get("line")) is not None
        })
        if not family_markets:
            status = "PRICE_API_NO_MARKET"
        elif not available_selection_rows:
            status = "PRICE_API_NO_SELECTION"
        elif line is not None and available_lines:
            status = "PRICE_API_NO_EXACT_LINE"
        else:
            status = "PRICE_API_NO_MARKET"
        row["price_resolution_status"] = status
        row["price_resolution_family"] = family
        row["price_resolution_selection"] = selection
        row["price_resolution_line"] = line
        row["price_resolution_available_lines"] = available_lines[:20]
        return status

    row["market_family"] = family
    row["market"] = offer.get("market")
    row["selection"] = selection
    row["line"] = line
    row["price"] = round(float(offer["decimal_price"]), 3)
    row["bookmaker"] = offer.get("bookmaker")
    row["p_market_fair"] = round(float(offer["fair_probability"]), 6)
    row["p_raw"] = round(float(p_raw), 6) if p_raw is not None else None
    row["prob_edge_pp"] = round((float(p_raw) - float(offer["fair_probability"])) * 100.0, 4) if p_raw is not None else None
    row["price_resolution_status"] = source_status
    row["price_resolution_source"] = offer.get("source")
    row["price_resolution_provider_update"] = offer.get("provider_update")
    row["price_resolution_bookmaker_count"] = offer.get("bookmaker_count")
    row["price_resolution_reference_policy"] = offer.get("reference_policy")
    row["price_resolution_calibrated_probability_added"] = False
    _apply_phase16_calibration(
        row,
        event,
        family=family,
        selection=selection,
        p_raw=p_raw,
        calibration_state=calibration_state or {},
        model_version=model_version,
    )
    return source_status


async def resolve_payload(
    payload: dict[str, Any],
    *,
    max_api_calls: int | None = None,
    calibration_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = payload.get("match_table_rows") if isinstance(payload.get("match_table_rows"), list) else []
    events = payload.get("events") if isinstance(payload.get("events"), list) else []
    api_key = os.getenv("API_FOOTBALL_KEY", "").strip()
    budget = max(0, int(DEFAULT_MAX_API_CALLS if max_api_calls is None else max_api_calls))
    calls = 0
    provider_daily_remaining: int | None = None
    counts: dict[str, int] = defaultdict(int)
    fixture_cache: dict[int, tuple[list[dict[str, Any]], str]] = {}

    targets = [
        row for row in rows
        if isinstance(row, dict)
        and str(row.get("execution_status") or "").upper() in ELIGIBLE_STATUSES
        and str(row.get("stage") or "").upper() != "POSTGAME"
        and row.get("fixture_id") is not None
    ]

    calibration_source = "INJECTED"
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_SECONDS, follow_redirects=True) as client:
        if calibration_state is None:
            calibration_state, calibration_source = await _load_research_calibration_state(client)
        for row in targets:
            fixture_id = int(row["fixture_id"])
            event_index = row.get("row_index")
            event = events[event_index] if isinstance(event_index, int) and 0 <= event_index < len(events) and isinstance(events[event_index], dict) else {}

            if fixture_id not in fixture_cache:
                cached = await asyncio.to_thread(_load_cached_markets, fixture_id, row.get("stage"))
                if cached:
                    fixture_cache[fixture_id] = (cached, "PRICE_CACHE_HIT")
                elif not api_key:
                    fixture_cache[fixture_id] = ([], "PRICE_API_KEY_MISSING")
                elif calls >= budget:
                    fixture_cache[fixture_id] = ([], "PRICE_BUDGET_EXHAUSTED")
                else:
                    try:
                        markets, used, status, observed_remaining = await _fetch_fixture_odds(
                            client,
                            fixture_id,
                            api_key=api_key,
                            remaining_calls=budget - calls,
                        )
                        calls += used
                        if observed_remaining is not None:
                            provider_daily_remaining = (
                                observed_remaining
                                if provider_daily_remaining is None
                                else min(provider_daily_remaining, observed_remaining)
                            )
                        fixture_cache[fixture_id] = (markets, status)
                    except Exception as exc:
                        fixture_cache[fixture_id] = ([], "PRICE_API_ERROR")
                        row["price_resolution_error"] = str(exc)[:180]

            markets, status = fixture_cache[fixture_id]
            if markets:
                _attach_market_to_event(event, markets, status)
                resolved_status = _enrich_row(
                    row,
                    event,
                    markets,
                    status,
                    calibration_state=calibration_state or {},
                    model_version=str(payload.get("model_version") or ""),
                )
            else:
                row["price_resolution_status"] = status
                resolved_status = status
            counts[resolved_status] += 1

    # Zero-provider-call calibration hydration for rows that already carried a
    # real price + de-vigged fair probability into the research table. These
    # rows do not need price resolution, but Phase16 still needs the current
    # OOS calibration/discrimination policy applied before ranking.
    existing_price_calibration_rows_considered = 0
    existing_price_calibrated_rows_added = 0
    existing_price_calibration_status_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        if not isinstance(row, dict) or str(row.get("stage") or "").upper() == "POSTGAME":
            continue
        if _num(row.get("price")) is None or _num(row.get("p_market_fair")) is None:
            continue
        if _num(row.get("p_model_calibrated")) is not None:
            continue

        event_index = row.get("row_index")
        event = (
            events[event_index]
            if isinstance(event_index, int)
            and 0 <= event_index < len(events)
            and isinstance(events[event_index], dict)
            else {}
        )
        family, selection, _line, p_raw = _desired_offer(row, event)
        if family not in {"1X2", "FT_TOTALS", "BTTS"} or selection is None or p_raw is None:
            continue

        existing_price_calibration_rows_considered += 1
        before = _num(row.get("p_model_calibrated"))
        row.setdefault("p_raw", round(float(p_raw), 6))
        _apply_phase16_calibration(
            row,
            event,
            family=family,
            selection=selection,
            p_raw=p_raw,
            calibration_state=calibration_state or {},
            model_version=str(payload.get("model_version") or ""),
        )
        after = _num(row.get("p_model_calibrated"))
        status = str(row.get("phase16_calibration_status") or "UNSPECIFIED")
        existing_price_calibration_status_counts[status] += 1
        if before is None and after is not None:
            existing_price_calibrated_rows_added += 1
            row["price_resolution_existing_price_calibration_added"] = True
            row["price_resolution_existing_price_calibration_source"] = "EXISTING_REAL_PRICE_AND_FAIR_PROBABILITY"

    # Zero-provider-call research hydration for derivative markets such as Team Totals.
    # Phase16 price targets above remain the only path allowed to spend API budget.
    cache_hydrated_research_fixtures = 0
    cache_hydrated_research_market_rows = 0
    target_fixture_ids = {int(row["fixture_id"]) for row in targets}
    for event in events:
        if not isinstance(event, dict) or str(event.get("stage") or "").upper() == "POSTGAME":
            continue
        if str(event.get("event_type") or "") != "SOCCER_REFRESH":
            continue
        fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
        fixture_id_value = fixture.get("fixture_id")
        if fixture_id_value is None:
            fixture_id_value = event.get("fixture_id")
        try:
            fixture_id = int(fixture_id_value)
        except (TypeError, ValueError):
            continue

        raw = event.get("raw_projection") if isinstance(event.get("raw_projection"), dict) else {}
        if _num(raw.get("raw_home_goal_rate")) is None and _num(raw.get("raw_away_goal_rate")) is None:
            continue

        if fixture_id in fixture_cache:
            markets, status = fixture_cache[fixture_id]
            if markets and not isinstance(event.get("market"), dict):
                _attach_market_to_event(event, markets, status)
            continue

        cached = await asyncio.to_thread(_load_cached_markets, fixture_id, event.get("stage"))
        fixture_cache[fixture_id] = (cached, "PRICE_CACHE_HIT_RESEARCH_ONLY" if cached else "PRICE_CACHE_MISS_RESEARCH_ONLY")
        if not cached:
            continue
        _attach_market_to_event(event, cached, "PRICE_CACHE_HIT_RESEARCH_ONLY")
        cache_hydrated_research_fixtures += 1
        cache_hydrated_research_market_rows += len(cached)

    _apply_quota_accounting(payload, calls, provider_daily_remaining)

    calibrated_rows_added = sum(
        1 for row in targets
        if row.get("price_resolution_calibrated_probability_added") is True
    )

    payload["price_resolution_v4"] = {
        "schema_version": "1.0.0",
        "model_version": MODEL_VERSION,
        "status": "ACTIVE_PRICE_RESOLVER",
        "candidate_rows": len(targets),
        "unique_candidate_fixtures": len({int(row["fixture_id"]) for row in targets}),
        "api_calls_added": calls,
        "max_api_calls": budget,
        "resolution_counts": dict(sorted(counts.items())),
        "provider": "API_FOOTBALL",
        "endpoint": "/odds?fixture=<id>",
        "catalog_policy": "/odds/bookmakers and /odds/bets are metadata catalogs; fixture /odds response is authoritative for prices.",
        "simulated_odds_allowed": False,
        "calibrated_probability_fabricated": False,
        "calibration_state_source": calibration_source,
        "calibration_state_loaded": bool(calibration_state),
        "calibrated_rows_added": calibrated_rows_added,
        "existing_price_calibration_rows_considered": existing_price_calibration_rows_considered,
        "existing_price_calibrated_rows_added": existing_price_calibrated_rows_added,
        "existing_price_calibration_status_counts": dict(sorted(existing_price_calibration_status_counts.items())),
        "existing_price_calibration_provider_requests_added": 0,
        "provider_requests_added": calls,
        "provider_daily_remaining_observed": provider_daily_remaining,
        "quota_accounting_included_in_api_calls_this_tick": True,
        "cache_hydrated_research_fixtures": cache_hydrated_research_fixtures,
        "cache_hydrated_research_market_rows": cache_hydrated_research_market_rows,
        "cache_hydration_provider_requests_added": 0,
        "cache_hydration_policy": "PREGAME_SOCCER_REFRESH_WITH_TEAM_LAMBDAS_FRESH_POSTGRES_SNAPSHOTS_ONLY",
    }
    payload["price_resolution_provider_requests_added"] = calls
    return payload["price_resolution_v4"]
