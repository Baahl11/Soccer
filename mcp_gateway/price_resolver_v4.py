from __future__ import annotations

import asyncio
import json
import math
import os
import re
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from mcp_gateway import calibration_v4, one_x_two_multiclass_oos_v4, persistence, research_derivative_postgres_audit as derivative_audit

MODEL_VERSION = "SOCCER_PRICE_RESOLVER_V4_1.15.0"
API_BASE_URL = os.getenv("API_BASE_URL", "https://v3.football.api-sports.io").rstrip("/")
DEFAULT_MAX_API_CALLS = int(os.getenv("SOCCER_PRICE_RESOLVER_MAX_API_CALLS", "25"))
DEFAULT_TIMEOUT_SECONDS = float(os.getenv("SOCCER_PRICE_RESOLVER_TIMEOUT_SECONDS", "12"))
TEAM_TOTALS_DIVERSITY_TARGET = max(1, int(os.getenv("SOCCER_TEAM_TOTALS_DIVERSITY_TARGET", "20")))
TEAM_TOTALS_DIVERSITY_LOOKAHEAD_HOURS = max(1, int(os.getenv("SOCCER_TEAM_TOTALS_DIVERSITY_LOOKAHEAD_HOURS", "36")))
TEAM_TOTALS_DIVERSITY_LOOKBACK_DAYS = max(1, int(os.getenv("SOCCER_TEAM_TOTALS_DIVERSITY_LOOKBACK_DAYS", "180")))
TEAM_TOTALS_DIVERSITY_BACKLOG_LIMIT = max(
    TEAM_TOTALS_DIVERSITY_TARGET,
    int(os.getenv("SOCCER_TEAM_TOTALS_DIVERSITY_BACKLOG_LIMIT", "80")),
)
TEAM_TOTALS_MATURATION_LOOKAHEAD_MINUTES = max(
    20,
    int(os.getenv("SOCCER_TEAM_TOTALS_MATURATION_LOOKAHEAD_MINUTES", "55")),
)
TEAM_TOTALS_MATURATION_BACKLOG_LIMIT = max(
    20,
    int(os.getenv("SOCCER_TEAM_TOTALS_MATURATION_BACKLOG_LIMIT", "80")),
)
TEAM_TOTALS_MATURATION_MAX_CALLS_PER_TICK = max(
    1,
    int(os.getenv("SOCCER_TEAM_TOTALS_MATURATION_MAX_CALLS_PER_TICK", "12")),
)
TEAM_TOTALS_SPILLOVER_EVENT_TYPE = "TEAM_TOTALS_RESEARCH_SPILLOVER"
PRIMARY_CLV_MATURATION_EVENT_TYPE = "PRIMARY_CLV_MATURATION_SPILLOVER"
PRIMARY_CLV_MATURATION_LOOKAHEAD_MINUTES = max(
    20,
    int(os.getenv("SOCCER_PRIMARY_CLV_MATURATION_LOOKAHEAD_MINUTES", "55")),
)
PRIMARY_CLV_MATURATION_BACKLOG_LIMIT = max(
    20,
    int(os.getenv("SOCCER_PRIMARY_CLV_MATURATION_BACKLOG_LIMIT", "80")),
)
PRIMARY_CLV_MATURATION_MAX_CALLS_PER_TICK = max(
    1,
    int(os.getenv("SOCCER_PRIMARY_CLV_MATURATION_MAX_CALLS_PER_TICK", "8")),
)

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

TEAM_TOTALS_RESEARCH_STAGES = {
    "EARLY_RESEARCH",
    "T-90",
    "T-60",
    "T-40",
    "T-30",
    "T-20",
    "T-10",
    "CLOSE",
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
        embedded_line = None
        embedded_match = re.search(
            r"\b(?:over|under)\s+([+-]?\d+(?:\.\d+)?)\b",
            str(raw_selection or ""),
            flags=re.IGNORECASE,
        )
        if embedded_match:
            embedded_line = _num(embedded_match.group(1))
        parsed.append({
            "selection": selection,
            "raw_selection": raw_selection,
            "line": line if line is not None else embedded_line,
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


def _is_ft_team_total_market(market: dict[str, Any]) -> bool:
    try:
        market_id = int(market.get("market_id")) if market.get("market_id") is not None else None
    except (TypeError, ValueError):
        market_id = None
    if market_id in {16, 17}:
        return True
    name = _norm(market.get("market"))
    return name in {
        "total - home",
        "total home",
        "total - away",
        "total away",
        "home team total goals",
        "away team total goals",
        "home team goals over/under",
        "away team goals over/under",
    }


def _has_ft_team_total_market(markets: list[dict[str, Any]]) -> bool:
    return any(isinstance(market, dict) and _is_ft_team_total_market(market) for market in markets)


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def _load_team_totals_diversity_backlog(
    *,
    target: int = TEAM_TOTALS_DIVERSITY_TARGET,
    lookahead_hours: int = TEAM_TOTALS_DIVERSITY_LOOKAHEAD_HOURS,
    lookback_days: int = TEAM_TOTALS_DIVERSITY_LOOKBACK_DAYS,
    limit: int = TEAM_TOTALS_DIVERSITY_BACKLOG_LIMIT,
) -> dict[str, Any]:
    """Load upcoming fixtures that already have a pre-kickoff model run but no FT Team Totals evidence.

    This loader is provider-call free. It deliberately reuses persisted model runs so
    diversity spillover spends only /odds requests and never creates extra sporting,
    lineup, injury, or model-input requests.
    """
    empty = {
        "existing_fixture_ids": set(),
        "existing_unique_fixtures": 0,
        "legacy_observed_unique_fixtures": 0,
        "target": int(target),
        "gap": int(target),
        "candidate_events": [],
        "candidate_count": 0,
        "source": "POSTGRES_NOT_CONFIGURED",
    }
    if not persistence.persistence_configured():
        return empty

    persistence.ensure_schema()
    now = datetime.now(timezone.utc)
    lookback_cutoff = now - timedelta(days=max(1, int(lookback_days)))
    lookahead_cutoff = now + timedelta(hours=max(1, int(lookahead_hours)))

    observed_rows_expr = """
        CASE
            WHEN jsonb_typeof(COALESCE(e.payload -> 'team_totals_intelligence' -> 'observed_exact_market_rows', '[]'::jsonb)) = 'array'
            THEN jsonb_array_length(COALESCE(e.payload -> 'team_totals_intelligence' -> 'observed_exact_market_rows', '[]'::jsonb))
            ELSE 0
        END
    """

    with persistence._connect() as conn:
        with conn.cursor() as cur:
            # Diversity is intentionally based only on the explicit strict-capture
            # marker introduced by resolver v1.9+. Historical
            # team_totals_intelligence.observed_exact_market_rows can contain
            # legacy/broader evidence and must never satisfy the 20-fixture gate.
            cur.execute(
                """
                SELECT DISTINCT e.fixture_id
                FROM soccer_refresh_events e
                JOIN soccer_fixtures f ON f.fixture_id = e.fixture_id
                WHERE e.generated_at >= %s
                  AND e.generated_at < f.kickoff
                  AND COALESCE(
                        e.payload -> 'team_totals_diversity_capture' ->> 'qualifies',
                        'false'
                      ) = 'true'
                """,
                (lookback_cutoff,),
            )
            existing_fixture_ids = {
                int(row[0])
                for row in cur.fetchall()
                if row and row[0] is not None
            }

            cur.execute(
                f"""
                SELECT COUNT(DISTINCT e.fixture_id)
                FROM soccer_refresh_events e
                JOIN soccer_fixtures f ON f.fixture_id = e.fixture_id
                WHERE e.generated_at >= %s
                  AND e.generated_at < f.kickoff
                  AND ({observed_rows_expr}) > 0
                """,
                (lookback_cutoff,),
            )
            legacy_row = cur.fetchone()
            legacy_observed_unique_fixtures = int((legacy_row or [0])[0] or 0)

            gap = max(0, int(target) - len(existing_fixture_ids))
            if gap <= 0:
                return {
                    "existing_fixture_ids": existing_fixture_ids,
                    "existing_unique_fixtures": len(existing_fixture_ids),
                    "legacy_observed_unique_fixtures": legacy_observed_unique_fixtures,
                    "target": int(target),
                    "gap": 0,
                    "candidate_events": [],
                    "candidate_count": 0,
                    "source": "POSTGRES_STRICT_TEAM_TOTAL_DIVERSITY_CAPTURE_BACKLOG",
                }

            cur.execute(
                f"""
                WITH latest_model AS (
                    SELECT DISTINCT ON (m.fixture_id)
                        m.fixture_id,
                        m.run_timestamp,
                        m.run_type,
                        m.model_version,
                        m.raw_projection,
                        f.league_id,
                        f.league,
                        f.country,
                        f.season,
                        f.round,
                        f.kickoff,
                        f.status,
                        f.status_long,
                        f.home_team_id,
                        f.home_team,
                        f.away_team_id,
                        f.away_team,
                        f.venue,
                        f.city
                    FROM soccer_model_runs m
                    JOIN soccer_fixtures f ON f.fixture_id = m.fixture_id
                    WHERE m.run_timestamp < f.kickoff
                      AND f.kickoff > %s
                      AND f.kickoff <= %s
                      AND COALESCE(f.status, 'NS') NOT IN ('FT','AET','PEN','CANC','PST','ABD','AWD','WO')
                      AND m.raw_projection IS NOT NULL
                      AND (
                            NULLIF(m.raw_projection ->> 'raw_home_goal_rate', '') IS NOT NULL
                            OR NULLIF(m.raw_projection ->> 'raw_away_goal_rate', '') IS NOT NULL
                      )
                    ORDER BY m.fixture_id, m.run_timestamp DESC
                )
                SELECT lm.*
                FROM latest_model lm
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM soccer_refresh_events e
                    JOIN soccer_fixtures f2 ON f2.fixture_id = e.fixture_id
                    WHERE e.fixture_id = lm.fixture_id
                      AND e.generated_at >= %s
                      AND e.generated_at < f2.kickoff
                      AND COALESCE(
                            e.payload -> 'team_totals_diversity_capture' ->> 'qualifies',
                            'false'
                          ) = 'true'
                )
                ORDER BY lm.kickoff ASC, lm.run_timestamp DESC
                LIMIT %s
                """,
                (now, lookahead_cutoff, lookback_cutoff, max(1, int(limit))),
            )
            rows = cur.fetchall()
            columns = [desc.name for desc in cur.description]

    candidate_events: list[dict[str, Any]] = []
    for raw_row in rows:
        row = dict(zip(columns, raw_row))
        raw_projection = _json_object(row.get("raw_projection"))
        if _num(raw_projection.get("raw_home_goal_rate")) is None and _num(raw_projection.get("raw_away_goal_rate")) is None:
            continue
        stage = str(row.get("run_type") or "EARLY_RESEARCH").upper()
        if stage not in TEAM_TOTALS_RESEARCH_STAGES:
            stage = "EARLY_RESEARCH"
        kickoff = row.get("kickoff")
        run_timestamp = row.get("run_timestamp")
        fixture = {
            "fixture_id": row.get("fixture_id"),
            "league_id": row.get("league_id"),
            "league": row.get("league"),
            "country": row.get("country"),
            "season": row.get("season"),
            "round": row.get("round"),
            "kickoff": kickoff.isoformat() if isinstance(kickoff, datetime) else kickoff,
            "status": row.get("status"),
            "status_long": row.get("status_long"),
            "home_team_id": row.get("home_team_id"),
            "home_team": row.get("home_team"),
            "away_team_id": row.get("away_team_id"),
            "away_team": row.get("away_team"),
            "venue": row.get("venue"),
            "city": row.get("city"),
        }
        candidate_events.append({
            "event_type": TEAM_TOTALS_SPILLOVER_EVENT_TYPE,
            "stage": stage,
            "fixture": fixture,
            "raw_projection": raw_projection,
            "classification": "RESEARCH_ONLY",
            "bet_eligible": False,
            "model_version": row.get("model_version"),
            "research_only": True,
            "decision_weight": 0.0,
            "team_totals_diversity_provenance": {
                "candidate_source": "POSTGRES_UPCOMING_MODELED_FIXTURE",
                "model_run_timestamp": run_timestamp.isoformat() if isinstance(run_timestamp, datetime) else run_timestamp,
                "provider_requests_before_price_resolver": 0,
                "model_recomputed": False,
                "primary_markets_preempted": False,
            },
        })

    return {
        "existing_fixture_ids": existing_fixture_ids,
        "existing_unique_fixtures": len(existing_fixture_ids),
        "legacy_observed_unique_fixtures": legacy_observed_unique_fixtures,
        "target": int(target),
        "gap": max(0, int(target) - len(existing_fixture_ids)),
        "candidate_events": candidate_events,
        "candidate_count": len(candidate_events),
        "source": "POSTGRES_UPCOMING_MODELED_FIXTURE_BACKLOG",
    }


def _maturation_stage(kickoff: Any, now: datetime) -> str:
    kickoff_dt = kickoff if isinstance(kickoff, datetime) else None
    if kickoff_dt is None and kickoff:
        try:
            kickoff_dt = datetime.fromisoformat(str(kickoff).replace("Z", "+00:00"))
        except ValueError:
            kickoff_dt = None
    if kickoff_dt is None:
        return "T-20"
    if kickoff_dt.tzinfo is None:
        kickoff_dt = kickoff_dt.replace(tzinfo=timezone.utc)
    minutes_to = (kickoff_dt.astimezone(timezone.utc) - now).total_seconds() / 60.0
    if minutes_to <= 7.5:
        return "CLOSE"
    if minutes_to <= 15.0:
        return "T-10"
    if minutes_to <= 27.5:
        return "T-20"
    return "T-40"


def _markets_have_provider_update_after(markets: list[dict[str, Any]], signal_generated_at: Any) -> bool:
    if not signal_generated_at:
        return False
    try:
        signal_at = (
            signal_generated_at
            if isinstance(signal_generated_at, datetime)
            else datetime.fromisoformat(str(signal_generated_at).replace("Z", "+00:00"))
        )
    except ValueError:
        return False
    if signal_at.tzinfo is None:
        signal_at = signal_at.replace(tzinfo=timezone.utc)
    signal_at = signal_at.astimezone(timezone.utc)
    for market in markets:
        if not isinstance(market, dict) or not _is_ft_team_total_market(market):
            continue
        raw_update = market.get("provider_update")
        if not raw_update:
            continue
        try:
            update = raw_update if isinstance(raw_update, datetime) else datetime.fromisoformat(str(raw_update).replace("Z", "+00:00"))
        except ValueError:
            continue
        if update.tzinfo is None:
            update = update.replace(tzinfo=timezone.utc)
        if update.astimezone(timezone.utc) > signal_at:
            return True
    return False


def _load_team_totals_maturation_backlog(
    *,
    lookback_days: int = TEAM_TOTALS_DIVERSITY_LOOKBACK_DAYS,
    lookahead_minutes: int = TEAM_TOTALS_MATURATION_LOOKAHEAD_MINUTES,
    limit: int = TEAM_TOTALS_MATURATION_BACKLOG_LIMIT,
) -> dict[str, Any]:
    """Load strict-captured, modeled Team Totals fixtures still missing a later real provider quote.

    Provider-call free. Candidates are bounded to the late pre-kickoff window so
    leftover /odds budget is spent on CLV maturation rather than extra diversity.
    """
    empty = {
        "candidate_events": [],
        "candidate_count": 0,
        "source": "POSTGRES_NOT_CONFIGURED",
    }
    if not persistence.persistence_configured():
        return empty

    persistence.ensure_schema()
    now = datetime.now(timezone.utc)
    lookback_cutoff = now - timedelta(days=max(1, int(lookback_days)))
    lookahead_cutoff = now + timedelta(minutes=max(20, int(lookahead_minutes)))

    with persistence._connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH strict_capture AS (
                    SELECT DISTINCT e.fixture_id
                    FROM soccer_refresh_events e
                    JOIN soccer_fixtures f ON f.fixture_id = e.fixture_id
                    WHERE e.generated_at >= %s
                      AND e.generated_at < f.kickoff
                      AND COALESCE(
                            e.payload -> 'team_totals_diversity_capture' ->> 'qualifies',
                            'false'
                          ) = 'true'
                ),
                modeled_signal AS (
                    SELECT
                        e.fixture_id,
                        MIN(e.generated_at) AS signal_generated_at
                    FROM soccer_refresh_events e
                    JOIN soccer_fixtures f ON f.fixture_id = e.fixture_id
                    WHERE e.generated_at >= %s
                      AND e.generated_at < f.kickoff
                      AND (
                            CASE
                                WHEN jsonb_typeof(
                                    COALESCE(
                                        e.payload -> 'team_totals_intelligence' -> 'observed_exact_market_rows',
                                        '[]'::jsonb
                                    )
                                ) = 'array'
                                THEN jsonb_array_length(
                                    COALESCE(
                                        e.payload -> 'team_totals_intelligence' -> 'observed_exact_market_rows',
                                        '[]'::jsonb
                                    )
                                )
                                ELSE 0
                            END
                          ) > 0
                    GROUP BY e.fixture_id
                )
                SELECT
                    f.fixture_id,
                    f.league_id,
                    f.league,
                    f.country,
                    f.season,
                    f.round,
                    f.kickoff,
                    f.status,
                    f.status_long,
                    f.home_team_id,
                    f.home_team,
                    f.away_team_id,
                    f.away_team,
                    f.venue,
                    f.city,
                    ms.signal_generated_at
                FROM strict_capture sc
                JOIN modeled_signal ms ON ms.fixture_id = sc.fixture_id
                JOIN soccer_fixtures f ON f.fixture_id = sc.fixture_id
                WHERE f.kickoff > %s
                  AND f.kickoff <= %s
                  AND COALESCE(f.status, 'NS') NOT IN ('FT','AET','PEN','CANC','PST','ABD','AWD','WO')
                  AND NOT EXISTS (
                      SELECT 1
                      FROM soccer_market_snapshots m
                      WHERE m.fixture_id = f.fixture_id
                        AND m.captured_at > ms.signal_generated_at
                        AND m.provider_update IS NOT NULL
                        AND m.provider_update > ms.signal_generated_at
                        AND m.captured_at < f.kickoff
                        AND (
                              m.market_id IN (16, 17)
                              OR LOWER(TRIM(COALESCE(m.market, ''))) IN (
                                  'total - home',
                                  'total home',
                                  'total - away',
                                  'total away',
                                  'home team total goals',
                                  'away team total goals',
                                  'home team goals over/under',
                                  'away team goals over/under'
                              )
                            )
                  )
                ORDER BY f.kickoff ASC, ms.signal_generated_at ASC
                LIMIT %s
                """,
                (
                    lookback_cutoff,
                    lookback_cutoff,
                    now,
                    lookahead_cutoff,
                    max(1, int(limit)),
                ),
            )
            rows = cur.fetchall()
            columns = [desc.name for desc in cur.description]

    candidate_events: list[dict[str, Any]] = []
    for raw_row in rows:
        row = dict(zip(columns, raw_row))
        kickoff = row.get("kickoff")
        signal_at = row.get("signal_generated_at")
        fixture = {
            "fixture_id": row.get("fixture_id"),
            "league_id": row.get("league_id"),
            "league": row.get("league"),
            "country": row.get("country"),
            "season": row.get("season"),
            "round": row.get("round"),
            "kickoff": kickoff.isoformat() if isinstance(kickoff, datetime) else kickoff,
            "status": row.get("status"),
            "status_long": row.get("status_long"),
            "home_team_id": row.get("home_team_id"),
            "home_team": row.get("home_team"),
            "away_team_id": row.get("away_team_id"),
            "away_team": row.get("away_team"),
            "venue": row.get("venue"),
            "city": row.get("city"),
        }
        candidate_events.append({
            "event_type": TEAM_TOTALS_SPILLOVER_EVENT_TYPE,
            "stage": _maturation_stage(kickoff, now),
            "fixture": fixture,
            "classification": "RESEARCH_ONLY",
            "bet_eligible": False,
            "research_only": True,
            "decision_weight": 0.0,
            "team_totals_clv_maturation": {
                "candidate_source": "POSTGRES_STRICT_CAPTURE_MODELED_NO_LATER_REAL_QUOTE",
                "signal_generated_at": signal_at.isoformat() if isinstance(signal_at, datetime) else signal_at,
                "provider_requests_before_price_resolver": 0,
                "primary_markets_preempted": False,
                "requires_provider_update_after_signal": True,
            },
        })

    return {
        "candidate_events": candidate_events,
        "candidate_count": len(candidate_events),
        "source": "POSTGRES_TEAM_TOTALS_CLV_MATURATION_BACKLOG",
    }


def _primary_market_matches_signal(market: dict[str, Any], signal: dict[str, Any]) -> bool:
    family = str(signal.get("market_family") or "").upper()
    expected_market = _norm(signal.get("market"))
    if family not in {"1X2", "FT_TOTALS", "BTTS"}:
        return False
    if _market_kind(str(market.get("market") or "")) != family:
        return False
    return not expected_market or _norm(market.get("market")) == expected_market


def _primary_signals_with_later_provider_quote(
    markets: list[dict[str, Any]],
    signals: list[dict[str, Any]],
) -> set[str]:
    matured: set[str] = set()
    for signal in signals:
        family = str(signal.get("market_family") or "").upper()
        raw_signal_at = signal.get("signal_generated_at")
        if family not in {"1X2", "FT_TOTALS", "BTTS"} or not raw_signal_at:
            continue
        try:
            signal_at = (
                raw_signal_at
                if isinstance(raw_signal_at, datetime)
                else datetime.fromisoformat(str(raw_signal_at).replace("Z", "+00:00"))
            )
        except ValueError:
            continue
        if signal_at.tzinfo is None:
            signal_at = signal_at.replace(tzinfo=timezone.utc)
        signal_at = signal_at.astimezone(timezone.utc)
        for market in markets:
            if not isinstance(market, dict) or not _primary_market_matches_signal(market, signal):
                continue
            raw_update = market.get("provider_update")
            if not raw_update:
                continue
            try:
                update = (
                    raw_update
                    if isinstance(raw_update, datetime)
                    else datetime.fromisoformat(str(raw_update).replace("Z", "+00:00"))
                )
            except ValueError:
                continue
            if update.tzinfo is None:
                update = update.replace(tzinfo=timezone.utc)
            if update.astimezone(timezone.utc) > signal_at:
                matured.add(family)
                break
    return matured


def _load_primary_clv_maturation_backlog(
    *,
    lookback_days: int = TEAM_TOTALS_DIVERSITY_LOOKBACK_DAYS,
    lookahead_minutes: int = PRIMARY_CLV_MATURATION_LOOKAHEAD_MINUTES,
    limit: int = PRIMARY_CLV_MATURATION_BACKLOG_LIMIT,
) -> dict[str, Any]:
    """Find upcoming primary-family signals that still lack a strictly later real quote.

    This query is provider-call free. It uses persisted Phase16 market mismatch
    candidates as the entry signal and asks only whether a later provider update
    for the same market exists before kickoff.
    """
    empty = {
        "candidate_events": [],
        "candidate_count": 0,
        "candidate_family_counts": {},
        "source": "POSTGRES_NOT_CONFIGURED",
    }
    if not persistence.persistence_configured():
        return empty

    persistence.ensure_schema()
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=max(1, int(lookback_days)))
    lookahead = now + timedelta(minutes=max(20, int(lookahead_minutes)))

    with persistence._connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH latest_signal AS (
                    SELECT DISTINCT ON (
                        (mm.row ->> 'fixture_id')::BIGINT,
                        UPPER(mm.row ->> 'market_family')
                    )
                        (mm.row ->> 'fixture_id')::BIGINT AS fixture_id,
                        UPPER(mm.row ->> 'market_family') AS market_family,
                        mm.row ->> 'market' AS market,
                        p.generated_at_utc AS signal_generated_at,
                        f.league_id,
                        f.league,
                        f.country,
                        f.season,
                        f.round,
                        f.kickoff,
                        f.status,
                        f.status_long,
                        f.home_team_id,
                        f.home_team,
                        f.away_team_id,
                        f.away_team,
                        f.venue,
                        f.city
                    FROM soccer_pipeline_runs p
                    CROSS JOIN LATERAL jsonb_array_elements(
                        CASE
                            WHEN jsonb_typeof(COALESCE(p.payload -> 'market_mismatch_rows', '[]'::jsonb)) = 'array'
                            THEN COALESCE(p.payload -> 'market_mismatch_rows', '[]'::jsonb)
                            ELSE '[]'::jsonb
                        END
                    ) AS mm(row)
                    JOIN soccer_fixtures f
                      ON f.fixture_id = (mm.row ->> 'fixture_id')::BIGINT
                    WHERE p.generated_at_utc >= %s
                      AND p.generated_at_utc < f.kickoff
                      AND f.kickoff > %s
                      AND f.kickoff <= %s
                      AND COALESCE(f.status, 'NS') NOT IN ('FT','AET','PEN','CANC','PST','ABD','AWD','WO')
                      AND UPPER(mm.row ->> 'market_family') IN ('1X2','FT_TOTALS','BTTS')
                      AND COALESCE((mm.row ->> 'rankable')::boolean, false) = true
                      AND NULLIF(mm.row ->> 'market', '') IS NOT NULL
                      AND NULLIF(mm.row ->> 'price', '') IS NOT NULL
                    ORDER BY
                        (mm.row ->> 'fixture_id')::BIGINT,
                        UPPER(mm.row ->> 'market_family'),
                        p.generated_at_utc DESC
                )
                SELECT ls.*
                FROM latest_signal ls
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM soccer_market_snapshots m
                    WHERE m.fixture_id = ls.fixture_id
                      AND m.captured_at > ls.signal_generated_at
                      AND m.captured_at < ls.kickoff
                      AND m.provider_update IS NOT NULL
                      AND m.provider_update > ls.signal_generated_at
                      AND LOWER(TRIM(COALESCE(m.market, ''))) = LOWER(TRIM(COALESCE(ls.market, '')))
                )
                ORDER BY ls.kickoff ASC, ls.fixture_id ASC, ls.market_family ASC
                LIMIT %s
                """,
                (cutoff, now, lookahead, max(1, int(limit))),
            )
            rows = cur.fetchall()
            columns = [desc.name for desc in cur.description]

    grouped: dict[int, dict[str, Any]] = {}
    family_counts: dict[str, int] = defaultdict(int)
    for raw_row in rows:
        row = dict(zip(columns, raw_row))
        try:
            fixture_id = int(row.get("fixture_id"))
        except (TypeError, ValueError):
            continue
        family = str(row.get("market_family") or "").upper()
        signal_at = row.get("signal_generated_at")
        kickoff = row.get("kickoff")
        family_counts[family] += 1
        record = grouped.setdefault(fixture_id, {
            "fixture": {
                "fixture_id": fixture_id,
                "league_id": row.get("league_id"),
                "league": row.get("league"),
                "country": row.get("country"),
                "season": row.get("season"),
                "round": row.get("round"),
                "kickoff": kickoff.isoformat() if isinstance(kickoff, datetime) else kickoff,
                "status": row.get("status"),
                "status_long": row.get("status_long"),
                "home_team_id": row.get("home_team_id"),
                "home_team": row.get("home_team"),
                "away_team_id": row.get("away_team_id"),
                "away_team": row.get("away_team"),
                "venue": row.get("venue"),
                "city": row.get("city"),
            },
            "kickoff": kickoff,
            "signals": [],
        })
        record["signals"].append({
            "market_family": family,
            "market": row.get("market"),
            "signal_generated_at": signal_at.isoformat() if isinstance(signal_at, datetime) else signal_at,
        })

    events: list[dict[str, Any]] = []
    for fixture_id, record in grouped.items():
        events.append({
            "event_type": PRIMARY_CLV_MATURATION_EVENT_TYPE,
            "stage": _maturation_stage(record.get("kickoff"), now),
            "fixture": record["fixture"],
            "classification": "RESEARCH_ONLY",
            "bet_eligible": False,
            "research_only": True,
            "decision_weight": 0.0,
            "primary_clv_maturation": {
                "candidate_source": "POSTGRES_PHASE16_PRIMARY_SIGNAL_NO_LATER_REAL_QUOTE",
                "signals": list(record["signals"]),
                "provider_requests_before_price_resolver": 0,
                "primary_markets_preempted": False,
                "requires_provider_update_after_signal": True,
            },
        })

    events.sort(key=lambda event: (
        str(((event.get("fixture") or {}).get("kickoff") or "")),
        int(((event.get("fixture") or {}).get("fixture_id") or 0)),
    ))
    return {
        "candidate_events": events,
        "candidate_count": len(events),
        "candidate_family_counts": dict(sorted(family_counts.items())),
        "source": "POSTGRES_PRIMARY_CLV_MATURATION_BACKLOG",
    }


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


def _research_derivative_subfamily(market_name: str) -> str | None:
    name = _norm(market_name)
    aggregate_player_markets = (
        "home player shots total",
        "away player shots total",
        "home player shots on target total",
        "away player shots on target total",
        "player shots total - home",
        "player shots total - away",
        "player shots on target total - home",
        "player shots on target total - away",
    )
    if any(token in name for token in aggregate_player_markets):
        return None
    if "first goal scorer" in name:
        return "GOALSCORER_FIRST"
    if "last goal scorer" in name:
        return "GOALSCORER_LAST"
    if any(token in name for token in ("anytime goal scorer", "anytime goalscorer", "player to score")):
        return "GOALSCORER_ANYTIME"
    if "goal scorer" in name or "goalscorer" in name:
        return "GOALSCORER_OTHER"
    if "shots on target - player" in name or "player shots on target" in name:
        return "SOT"
    if "player shots" in name or "player shot" in name:
        return "SHOTS"
    if "goalkeeper saves" in name or "keeper saves" in name or "gk saves" in name:
        return "GK_SAVES"
    if "player assists" in name or "player assist" in name:
        return "ASSISTS"
    if any(token in name for token in ("player cards", "player card", "player booked", "player booking")):
        return "PLAYER_CARDS"
    return None


def _research_derivative_family(market_name: str) -> str | None:
    name = _norm(market_name)
    if _research_derivative_subfamily(name) is not None:
        return "PLAYER_PROPS"
    card_tokens = (
        "cards over/under",
        "card over/under",
        "total cards",
        "total yellow cards",
        "yellow cards",
        "red card",
        "team cards",
        "booking points",
        "bookings",
        "cards asian handicap",
        "cards european handicap",
        "first card received",
    )
    if name == "rcard" or any(token in name for token in card_tokens):
        return "CARDS"
    return None


def _attach_market_to_event(event: dict[str, Any], markets: list[dict[str, Any]], source_status: str) -> None:
    if not markets:
        return
    canonical: list[dict[str, Any]] = []
    card_rows: list[dict[str, Any]] = []
    prop_rows: list[dict[str, Any]] = []
    for market in markets:
        if not isinstance(market, dict):
            continue
        family = _research_derivative_family(str(market.get("market") or ""))
        if family is None:
            canonical.append(market)
            continue
        subfamily = (
            _research_derivative_subfamily(str(market.get("market") or ""))
            if family == "PLAYER_PROPS" else None
        )
        row = {
            **market,
            "research_only": True,
            "research_family": family,
            "research_subfamily": subfamily,
            "decision_weight": 0.0,
            "production_promotion_allowed": False,
        }
        if family == "PLAYER_PROPS":
            lineup = event.get("lineups") if isinstance(event.get("lineups"), dict) else None
            aligned_values = []
            for value in (market.get("values") or []):
                if not isinstance(value, dict):
                    aligned_values.append(value)
                    continue
                normalized_value = dict(value)
                if (
                    subfamily in {"SHOTS", "SOT", "GK_SAVES"}
                    and derivative_audit.value_line(normalized_value) is not None
                    and normalized_value.get("parsed_line") is None
                    and normalized_value.get("line") is None
                ):
                    normalized_value["parsed_line"] = derivative_audit.value_line(normalized_value)
                    normalized_value["line_basis"] = (
                        "PLAYER_THRESHOLD_N_PLUS"
                        if re.match(
                            r"^.+?\s+-\s+\d+\s*$",
                            str(normalized_value.get("selection") or normalized_value.get("value") or "").strip(),
                        )
                        else "DERIVED_FROM_SELECTION"
                    )
                aligned_values.append(
                    derivative_audit.align_value_to_confirmed_xi(
                        normalized_value,
                        lineup_payload=lineup,
                        family=str(subfamily or ""),
                    )
                )
            row["values"] = aligned_values
            row["confirmed_xi_at_quote"] = bool(
                isinstance(lineup, dict) and lineup.get("both_xi_confirmed") is True
            )
            row["xi_aligned_value_rows"] = sum(
                1 for value in aligned_values
                if isinstance(value, dict)
                and value.get("xi_alignment_status") == "MATCHED_CONFIRMED_XI"
            )
        if family == "CARDS":
            row["bookmaker_scoring_rule_required"] = "booking point" in _norm(market.get("market"))
            card_rows.append(row)
        else:
            prop_rows.append(row)

    kept_cards = card_rows[:20]
    kept_props = prop_rows[:40]
    research_rows = kept_cards + kept_props
    event["market"] = {
        "source": "API_FOOTBALL_ODDS_V3" if source_status == "PRICE_API_RESOLVED" else "POSTGRES_MARKET_SNAPSHOT_CACHE",
        "resolution_status": source_status,
        "markets": canonical,
        "research_cards_props_markets": research_rows,
        "card_research_market_rows": len(kept_cards),
        "player_prop_research_market_rows": len(kept_props),
        "research_derivative_sidecar_rows": len(research_rows),
        "research_derivative_sidecar_provider_requests_added": 0,
        "research_derivative_sidecar_decision_weight": 0.0,
        "research_derivative_sidecar_production_promotion_allowed": False,
        "research_derivative_sidecar_policy": "SPLIT_FROM_ALREADY_PAID_PRICE_RESOLVER_RESPONSE; BOUNDED_20_CARD_40_PLAYER_PROP; XI_ALIGN_WHEN_CONFIRMED; RESEARCH_ONLY",
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

    # Primary-family true-CLV maturation.
    #
    # Primary price targets above always execute first. This pass then uses only
    # the same reserved price budget that remains to obtain a strictly later
    # provider quote for existing 1X2 / FT_TOTALS / BTTS Phase16 signals.
    # Cache replay is intentionally not accepted as new closing evidence.
    primary_maturation = await asyncio.to_thread(_load_primary_clv_maturation_backlog)
    primary_maturation_events = [
        event
        for event in (primary_maturation.get("candidate_events") or [])
        if isinstance(event, dict)
    ]
    primary_maturation_candidates = len(primary_maturation_events)
    primary_maturation_api_calls_added = 0
    primary_maturation_fixtures_refreshed = 0
    primary_maturation_family_refresh_counts: dict[str, int] = defaultdict(int)
    primary_maturation_cache_replays_ignored = 0
    primary_maturation_unchanged_provider_updates = 0
    primary_maturation_budget_exhausted = 0
    primary_maturation_primary_payload_reuse_fixtures = 0
    primary_maturation_synthetic_events_added = 0

    if primary_maturation_events:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_SECONDS, follow_redirects=True) as primary_maturation_client:
            primary_maturation_calls_this_tick = 0
            for event in primary_maturation_events:
                fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
                try:
                    fixture_id = int(fixture.get("fixture_id"))
                except (TypeError, ValueError):
                    continue
                meta = event.get("primary_clv_maturation") if isinstance(event.get("primary_clv_maturation"), dict) else {}
                signals = [row for row in (meta.get("signals") or []) if isinstance(row, dict)]
                if not signals:
                    continue

                markets: list[dict[str, Any]] = []
                status = ""
                provider_calls_for_event = 0
                reused_paid_primary_payload = False
                cached_tuple = fixture_cache.get(fixture_id)
                if cached_tuple is not None:
                    candidate_markets, candidate_status = cached_tuple
                    if str(candidate_status).startswith("PRICE_API") and candidate_markets:
                        matured = _primary_signals_with_later_provider_quote(candidate_markets, signals)
                        if matured:
                            markets = candidate_markets
                            status = str(candidate_status)
                            reused_paid_primary_payload = True
                            primary_maturation_primary_payload_reuse_fixtures += 1
                    elif candidate_markets:
                        primary_maturation_cache_replays_ignored += 1

                if not markets:
                    if not api_key:
                        continue
                    if calls >= budget or primary_maturation_calls_this_tick >= PRIMARY_CLV_MATURATION_MAX_CALLS_PER_TICK:
                        primary_maturation_budget_exhausted += 1
                        continue
                    remaining = min(
                        budget - calls,
                        PRIMARY_CLV_MATURATION_MAX_CALLS_PER_TICK - primary_maturation_calls_this_tick,
                    )
                    try:
                        markets, used, status, observed_remaining = await _fetch_fixture_odds(
                            primary_maturation_client,
                            fixture_id,
                            api_key=api_key,
                            remaining_calls=remaining,
                        )
                        calls += used
                        primary_maturation_calls_this_tick += used
                        primary_maturation_api_calls_added += used
                        provider_calls_for_event = used
                        fixture_cache[fixture_id] = (markets, status)
                        if observed_remaining is not None:
                            provider_daily_remaining = (
                                observed_remaining
                                if provider_daily_remaining is None
                                else min(provider_daily_remaining, observed_remaining)
                            )
                    except Exception as exc:
                        event["primary_clv_maturation_error"] = str(exc)[:180]
                        continue

                matured_families = _primary_signals_with_later_provider_quote(markets, signals)
                if not matured_families:
                    primary_maturation_unchanged_provider_updates += 1
                    continue

                exact_markets = [
                    market
                    for market in markets
                    if isinstance(market, dict)
                    and any(
                        str(signal.get("market_family") or "").upper() in matured_families
                        and _primary_market_matches_signal(market, signal)
                        for signal in signals
                    )
                ]
                if not exact_markets:
                    continue
                event["market"] = {
                    "source": "API_FOOTBALL_ODDS_V3",
                    "resolution_status": status or "PRIMARY_CLV_MATURATION_PROVIDER_REFRESH",
                    "markets": exact_markets,
                }
                event["primary_clv_maturation"]["matured_families"] = sorted(matured_families)
                event["primary_clv_maturation"]["provider_update_after_signal"] = True
                event["primary_clv_maturation"]["provider_requests_added"] = provider_calls_for_event
                event["primary_clv_maturation"]["reused_paid_primary_payload"] = reused_paid_primary_payload
                events.append(event)
                primary_maturation_synthetic_events_added += 1
                primary_maturation_fixtures_refreshed += 1
                for family in matured_families:
                    primary_maturation_family_refresh_counts[family] += 1

    # Derivative-market research hydration (currently Team Totals).
    #
    # Primary Phase16 price targets are always resolved first. Only the provider
    # budget left after that complete primary loop may be used below.
    #
    # Diversity policy:
    #   1) Reuse current due-event fixtures and fresh cache at zero provider cost.
    #   2) Load upcoming fixtures that already have a persisted pre-kickoff team
    #      lambda but no observed FT Team Totals evidence.
    #   3) Prioritize uncovered fixtures until 20+ unique fixtures are reached.
    #   4) Keep lifecycle refreshes for already-covered current due events behind
    #      new diversity work so later CLV snapshots can still accumulate.
    #   5) Never issue sporting/model/lineup calls here: /odds only.
    cache_hydrated_research_fixtures = 0
    cache_hydrated_research_market_rows = 0
    research_spillover_candidate_fixtures: set[int] = set()
    research_spillover_cache_hits = 0
    research_spillover_api_calls_added = 0
    research_spillover_fixtures_fetched = 0
    research_spillover_market_rows_fetched = 0
    research_spillover_budget_exhausted_fixtures = 0
    research_spillover_api_errors = 0
    research_spillover_synthetic_events_added = 0
    research_spillover_exact_team_total_fixture_ids: set[int] = set()
    research_spillover_new_unique_fixture_ids: set[int] = set()
    research_spillover_current_event_candidates = 0
    research_spillover_scanned_upcoming_candidates = 0
    research_spillover_market_capture_only_candidates = 0
    research_spillover_ft_team_total_market_rows_attached = 0
    research_spillover_primary_payload_reuse_fixtures = 0
    research_spillover_primary_payload_reuse_market_rows = 0
    research_spillover_maturation_candidates = 0
    research_spillover_maturation_api_calls_added = 0
    research_spillover_maturation_later_real_quote_refreshes = 0
    research_spillover_maturation_cache_replays_ignored = 0
    research_spillover_maturation_unchanged_provider_updates = 0
    research_spillover_maturation_budget_exhausted = 0

    maturation = await asyncio.to_thread(_load_team_totals_maturation_backlog)
    maturation_events = [
        event
        for event in (maturation.get("candidate_events") or [])
        if isinstance(event, dict)
    ]
    maturation_by_fixture: dict[int, dict[str, Any]] = {}
    for event in maturation_events:
        fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
        try:
            fixture_id = int(fixture.get("fixture_id"))
        except (TypeError, ValueError):
            continue
        maturation_by_fixture[fixture_id] = event
    research_spillover_maturation_candidates = len(maturation_by_fixture)

    diversity = await asyncio.to_thread(_load_team_totals_diversity_backlog)
    existing_team_total_fixture_ids = {
        int(value)
        for value in (diversity.get("existing_fixture_ids") or set())
        if value is not None
    }
    diversity_target = int(diversity.get("target") or TEAM_TOTALS_DIVERSITY_TARGET)
    persisted_backlog_events = [
        event
        for event in (diversity.get("candidate_events") or [])
        if isinstance(event, dict)
    ]

    current_candidates: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict) or str(event.get("stage") or "").upper() == "POSTGAME":
            continue
        if str(event.get("event_type") or "") != "SOCCER_REFRESH":
            continue
        stage = str(event.get("stage") or "").upper()
        if stage not in TEAM_TOTALS_RESEARCH_STAGES:
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
        maturation_event = maturation_by_fixture.get(fixture_id)
        if maturation_event is not None:
            maturation_meta = maturation_event.get("team_totals_clv_maturation")
            if isinstance(maturation_meta, dict):
                event["team_totals_clv_maturation"] = dict(maturation_meta)
            source = "CURRENT_DUE_EVENT_MATURATION"
            priority = -1
            signal_generated_at = (maturation_meta or {}).get("signal_generated_at") if isinstance(maturation_meta, dict) else None
        else:
            source = "CURRENT_DUE_EVENT"
            priority = 0 if fixture_id not in existing_team_total_fixture_ids else 3
            signal_generated_at = None
        current_candidates.append({
            "fixture_id": fixture_id,
            "event": event,
            "source": source,
            "priority": priority,
            "signal_generated_at": signal_generated_at,
        })
        research_spillover_current_event_candidates += 1

    candidate_records: list[dict[str, Any]] = []
    seen_candidate_fixture_ids: set[int] = set()
    for record in current_candidates:
        fixture_id = int(record["fixture_id"])
        if fixture_id in seen_candidate_fixture_ids:
            continue
        seen_candidate_fixture_ids.add(fixture_id)
        candidate_records.append(record)

    for event in maturation_events:
        fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
        try:
            fixture_id = int(fixture.get("fixture_id"))
        except (TypeError, ValueError):
            continue
        if fixture_id in seen_candidate_fixture_ids:
            continue
        maturation_meta = event.get("team_totals_clv_maturation")
        seen_candidate_fixture_ids.add(fixture_id)
        candidate_records.append({
            "fixture_id": fixture_id,
            "event": event,
            "source": "PERSISTED_CLV_MATURATION_BACKLOG",
            "priority": -1,
            "signal_generated_at": (
                maturation_meta.get("signal_generated_at")
                if isinstance(maturation_meta, dict)
                else None
            ),
        })

    for event in persisted_backlog_events:
        fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
        try:
            fixture_id = int(fixture.get("fixture_id"))
        except (TypeError, ValueError):
            continue
        if fixture_id in seen_candidate_fixture_ids or fixture_id in existing_team_total_fixture_ids:
            continue
        seen_candidate_fixture_ids.add(fixture_id)
        candidate_records.append({
            "fixture_id": fixture_id,
            "event": event,
            "source": "PERSISTED_MODELED_BACKLOG",
            "priority": 1,
        })

    # The base scheduler already paid for the fixture slate scan. Reuse those
    # upcoming fixture identities here at zero provider cost, even before a
    # Team Totals model run exists. This captures the exact FT Team Totals
    # market early; it does NOT count as Phase19 directional evidence until a
    # pre-kickoff model signal and later close are available.
    scanned_upcoming = (
        payload.get("upcoming_market_capture_fixtures")
        if isinstance(payload.get("upcoming_market_capture_fixtures"), list)
        else []
    )
    for fixture in scanned_upcoming:
        if not isinstance(fixture, dict):
            continue
        try:
            fixture_id = int(fixture.get("fixture_id"))
        except (TypeError, ValueError):
            continue
        if fixture_id in seen_candidate_fixture_ids or fixture_id in existing_team_total_fixture_ids:
            continue
        event = {
            "event_type": TEAM_TOTALS_SPILLOVER_EVENT_TYPE,
            "stage": "EARLY_RESEARCH",
            "fixture": dict(fixture),
            "classification": "RESEARCH_ONLY",
            "bet_eligible": False,
            "research_only": True,
            "decision_weight": 0.0,
            "team_totals_diversity_provenance": {
                "candidate_source": "CURRENT_TICK_SCANNED_UPCOMING_FIXTURE",
                "market_capture_only": True,
                "model_recomputed": False,
                "provider_requests_before_price_resolver": 0,
                "primary_markets_preempted": False,
                "phase19_directional_evidence": False,
            },
        }
        seen_candidate_fixture_ids.add(fixture_id)
        candidate_records.append({
            "fixture_id": fixture_id,
            "event": event,
            "source": "SCANNED_UPCOMING_FIXTURE",
            "priority": 2,
        })
        research_spillover_scanned_upcoming_candidates += 1
        research_spillover_market_capture_only_candidates += 1

    # The fixture list is an intra-tick handoff only. Keep the scalar count but
    # drop the bounded list before Postgres persists the final pipeline payload.
    payload.pop("upcoming_market_capture_fixtures", None)

    candidate_records.sort(key=lambda record: (
        int(record.get("priority") or 0),
        str(((record.get("event") or {}).get("fixture") or {}).get("kickoff") or ""),
        int(record.get("fixture_id") or 0),
    ))
    research_spillover_candidate_fixtures = {
        int(record["fixture_id"]) for record in candidate_records
    }

    def _record_exact_team_total_coverage(
        fixture_id: int,
        markets: list[dict[str, Any]],
        event: dict[str, Any],
        *,
        candidate_source: str,
        resolution_status: str,
    ) -> None:
        if not _has_ft_team_total_market(markets):
            return
        research_spillover_exact_team_total_fixture_ids.add(fixture_id)
        event["team_totals_diversity_capture"] = {
            "schema_version": "1.0.0",
            "qualifies": True,
            "criterion": "STRICT_FT_TEAM_TOTAL_MARKET_ID_16_17_OR_CANONICAL_GOAL_LABEL",
            "captured_by_price_resolver": MODEL_VERSION,
            "candidate_source": candidate_source,
            "resolution_status": resolution_status,
            "research_only": True,
            "phase19_true_clv_qualified": False,
            "phase19_true_clv_requires_later_pre_kickoff_close": True,
        }
        if fixture_id not in existing_team_total_fixture_ids:
            research_spillover_new_unique_fixture_ids.add(fixture_id)

    def _attach_spillover_event(record: dict[str, Any], markets: list[dict[str, Any]], status: str) -> None:
        nonlocal research_spillover_synthetic_events_added
        nonlocal research_spillover_ft_team_total_market_rows_attached
        event = record["event"]
        source = str(record.get("source") or "")
        exact_team_total_markets = [
            market for market in markets
            if isinstance(market, dict) and _is_ft_team_total_market(market)
        ]

        if source == "CURRENT_DUE_EVENT":
            if markets and not isinstance(event.get("market"), dict):
                _attach_market_to_event(event, markets, status)
            return

        if source == "CURRENT_DUE_EVENT_MATURATION":
            if not exact_team_total_markets:
                return
            existing_market = event.get("market") if isinstance(event.get("market"), dict) else None
            if existing_market is None:
                _attach_market_to_event(event, exact_team_total_markets, status)
            else:
                existing_rows = [
                    row for row in (existing_market.get("markets") or [])
                    if isinstance(row, dict)
                ]
                seen_keys = {
                    (
                        row.get("bookmaker_id"),
                        row.get("market_id"),
                        str(row.get("provider_update") or ""),
                        str(row.get("values") or ""),
                    )
                    for row in existing_rows
                }
                for row in exact_team_total_markets:
                    key = (
                        row.get("bookmaker_id"),
                        row.get("market_id"),
                        str(row.get("provider_update") or ""),
                        str(row.get("values") or ""),
                    )
                    if key not in seen_keys:
                        existing_rows.append(row)
                        seen_keys.add(key)
                existing_market["markets"] = existing_rows
                existing_market["source"] = (
                    "API_FOOTBALL_ODDS_V3"
                    if status == "PRICE_API_RESOLVED"
                    else existing_market.get("source") or "POSTGRES_MARKET_SNAPSHOT_CACHE"
                )
                existing_market["resolution_status"] = status
            research_spillover_ft_team_total_market_rows_attached += len(exact_team_total_markets)
            return

        # Synthetic backlog/capture events persist only the derivative market we
        # actually need. Avoid writing the entire /odds catalog for 20+ fixtures.
        if not exact_team_total_markets:
            return
        if not isinstance(event.get("market"), dict):
            _attach_market_to_event(event, exact_team_total_markets, status)
        research_spillover_ft_team_total_market_rows_attached += len(exact_team_total_markets)
        if event not in events:
            events.append(event)
            research_spillover_synthetic_events_added += 1

    # First pass: consume only already-fetched primary payloads or fresh Postgres
    # market snapshots. This can increase fixture diversity with zero provider calls.
    unresolved_records: list[dict[str, Any]] = []
    for record in candidate_records:
        fixture_id = int(record["fixture_id"])
        event = record["event"]

        source = str(record.get("source") or "")

        # Reuse Team Totals already present in the paid primary /odds payload
        # attached to this current event. This is the cheapest possible path:
        # zero provider calls, zero cache roundtrip, and primary resolution has
        # already happened before this research-only diversity pass.
        is_maturation = source in {"CURRENT_DUE_EVENT_MATURATION", "PERSISTED_CLV_MATURATION_BACKLOG"}
        signal_generated_at = record.get("signal_generated_at")

        if source in {"CURRENT_DUE_EVENT", "CURRENT_DUE_EVENT_MATURATION"}:
            event_market = event.get("market") if isinstance(event.get("market"), dict) else {}
            event_markets = [
                market
                for market in (event_market.get("markets") or [])
                if isinstance(market, dict)
            ]
            if _has_ft_team_total_market(event_markets):
                _record_exact_team_total_coverage(
                    fixture_id,
                    event_markets,
                    event,
                    candidate_source="PRIMARY_ODDS_PAYLOAD_REUSE",
                    resolution_status=str(event_market.get("resolution_status") or "PRIMARY_ODDS_PAYLOAD_REUSE"),
                )
                research_spillover_primary_payload_reuse_fixtures += 1
                research_spillover_primary_payload_reuse_market_rows += sum(
                    1 for market in event_markets if _is_ft_team_total_market(market)
                )
                if not is_maturation:
                    continue
                if _markets_have_provider_update_after(event_markets, signal_generated_at):
                    research_spillover_maturation_later_real_quote_refreshes += 1
                    event["team_totals_clv_maturation"]["provider_update_after_signal"] = True
                    event["team_totals_clv_maturation"]["resolution_source"] = "PRIMARY_ODDS_PAYLOAD_REUSE"
                    continue

        if fixture_id in fixture_cache:
            markets, status = fixture_cache[fixture_id]
            if markets:
                if is_maturation and str(status).startswith("PRICE_CACHE"):
                    research_spillover_maturation_cache_replays_ignored += 1
                    unresolved_records.append(record)
                    continue
                else:
                    _attach_spillover_event(record, markets, status)
                    _record_exact_team_total_coverage(
                        fixture_id,
                        markets,
                        event,
                        candidate_source=str(record.get("source") or ""),
                        resolution_status=str(status),
                    )
                    if str(status).startswith("PRICE_CACHE"):
                        research_spillover_cache_hits += 1
                    if not is_maturation:
                        continue
                    if _markets_have_provider_update_after(markets, signal_generated_at):
                        research_spillover_maturation_later_real_quote_refreshes += 1
                        event["team_totals_clv_maturation"]["provider_update_after_signal"] = True
                        event["team_totals_clv_maturation"]["resolution_source"] = str(status)
                    else:
                        research_spillover_maturation_unchanged_provider_updates += 1
                    continue

        cached = await asyncio.to_thread(_load_cached_markets, fixture_id, event.get("stage"))
        if cached:
            if is_maturation:
                # A persisted cache row is not new closing evidence. The maturation
                # loader selected this fixture specifically because no later real
                # provider update exists yet, so force a fresh /odds attempt below.
                research_spillover_maturation_cache_replays_ignored += 1
            else:
                fixture_cache[fixture_id] = (cached, "PRICE_CACHE_HIT_RESEARCH_ONLY")
                _attach_spillover_event(record, cached, "PRICE_CACHE_HIT_RESEARCH_ONLY")
                _record_exact_team_total_coverage(
                    fixture_id,
                    cached,
                    event,
                    candidate_source=str(record.get("source") or ""),
                    resolution_status="PRICE_CACHE_HIT_RESEARCH_ONLY",
                )
                cache_hydrated_research_fixtures += 1
                cache_hydrated_research_market_rows += len(cached)
                research_spillover_cache_hits += 1
                continue

        unresolved_records.append(record)

    # Second pass: only now can leftover provider budget be spent. Use a fresh
    # client because the primary resolver client above has already exited its
    # context manager; reusing that closed client would make cache misses fail.
    if unresolved_records:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_SECONDS, follow_redirects=True) as spillover_client:
            maturation_calls_this_tick = 0
            for record in unresolved_records:
                fixture_id = int(record["fixture_id"])
                event = record["event"]
                source = str(record.get("source") or "")
                is_maturation = source in {"CURRENT_DUE_EVENT_MATURATION", "PERSISTED_CLV_MATURATION_BACKLOG"}
                signal_generated_at = record.get("signal_generated_at")

                # Once the accumulated diversity target is satisfied, do not spend
                # extra calls on synthetic backlog. Current due-event lifecycle
                # refreshes may still use any budget that remains for future CLV.
                diversity_progress = len(existing_team_total_fixture_ids | research_spillover_new_unique_fixture_ids)
                catchup_overflow_active = int(payload.get("team_totals_diversity_catchup_overflow_budget") or 0) > 0
                if catchup_overflow_active and diversity_progress >= diversity_target and not is_maturation:
                    # Temporary overflow exists only to close the strict
                    # diversity gap. Once 20 fixtures are reached, never spend
                    # the remaining overflow on lifecycle refreshes.
                    continue
                if source in {"PERSISTED_MODELED_BACKLOG", "SCANNED_UPCOMING_FIXTURE"} and diversity_progress >= diversity_target:
                    continue

                if is_maturation and maturation_calls_this_tick >= TEAM_TOTALS_MATURATION_MAX_CALLS_PER_TICK:
                    research_spillover_maturation_budget_exhausted += 1
                    continue

                if not api_key:
                    fixture_cache[fixture_id] = ([], "PRICE_API_KEY_MISSING_RESEARCH_ONLY")
                    continue
                if calls >= budget:
                    fixture_cache[fixture_id] = ([], "PRICE_BUDGET_EXHAUSTED_RESEARCH_ONLY")
                    research_spillover_budget_exhausted_fixtures += 1
                    continue

                try:
                    markets, used, status, observed_remaining = await _fetch_fixture_odds(
                        spillover_client,
                        fixture_id,
                        api_key=api_key,
                        remaining_calls=budget - calls,
                    )
                    calls += used
                    research_spillover_api_calls_added += used
                    if is_maturation:
                        maturation_calls_this_tick += used
                        research_spillover_maturation_api_calls_added += used
                    if observed_remaining is not None:
                        provider_daily_remaining = (
                            observed_remaining
                            if provider_daily_remaining is None
                            else min(provider_daily_remaining, observed_remaining)
                        )
                    fixture_cache[fixture_id] = (markets, status)
                    if markets:
                        _attach_spillover_event(record, markets, status)
                        _record_exact_team_total_coverage(
                            fixture_id,
                            markets,
                            event,
                            candidate_source=source,
                            resolution_status=str(status),
                        )
                        research_spillover_fixtures_fetched += 1
                        research_spillover_market_rows_fetched += len(markets)
                        event["research_price_spillover"] = {
                            "source": "API_FOOTBALL_ODDS_V3",
                            "provider_requests_added": used,
                            "policy": "LEFTOVER_PRICE_RESOLVER_BUDGET_AFTER_PRIMARY_TARGETS",
                            "research_only": True,
                            "diversity_priority": source in {"PERSISTED_MODELED_BACKLOG", "SCANNED_UPCOMING_FIXTURE"},
                            "clv_maturation_priority": is_maturation,
                            "ft_team_totals_present": _has_ft_team_total_market(markets),
                        }
                        if is_maturation:
                            later_real = _markets_have_provider_update_after(markets, signal_generated_at)
                            event["team_totals_clv_maturation"]["provider_update_after_signal"] = later_real
                            event["team_totals_clv_maturation"]["resolution_source"] = "API_FOOTBALL_ODDS_V3"
                            if later_real and _has_ft_team_total_market(markets):
                                research_spillover_maturation_later_real_quote_refreshes += 1
                            else:
                                research_spillover_maturation_unchanged_provider_updates += 1
                except Exception as exc:
                    research_spillover_api_errors += 1
                    fixture_cache[fixture_id] = ([], "PRICE_API_ERROR_RESEARCH_ONLY")
                    event["research_price_spillover_error"] = str(exc)[:180]

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
        "primary_clv_maturation_source": primary_maturation.get("source"),
        "primary_clv_maturation_candidates": primary_maturation_candidates,
        "primary_clv_maturation_candidate_family_counts": dict(primary_maturation.get("candidate_family_counts") or {}),
        "primary_clv_maturation_max_calls_per_tick": PRIMARY_CLV_MATURATION_MAX_CALLS_PER_TICK,
        "primary_clv_maturation_api_calls_added": primary_maturation_api_calls_added,
        "primary_clv_maturation_fixtures_refreshed": primary_maturation_fixtures_refreshed,
        "primary_clv_maturation_family_refresh_counts": dict(sorted(primary_maturation_family_refresh_counts.items())),
        "primary_clv_maturation_cache_replays_ignored": primary_maturation_cache_replays_ignored,
        "primary_clv_maturation_unchanged_provider_updates": primary_maturation_unchanged_provider_updates,
        "primary_clv_maturation_budget_exhausted": primary_maturation_budget_exhausted,
        "primary_clv_maturation_primary_payload_reuse_fixtures": primary_maturation_primary_payload_reuse_fixtures,
        "primary_clv_maturation_synthetic_events_added": primary_maturation_synthetic_events_added,
        "primary_clv_maturation_policy": "PRIMARY_TARGETS_FIRST;THEN_1X2_FT_TOTALS_BTTS_LATER_REAL_QUOTE;CACHE_REPLAY_NOT_CLOSE;THEN_TEAM_TOTALS;SAME_GLOBAL_PRICE_BUDGET_ONLY",
        "provider_requests_added": calls,
        "provider_daily_remaining_observed": provider_daily_remaining,
        "quota_accounting_included_in_api_calls_this_tick": True,
        "cache_hydrated_research_fixtures": cache_hydrated_research_fixtures,
        "cache_hydrated_research_market_rows": cache_hydrated_research_market_rows,
        "cache_hydration_provider_requests_added": 0,
        "cache_hydration_policy": "PREGAME_TEAM_LAMBDA_FIXTURES_FRESH_POSTGRES_SNAPSHOTS_FIRST",
        "research_spillover_candidate_fixtures": len(research_spillover_candidate_fixtures),
        "research_spillover_current_event_candidates": research_spillover_current_event_candidates,
        "research_spillover_persisted_backlog_candidates": int(diversity.get("candidate_count") or 0),
        "research_spillover_scanned_upcoming_candidates": research_spillover_scanned_upcoming_candidates,
        "research_spillover_market_capture_only_candidates": research_spillover_market_capture_only_candidates,
        "research_spillover_diversity_source": diversity.get("source"),
        "research_spillover_unique_fixture_target": diversity_target,
        "research_spillover_existing_unique_fixtures": len(existing_team_total_fixture_ids),
        "research_spillover_legacy_observed_unique_fixtures": int(diversity.get("legacy_observed_unique_fixtures") or 0),
        "research_spillover_diversity_counter_semantics": "EXPLICIT_STRICT_FT_TEAM_TOTAL_CAPTURE_MARKER_V1_9_PLUS;NOT_PHASE19_TRUE_CLV",
        "research_spillover_phase19_true_clv_gate_separate": True,
        "research_spillover_new_unique_fixtures_this_tick": len(research_spillover_new_unique_fixture_ids),
        "research_spillover_projected_unique_fixtures": len(existing_team_total_fixture_ids | research_spillover_new_unique_fixture_ids),
        "research_spillover_diversity_gap_remaining": max(0, diversity_target - len(existing_team_total_fixture_ids | research_spillover_new_unique_fixture_ids)),
        "research_spillover_exact_team_total_fixtures_attached": len(research_spillover_exact_team_total_fixture_ids),
        "research_spillover_synthetic_events_added": research_spillover_synthetic_events_added,
        "research_spillover_ft_team_total_market_rows_attached": research_spillover_ft_team_total_market_rows_attached,
        "research_spillover_primary_payload_reuse_fixtures": research_spillover_primary_payload_reuse_fixtures,
        "research_spillover_primary_payload_reuse_market_rows": research_spillover_primary_payload_reuse_market_rows,
        "research_spillover_maturation_source": maturation.get("source"),
        "research_spillover_maturation_candidates": research_spillover_maturation_candidates,
        "research_spillover_maturation_max_calls_per_tick": TEAM_TOTALS_MATURATION_MAX_CALLS_PER_TICK,
        "research_spillover_maturation_api_calls_added": research_spillover_maturation_api_calls_added,
        "research_spillover_maturation_later_real_quote_refreshes": research_spillover_maturation_later_real_quote_refreshes,
        "research_spillover_maturation_cache_replays_ignored": research_spillover_maturation_cache_replays_ignored,
        "research_spillover_maturation_unchanged_provider_updates": research_spillover_maturation_unchanged_provider_updates,
        "research_spillover_maturation_budget_exhausted": research_spillover_maturation_budget_exhausted,
        "research_spillover_cache_hits": research_spillover_cache_hits,
        "research_spillover_api_calls_added": research_spillover_api_calls_added,
        "research_spillover_fixtures_fetched": research_spillover_fixtures_fetched,
        "research_spillover_market_rows_fetched": research_spillover_market_rows_fetched,
        "research_spillover_budget_exhausted_fixtures": research_spillover_budget_exhausted_fixtures,
        "research_spillover_api_errors": research_spillover_api_errors,
        "research_spillover_provider_requests_included_in_api_calls_added": True,
        "research_spillover_primary_markets_preempted": False,
        "research_spillover_only_odds_provider_calls": True,
        "research_spillover_standard_leftover_budget": int(payload.get("price_resolver_leftover_budget") or 0),
        "research_spillover_diversity_catchup_overflow_budget": int(payload.get("team_totals_diversity_catchup_overflow_budget") or 0),
        "research_spillover_total_price_resolver_budget": budget,
        "research_spillover_catchup_stops_at_diversity_target": True,
        "research_spillover_clv_maturation_continues_after_diversity_target": True,
        "research_spillover_policy": "PRIMARY_PRICE_TARGETS_COMPLETE_FIRST;TEAM_TOTALS_CLV_MATURATION_USES_ONLY_POST_PRIMARY_LEFTOVER;MATURATION_IGNORES_CACHE_REPLAY_AS_NEW_CLOSE;MATURATION_MAX_12_PROVIDER_CALLS_PER_TICK;PRIMARY_ODDS_PAYLOAD_REUSE_ZERO_EXTRA_CALLS_WHEN_PROVIDER_UPDATE_IS_LATER;DIVERSITY_CAPTURE_STOPS_AT_20;LEGACY_OBSERVED_ROWS_DO_NOT_SATISFY_GATE;MARKET_CAPTURE_WITHOUT_MODEL_IS_NOT_PHASE19_DIRECTIONAL_EVIDENCE;API_FOOTBALL_ODDS_ONLY;RESEARCH_ONLY",
    }
    payload["price_resolution_provider_requests_added"] = calls
    return payload["price_resolution_v4"]
