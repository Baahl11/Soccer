from __future__ import annotations

from typing import Any

from mcp_gateway import market_mismatch_v4

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_PRODUCT_VIEWS_V4_1.0.0"
MAX_ROWS_PER_VIEW = 25

VIEW_NAMES = (
    "todays_slate",
    "strong_sport_signals",
    "value_plays",
    "waiting_for_price",
    "waiting_for_xi",
    "team_totals",
    "first_half",
    "second_half",
    "corners",
    "player_props",
    "closing_line",
    "performance",
    "model_health",
    "data_health",
)


def _rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    source = payload.get("match_table_rows")
    return [row for row in source if isinstance(row, dict)] if isinstance(source, list) else []


def _active_fixture(row: dict[str, Any]) -> bool:
    status = str(row.get("status") or "").upper()
    stage = str(row.get("stage") or "").upper()
    return status not in {"FT", "AET", "PEN", "CANC", "PST"} and stage != "POSTGAME"


def _canonical_family(row: dict[str, Any]) -> str | None:
    return market_mismatch_v4.canonical_market_family(row)


def _take(rows: list[dict[str, Any]], limit: int = MAX_ROWS_PER_VIEW) -> list[dict[str, Any]]:
    return rows[: max(int(limit), 0)]


def build_views(payload: dict[str, Any], *, limit: int = MAX_ROWS_PER_VIEW) -> dict[str, Any]:
    rows = _rows(payload)
    active = [row for row in rows if _active_fixture(row)]

    strong = [
        row for row in active
        if str(row.get("model_signal") or "").upper() in {"STRONG", "VERY_STRONG"}
    ]
    strong.sort(key=lambda row: float(row.get("model_signal_score") or 0.0), reverse=True)

    value_plays = [
        row for row in (payload.get("market_mismatch_rows") or [])
        if isinstance(row, dict)
    ]
    value_plays.sort(key=lambda row: float(row.get("mismatch_score") or 0.0), reverse=True)

    waiting_price = [
        row for row in active
        if str(row.get("execution_status") or "").upper() in {"WAIT_PRICE", "WAIT_FRESH_QUOTE"}
    ]
    waiting_xi = [
        row for row in active
        if str(row.get("execution_status") or "").upper() in {"WAIT_XI", "WAIT_GK", "WAIT_AVAILABILITY"}
    ]

    family_rows: dict[str, list[dict[str, Any]]] = {}
    for row in active:
        family = _canonical_family(row)
        if family:
            family_rows.setdefault(family, []).append(row)

    team_totals = family_rows.get("HOME_TT", []) + family_rows.get("AWAY_TT", [])
    first_half = family_rows.get("1H", [])
    second_half = family_rows.get("2H", [])
    corners = family_rows.get("FT_CORNERS", []) + family_rows.get("TEAM_CORNERS", [])
    player_props: list[dict[str, Any]] = []
    for family in ("SHOTS", "SOT", "GOALSCORER", "ASSISTS", "PLAYER_CARDS", "GK_SAVES"):
        player_props.extend(family_rows.get(family, []))

    views = {
        "todays_slate": {
            "rows": _take(active, limit),
            "total": len(active),
        },
        "strong_sport_signals": {
            "rows": _take(strong, limit),
            "total": len(strong),
        },
        "value_plays": {
            "rows": _take(value_plays, limit),
            "total": len(value_plays),
            "requires_calibrated_probability": True,
        },
        "waiting_for_price": {
            "rows": _take(waiting_price, limit),
            "total": len(waiting_price),
        },
        "waiting_for_xi": {
            "rows": _take(waiting_xi, limit),
            "total": len(waiting_xi),
        },
        "team_totals": {
            "rows": _take(team_totals, limit),
            "total": len(team_totals),
        },
        "first_half": {
            "rows": _take(first_half, limit),
            "total": len(first_half),
        },
        "second_half": {
            "rows": _take(second_half, limit),
            "total": len(second_half),
        },
        "corners": {
            "rows": _take(corners, limit),
            "total": len(corners),
        },
        "player_props": {
            "rows": _take(player_props, limit),
            "total": len(player_props),
        },
        "closing_line": {
            "status": (payload.get("phase17_clv_engine") or {}).get("status")
            if isinstance(payload.get("phase17_clv_engine"), dict)
            else None,
            "source": "phase17_clv_engine_report",
        },
        "performance": {
            "oos_status": (payload.get("phase18_oos_backtest_framework") or {}).get("status")
            if isinstance(payload.get("phase18_oos_backtest_framework"), dict)
            else None,
            "promotion_status": (payload.get("phase19_promotion_framework") or {}).get("status")
            if isinstance(payload.get("phase19_promotion_framework"), dict)
            else None,
            "risk_status": (payload.get("phase20_bankroll_risk_engine") or {}).get("status")
            if isinstance(payload.get("phase20_bankroll_risk_engine"), dict)
            else None,
        },
        "model_health": {
            "calibration": (payload.get("v4_013_calibration_engine_v1") or {}).get("status")
            if isinstance(payload.get("v4_013_calibration_engine_v1"), dict)
            else None,
            "disagreement": (payload.get("v4_014_model_disagreement_engine") or {}).get("status")
            if isinstance(payload.get("v4_014_model_disagreement_engine"), dict)
            else None,
            "confidence": (payload.get("v4_015_confidence_engine_v1") or {}).get("status")
            if isinstance(payload.get("v4_015_confidence_engine_v1"), dict)
            else None,
            "registry": (payload.get("phase23_mlops_model_registry") or {}).get("status")
            if isinstance(payload.get("phase23_mlops_model_registry"), dict)
            else None,
        },
        "data_health": {
            "database_persisted": bool(payload.get("database_persisted")),
            "api_calls_this_tick": payload.get("api_calls_this_tick"),
            "max_api_calls_per_tick": payload.get("max_api_calls_per_tick"),
            "last_daily_remaining": payload.get("last_daily_remaining"),
            "fixture_scan_count": payload.get("fixture_scan_count"),
            "due_fixture_count": payload.get("due_fixture_count"),
            "deferred_due_to_budget": payload.get("deferred_due_to_budget"),
            "deferred_due_to_priority": payload.get("deferred_due_to_priority"),
        },
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": "API_VIEW_CONTRACT_IMPLEMENTED_UI_PENDING",
        "view_names": list(VIEW_NAMES),
        "views": views,
        "row_limit_per_view": int(limit),
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
        "canonical_bet_logic_changed": False,
        "model_weights_changed": False,
    }
