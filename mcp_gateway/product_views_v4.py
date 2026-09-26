from __future__ import annotations

from typing import Any

from mcp_gateway import market_mismatch_v4

SCHEMA_VERSION = "1.2.0"
MODEL_VERSION = "SOCCER_PRODUCT_VIEWS_V4_1.2.0"
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
    "control_tower",
)

PHASES = (
    ("14", "Cards", "phase14_cards_referee_validation"),
    ("15", "Player Props", "phase15_player_props_validation"),
    ("16", "Market Mismatch", "phase16_market_mismatch_finder"),
    ("17", "CLV", "phase17_clv_engine"),
    ("18", "OOS", "phase18_oos_backtest_framework"),
    ("19", "Promotion", "phase19_promotion_framework"),
    ("20", "Risk", "phase20_bankroll_risk_engine"),
    ("21", "Alerts", "phase21_alert_engine"),
    ("22", "Live", "phase22_live_inplay_engine"),
    ("23", "MLOps", "phase23_mlops_model_registry"),
    ("24", "Product", "phase24_final_product_experience"),
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


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _phase_cards(payload: dict[str, Any]) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for phase, name, key in PHASES:
        raw = payload.get(key)
        node = raw if isinstance(raw, dict) else {}
        cards.append({
            "phase": phase,
            "name": name,
            "status": node.get("status") or "NOT_VERIFIED",
            "model_version": node.get("model_version"),
            "production_promotion_allowed": bool(node.get("production_promotion_allowed")),
            "source_key": key,
        })
    return cards


def _pipeline_errors(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for row in rows:
        fields = (
            row.get("row_type"),
            row.get("event_type"),
            row.get("classification"),
            row.get("event_classification"),
            row.get("reason"),
        )
        blockers = row.get("blockers") if isinstance(row.get("blockers"), list) else []
        haystack = " ".join(str(value or "") for value in (*fields, *blockers)).upper()
        if "PIPELINE_ERROR" not in haystack:
            continue
        errors.append({
            "fixture_id": row.get("fixture_id"),
            "home": row.get("home"),
            "away": row.get("away"),
            "league": row.get("league"),
            "stage": row.get("stage"),
            "reason": row.get("reason") or "PIPELINE_ERROR",
        })
    return errors[:10]


def _max_phase16_calibration_rows(rows: list[dict[str, Any]]) -> int | None:
    values: list[int] = []
    for row in rows:
        binary = row.get("phase16_binary_calibration_diagnostics")
        if isinstance(binary, dict):
            value = _int_or_none(binary.get("rows"))
            if value is not None:
                values.append(value)
        multiclass = row.get("phase16_1x2_class_discrimination_diagnostics")
        if isinstance(multiclass, dict):
            for diag in multiclass.values():
                if isinstance(diag, dict):
                    value = _int_or_none(diag.get("rows"))
                    if value is not None:
                        values.append(value)
    return max(values) if values else None


def _validation_gates(payload: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def required(key: str, field: str) -> int | None:
        node = payload.get(key)
        return _int_or_none(node.get(field)) if isinstance(node, dict) else None

    # Current values are emitted only when they are present in the persisted
    # runtime payload. Missing validation-report counters remain None/N/V.
    return [
        {
            "key": "phase16_calibration_sample",
            "label": "Core calibration sample",
            "current": _max_phase16_calibration_rows(rows),
            "target": required("v4_017_1x2_calibration_validation", "minimum_calibration_sample"),
            "unit": "rows",
            "source": "latest_runtime_diagnostics",
        },
        {
            "key": "1x2_true_clv",
            "label": "1X2 True CLV",
            "current": None,
            "target": required("v4_017_1x2_calibration_validation", "minimum_family_specific_true_clv_rows"),
            "unit": "rows",
            "source": "validation_report_required",
        },
        {
            "key": "btts_true_clv",
            "label": "BTTS True CLV",
            "current": None,
            "target": required("v4_018_btts_calibration_validation", "minimum_family_specific_true_clv_rows"),
            "unit": "rows",
            "source": "validation_report_required",
        },
        {
            "key": "team_totals_true_clv",
            "label": "Team Totals True CLV",
            "current": None,
            "target": required("v4_019_team_totals_oos_validation", "minimum_family_specific_true_clv_rows"),
            "unit": "rows",
            "source": "validation_report_required",
        },
        {
            "key": "1h_true_clv",
            "label": "1H True CLV",
            "current": None,
            "target": required("v4_020_1h_oos_validation", "minimum_family_specific_true_clv_rows"),
            "unit": "rows",
            "source": "validation_report_required",
        },
        {
            "key": "corners_formation",
            "label": "Corners formation-adjusted",
            "current": None,
            "target": required("v4_022_corners_oos_validation", "minimum_formation_adjusted"),
            "unit": "fixtures",
            "source": "validation_report_required",
        },
        {
            "key": "player_props_true_clv",
            "label": "Player Props True CLV / family",
            "current": None,
            "target": required("phase15_player_props_validation", "minimum_prop_true_clv_rows"),
            "unit": "rows",
            "source": "validation_report_required",
        },
    ]


def _control_tower(payload: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = _pipeline_errors(rows)
    slate = (
        payload.get("core_slate_floor_reconciliation")
        if isinstance(payload.get("core_slate_floor_reconciliation"), dict)
        else {}
    )
    db_ok = bool(payload.get("database_persisted")) and not payload.get("database_error")
    tick_ok = str(payload.get("status") or "").lower() == "ok"
    daily_remaining = _int_or_none(payload.get("last_daily_remaining"))
    effective_cap = _int_or_none(payload.get("effective_max_api_calls_per_tick"))
    if effective_cap is None:
        effective_cap = _int_or_none(payload.get("max_api_calls_per_tick"))
    fair = payload.get("fair_scheduler") if isinstance(payload.get("fair_scheduler"), dict) else {}
    fair_processed = fair.get("processed_category_counts") if isinstance(fair.get("processed_category_counts"), dict) else {}
    price_checkpoint = (
        payload.get("price_resolution_checkpoint")
        if isinstance(payload.get("price_resolution_checkpoint"), dict)
        else {}
    )

    return {
        "schema_version": "2.0.0",
        "status": "LIVE" if tick_ok and db_ok and not errors else "DEGRADED",
        "generated_at_utc": payload.get("generated_at_utc"),
        "generated_at_local": payload.get("generated_at_local"),
        "pipeline_version": payload.get("version"),
        "model_version": payload.get("model_version"),
        "system_health": {
            "runtime": "HEALTHY" if tick_ok else "DEGRADED",
            "postgres": "HEALTHY" if db_ok else "DEGRADED",
            "scheduler": "TICK_OBSERVED" if payload.get("generated_at_utc") else "NOT_VERIFIED",
            "api_football": "HEALTHY" if daily_remaining is not None and daily_remaining > 0 else "NOT_VERIFIED",
            "api_football_remaining": daily_remaining,
            "daily_budget_mode": payload.get("daily_budget_mode"),
            "galaxy": "NOT_VERIFIED",
            "last_tick": payload.get("generated_at_local") or payload.get("generated_at_utc"),
        },
        "pipeline": {
            "fixtures_scanned": _int_or_none(payload.get("fixture_scan_count")),
            "due": _int_or_none(payload.get("due_fixture_count")),
            "events": _int_or_none(payload.get("event_count")),
            "actionable_refreshes": _int_or_none(payload.get("actionable_refresh_count")),
            "deep_dives": _int_or_none(payload.get("deep_dive_processed_count")),
            "research_visible": _int_or_none(payload.get("research_visible_count")),
            "bet_candidates": _int_or_none(payload.get("bet_candidate_count")),
            "api_calls": _int_or_none(payload.get("api_calls_this_tick")),
            "api_call_cap": effective_cap,
            "deferred_budget": _int_or_none(payload.get("deferred_due_to_budget")),
            "deferred_priority": _int_or_none(payload.get("deferred_due_to_priority")),
            "unique_leagues_scanned": _int_or_none(slate.get("unique_leagues_scanned")),
            "unique_countries_scanned": _int_or_none(slate.get("unique_countries_scanned")),
            "scan_dates": list(slate.get("scan_dates") or []),
            "scan_date_counts": dict(slate.get("scan_date_counts") or {}),
            "future_prefetch_fixture_count": _int_or_none(slate.get("future_prefetch_fixture_count")),
            "future_prefetch_cache_hits": _int_or_none(slate.get("future_prefetch_cache_hits")),
            "future_prefetch_cache_misses": _int_or_none(slate.get("future_prefetch_cache_misses")),
            "market_capture_handoff_fixture_count": _int_or_none(slate.get("market_capture_handoff_fixture_count")),
            "market_capture_handoff_tier_counts": dict(slate.get("market_capture_handoff_tier_counts") or {}),
            "league_allowlist_applied": bool(slate.get("league_allowlist_applied")),
            "scheduler_schema_version": fair.get("schema_version"),
            "scheduler_mode": fair.get("scheduling_mode") or fair.get("policy"),
            "scheduler_effective_weights_pct": dict(fair.get("effective_weights_pct") or fair.get("weights_pct") or {}),
            "scheduler_planned_unique_leagues": _int_or_none(fair.get("planned_unique_leagues")),
            "scheduler_urgent_actionable_count": _int_or_none(fair.get("urgent_actionable_count")),
            "scheduler_unseen_processed": _int_or_none(fair_processed.get("unseen")),
            "scheduler_actionable_processed": _int_or_none(fair_processed.get("actionable")),
            "scheduler_starvation_count": _int_or_none(fair.get("starvation_count")),
            "scheduler_due_analyzed_pct": fair.get("due_analyzed_pct"),
            "primary_clv_maturation_candidates": _int_or_none(price_checkpoint.get("primary_clv_maturation_candidates")),
            "primary_clv_maturation_refreshed": _int_or_none(price_checkpoint.get("primary_clv_maturation_fixtures_refreshed")),
            "primary_clv_maturation_unchanged_provider_updates": _int_or_none(price_checkpoint.get("primary_clv_maturation_unchanged_provider_updates")),
            "team_totals_maturation_candidates": _int_or_none(price_checkpoint.get("research_spillover_maturation_candidates")),
            "team_totals_maturation_api_calls_added": _int_or_none(price_checkpoint.get("research_spillover_maturation_api_calls_added")),
            "team_totals_later_quote_refreshes": _int_or_none(price_checkpoint.get("research_spillover_maturation_later_real_quote_refreshes")),
            "team_totals_unchanged_provider_updates": _int_or_none(price_checkpoint.get("research_spillover_maturation_unchanged_provider_updates")),
        },
        "errors": {
            "count": len(errors),
            "rows": errors,
        },
        "validation_gates": _validation_gates(payload, rows),
        "phases": _phase_cards(payload),
        "production_valid_market_count": 0,
        "production_promotion_allowed": False,
        "source": "POSTGRES_LATEST_PIPELINE_RUN",
        "note": (
            "Live telemetry is sourced from the latest persisted pipeline run. "
            "Validation counters not present in that runtime payload are shown as N/V rather than inferred."
        ),
    }


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

    data_health = {
        "database_persisted": bool(payload.get("database_persisted")),
        "database_error": payload.get("database_error"),
        "api_calls_this_tick": payload.get("api_calls_this_tick"),
        "max_api_calls_per_tick": payload.get("max_api_calls_per_tick"),
        "effective_max_api_calls_per_tick": payload.get("effective_max_api_calls_per_tick"),
        "last_daily_remaining": payload.get("last_daily_remaining"),
        "daily_budget_mode": payload.get("daily_budget_mode"),
        "fixture_scan_count": payload.get("fixture_scan_count"),
        "due_fixture_count": payload.get("due_fixture_count"),
        "deferred_due_to_budget": payload.get("deferred_due_to_budget"),
        "deferred_due_to_priority": payload.get("deferred_due_to_priority"),
    }

    views = {
        "todays_slate": {"rows": _take(active, limit), "total": len(active)},
        "strong_sport_signals": {"rows": _take(strong, limit), "total": len(strong)},
        "value_plays": {
            "rows": _take(value_plays, limit),
            "total": len(value_plays),
            "requires_calibrated_probability": True,
        },
        "waiting_for_price": {"rows": _take(waiting_price, limit), "total": len(waiting_price)},
        "waiting_for_xi": {"rows": _take(waiting_xi, limit), "total": len(waiting_xi)},
        "team_totals": {"rows": _take(team_totals, limit), "total": len(team_totals)},
        "first_half": {"rows": _take(first_half, limit), "total": len(first_half)},
        "second_half": {"rows": _take(second_half, limit), "total": len(second_half)},
        "corners": {"rows": _take(corners, limit), "total": len(corners)},
        "player_props": {"rows": _take(player_props, limit), "total": len(player_props)},
        "closing_line": {
            "status": (payload.get("phase17_clv_engine") or {}).get("status")
            if isinstance(payload.get("phase17_clv_engine"), dict) else None,
            "source": "phase17_clv_engine_report",
        },
        "performance": {
            "oos_status": (payload.get("phase18_oos_backtest_framework") or {}).get("status")
            if isinstance(payload.get("phase18_oos_backtest_framework"), dict) else None,
            "promotion_status": (payload.get("phase19_promotion_framework") or {}).get("status")
            if isinstance(payload.get("phase19_promotion_framework"), dict) else None,
            "risk_status": (payload.get("phase20_bankroll_risk_engine") or {}).get("status")
            if isinstance(payload.get("phase20_bankroll_risk_engine"), dict) else None,
        },
        "model_health": {
            "calibration": (payload.get("v4_013_calibration_engine_v1") or {}).get("status")
            if isinstance(payload.get("v4_013_calibration_engine_v1"), dict) else None,
            "disagreement": (payload.get("v4_014_model_disagreement_engine") or {}).get("status")
            if isinstance(payload.get("v4_014_model_disagreement_engine"), dict) else None,
            "confidence": (payload.get("v4_015_confidence_engine_v1") or {}).get("status")
            if isinstance(payload.get("v4_015_confidence_engine_v1"), dict) else None,
            "registry": (payload.get("phase23_mlops_model_registry") or {}).get("status")
            if isinstance(payload.get("phase23_mlops_model_registry"), dict) else None,
        },
        "data_health": data_health,
        "control_tower": _control_tower(payload, rows),
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": "CONTROL_TOWER_V1_CONTRACT",
        "view_names": list(VIEW_NAMES),
        "views": views,
        "row_limit_per_view": int(limit),
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
        "canonical_bet_logic_changed": False,
        "model_weights_changed": False,
    }
