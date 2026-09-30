from __future__ import annotations

import asyncio
from collections import Counter
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

from mcp_gateway import market_mismatch_v4
from mcp_gateway import persistence as persistence_base
from mcp_gateway import product_views_v4
from mcp_gateway import subscriber_ui_contract_v231
from mcp_gateway import subscription_entitlements_v4
from mcp_gateway import supabase_auth_v4

SCHEMA_VERSION = "1.3.0"
MODEL_VERSION = "SOCCER_SUBSCRIBER_PREVIEW_DATA_V231"


def _rows(node: Any) -> list[dict[str, Any]]:
    if not isinstance(node, dict):
        return []
    value = node.get("rows")
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _first(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return None


def _number(value: Any) -> float | None:
    try:
        return float(value) if value is not None and value != "" else None
    except (TypeError, ValueError):
        return None


def _probability(value: Any) -> float | None:
    number = _number(value)
    if number is None:
        return None
    if abs(number) > 1.0 and abs(number) <= 100.0:
        number /= 100.0
    return number if 0.0 <= number <= 1.0 else None


def _score_value(value: Any) -> float | None:
    number = _number(value)
    if number is None:
        return None
    if abs(number) <= 1.0:
        number *= 100.0
    return max(0.0, min(100.0, number))


def _edge_sort_key(row: dict[str, Any]) -> tuple[int, float]:
    pricing = row.get("pricing") if isinstance(row.get("pricing"), dict) else {}
    edge = pricing.get("edge_pp")
    try:
        value = float(edge)
    except (TypeError, ValueError):
        return (0, -10_000.0)
    return (1, value)


def _dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, Any, Any]] = set()
    result: list[dict[str, Any]] = []
    for row in rows:
        match = row.get("match") if isinstance(row.get("match"), dict) else {}
        market = row.get("market") if isinstance(row.get("market"), dict) else {}
        key = (match.get("fixture_id"), market.get("family"), market.get("selection"))
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def _active_raw_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    source = payload.get("match_table_rows")
    rows = [row for row in source if isinstance(row, dict)] if isinstance(source, list) else []
    active: list[dict[str, Any]] = []
    for row in rows:
        status = str(row.get("status") or "").upper()
        stage = str(row.get("stage") or "").upper()
        if status in {"FT", "AET", "PEN", "CANC", "PST"} or stage == "POSTGAME":
            continue
        active.append(row)
    return active


def _market_catalog(payload: dict[str, Any], maturation: dict[str, Any]) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    for row in _active_raw_rows(payload):
        family = market_mismatch_v4.canonical_market_family(row)
        if family:
            counts[family] += 1

    maturity_by_label: dict[str, dict[str, Any]] = {}
    for family in maturation.get("families") or []:
        if isinstance(family, dict):
            maturity_by_label[str(family.get("label") or family.get("key") or "").upper()] = family

    specs = [
        ("1X2", ("1X2",), "Match result probabilities, draw-aware calibration"),
        ("BTTS", ("BTTS",), "Both Teams To Score · de-vig market compare"),
        ("FT Totals", ("FT_TOTALS",), "Full-time totals ladder and price history"),
        ("Team Totals", ("HOME_TT", "AWAY_TT"), "Home/Away team scoring markets"),
        ("1H", ("1H",), "First-half goals and match markets"),
        ("Corners", ("FT_CORNERS", "TEAM_CORNERS"), "Formation-aware corner intelligence"),
        ("2H", ("2H",), "Second-half goals intelligence"),
        ("Cards", ("CARDS",), "Match/team cards and referee context"),
        ("Player Props", ("SHOTS", "SOT", "GOALSCORER", "ASSISTS", "PLAYER_CARDS", "GK_SAVES"), "XI-aligned player market research"),
    ]

    result: list[dict[str, Any]] = []
    for label, families, description in specs:
        maturity = maturity_by_label.get(label.upper()) or {}
        result.append({
            "label": label,
            "live_rows": sum(counts[name] for name in families),
            "maturity_current": maturity.get("current"),
            "maturity_target": maturity.get("target"),
            "maturity_status": maturity.get("status"),
            "blocker": maturity.get("blocker"),
            "evidence_kind": maturity.get("evidence_kind"),
            "description": description,
        })
    return result


def _fixture_rows(payload: dict[str, Any], fixture_id: Any) -> list[dict[str, Any]]:
    if fixture_id is None:
        return []
    return [row for row in _active_raw_rows(payload) if row.get("fixture_id") == fixture_id]


def _find_probability(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> float | None:
    for row in rows:
        direct = _probability(_first(row, *keys))
        if direct is not None:
            return direct
        for node_key in ("probabilities", "one_x_two", "one_x_two_probabilities", "match_result_probabilities"):
            node = row.get(node_key)
            if not isinstance(node, dict):
                continue
            direct = _probability(_first(node, *keys))
            if direct is not None:
                return direct
    return None


def _find_number(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> float | None:
    for row in rows:
        value = _number(_first(row, *keys))
        if value is not None:
            return value
        for node_key in ("expected_goals", "xg", "goal_model", "poisson"):
            node = row.get(node_key)
            if isinstance(node, dict):
                value = _number(_first(node, *keys))
                if value is not None:
                    return value
    return None


def _find_any(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> Any:
    for row in rows:
        value = _first(row, *keys)
        if value is not None:
            return value
    return None


def _score_matrix(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[tuple[str, float]] = []
    raw_matrix: Any = None
    for row in rows:
        for key in ("score_matrix", "scoreline_matrix", "score_probabilities", "exact_score_probabilities"):
            value = row.get(key)
            if isinstance(value, (dict, list)) and value:
                raw_matrix = value
                break
        if raw_matrix is not None:
            break

    if isinstance(raw_matrix, dict):
        for key, value in raw_matrix.items():
            probability = _probability(value)
            if probability is None:
                continue
            label = str(key).replace("_", "-").replace(":", "-")
            candidates.append((label, probability))
    elif isinstance(raw_matrix, list):
        for item in raw_matrix:
            if not isinstance(item, dict):
                continue
            home = _first(item, "home", "home_goals", "h")
            away = _first(item, "away", "away_goals", "a")
            probability = _probability(_first(item, "probability", "prob", "p"))
            if home is None or away is None or probability is None:
                continue
            candidates.append((f"{home}-{away}", probability))

    candidates.sort(key=lambda item: item[1], reverse=True)
    return [{"score": score, "probability": probability} for score, probability in candidates[:9]]


def _sport_profile(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aliases = [
        ("Attack", ("attack_strength", "attack_score", "attack_momentum", "attacking_strength")),
        ("Defense", ("defense_strength", "defense_score", "defensive_balance", "defensive_strength")),
        ("Territory", ("territorial_control", "field_tilt", "field_tilt_score", "possession_control")),
        ("Form", ("form_score", "recent_form_score", "form_strength")),
        ("Set pieces", ("set_piece_score", "set_pieces_score", "set_piece_strength")),
    ]
    result: list[dict[str, Any]] = []
    for label, keys in aliases:
        value: float | None = None
        source: str | None = None
        for row in rows:
            profile = row.get("sport_profile") if isinstance(row.get("sport_profile"), dict) else {}
            raw = _first(profile, *keys)
            if raw is None:
                raw = _first(row, *keys)
            score = _score_value(raw)
            if score is not None:
                value = score
                source = next((key for key in keys if profile.get(key) is not None or row.get(key) is not None), None)
                break
        if value is not None:
            result.append({"label": label, "score": value, "source": source})
    return result


def _normalize_lineup(value: Any) -> str | None:
    if isinstance(value, bool):
        return "CONFIRMED" if value else "NOT_CONFIRMED"
    if value is None or value == "":
        return None
    return str(value)


def _match_detail(payload: dict[str, Any], selected: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(selected, dict):
        return None
    match = selected.get("match") if isinstance(selected.get("match"), dict) else {}
    fixture_id = match.get("fixture_id")
    rows = _fixture_rows(payload, fixture_id)
    selected_raw = selected.get("raw") if isinstance(selected.get("raw"), dict) else {}
    if selected_raw:
        rows = [selected_raw] + [row for row in rows if row is not selected_raw]

    home = _find_probability(rows, ("prob_home_win", "home_win_probability", "p_home", "home"))
    draw = _find_probability(rows, ("prob_draw", "draw_probability", "p_draw", "draw"))
    away = _find_probability(rows, ("prob_away_win", "away_win_probability", "p_away", "away"))
    outcome = None
    if home is not None and draw is not None and away is not None:
        total = home + draw + away
        if total > 0:
            outcome = {"home": home / total, "draw": draw / total, "away": away / total}

    lambda_home = _find_number(rows, ("lambda_home", "predicted_home_goals", "xg_home", "home_xg", "home"))
    lambda_away = _find_number(rows, ("lambda_away", "predicted_away_goals", "xg_away", "away_xg", "away"))
    xg = None
    if lambda_home is not None or lambda_away is not None:
        xg = {
            "home": lambda_home,
            "away": lambda_away,
            "total": (lambda_home + lambda_away) if lambda_home is not None and lambda_away is not None else None,
        }

    confidence = _number(_find_any(rows, ("confidence_score", "model_confidence", "confidence", "model_signal_score")))
    if confidence is not None and abs(confidence) <= 1.0:
        confidence *= 100.0
    if confidence is not None:
        confidence = max(0.0, min(100.0, confidence))

    data_quality = _find_any(rows, ("data_tier", "quality_tier", "data_quality"))
    lineup = _normalize_lineup(_find_any(rows, ("lineup_status", "xi_status", "confirmed_xi", "lineups_confirmed")))
    disagreement = _find_any(rows, ("model_disagreement", "disagreement_label", "ensemble_disagreement"))
    agreement_count = _number(_find_any(rows, ("model_agreement_count", "models_agreeing", "agreement_count")))
    model_count = _number(_find_any(rows, ("model_count", "models_total", "ensemble_model_count")))
    provider_update = _find_any(rows, ("provider_update", "provider_updated_at", "market_updated_at", "odds_updated_at"))
    bookmaker = _find_any(rows, ("bookmaker", "provider"))

    return {
        "selected": selected,
        "fixture_row_count": len(rows),
        "outcome_probabilities": outcome,
        "expected_goals": xg,
        "score_matrix": _score_matrix(rows),
        "sport_profile": _sport_profile(rows),
        "model_context": {
            "confidence": confidence,
            "data_quality": data_quality,
            "lineup": lineup,
            "model_disagreement": disagreement,
            "models_agreeing": int(agreement_count) if agreement_count is not None else None,
            "models_total": int(model_count) if model_count is not None else None,
            "model_version": (selected.get("model") or {}).get("version") if isinstance(selected.get("model"), dict) else None,
            "stage": (selected.get("state") or {}).get("stage") if isinstance(selected.get("state"), dict) else None,
            "reason": (selected.get("state") or {}).get("reason") if isinstance(selected.get("state"), dict) else None,
            "provider_update": provider_update,
            "bookmaker": bookmaker,
        },
        "source": "PERSISTED_ROWS_FOR_SELECTED_FIXTURE",
    }


def _system_alerts(
    waiting_price: list[dict[str, Any]],
    waiting_xi: list[dict[str, Any]],
    maturation: dict[str, Any],
) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    for row in waiting_price[:6]:
        alerts.append({
            "type": "PRICE",
            "severity": "WATCH",
            "match": (row.get("match") or {}).get("label"),
            "market": (row.get("market") or {}).get("selection") or (row.get("market") or {}).get("family"),
            "message": (row.get("state") or {}).get("reason") or "Waiting for a usable market price",
        })
    for row in waiting_xi[:6]:
        alerts.append({
            "type": "XI",
            "severity": "WATCH",
            "match": (row.get("match") or {}).get("label"),
            "market": (row.get("market") or {}).get("selection") or (row.get("market") or {}).get("family"),
            "message": (row.get("state") or {}).get("reason") or "Waiting for lineup evidence",
        })
    for family in maturation.get("families") or []:
        if not isinstance(family, dict) or not family.get("blocker"):
            continue
        alerts.append({
            "type": "MATURATION",
            "severity": str(family.get("status") or "WATCH"),
            "match": family.get("label"),
            "market": family.get("evidence_kind"),
            "message": family.get("blocker"),
        })
    return alerts[:12]


def build_preview_payload(payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(payload)
    payload.setdefault("status", "ok")
    payload["database_persisted"] = True
    payload["database_error"] = None

    product = product_views_v4.build_views(payload, limit=product_views_v4.MAX_ROWS_PER_VIEW)
    views = product.get("views") if isinstance(product.get("views"), dict) else {}

    slate = subscriber_ui_contract_v231.adapt_rows(_rows(views.get("todays_slate")))
    strong = subscriber_ui_contract_v231.adapt_rows(_rows(views.get("strong_sport_signals")))
    value = subscriber_ui_contract_v231.adapt_rows(_rows(views.get("value_plays")))
    waiting_price = subscriber_ui_contract_v231.adapt_rows(_rows(views.get("waiting_for_price")))
    waiting_xi = subscriber_ui_contract_v231.adapt_rows(_rows(views.get("waiting_for_xi")))

    feed = _dedupe(strong + value + slate)
    feed.sort(key=_edge_sort_key, reverse=True)
    top_edge = next(
        (
            row
            for row in feed
            if (row.get("model") or {}).get("probability") is not None
            and (row.get("pricing") or {}).get("market_probability") is not None
            and (row.get("pricing") or {}).get("edge_pp") is not None
        ),
        None,
    )
    selected_match = top_edge or (feed[0] if feed else None)

    control = views.get("control_tower") if isinstance(views.get("control_tower"), dict) else {}
    pipeline = control.get("pipeline") if isinstance(control.get("pipeline"), dict) else {}
    health = control.get("system_health") if isinstance(control.get("system_health"), dict) else {}
    errors = control.get("errors") if isinstance(control.get("errors"), dict) else {}
    maturity_snapshot = control.get("maturity_snapshot") if isinstance(control.get("maturity_snapshot"), dict) else {}
    maturation = maturity_snapshot.get("maturation_control_tower") if isinstance(maturity_snapshot.get("maturation_control_tower"), dict) else {}
    monitoring = maturation.get("monitoring") if isinstance(maturation.get("monitoring"), dict) else {}

    pass_count = 0
    for row in slate:
        state = row.get("state") if isinstance(row.get("state"), dict) else {}
        if str(state.get("status") or "").upper() == "PASS":
            pass_count += 1

    detail = _match_detail(payload, selected_match)
    selected_family = None
    if isinstance(selected_match, dict):
        selected_family = (selected_match.get("market") or {}).get("family")

    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "source": "POSTGRES_LATEST_PIPELINE_RUN+UI_CONTRACT_V231",
        "generated_at_utc": payload.get("generated_at_utc"),
        "generated_at_local": payload.get("generated_at_local"),
        "today": {
            "metrics": {
                "matches_scanned": pipeline.get("fixtures_scanned"),
                "deep_analyzed": pipeline.get("deep_dives"),
                "strong_edges": len(strong),
                "waiting_xi": len(waiting_xi),
                "pass": pass_count,
            },
            "top_edge": top_edge,
            "strong_signals": strong[:5],
            "price_opportunities": waiting_price[:5],
            "waiting_xi": waiting_xi[:5],
            "upcoming": slate[:6],
        },
        "edge_feed": {"rows": feed[:25], "total": len(feed)},
        "match_center": {
            "selected": selected_match,
            "detail": detail,
            "source": "TOP_COMPARABLE_EDGE_OR_FIRST_PERSISTED_FEED_ROW",
            "wired": [
                "match_header",
                "selected_market_probability",
                "market_probability",
                "edge",
                "price",
                "confidence",
                "status",
                "outcome_probabilities_when_persisted",
                "expected_goals_when_persisted",
                "score_matrix_when_persisted",
                "sport_profile_when_persisted",
            ],
        },
        "markets": {
            "families": _market_catalog(payload, maturation),
            "source": "ACTIVE_PERSISTED_ROWS+MATURATION_REPORTS",
        },
        "research_lab": {
            "selected_fixture": detail,
            "selected_family": selected_family,
            "runtime_model_version": payload.get("model_version"),
            "pipeline_version": payload.get("version"),
            "firewall": {
                "decision_weight": 0.0,
                "production_promotion_allowed": False,
                "canonical_bet_logic_changed": False,
                "model_weights_changed": False,
                "provider_requests_added": 0,
                "strict_close_changed": False,
            },
        },
        "my_edge": {
            "system_alerts": _system_alerts(waiting_price, waiting_xi, maturation),
            "storage_mode": "DEVICE_LOCAL_PREVIEW",
            "note": "Saved signals and tracked markets are stored only in this browser during preview wiring.",
        },
        "control_tower": {
            "status": control.get("status"),
            "runtime_generated_at_utc": control.get("generated_at_utc") or payload.get("generated_at_utc"),
            "runtime_generated_at_local": control.get("generated_at_local") or payload.get("generated_at_local"),
            "pipeline_version": control.get("pipeline_version") or payload.get("version"),
            "model_version": control.get("model_version") or payload.get("model_version"),
            "system_health": health,
            "pipeline": pipeline,
            "errors": errors,
            "maturation": {
                "status": maturation.get("status"),
                "families": list(maturation.get("families") or []),
                "monitoring": monitoring,
                "snapshot_generated_at_utc": maturity_snapshot.get("generated_at_utc"),
                "reports_loaded": maturity_snapshot.get("reports_loaded"),
                "reports_expected": maturity_snapshot.get("reports_expected"),
                "errors": maturity_snapshot.get("errors") if isinstance(maturity_snapshot.get("errors"), dict) else {},
            },
        },
        "provider_requests_added": 0,
        "canonical_bet_logic_changed": False,
        "model_weights_changed": False,
        "production_promotion_allowed": False,
    }


async def preview_data(request: Request) -> JSONResponse:
    token = supabase_auth_v4.bearer_token(request.headers.get("authorization"))
    if not token:
        return JSONResponse({"error": "AUTH_REQUIRED"}, status_code=401)

    entitlement = await asyncio.to_thread(subscription_entitlements_v4.resolve_entitlement, token)
    if not entitlement.get("ok") or not entitlement.get("authenticated"):
        return JSONResponse({"error": entitlement.get("status") or "AUTH_REQUIRED"}, status_code=401)

    is_owner = bool(entitlement.get("owner")) or (entitlement.get("user") or {}).get("role") == "OWNER"
    is_pro = str(entitlement.get("effective_plan") or "").upper() == subscription_entitlements_v4.PRO_PLAN
    if not (is_owner or is_pro):
        return JSONResponse({"error": "PREVIEW_REQUIRES_PRO"}, status_code=403)

    try:
        payload = await asyncio.to_thread(persistence_base.load_latest_pipeline_payload)
    except Exception as exc:
        return JSONResponse({"error": "PREVIEW_DATA_UNAVAILABLE", "detail": str(exc)[:200]}, status_code=503)
    if not isinstance(payload, dict):
        return JSONResponse({"error": "NO_PERSISTED_PIPELINE_RUN"}, status_code=503)

    result = build_preview_payload(payload)
    result["user"] = entitlement.get("user")
    result["owner"] = is_owner
    result["effective_plan"] = entitlement.get("effective_plan")
    return JSONResponse(result)
