from __future__ import annotations

import asyncio
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

from mcp_gateway import persistence as persistence_base
from mcp_gateway import product_views_v4
from mcp_gateway import subscriber_ui_contract_v231
from mcp_gateway import subscription_entitlements_v4
from mcp_gateway import supabase_auth_v4

SCHEMA_VERSION = "1.1.0"
MODEL_VERSION = "SOCCER_SUBSCRIBER_PREVIEW_DATA_V231"


def _rows(node: Any) -> list[dict[str, Any]]:
    if not isinstance(node, dict):
        return []
    value = node.get("rows")
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


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
            row for row in feed
            if (row.get("model") or {}).get("probability") is not None
            and (row.get("pricing") or {}).get("market_probability") is not None
            and (row.get("pricing") or {}).get("edge_pp") is not None
        ),
        None,
    )

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
        "edge_feed": {
            "rows": feed[:25],
            "total": len(feed),
        },
        "control_tower": {
            "status": control.get("status"),
            "runtime_generated_at_utc": control.get("generated_at_utc") or payload.get("generated_at_utc"),
            "runtime_generated_at_local": control.get("generated_at_local") or payload.get("generated_at_local"),
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
