from __future__ import annotations

import html
from datetime import datetime, timezone
from typing import Any

import httpx

from mcp_gateway import supabase_auth_v4

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_SUBSCRIPTION_ENTITLEMENTS_V4_1.0.0"
REQUEST_TIMEOUT_SECONDS = 3.0

FREE_PLAN = "FREE"
PRO_PLAN = "PRO"
PRO_ACTIVE_STATUSES = frozenset({"ACTIVE", "TRIALING"})

FREE_FEATURE_IDS = (
    "verified_slate",
    "market_readiness_radar",
    "model_maturity_overview",
    "public_intelligence",
)

PRO_FEATURE_IDS = FREE_FEATURE_IDS + (
    "strong_signal_desk",
    "calibrated_value_plays",
    "advanced_market_families",
    "premium_match_detail",
    "verified_performance_history",
    "favorites_alerts",
)

FEATURE_LABELS = {
    "verified_slate": "Today's verified slate",
    "market_readiness_radar": "Market readiness radar",
    "model_maturity_overview": "Model maturity overview",
    "public_intelligence": "Read-only public intelligence",
    "strong_signal_desk": "Strong signal desk",
    "calibrated_value_plays": "Calibrated value plays",
    "advanced_market_families": "Advanced market families",
    "premium_match_detail": "Premium match detail",
    "verified_performance_history": "Verified performance history",
    "favorites_alerts": "Favorites and alerts",
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_datetime(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def plan_contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "default_plan": FREE_PLAN,
        "billing_enabled": False,
        "entitlements_enforced": True,
        "authorization_source": "SUPABASE_RLS_SUBSCRIPTION_ENTITLEMENTS",
        "missing_row_policy": "FREE",
        "free_features": [FEATURE_LABELS[key] for key in FREE_FEATURE_IDS],
        "pro_features": [FEATURE_LABELS[key] for key in PRO_FEATURE_IDS if key not in FREE_FEATURE_IDS],
        "provider_requests_added": 0,
    }


def effective_plan(row: dict[str, Any] | None, *, now: datetime | None = None) -> tuple[str, str]:
    if not isinstance(row, dict):
        return FREE_PLAN, "DEFAULT_FREE_NO_ENTITLEMENT_ROW"

    plan = str(row.get("plan") or FREE_PLAN).strip().upper()
    status = str(row.get("status") or "INACTIVE").strip().upper()
    if plan != PRO_PLAN:
        return FREE_PLAN, "PERSISTED_FREE"
    if status not in PRO_ACTIVE_STATUSES:
        return FREE_PLAN, f"PRO_STATUS_{status or 'INACTIVE'}"

    valid_until = _parse_datetime(row.get("valid_until"))
    current = (now or _utc_now()).astimezone(timezone.utc)
    if valid_until is not None and valid_until <= current:
        return FREE_PLAN, "PRO_EXPIRED"
    return PRO_PLAN, f"PRO_{status}"


def feature_access(plan: str) -> dict[str, bool]:
    effective = str(plan or FREE_PLAN).strip().upper()
    allowed = set(PRO_FEATURE_IDS if effective == PRO_PLAN else FREE_FEATURE_IDS)
    return {key: key in allowed for key in FEATURE_LABELS}


def resolve_entitlement(
    access_token: str,
    *,
    client: httpx.Client | None = None,
    auth_client: httpx.Client | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    auth_result = supabase_auth_v4.verify_access_token(access_token, client=auth_client)
    if not auth_result.get("ok"):
        return {
            "ok": False,
            "status": auth_result.get("status") or "AUTH_REQUIRED",
            "authenticated": False,
            "effective_plan": None,
            "feature_access": {},
            "billing_enabled": False,
            "entitlements_enforced": True,
            "provider_requests_added": 0,
        }

    config = supabase_auth_v4.auth_config()
    if not config.get("configured"):
        return {
            "ok": False,
            "status": "AUTH_NOT_CONFIGURED",
            "authenticated": True,
            "effective_plan": None,
            "feature_access": {},
            "billing_enabled": False,
            "entitlements_enforced": True,
            "provider_requests_added": 0,
        }

    user = auth_result.get("user") if isinstance(auth_result.get("user"), dict) else {}
    user_id = str(user.get("id") or "").strip()
    if not user_id:
        return {
            "ok": False,
            "status": "AUTH_INVALID_USER_PAYLOAD",
            "authenticated": False,
            "effective_plan": None,
            "feature_access": {},
            "billing_enabled": False,
            "entitlements_enforced": True,
            "provider_requests_added": 0,
        }

    owned_client = client is None
    http = client or httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS, follow_redirects=False)
    try:
        response = http.get(
            f"{config['project_url']}/rest/v1/subscription_entitlements",
            headers={
                "apikey": str(config["publishable_key"]),
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json",
            },
            params={
                "select": "user_id,plan,status,source,starts_at,valid_until,updated_at",
                "user_id": f"eq.{user_id}",
                "limit": "1",
            },
        )
        if response.status_code != 200:
            return {
                "ok": False,
                "status": f"ENTITLEMENT_LOOKUP_REJECTED_{response.status_code}",
                "authenticated": True,
                "user": {"id": user_id, "email": user.get("email")},
                "effective_plan": FREE_PLAN,
                "feature_access": feature_access(FREE_PLAN),
                "billing_enabled": False,
                "entitlements_enforced": True,
                "provider_requests_added": 0,
            }

        payload = response.json()
        rows = payload if isinstance(payload, list) else []
        row = rows[0] if rows and isinstance(rows[0], dict) else None
        plan, reason = effective_plan(row, now=now)
        return {
            "ok": True,
            "status": "ENTITLEMENT_RESOLVED",
            "authenticated": True,
            "user": {"id": user_id, "email": user.get("email")},
            "persisted_entitlement": row,
            "effective_plan": plan,
            "effective_plan_reason": reason,
            "feature_access": feature_access(plan),
            "billing_enabled": False,
            "entitlements_enforced": True,
            "authorization_source": "SUPABASE_RLS_SUBSCRIPTION_ENTITLEMENTS",
            "provider_requests_added": 0,
        }
    except (httpx.HTTPError, ValueError):
        return {
            "ok": False,
            "status": "ENTITLEMENT_LOOKUP_UNAVAILABLE",
            "authenticated": True,
            "user": {"id": user_id, "email": user.get("email")},
            "effective_plan": FREE_PLAN,
            "feature_access": feature_access(FREE_PLAN),
            "billing_enabled": False,
            "entitlements_enforced": True,
            "provider_requests_added": 0,
        }
    finally:
        if owned_client:
            http.close()


def can_access(entitlement: dict[str, Any], feature_id: str) -> bool:
    access = entitlement.get("feature_access") if isinstance(entitlement, dict) else {}
    return bool(access.get(str(feature_id))) if isinstance(access, dict) else False


def _esc(value: Any) -> str:
    return "N/V" if value is None else html.escape(str(value), quote=True)


def render_fragment() -> str:
    auth_ready = bool(supabase_auth_v4.auth_config().get("configured"))
    status = "ENTITLEMENTS_READY" if auth_ready else "AUTH_REQUIRED"
    return f"""
<style>
.v219{{margin:14px 0;border:1px solid #1d3c52;background:#081722;border-radius:14px;padding:16px 18px;display:flex;justify-content:space-between;gap:18px;align-items:center}}
.v219 h2{{margin:3px 0 5px;font-size:18px}}.v219 p{{margin:0;color:#7890a5;font-size:11px;line-height:1.5;max-width:860px}}
.v219-badge{{flex:0 0 auto;padding:6px 9px;border-radius:999px;font-size:9px;font-weight:900;border:1px solid #155c48;color:#63ecc1;background:#0c382d}}
@media(max-width:620px){{.v219{{display:block}}.v219-badge{{display:inline-block;margin-top:10px}}}}
</style>
<section class="v219" id="subscription-entitlements"><div><div class="eyebrow">SUBSCRIPTION ENTITLEMENTS · V219</div><h2>Free by default. Pro by persisted entitlement.</h2><p>Supabase RLS is authoritative. Signed-in users can read only their own entitlement; browser clients cannot create or modify Pro access. Billing remains disabled until V220.</p></div><span class="v219-badge">{_esc(status)}</span></section>
"""
