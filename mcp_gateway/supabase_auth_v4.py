from __future__ import annotations

import html
import os
from typing import Any

import httpx

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_SUPABASE_AUTH_V4_1.0.1"
REQUEST_TIMEOUT_SECONDS = 3.0


def _clean_url(value: str | None) -> str:
    return str(value or "").strip().rstrip("/")


def auth_config() -> dict[str, Any]:
    url = _clean_url(os.getenv("SOCCER_SUPABASE_URL"))
    publishable_key = str(os.getenv("SOCCER_SUPABASE_PUBLISHABLE_KEY") or "").strip()
    configured = bool(url and publishable_key)
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": "AUTH_READY" if configured else "AUTH_NOT_CONFIGURED",
        "configured": configured,
        "project_url": url if configured else None,
        "publishable_key": publishable_key if configured else None,
        "publishable_key_present": bool(publishable_key),
        "secret_key_required": False,
        "service_role_required": False,
        "billing_enabled": False,
        "entitlements_enforced": False,
        "authorization_metadata_source": "DELEGATED_TO_SUBSCRIPTION_ENTITLEMENTS_V4",
        "provider_requests_added": 0,
    }


def public_auth_config() -> dict[str, Any]:
    config = auth_config()
    return {
        "schema_version": config["schema_version"],
        "model_version": config["model_version"],
        "status": config["status"],
        "configured": config["configured"],
        "project_url": config["project_url"],
        "publishable_key": config["publishable_key"],
        "billing_enabled": False,
        "entitlements_enforced": False,
        "provider_requests_added": 0,
    }


def verify_access_token(token: str, *, client: httpx.Client | None = None) -> dict[str, Any]:
    config = auth_config()
    if not config["configured"]:
        return {"ok": False, "status": "AUTH_NOT_CONFIGURED", "user": None}
    value = str(token or "").strip()
    if not value:
        return {"ok": False, "status": "MISSING_ACCESS_TOKEN", "user": None}

    owned_client = client is None
    http = client or httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS, follow_redirects=False)
    try:
        response = http.get(
            f"{config['project_url']}/auth/v1/user",
            headers={
                "apikey": str(config["publishable_key"]),
                "Authorization": f"Bearer {value}",
                "Accept": "application/json",
            },
        )
        if response.status_code != 200:
            return {"ok": False, "status": f"AUTH_REJECTED_{response.status_code}", "user": None}
        payload = response.json()
        if not isinstance(payload, dict) or not payload.get("id"):
            return {"ok": False, "status": "AUTH_INVALID_USER_PAYLOAD", "user": None}
        return {
            "ok": True,
            "status": "AUTHENTICATED",
            "user": {
                "id": payload.get("id"),
                "email": payload.get("email"),
                "aud": payload.get("aud"),
                "role": payload.get("role"),
            },
            "authorization_metadata_ignored": True,
            "billing_enabled": False,
            "entitlements_enforced": False,
        }
    except (httpx.HTTPError, ValueError):
        return {"ok": False, "status": "AUTH_VERIFICATION_UNAVAILABLE", "user": None}
    finally:
        if owned_client:
            http.close()


def bearer_token(authorization_header: str | None) -> str:
    value = str(authorization_header or "").strip()
    if not value.lower().startswith("bearer "):
        return ""
    return value[7:].strip()


def _esc(value: Any) -> str:
    return "N/V" if value is None else html.escape(str(value), quote=True)


def render_fragment() -> str:
    config = auth_config()
    configured = bool(config["configured"])
    status = str(config["status"])
    headline = "Account infrastructure ready" if configured else "Account infrastructure staged"
    body = (
        "Dedicated Supabase Auth configuration is present and user JWT verification is available. Billing remains disabled; subscription authorization is handled separately by V219 entitlements."
        if configured
        else "No dedicated Soccer Edge Supabase project is configured yet. Sign-in stays disabled; existing projects are not reused and no account data is created."
    )
    badge_border = "#155c48" if configured else "#5a431f"
    badge_color = "#63ecc1" if configured else "#f6ca73"
    badge_bg = "#0c382d" if configured else "#362914"
    return f"""
<style>
.v218{{margin:14px 0;border:1px solid #1b3448;background:#08141e;border-radius:14px;padding:16px 18px;display:flex;justify-content:space-between;gap:18px;align-items:center}}
.v218 h2{{margin:3px 0 5px;font-size:18px}}.v218 p{{margin:0;color:#7890a5;font-size:11px;line-height:1.5;max-width:840px}}
.v218-badge{{flex:0 0 auto;padding:6px 9px;border-radius:999px;font-size:9px;font-weight:900;border:1px solid {badge_border};color:{badge_color};background:{badge_bg}}}
@media(max-width:620px){{.v218{{display:block}}.v218-badge{{display:inline-block;margin-top:10px}}}}
</style>
<section class="v218" id="account-readiness"><div><div class="eyebrow">AUTH + ACCOUNTS · V218</div><h2>{_esc(headline)}</h2><p>{_esc(body)}</p></div><span class="v218-badge">{_esc(status)}</span></section>
"""
