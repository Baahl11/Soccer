from __future__ import annotations

import os
from typing import Any

import httpx

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_SUPABASE_AUTH_V4_1.0.0"
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
        "authorization_metadata_source": "NONE_UNTIL_V219",
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
