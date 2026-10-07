from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import httpx

from mcp_gateway import supabase_auth_v4

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_SUBSCRIBER_SAVED_ITEMS_V4_1.0.0"
TABLE = "subscriber_saved_items"
REQUEST_TIMEOUT_SECONDS = 4.0
ALLOWED_TYPES = {"MATCH", "BET", "LEAN", "WATCH"}


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text(value: Any, *, limit: int = 180) -> str | None:
    if value in (None, ""):
        return None
    out = " ".join(str(value).strip().split())
    return out[:limit] or None


def _number(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _auth_user(access_token: str) -> dict[str, Any]:
    result = supabase_auth_v4.verify_access_token(access_token)
    if not result.get("ok"):
        return {
            "ok": False,
            "status": result.get("status") or "AUTH_REQUIRED",
            "user": None,
        }
    user = _dict(result.get("user"))
    user_id = _text(user.get("id"), limit=64)
    if not user_id:
        return {"ok": False, "status": "AUTH_INVALID_USER_PAYLOAD", "user": None}
    return {"ok": True, "status": "AUTHENTICATED", "user": user}


def _item_key(item: dict[str, Any]) -> str:
    item_type = str(item.get("item_type") or "").upper()
    fixture_id = int(item.get("fixture_id") or 0)
    identity = {
        "item_type": item_type,
        "fixture_id": fixture_id,
        "market_family": _text(item.get("market_family"), limit=80),
        "market_name": _text(item.get("market_name"), limit=120),
        "selection": _text(item.get("selection"), limit=120),
        "line": _number(item.get("line")),
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]
    return f"{item_type.lower()}:{fixture_id}:{digest}"


def sanitize_item(item: dict[str, Any]) -> dict[str, Any]:
    raw = _dict(item)
    item_type = str(raw.get("item_type") or "").strip().upper()
    if item_type not in ALLOWED_TYPES:
        raise ValueError("ITEM_TYPE_NOT_ALLOWED")
    try:
        fixture_id = int(raw.get("fixture_id") or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError("VALID_FIXTURE_ID_REQUIRED") from exc
    if fixture_id <= 0:
        raise ValueError("VALID_FIXTURE_ID_REQUIRED")

    payload_raw = _dict(raw.get("payload"))
    payload = {
        "home_team": _text(payload_raw.get("home_team"), limit=120),
        "away_team": _text(payload_raw.get("away_team"), limit=120),
        "league": _text(payload_raw.get("league"), limit=120),
        "kickoff": _text(payload_raw.get("kickoff"), limit=80),
        "classification": _text(payload_raw.get("classification"), limit=20),
        "tier": _text(payload_raw.get("tier"), limit=8),
        "reason_display": _text(payload_raw.get("reason_display"), limit=280),
        "price_display": _text(payload_raw.get("price_display"), limit=100),
    }
    payload = {key: value for key, value in payload.items() if value is not None}

    normalized = {
        "item_type": item_type,
        "fixture_id": fixture_id,
        "market_family": _text(raw.get("market_family"), limit=80),
        "market_name": _text(raw.get("market_name"), limit=120),
        "selection": _text(raw.get("selection"), limit=120),
        "line": _number(raw.get("line")),
        "source_snapshot_at": _text(raw.get("source_snapshot_at"), limit=80),
        "payload": payload,
    }
    normalized["item_key"] = _item_key(normalized)
    return normalized


def _config() -> dict[str, Any]:
    config = supabase_auth_v4.auth_config()
    if not config.get("configured"):
        raise RuntimeError("AUTH_NOT_CONFIGURED")
    return config


def _headers(config: dict[str, Any], access_token: str) -> dict[str, str]:
    return {
        "apikey": str(config["publishable_key"]),
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def list_saved(access_token: str, *, client: httpx.Client | None = None) -> dict[str, Any]:
    auth = _auth_user(access_token)
    if not auth["ok"]:
        return {
            "ok": False,
            "status": auth["status"],
            "rows": [],
            "provider_requests_added": 0,
        }
    config = _config()
    user_id = str(_dict(auth.get("user")).get("id"))
    owned = client is None
    http = client or httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS, follow_redirects=False)
    try:
        response = http.get(
            f"{config['project_url']}/rest/v1/{TABLE}",
            headers=_headers(config, access_token),
            params={
                "select": (
                    "item_key,item_type,fixture_id,market_family,market_name,"
                    "selection,line,source_snapshot_at,payload,created_at,updated_at"
                ),
                "user_id": f"eq.{user_id}",
                "order": "updated_at.desc",
                "limit": "200",
            },
        )
        if response.status_code != 200:
            return {
                "ok": False,
                "status": f"SAVED_ITEMS_LOOKUP_REJECTED_{response.status_code}",
                "rows": [],
                "provider_requests_added": 0,
            }
        payload = response.json()
        rows = payload if isinstance(payload, list) else []
        return {
            "ok": True,
            "status": "SAVED_ITEMS_READY",
            "rows": [row for row in rows if isinstance(row, dict)],
            "provider_requests_added": 0,
        }
    except (httpx.HTTPError, ValueError):
        return {
            "ok": False,
            "status": "SAVED_ITEMS_LOOKUP_UNAVAILABLE",
            "rows": [],
            "provider_requests_added": 0,
        }
    finally:
        if owned:
            http.close()


def save_item(
    access_token: str,
    item: dict[str, Any],
    *,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    auth = _auth_user(access_token)
    if not auth["ok"]:
        return {
            "ok": False,
            "status": auth["status"],
            "row": None,
            "provider_requests_added": 0,
        }
    try:
        normalized = sanitize_item(item)
    except ValueError as exc:
        return {
            "ok": False,
            "status": str(exc),
            "row": None,
            "provider_requests_added": 0,
        }

    config = _config()
    user_id = str(_dict(auth.get("user")).get("id"))
    body = {"user_id": user_id, **normalized}
    owned = client is None
    http = client or httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS, follow_redirects=False)
    try:
        response = http.post(
            f"{config['project_url']}/rest/v1/{TABLE}",
            headers={
                **_headers(config, access_token),
                "Prefer": "resolution=merge-duplicates,return=representation",
            },
            params={"on_conflict": "user_id,item_key"},
            json=body,
        )
        if response.status_code not in {200, 201}:
            return {
                "ok": False,
                "status": f"SAVED_ITEM_WRITE_REJECTED_{response.status_code}",
                "row": None,
                "provider_requests_added": 0,
            }
        payload = response.json()
        rows = payload if isinstance(payload, list) else []
        row = rows[0] if rows and isinstance(rows[0], dict) else None
        return {
            "ok": True,
            "status": "SAVED_ITEM_UPSERTED",
            "row": row or body,
            "provider_requests_added": 0,
        }
    except (httpx.HTTPError, ValueError):
        return {
            "ok": False,
            "status": "SAVED_ITEM_WRITE_UNAVAILABLE",
            "row": None,
            "provider_requests_added": 0,
        }
    finally:
        if owned:
            http.close()


def delete_item(
    access_token: str,
    item_key: str,
    *,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    auth = _auth_user(access_token)
    if not auth["ok"]:
        return {"ok": False, "status": auth["status"], "provider_requests_added": 0}
    key = _text(item_key, limit=220)
    if not key:
        return {
            "ok": False,
            "status": "ITEM_KEY_REQUIRED",
            "provider_requests_added": 0,
        }

    config = _config()
    user_id = str(_dict(auth.get("user")).get("id"))
    owned = client is None
    http = client or httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS, follow_redirects=False)
    try:
        response = http.delete(
            f"{config['project_url']}/rest/v1/{TABLE}",
            headers={**_headers(config, access_token), "Prefer": "return=minimal"},
            params={"user_id": f"eq.{user_id}", "item_key": f"eq.{key}"},
        )
        if response.status_code not in {200, 204}:
            return {
                "ok": False,
                "status": f"SAVED_ITEM_DELETE_REJECTED_{response.status_code}",
                "provider_requests_added": 0,
            }
        return {
            "ok": True,
            "status": "SAVED_ITEM_DELETED",
            "item_key": key,
            "provider_requests_added": 0,
        }
    except httpx.HTTPError:
        return {
            "ok": False,
            "status": "SAVED_ITEM_DELETE_UNAVAILABLE",
            "provider_requests_added": 0,
        }
    finally:
        if owned:
            http.close()


def contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "table": TABLE,
        "allowed_types": sorted(ALLOWED_TYPES),
        "rls_required": True,
        "auth_uid_owns_rows": True,
        "model_input_allowed": False,
        "provider_requests_added": 0,
        "canonical_bet_logic_changed": False,
        "model_weights_changed": False,
        "production_promotion_allowed": False,
    }
