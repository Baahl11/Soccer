"""RLS-isolated analyst notes. Product state, not model input or value signal."""
from __future__ import annotations
from typing import Any
import httpx
from mcp_gateway import supabase_auth_v4

TABLE = "subscriber_match_notes"
MAX_TEXT = 2000

def validate_note(payload: dict[str, Any]) -> tuple[int, str]:
    if not isinstance(payload, dict):
        raise ValueError("NOTE_PAYLOAD_INVALID")
    fid = payload.get("fixture_id")
    if type(fid) is not int or not 0 < fid < 10**12:
        raise ValueError("NOTE_FIXTURE_ID_INVALID")
    note = payload.get("note_text")
    if not isinstance(note, str) or not note.strip() or len(note.strip()) > MAX_TEXT:
        raise ValueError("NOTE_TEXT_LENGTH_INVALID")
    return fid, note.strip()

def _http_config(token: str) -> tuple[str, dict[str, str]]:
    conf = supabase_auth_v4.auth_config()
    if not conf.get("configured") or not token:
        raise RuntimeError("AUTH_NOT_CONFIGURED")
    headers = {
        "apikey": str(conf["publishable_key"]),
        "Authorization": "Bearer " + token,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    return str(conf["project_url"]).rstrip("/") + "/rest/v1/" + TABLE, headers

def operate(
    method: str, token: str, user_id: str, *, fixture_id: int | None = None,
    body: dict[str, Any] | None = None, client: httpx.Client | None = None,
) -> dict[str, Any]:
    """User identity MUST be verified by the existing subscription/auth route."""
    if not isinstance(user_id, str) or len(user_id) < 20 or len(user_id) > 64:
        return {"ok": False, "status": "AUTH_INVALID_USER", "provider_requests_added": 0}
    method = method.upper()
    if method not in {"GET", "POST", "DELETE"}:
        return {"ok": False, "status": "METHOD_NOT_ALLOWED", "provider_requests_added": 0}
    try:
        if method == "POST":
            fid, text = validate_note(body or {})
        else:
            fid = fixture_id
            if fid is not None and (type(fid) is not int or not 0 < fid < 10**12):
                raise ValueError("NOTE_FIXTURE_ID_INVALID")
            if method == "DELETE" and fid is None:
                raise ValueError("NOTE_FIXTURE_ID_REQUIRED")
            text = None
        url, headers = _http_config(token)
    except (ValueError, RuntimeError) as exc:
        return {"ok": False, "status": str(exc), "provider_requests_added": 0}

    owned = client is None
    http = client or httpx.Client(timeout=12, follow_redirects=False)
    params: dict[str, str] = {"user_id": "eq." + user_id}
    if fid is not None:
        params["fixture_id"] = "eq." + str(fid)
    try:
        if method == "GET":
            response = http.get(url, headers=headers, params={
                **params, "select": "fixture_id,note_text,created_at,updated_at",
                "order": "updated_at.desc", "limit": "200",
            })
            if response.status_code != 200:
                raise RuntimeError("NOTES_READ_REJECTED_" + str(response.status_code))
            rows = response.json()
            if not isinstance(rows, list):
                raise RuntimeError("NOTES_INVALID_RESPONSE")
            return {"ok": True, "status": "NOTES_READY",
                    "rows": [r for r in rows if isinstance(r, dict)],
                    "provider_requests_added": 0, "model_inputs_changed": False}
        if method == "POST":
            response = http.post(
                url, headers={**headers, "Prefer": "resolution=merge-duplicates,return=representation"},
                params={"on_conflict": "user_id,fixture_id"},
                json={"user_id": user_id, "fixture_id": fid, "note_text": text},
            )
            if response.status_code not in {200, 201}:
                raise RuntimeError("NOTE_WRITE_REJECTED_" + str(response.status_code))
            rows = response.json()
            return {"ok": True, "status": "NOTE_UPSERTED",
                    "row": rows[0] if isinstance(rows, list) and rows else None,
                    "provider_requests_added": 0, "model_inputs_changed": False}
        response = http.delete(url, headers={**headers, "Prefer": "return=minimal"}, params=params)
        if response.status_code not in {200, 204}:
            raise RuntimeError("NOTE_DELETE_REJECTED_" + str(response.status_code))
        return {"ok": True, "status": "NOTE_DELETED", "fixture_id": fid,
                "provider_requests_added": 0, "model_inputs_changed": False}
    except (httpx.HTTPError, RuntimeError, ValueError) as exc:
        status = str(exc) if isinstance(exc, RuntimeError) else "NOTES_PROVIDER_UNAVAILABLE"
        return {"ok": False, "status": status, "provider_requests_added": 0}
    finally:
        if owned:
            http.close()
