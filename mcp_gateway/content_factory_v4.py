from __future__ import annotations

import hashlib
import json
from typing import Any

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_CONTENT_FACTORY_V4_1.0.0"
MAX_PACKAGES = 12


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _rows(view: Any) -> list[dict[str, Any]]:
    rows = _dict(view).get("rows")
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _first(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return None


def _float(value: Any) -> float | None:
    try:
        number = float(value)
        return number if number == number else None
    except (TypeError, ValueError):
        return None


def _prob(value: Any) -> float | None:
    number = _float(value)
    return number if number is not None and 0.0 <= number <= 1.0 else None


def _pct(value: float | None) -> str | None:
    return f"{value * 100:.1f}%" if value is not None else None


def _pp(value: float | None) -> str | None:
    return f"{value:+.1f} pp" if value is not None else None


def _teams(row: dict[str, Any]) -> tuple[str | None, str | None]:
    home = _first(row, "home", "home_team")
    away = _first(row, "away", "away_team")
    return (str(home) if home else None, str(away) if away else None)


def _market(row: dict[str, Any]) -> str | None:
    value = _first(row, "market", "market_family")
    return str(value) if value else None


def _selection(row: dict[str, Any]) -> str | None:
    value = row.get("selection")
    line = row.get("line")
    if value is None and line is None:
        return None
    parts = [str(item) for item in (value, line) if item is not None and item != ""]
    return " ".join(parts) or None


def _blockers(row: dict[str, Any]) -> list[str]:
    raw = row.get("blockers")
    if not isinstance(raw, list):
        return []
    return [str(item)[:160] for item in raw if item is not None and str(item).strip()][:5]


def _identity(row: dict[str, Any], format_name: str) -> str:
    parts = (
        format_name,
        str(row.get("fixture_id") or ""),
        str(_market(row) or ""),
        str(_selection(row) or ""),
        str(row.get("kickoff") or ""),
    )
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:20]


def _facts(row: dict[str, Any]) -> dict[str, Any]:
    home, away = _teams(row)
    p_market = _prob(_first(row, "p_market_fair", "p_market_devig"))
    p_model = _prob(_first(row, "p_model_calibrated", "p_calibrated", "calibrated_probability", "model_probability_calibrated"))
    edge_pp = _float(_first(row, "calibrated_edge_pp", "prob_edge_pp"))
    if edge_pp is None and p_market is not None and p_model is not None:
        edge_pp = (p_model - p_market) * 100.0
    return {
        "fixture_id": row.get("fixture_id"),
        "home": home,
        "away": away,
        "league": row.get("league"),
        "kickoff": row.get("kickoff"),
        "market": _market(row),
        "selection": _selection(row),
        "price": row.get("price"),
        "bookmaker": row.get("bookmaker"),
        "market_probability": p_market,
        "market_probability_display": _pct(p_market),
        "model_probability": p_model,
        "model_probability_display": _pct(p_model),
        "edge_pp": edge_pp,
        "edge_display": _pp(edge_pp),
        "model_signal": row.get("model_signal"),
        "model_signal_score": row.get("model_signal_score"),
        "execution_status": row.get("execution_status"),
        "reason": row.get("reason"),
        "blockers": _blockers(row),
        "price_source": row.get("price_resolution_source"),
        "provider_update": row.get("price_resolution_provider_update"),
    }


def _model_vs_market(row: dict[str, Any]) -> dict[str, Any] | None:
    facts = _facts(row)
    if not all((facts["home"], facts["away"], facts["market"], facts["market_probability_display"], facts["model_probability_display"], facts["edge_display"])):
        return None
    fixture = f'{facts["home"]} vs {facts["away"]}'
    market = str(facts["market"])
    selection = f' · {facts["selection"]}' if facts["selection"] else ""
    price = f' @ {facts["price"]}' if facts["price"] is not None else ""
    en = {
        "hook": f'Market {facts["market_probability_display"]}. Soccer Edge {facts["model_probability_display"]}.' ,
        "voiceover": (
            f'{fixture}. {market}{selection}{price}. The de-vigged market probability is {facts["market_probability_display"]}. '
            f'The calibrated Soccer Edge probability is {facts["model_probability_display"]}, a gap of {facts["edge_display"]}. '
            f'Execution status: {facts["execution_status"] or "not verified"}. Soccer Edge: model versus market.'
        ),
        "caption": f'{fixture} — {market}{selection}. Market {facts["market_probability_display"]} vs model {facts["model_probability_display"]}. Gap {facts["edge_display"]}. #SoccerEdge #ModelVsMarket',
        "x_post": f'{fixture}\n{market}{selection}{price}\nMarket fair: {facts["market_probability_display"]}\nSoccer Edge: {facts["model_probability_display"]}\nGap: {facts["edge_display"]}\nStatus: {facts["execution_status"] or "N/V"}',
    }
    es = {
        "hook": f'Mercado {facts["market_probability_display"]}. Soccer Edge {facts["model_probability_display"]}.',
        "voiceover": (
            f'{fixture}. {market}{selection}{price}. La probabilidad de-vig del mercado es {facts["market_probability_display"]}. '
            f'La probabilidad calibrada de Soccer Edge es {facts["model_probability_display"]}, una diferencia de {facts["edge_display"]}. '
            f'Estado de ejecución: {facts["execution_status"] or "no verificado"}. Soccer Edge: modelo contra mercado.'
        ),
        "caption": f'{fixture} — {market}{selection}. Mercado {facts["market_probability_display"]} vs modelo {facts["model_probability_display"]}. Diferencia {facts["edge_display"]}. #SoccerEdge #ModeloVsMercado',
        "x_post": f'{fixture}\n{market}{selection}{price}\nMercado fair: {facts["market_probability_display"]}\nSoccer Edge: {facts["model_probability_display"]}\nDiferencia: {facts["edge_display"]}\nEstado: {facts["execution_status"] or "N/V"}',
    }
    return _package(row, "MODEL_VS_MARKET", facts, en, es)


def _why_passed(row: dict[str, Any]) -> dict[str, Any] | None:
    facts = _facts(row)
    status = str(facts.get("execution_status") or "").upper()
    if status not in {"WAIT_PRICE", "WAIT_FRESH_QUOTE", "WAIT_XI", "WAIT_GK", "WAIT_AVAILABILITY", "NO_BET"}:
        return None
    if not facts["home"] or not facts["away"]:
        return None
    fixture = f'{facts["home"]} vs {facts["away"]}'
    reason = str(facts["reason"] or "No verified execution reason available")
    blocker_text = "; ".join(facts["blockers"]) if facts["blockers"] else reason
    en = {
        "hook": f'Why Soccer Edge did not fire on {fixture}.',
        "voiceover": f'{fixture}. Status: {status}. We do not force a pick when the evidence is incomplete. Verified blocker: {blocker_text}. Passing is a decision too.',
        "caption": f'{fixture}: {status}. {blocker_text}. No forced picks. #SoccerEdge #NoBet',
        "x_post": f'{fixture}\nStatus: {status}\nWhy we passed: {blocker_text}\nNo forced picks.',
    }
    es = {
        "hook": f'Por qué Soccer Edge no disparó en {fixture}.',
        "voiceover": f'{fixture}. Estado: {status}. No forzamos una apuesta cuando la evidencia está incompleta. Bloqueador verificado: {blocker_text}. Pasar también es una decisión.',
        "caption": f'{fixture}: {status}. {blocker_text}. Sin picks forzados. #SoccerEdge #NoBet',
        "x_post": f'{fixture}\nEstado: {status}\nPor qué pasamos: {blocker_text}\nSin picks forzados.',
    }
    return _package(row, "WHY_WE_PASSED", facts, en, es)


def _package(row: dict[str, Any], format_name: str, facts: dict[str, Any], en: dict[str, str], es: dict[str, str]) -> dict[str, Any]:
    return {
        "content_id": _identity(row, format_name),
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "format": format_name,
        "facts": facts,
        "copy": {"en": en, "es": es},
        "render_spec": {
            "width": 1080,
            "height": 1920,
            "fps": 30,
            "duration_seconds": 30,
            "safe_zone": {"top": 180, "bottom": 260, "left": 80, "right": 80},
            "brand": "Soccer Edge",
            "visual_system": "DARK_MARKET_TERMINAL",
            "panels": ["HOOK", "FIXTURE", "MARKET_VS_MODEL", "DECISION", "CTA"],
        },
        "platforms": {
            "tiktok": {"aspect_ratio": "9:16", "duration_seconds": 30},
            "instagram_reels": {"aspect_ratio": "9:16", "duration_seconds": 30},
            "youtube_shorts": {"aspect_ratio": "9:16", "duration_seconds": 30},
            "x": {"card_aspect_ratio": "16:9", "text": True},
        },
        "evidence_policy": {
            "numbers_from_persisted_row_only": True,
            "ai_may_modify_numeric_facts": False,
            "provider_requests_added": 0,
            "production_promotion_allowed": False,
        },
    }


def build_content_packages(product_payload: dict[str, Any], *, limit: int = MAX_PACKAGES) -> dict[str, Any]:
    views = _dict(product_payload.get("views"))
    output: list[dict[str, Any]] = []
    seen: set[str] = set()

    for row in _rows(views.get("strong_sport_signals")) + _rows(views.get("value_plays")):
        package = _model_vs_market(row)
        if package and package["content_id"] not in seen:
            seen.add(package["content_id"])
            output.append(package)
            if len(output) >= max(0, int(limit)):
                break

    if len(output) < max(0, int(limit)):
        wait_rows = _rows(views.get("waiting_for_price")) + _rows(views.get("waiting_for_xi")) + _rows(views.get("todays_slate"))
        for row in wait_rows:
            package = _why_passed(row)
            if package and package["content_id"] not in seen:
                seen.add(package["content_id"])
                output.append(package)
                if len(output) >= max(0, int(limit)):
                    break

    counts: dict[str, int] = {}
    for item in output:
        counts[item["format"]] = counts.get(item["format"], 0) + 1

    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": "CONTENT_PACKAGES_READY" if output else "NO_ELIGIBLE_CONTENT_ROWS",
        "packages": output,
        "package_count": len(output),
        "format_counts": counts,
        "provider_requests_added": 0,
        "canonical_bet_logic_changed": False,
        "model_weights_changed": False,
        "production_promotion_allowed": False,
    }


def dumps(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
