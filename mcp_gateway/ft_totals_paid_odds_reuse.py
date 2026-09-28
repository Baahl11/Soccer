from __future__ import annotations

from collections import defaultdict
from typing import Any, Awaitable, Callable

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_FT_TOTALS_PAID_ODDS_REUSE_V1.0.0"


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def is_strict_ft_totals_market(row: dict[str, Any]) -> bool:
    name = _norm(row.get("market"))
    if name not in {"goals over/under", "over/under"}:
        return False
    if any(
        token in name
        for token in (
            "first half",
            "1st half",
            "second half",
            "2nd half",
            "home team",
            "away team",
        )
    ):
        return False
    return True


def collect_fixture_markets(
    captured: dict[int, list[dict[str, Any]]],
    fixture_id: int,
    markets: list[dict[str, Any]],
    status: str,
) -> None:
    if str(status or "").upper() != "PRICE_API_RESOLVED":
        return
    exact = [dict(row) for row in markets if isinstance(row, dict) and is_strict_ft_totals_market(row)]
    if not exact:
        return
    captured[int(fixture_id)].extend(exact)


def install_fetch_observer(
    resolver_module: Any,
) -> tuple[Callable[..., Awaitable[Any]], dict[int, list[dict[str, Any]]]]:
    original = resolver_module._fetch_fixture_odds
    captured: dict[int, list[dict[str, Any]]] = defaultdict(list)

    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        result = await original(*args, **kwargs)
        try:
            fixture_id = int(args[1] if len(args) > 1 else kwargs.get("fixture_id"))
            markets, _used, status, _remaining = result
            if isinstance(markets, list):
                collect_fixture_markets(captured, fixture_id, markets, str(status or ""))
        except (TypeError, ValueError, IndexError):
            pass
        return result

    resolver_module._fetch_fixture_odds = wrapped
    return original, captured


def restore_fetch_observer(resolver_module: Any, original: Callable[..., Awaitable[Any]]) -> None:
    resolver_module._fetch_fixture_odds = original


def _fixture_id(event: dict[str, Any]) -> int | None:
    fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
    value = fixture.get("fixture_id")
    if value is None:
        value = event.get("fixture_id")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _market_key(row: dict[str, Any]) -> tuple[Any, Any, str, str]:
    return (
        row.get("bookmaker_id"),
        row.get("market_id"),
        str(row.get("provider_update") or ""),
        str(row.get("values") or ""),
    )


def attach(payload: dict[str, Any], captured: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    events = payload.get("events") if isinstance(payload.get("events"), list) else []
    by_fixture: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if not isinstance(event, dict):
            continue
        fixture_id = _fixture_id(event)
        if fixture_id is not None:
            by_fixture[fixture_id].append(event)

    captured_fixtures = 0
    attached_fixtures = 0
    attached_rows = 0
    already_present_rows = 0
    missing_event_fixtures = 0

    for fixture_id, rows in captured.items():
        exact_rows = [row for row in rows if isinstance(row, dict) and is_strict_ft_totals_market(row)]
        if not exact_rows:
            continue
        captured_fixtures += 1
        candidates = by_fixture.get(int(fixture_id), [])
        if not candidates:
            missing_event_fixtures += 1
            continue

        # Prefer the event that already carries the paid provider market payload.
        event = max(
            candidates,
            key=lambda item: int(
                isinstance(item.get("market"), dict)
                and str((item.get("market") or {}).get("source") or "").upper() == "API_FOOTBALL_ODDS_V3"
            ),
        )
        market = event.get("market") if isinstance(event.get("market"), dict) else {}
        existing = [row for row in (market.get("markets") or []) if isinstance(row, dict)]
        seen = {_market_key(row) for row in existing}
        added_here = 0
        for row in exact_rows:
            key = _market_key(row)
            if key in seen:
                already_present_rows += 1
                continue
            existing.append(row)
            seen.add(key)
            added_here += 1

        market["markets"] = existing
        if not market.get("source"):
            market["source"] = "API_FOOTBALL_ODDS_V3"
        market["ft_totals_paid_odds_reuse"] = {
            "schema_version": SCHEMA_VERSION,
            "captured_from_already_paid_odds": True,
            "provider_requests_added": 0,
            "decision_weight": 0.0,
            "production_promotion_allowed": False,
            "rows_added": added_here,
        }
        event["market"] = market
        if added_here:
            attached_fixtures += 1
            attached_rows += added_here

    result = {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": "FT_TOTALS_PAID_ODDS_REUSE_ACTIVE",
        "captured_fixtures": captured_fixtures,
        "attached_fixtures": attached_fixtures,
        "attached_market_rows": attached_rows,
        "already_present_market_rows": already_present_rows,
        "missing_event_fixtures": missing_event_fixtures,
        "provider_requests_added": 0,
        "decision_weight": 0.0,
        "production_promotion_allowed": False,
        "canonical_bet_logic_changed": False,
        "model_weights_changed": False,
        "policy": "OBSERVE_ALREADY_PAID_PRICE_API_FETCHES; FRESH_PRICE_API_RESOLVED_ONLY; STRICT_FT_TOTALS_ONLY; DEDUPE_BEFORE_PERSISTENCE; ZERO_EXTRA_PROVIDER_CALLS; RESEARCH_ONLY",
    }
    payload["ft_totals_paid_odds_reuse"] = result
    return result
