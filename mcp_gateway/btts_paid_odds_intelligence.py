from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any, Awaitable, Callable

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_BTTS_PAID_ODDS_INTELLIGENCE_V1.0.0"
SIGNAL_STAGES = {"T-40", "T-20", "T-10"}
SIGNAL_CLASSES = {"BET", "LEAN", "WATCH"}


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def is_strict_btts_market(row: dict[str, Any]) -> bool:
    return _norm(row.get("market")) in {"both teams score", "both teams to score"}


def collect_fixture_markets(
    captured: dict[int, list[dict[str, Any]]],
    fixture_id: int,
    markets: list[dict[str, Any]],
    status: str,
) -> None:
    """Keep only fresh real-price BTTS markets from an already-paid resolver call."""
    if str(status or "").upper() != "PRICE_API_RESOLVED":
        return
    exact = [dict(row) for row in markets if isinstance(row, dict) and is_strict_btts_market(row)]
    if exact:
        captured[int(fixture_id)].extend(exact)


def install_fetch_observer(
    resolver_module: Any,
) -> tuple[Callable[..., Awaitable[Any]], dict[int, list[dict[str, Any]]]]:
    """Observe the existing resolver fetch path without initiating a provider call."""
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


def _screen(event: dict[str, Any]) -> dict[str, Any]:
    for key in ("sporting_shortlist", "sporting_screen_refined", "sporting_screen_initial"):
        value = event.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _tracks(event: dict[str, Any]) -> list[str]:
    tracks = _screen(event).get("tracks")
    if not isinstance(tracks, list):
        return []
    return [str(track).upper() for track in tracks if track]


def _best_market_family(event: dict[str, Any]) -> str:
    best = event.get("best_market")
    if not isinstance(best, dict):
        decision = event.get("market_decision")
        best = decision.get("best_decision") if isinstance(decision, dict) else {}
    if not isinstance(best, dict):
        return ""
    return str(best.get("family") or best.get("market_family") or "").upper()


def _has_btts_research_signal(event: dict[str, Any]) -> bool:
    if "TWO_WAY" in _tracks(event):
        return True
    return _best_market_family(event) in {"BTTS", "FT_BTTS", "FT_BTTS_RESEARCH"}


def _raw_btts_yes_probability(event: dict[str, Any]) -> float | None:
    raw = event.get("raw_projection")
    if not isinstance(raw, dict):
        return None
    probability = _num(raw.get("raw_btts_yes_prob"))
    if probability is None or not 0.0 < probability < 1.0:
        return None
    return probability


def _selection_value(values: list[dict[str, Any]], target: str) -> dict[str, Any] | None:
    target_n = _norm(target)
    for value in values:
        if not isinstance(value, dict):
            continue
        selection = value.get("selection")
        if selection is None:
            selection = value.get("value")
        if _norm(selection) == target_n:
            return value
    return None


def _price(value: dict[str, Any] | None) -> float | None:
    if not isinstance(value, dict):
        return None
    for key in ("decimal_price", "odd", "price"):
        price = _num(value.get(key))
        if price is not None and 1.0 < price <= 1000.0:
            return price
    return None


def _offer_from_market(market: dict[str, Any]) -> dict[str, Any] | None:
    if not is_strict_btts_market(market):
        return None
    values = [value for value in (market.get("values") or []) if isinstance(value, dict)]
    yes = _selection_value(values, "yes")
    no = _selection_value(values, "no")
    yes_price = _price(yes)
    no_price = _price(no)
    if yes_price is None or no_price is None:
        return None

    yes_implied = 1.0 / yes_price
    no_implied = 1.0 / no_price
    total = yes_implied + no_implied
    if total <= 0.0:
        return None
    fair = yes_implied / total

    return {
        "market_family": "BTTS",
        "market": market.get("market") or "Both Teams Score",
        "selection": "yes",
        "line": None,
        "decimal_price": round(yes_price, 6),
        "price": round(yes_price, 6),
        "p_market_fair": round(fair, 8),
        "market_fair_probability": round(fair, 8),
        "bookmaker": market.get("bookmaker"),
        "bookmaker_id": market.get("bookmaker_id"),
        "market_id": market.get("market_id"),
        "provider_update": market.get("provider_update"),
        "market_source": market.get("source") or "API_FOOTBALL_ODDS_V3",
        "research_only": True,
        "actionable": False,
        "decision_weight": 0.0,
        "production_promotion_allowed": False,
    }


def _representative_offer(markets: list[dict[str, Any]]) -> dict[str, Any] | None:
    offers = [offer for offer in (_offer_from_market(market) for market in markets) if offer is not None]
    if not offers:
        return None
    median_price = statistics.median(float(offer["decimal_price"]) for offer in offers)
    return min(
        offers,
        key=lambda offer: (
            abs(float(offer["decimal_price"]) - median_price),
            str(offer.get("bookmaker") or ""),
            str(offer.get("market_id") or ""),
        ),
    )


def _eligible_event(event: dict[str, Any]) -> bool:
    return (
        event.get("event_type") == "SOCCER_REFRESH"
        and str(event.get("stage") or "").upper() in SIGNAL_STAGES
        and str(event.get("classification") or "").upper() in SIGNAL_CLASSES
        and _has_btts_research_signal(event)
        and _raw_btts_yes_probability(event) is not None
    )


def _existing_priced_btts_row(rows: list[dict[str, Any]], fixture_id: int) -> bool:
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            row_fixture = int(row.get("fixture_id"))
        except (TypeError, ValueError):
            continue
        if row_fixture != fixture_id:
            continue
        if _norm(row.get("market")) not in {"both teams score", "both teams to score"}:
            continue
        if _norm(row.get("selection")) != "yes":
            continue
        price = _num(row.get("decimal_price"))
        if price is None:
            price = _num(row.get("price"))
        if price is not None and price > 1.0:
            return True
    return False


def _visibility_row(event: dict[str, Any], offer: dict[str, Any], row_index: int) -> dict[str, Any]:
    fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
    coverage = event.get("coverage") if isinstance(event.get("coverage"), dict) else {}
    screen = _screen(event)
    raw_probability = _raw_btts_yes_probability(event)
    return {
        "row_type": "research_visibility",
        "row_index": row_index,
        "event_type": event.get("event_type"),
        "stage": event.get("stage"),
        "classification": "WATCH",
        "event_classification": str(event.get("classification") or "WATCH").upper(),
        "fixture_id": fixture.get("fixture_id"),
        "kickoff": fixture.get("kickoff"),
        "league": fixture.get("league"),
        "country": fixture.get("country"),
        "home": fixture.get("home_team") or fixture.get("home") or "N/V",
        "away": fixture.get("away_team") or fixture.get("away") or "N/V",
        "status": fixture.get("status"),
        "data_tier": coverage.get("data_tier") or event.get("data_tier") or "N/V",
        "market_family": "BTTS",
        "market": offer.get("market") or "Both Teams Score",
        "selection": "yes",
        "line": None,
        "price": offer.get("decimal_price"),
        "decimal_price": offer.get("decimal_price"),
        "bookmaker": offer.get("bookmaker"),
        "bookmaker_id": offer.get("bookmaker_id"),
        "market_id": offer.get("market_id"),
        "provider_update": offer.get("provider_update"),
        "tier": event.get("tier"),
        "stake_units": 0.0,
        "bet_eligible": False,
        "side_score": screen.get("side_edge_score"),
        "goals_score": screen.get("goal_environment_score"),
        "two_way_score": screen.get("two_way_scoring_score"),
        "shortlist_rank": screen.get("rank"),
        "tracks": _tracks(event),
        "p_market_fair": offer.get("p_market_fair"),
        "market_fair_probability": offer.get("market_fair_probability"),
        "p_raw": round(raw_probability, 8) if raw_probability is not None else None,
        "model_signal": event.get("model_signal"),
        "execution_status": "RESEARCH_ONLY",
        "reason": "BTTS_PAID_ODDS_RESEARCH_ENTRY",
        "market_use": "BTTS_TRUE_CLV_ENTRY_RESEARCH_ONLY",
        "blockers": ["RESEARCH_ONLY", "NO_PRODUCTION_PROMOTION"],
        "research_only": True,
        "actionable": False,
        "decision_weight": 0.0,
        "production_promotion_allowed": False,
    }


def attach(payload: dict[str, Any], captured: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    events = payload.get("events") if isinstance(payload.get("events"), list) else []
    rows = payload.get("match_table_rows") if isinstance(payload.get("match_table_rows"), list) else []
    by_fixture: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if not isinstance(event, dict) or not _eligible_event(event):
            continue
        fixture_id = _fixture_id(event)
        if fixture_id is not None:
            by_fixture[fixture_id].append(event)

    captured_fixtures = 0
    eligible_signal_fixtures = 0
    attached_fixtures = 0
    entry_rows_added = 0
    already_priced_rows = 0
    missing_signal_fixtures = 0
    incomplete_yes_no_markets = 0

    stage_rank = {"T-10": 0, "T-20": 1, "T-40": 2}
    for fixture_id, markets in captured.items():
        exact = [market for market in markets if isinstance(market, dict) and is_strict_btts_market(market)]
        if not exact:
            continue
        captured_fixtures += 1
        candidates = by_fixture.get(int(fixture_id), [])
        if not candidates:
            missing_signal_fixtures += 1
            continue
        eligible_signal_fixtures += 1

        event = min(
            candidates,
            key=lambda item: stage_rank.get(str(item.get("stage") or "").upper(), 9),
        )
        offer = _representative_offer(exact)
        if offer is None:
            incomplete_yes_no_markets += 1
            continue

        raw_probability = _raw_btts_yes_probability(event)
        observed = dict(offer)
        observed["p_raw"] = round(raw_probability, 8) if raw_probability is not None else None
        observed["model_probability_source"] = "raw_projection.raw_btts_yes_prob"
        observed["entry_signal_source"] = "EXISTING_TWO_WAY_SPORTING_SIGNAL_PLUS_ALREADY_PAID_PRICE"
        event["btts_paid_odds_intelligence"] = {
            "schema_version": SCHEMA_VERSION,
            "model_version": MODEL_VERSION,
            "status": "BTTS_PAID_ENTRY_CAPTURED",
            "observed_market_rows": [observed],
            "provider_requests_added": 0,
            "research_only": True,
            "actionable": False,
            "decision_weight": 0.0,
            "production_promotion_allowed": False,
            "canonical_bet_logic_changed": False,
            "model_weights_changed": False,
        }

        if _existing_priced_btts_row(rows, int(fixture_id)):
            already_priced_rows += 1
            continue
        rows.append(_visibility_row(event, offer, len(rows)))
        attached_fixtures += 1
        entry_rows_added += 1

    payload["match_table_rows"] = rows
    result = {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": "BTTS_PAID_ODDS_ENTRY_CAPTURE_ACTIVE",
        "captured_fixtures": captured_fixtures,
        "eligible_signal_fixtures": eligible_signal_fixtures,
        "attached_fixtures": attached_fixtures,
        "entry_rows_added": entry_rows_added,
        "already_priced_btts_rows": already_priced_rows,
        "missing_signal_fixtures": missing_signal_fixtures,
        "incomplete_yes_no_markets": incomplete_yes_no_markets,
        "provider_requests_added": 0,
        "provider_budget_changed": False,
        "research_only": True,
        "actionable": False,
        "decision_weight": 0.0,
        "production_promotion_allowed": False,
        "canonical_bet_logic_changed": False,
        "model_weights_changed": False,
        "thresholds_changed": False,
        "gates_changed": False,
        "strict_close_semantics_changed": False,
        "policy": (
            "OBSERVE_ALREADY_PAID_PRICE_API_RESOLVED_BTTS_ONLY; REQUIRE_EXISTING_TWO_WAY_RESEARCH_SIGNAL; "
            "REQUIRE_REAL_YES_AND_NO_PRICES; DE_VIG_ENTRY_PRICE; APPEND_RESEARCH_ONLY_PRICED_BTTS_ROW; "
            "LATER_TRUE_CLV_STILL_REQUIRES_STRICTLY_LATER_PREKICKOFF_PROVIDER_UPDATE; ZERO_EXTRA_PROVIDER_CALLS"
        ),
    }
    payload["btts_paid_odds_entry_capture"] = result
    return result
