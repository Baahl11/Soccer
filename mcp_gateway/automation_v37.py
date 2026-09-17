from __future__ import annotations

import itertools
import math
from datetime import datetime, timedelta, timezone as dt_timezone
from typing import Any

from mcp_gateway import automation as base
from mcp_gateway import automation_v6 as v6
from mcp_gateway import automation_v36 as v36
from mcp_gateway import galaxy_builder_v2 as v2
from mcp_gateway import galaxy_builder_v4 as v4

MODEL_VERSION = v36.MODEL_VERSION
AUTOMATION_VERSION = "3.13.0"
GALAXY_SCHEMA_VERSION = "0.5.0"

LEG_POOL_CACHE_KEY = "_galaxy_rolling_leg_pool_v1"
LEG_POOL_TTL = timedelta(hours=14)
DEFAULT_MARKET_FRESHNESS_MINUTES = 20.0
MIN_RESEARCH_AVAILABILITY = 0.70
MIN_ACTIONABLE_AVAILABILITY = 0.85
MIN_LEG_PROBABILITY = 0.55
MIN_COMPONENT_EDGE_PP = 1.0
TARGET_EDGE_PP = v2.TARGET_EDGE_PP
TARGET_DECIMAL = v2.TARGET_DECIMAL
MAX_LEGS_PER_FIXTURE = 3
MAX_ACTIVE_POOL_ENTRIES = 80
MAX_COMBINATION_POOL_ENTRIES = 30
MAX_ROLLING_MULTI = 6
MAX_REFERENCE_BY_LEG_COUNT = {2: 3.50, 3: 4.50}
_ALLOWED_MARKET_SOURCES = {"GALAXY_ODDS", "API_FALLBACK_ODDS"}


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        out = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if out.tzinfo is None:
        out = out.replace(tzinfo=dt_timezone.utc)
    return out.astimezone(dt_timezone.utc)


def _leg_key(row: dict[str, Any]) -> str:
    return ":".join(
        [
            str(row.get("fixture_id") or ""),
            str(row.get("family") or ""),
            str(row.get("selection") or ""),
            str(row.get("line") if row.get("line") is not None else ""),
        ]
    )


def _quote_map(quotes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for quote in quotes:
        if not isinstance(quote, dict):
            continue
        bookmaker = str(quote.get("bookmaker") or "").strip()
        price = _num(quote.get("price"))
        if not bookmaker or price is None or price <= 1:
            continue
        current = best.get(bookmaker)
        if current is None or price > float(current.get("price") or 0):
            best[bookmaker] = {
                "bookmaker": bookmaker,
                "market": quote.get("market"),
                "selection_text": quote.get("selection_text"),
                "price": round(price, 4),
                "provider_update": quote.get("provider_update"),
            }
    return best


def _best_quotes(quotes: list[dict[str, Any]], limit: int = 6) -> list[dict[str, Any]]:
    rows = list(_quote_map(quotes).values())
    rows.sort(key=lambda q: float(q.get("price") or 0), reverse=True)
    return rows[:limit]


def _leg_best_edge_pp(leg: dict[str, Any]) -> float | None:
    probability = _num(leg.get("probability"))
    if probability is None:
        return None
    edges = []
    for quote in leg.get("quotes") or []:
        price = _num(quote.get("price"))
        if price is None or price <= 1:
            continue
        edges.append((probability - 1.0 / price) * 100.0)
    return max(edges) if edges else None


def _entry_rank(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        1.0 if row.get("actionable_model") else 0.0,
        float(row.get("best_component_edge_pp") or -999.0),
        float(row.get("probability") or 0.0),
    )


def _fixture_context(event: dict[str, Any]) -> dict[str, Any]:
    fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
    coverage = event.get("coverage") if isinstance(event.get("coverage"), dict) else {}
    availability = _num(event.get("availability_confidence"))
    return {
        "fixture_id": fixture.get("fixture_id"),
        "kickoff": fixture.get("kickoff"),
        "country": fixture.get("country"),
        "league": fixture.get("league"),
        "home_team_id": fixture.get("home_team_id"),
        "away_team_id": fixture.get("away_team_id"),
        "match": f"{fixture.get('home_team') or fixture.get('home_name')} vs {fixture.get('away_team') or fixture.get('away_name')}",
        "data_tier": coverage.get("data_tier") or event.get("data_tier") or event.get("tier"),
        "availability_confidence": availability,
        "stage": event.get("stage"),
    }


def _extract_event_entries(
    event: dict[str, Any],
    now: datetime,
    market_freshness_minutes: float,
) -> list[dict[str, Any]]:
    if event.get("event_type") != "SOCCER_REFRESH":
        return []
    if event.get("stage") in {"POSTGAME", "CLOSE"}:
        return []

    provenance = event.get("market_provenance") if isinstance(event.get("market_provenance"), dict) else {}
    if provenance.get("source") not in _ALLOWED_MARKET_SOURCES or provenance.get("fresh") is not True:
        return []

    context = _fixture_context(event)
    if context.get("fixture_id") is None:
        return []
    if context.get("data_tier") not in {"A", "B"}:
        return []
    availability = context.get("availability_confidence")
    if availability is None or availability < MIN_RESEARCH_AVAILABILITY:
        return []

    kickoff = _dt(context.get("kickoff"))
    if kickoff is None or kickoff <= now:
        return []

    backed, _ = v4._market_backed_legs(event)
    ranked: list[dict[str, Any]] = []
    for leg in backed:
        probability = _num(leg.get("probability"))
        if probability is None or probability < MIN_LEG_PROBABILITY or probability >= 1:
            continue
        quotes = _best_quotes(v4._verified_quotes(leg))
        if not quotes:
            continue
        row = {
            **context,
            "family": leg.get("family"),
            "selection": leg.get("selection"),
            "line": leg.get("line"),
            "probability": round(probability, 6),
            "probability_source": leg.get("probability_source"),
            "actionable_model": bool(leg.get("actionable_model")),
            "research_only_reason": leg.get("research_only_reason"),
            "quotes": quotes,
            "market_source": provenance.get("source"),
            "market_fresh_at_capture": True,
            "captured_at_utc": now.isoformat(),
            "freshness_limit_minutes": market_freshness_minutes,
        }
        best_edge = _leg_best_edge_pp(row)
        if best_edge is None or best_edge < MIN_COMPONENT_EDGE_PP:
            continue
        row["best_component_edge_pp"] = round(best_edge, 2)
        ranked.append(row)

    ranked.sort(key=_entry_rank, reverse=True)
    return ranked[:MAX_LEGS_PER_FIXTURE]


def _pool_record(now: datetime) -> dict[str, Any]:
    cached = base._cache_get("sport_shortlist", LEG_POOL_CACHE_KEY, LEG_POOL_TTL, now)
    if not isinstance(cached, dict):
        cached = {}
    entries = cached.get("entries") if isinstance(cached.get("entries"), dict) else {}
    return {
        "schema_version": "1.0.0",
        "entries": dict(entries),
        "updated_at_utc": cached.get("updated_at_utc"),
    }


def _entry_future_and_fresh(
    row: dict[str, Any],
    now: datetime,
    market_freshness_minutes: float,
) -> bool:
    kickoff = _dt(row.get("kickoff"))
    captured = _dt(row.get("captured_at_utc"))
    if kickoff is None or kickoff <= now or captured is None:
        return False
    age_minutes = (now - captured).total_seconds() / 60.0
    return -1.0 <= age_minutes <= market_freshness_minutes


def _update_pool(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int | float]]:
    now = _dt(payload.get("generated_at_utc")) or datetime.now(dt_timezone.utc)
    policy = payload.get("odds_source_policy") if isinstance(payload.get("odds_source_policy"), dict) else {}
    market_freshness_minutes = _num(policy.get("galaxy_freshness_limit_minutes")) or DEFAULT_MARKET_FRESHNESS_MINUTES

    record = _pool_record(now)
    entries = record["entries"]

    due_fixture_ids: set[str] = set()
    current_events = [
        event for event in payload.get("events") or []
        if isinstance(event, dict) and event.get("event_type") == "SOCCER_REFRESH"
    ]
    for event in current_events:
        context = _fixture_context(event)
        fixture_id = context.get("fixture_id")
        if fixture_id is not None:
            due_fixture_ids.add(str(fixture_id))

    invalidated_due = 0
    expired = 0
    for key, row in list(entries.items()):
        if not isinstance(row, dict):
            entries.pop(key, None)
            expired += 1
            continue
        if str(row.get("fixture_id")) in due_fixture_ids:
            entries.pop(key, None)
            invalidated_due += 1
            continue
        if not _entry_future_and_fresh(row, now, market_freshness_minutes):
            entries.pop(key, None)
            expired += 1

    added = 0
    for event in current_events:
        for row in _extract_event_entries(event, now, market_freshness_minutes):
            entries[_leg_key(row)] = row
            added += 1

    active = [
        row for row in entries.values()
        if isinstance(row, dict) and _entry_future_and_fresh(row, now, market_freshness_minutes)
    ]
    active.sort(key=_entry_rank, reverse=True)
    active = active[:MAX_ACTIVE_POOL_ENTRIES]

    record["entries"] = {_leg_key(row): row for row in active}
    record["updated_at_utc"] = now.isoformat()
    record["policy"] = (
        "MARKET_BACKED_CANONICAL_LEGS_ONLY; SAME_FIXTURE_REFRESH_REPLACES_PRIOR_LEGS; "
        "PRICE_FRESHNESS_REQUIRED; KICKOFF_EXPIRES; ZERO_EXTRA_PROVIDER_REQUESTS"
    )
    base._cache_set("sport_shortlist", LEG_POOL_CACHE_KEY, record, now)

    metrics: dict[str, int | float] = {
        "active_leg_count": len(active),
        "fixtures_with_active_legs": len({str(x.get("fixture_id")) for x in active}),
        "added_this_tick": added,
        "invalidated_due_this_tick": invalidated_due,
        "expired_or_stale_this_tick": expired,
        "market_freshness_minutes": market_freshness_minutes,
    }
    return active, metrics


def _team_ids(row: dict[str, Any]) -> set[str]:
    out = set()
    for key in ("home_team_id", "away_team_id"):
        value = row.get(key)
        if value is not None:
            out.add(str(value))
    return out


def _common_book(combo: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]], float] | None:
    maps = [_quote_map(list(row.get("quotes") or [])) for row in combo]
    if any(not mapping for mapping in maps):
        return None
    common = set(maps[0])
    for mapping in maps[1:]:
        common &= set(mapping)
    if not common:
        return None

    viable: list[tuple[float, str, list[dict[str, Any]]]] = []
    for book in common:
        quotes = [mapping[book] for mapping in maps]
        component_edges = []
        for row, quote in zip(combo, quotes):
            probability = float(row.get("probability") or 0.0)
            price = float(quote.get("price") or 0.0)
            component_edges.append((probability - 1.0 / price) * 100.0)
        if min(component_edges) < MIN_COMPONENT_EDGE_PP:
            continue
        product = math.prod(float(quote["price"]) for quote in quotes)
        viable.append((product, book, quotes))
    if not viable:
        return None
    product, book, quotes = max(viable, key=lambda x: x[0])
    return book, quotes, product


def _minimum_decimal_for_edge(probability: float) -> float | None:
    max_market_probability = probability - TARGET_EDGE_PP / 100.0
    if max_market_probability <= 0:
        return None
    return 1.0 / max_market_probability


def _build_rolling_multis(active: list[dict[str, Any]]) -> list[dict[str, Any]]:
    combine_pool = sorted(active, key=_entry_rank, reverse=True)[:MAX_COMBINATION_POOL_ENTRIES]
    best_by_fixture_set: dict[tuple[str, ...], dict[str, Any]] = {}

    for leg_count in (2, 3):
        for combo_tuple in itertools.combinations(combine_pool, leg_count):
            combo = list(combo_tuple)
            fixture_ids = [str(row.get("fixture_id")) for row in combo]
            if len(set(fixture_ids)) != leg_count:
                continue

            team_sets = [_team_ids(row) for row in combo]
            if any(team_sets[i] & team_sets[j] for i in range(leg_count) for j in range(i + 1, leg_count)):
                continue

            common = _common_book(combo)
            if common is None:
                continue
            bookmaker, quotes, decimal_reference = common
            max_reference = MAX_REFERENCE_BY_LEG_COUNT[leg_count]
            if not (TARGET_DECIMAL <= decimal_reference <= max_reference):
                continue

            raw_joint = math.prod(float(row["probability"]) for row in combo)
            conservative_joint = raw_joint * 0.98
            if conservative_joint <= 0 or conservative_joint >= 1:
                continue
            reference_edge_pp = (conservative_joint - 1.0 / decimal_reference) * 100.0
            if reference_edge_pp < TARGET_EDGE_PP:
                continue

            all_actionable = all(bool(row.get("actionable_model")) for row in combo)
            min_availability = min(float(row.get("availability_confidence") or 0.0) for row in combo)
            blocks = ["FINAL_PARLAY_QUOTE_NOT_VERIFIED"]
            if not all_actionable:
                blocks.append("RESEARCH_ONLY_LEG_PRESENT")
            if min_availability < MIN_ACTIONABLE_AVAILABILITY:
                blocks.append("AVAILABILITY_BELOW_0.85")

            public_legs = []
            for row, quote in zip(combo, quotes):
                probability = float(row["probability"])
                price = float(quote["price"])
                public_legs.append(
                    {
                        "fixture_id": row.get("fixture_id"),
                        "match": row.get("match"),
                        "kickoff": row.get("kickoff"),
                        "league": row.get("league"),
                        "data_tier": row.get("data_tier"),
                        "availability_confidence": row.get("availability_confidence"),
                        "family": row.get("family"),
                        "selection": row.get("selection"),
                        "line": row.get("line"),
                        "probability": row.get("probability"),
                        "probability_source": row.get("probability_source"),
                        "actionable_model": row.get("actionable_model"),
                        "research_only_reason": row.get("research_only_reason"),
                        "bookmaker": bookmaker,
                        "decimal_price": round(price, 4),
                        "selection_text": quote.get("selection_text"),
                        "market": quote.get("market"),
                        "provider_update": quote.get("provider_update"),
                        "component_edge_pp": round((probability - 1.0 / price) * 100.0, 2),
                    }
                )

            minimum_decimal = _minimum_decimal_for_edge(conservative_joint)
            candidate = {
                "type": "MULTI_MATCH_PARLAY",
                "candidate_source": "GALAXY_BUILDER_V0.5_ROLLING_LEG_POOL",
                "bookmaker": bookmaker,
                "leg_count": leg_count,
                "legs": public_legs,
                "component_product_decimal_reference": round(decimal_reference, 4),
                "component_product_american_reference": v2._american(decimal_reference),
                "component_product_is_exact_parlay_quote": False,
                "raw_independence_probability": round(raw_joint, 6),
                "conservative_joint_probability": round(conservative_joint, 6),
                "probability_edge_vs_component_product_pp": round(reference_edge_pp, 2),
                "minimum_parlay_decimal_for_target_edge": round(minimum_decimal, 4) if minimum_decimal else None,
                "minimum_parlay_american_for_target_edge": v2._american(minimum_decimal) if minimum_decimal else None,
                "exact_parlay_quote": None,
                "status": "GALAXY ROLLING MULTI WATCH — FINAL PARLAY QUOTE NEEDED",
                "bet_eligible": False,
                "all_leg_models_actionable": all_actionable,
                "availability_confidence": round(min_availability, 3),
                "block_reasons": blocks,
                "joint_probability_method": "DISTINCT_FIXTURES_INDEPENDENCE_WITH_2_PERCENT_UNCERTAINTY_HAIRCUT",
                "rolling_leg_pool": True,
            }

            key = tuple(sorted(fixture_ids))
            current = best_by_fixture_set.get(key)
            new_rank = (
                1.0 if all_actionable else 0.0,
                -float(leg_count),
                float(reference_edge_pp),
                float(conservative_joint),
            )
            if current is None:
                candidate["_rank"] = new_rank
                best_by_fixture_set[key] = candidate
            elif new_rank > tuple(current.get("_rank") or (0, -99, 0, 0)):
                candidate["_rank"] = new_rank
                best_by_fixture_set[key] = candidate

    rows = list(best_by_fixture_set.values())
    rows.sort(key=lambda row: tuple(row.get("_rank") or (0, -99, 0, 0)), reverse=True)
    for row in rows:
        row.pop("_rank", None)
    return rows[:MAX_ROLLING_MULTI]


def _candidate_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    parts = []
    for leg in row.get("legs") or []:
        if not isinstance(leg, dict):
            continue
        parts.append(
            (
                str(leg.get("fixture_id") or ""),
                str(leg.get("family") or ""),
                str(leg.get("selection") or ""),
                str(leg.get("line") if leg.get("line") is not None else ""),
            )
        )
    parts.sort()
    return (str(row.get("type") or ""), str(row.get("bookmaker") or ""), tuple(parts))


def _merge_multis(existing: list[dict[str, Any]], rolling: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in [*existing, *rolling]:
        if not isinstance(row, dict):
            continue
        key = _candidate_signature(row)
        current = best.get(key)
        if current is None:
            best[key] = row
            continue
        current_rank = (
            1.0 if current.get("all_leg_models_actionable") else 0.0,
            float(current.get("probability_edge_vs_component_product_pp") or 0.0),
            float(current.get("conservative_joint_probability") or 0.0),
        )
        new_rank = (
            1.0 if row.get("all_leg_models_actionable") else 0.0,
            float(row.get("probability_edge_vs_component_product_pp") or 0.0),
            float(row.get("conservative_joint_probability") or 0.0),
        )
        if new_rank > current_rank:
            best[key] = row

    rows = list(best.values())
    rows.sort(
        key=lambda row: (
            1.0 if row.get("all_leg_models_actionable") else 0.0,
            -float(row.get("leg_count") or len(row.get("legs") or [])),
            float(row.get("probability_edge_vs_component_product_pp") or 0.0),
            float(row.get("conservative_joint_probability") or 0.0),
        ),
        reverse=True,
    )
    return rows[:MAX_ROLLING_MULTI]


async def run_tick() -> dict[str, Any]:
    payload = await v36.run_tick()

    active_legs, pool_metrics = _update_pool(payload)
    rolling_multis = _build_rolling_multis(active_legs)

    builder = payload.get("galaxy_builder") if isinstance(payload.get("galaxy_builder"), dict) else {}
    existing_multis = [
        row for row in builder.get("multi_match_candidates") or []
        if isinstance(row, dict)
    ]
    merged_multis = _merge_multis(existing_multis, rolling_multis)

    builder = dict(builder)
    same_game = [
        row for row in builder.get("same_game_candidates") or []
        if isinstance(row, dict)
    ]
    builder["schema_version"] = GALAXY_SCHEMA_VERSION
    builder["multi_match_candidates"] = merged_multis
    builder["multi_match_candidate_count"] = len(merged_multis)
    builder["candidate_count"] = len(same_game) + len(merged_multis)
    builder["rolling_leg_pool"] = {
        "schema_version": "1.0.0",
        "active": True,
        **pool_metrics,
        "rolling_multi_candidate_count": len(rolling_multis),
        "merged_multi_candidate_count": len(merged_multis),
        "minimum_leg_probability": MIN_LEG_PROBABILITY,
        "minimum_component_edge_pp": MIN_COMPONENT_EDGE_PP,
        "minimum_research_availability": MIN_RESEARCH_AVAILABILITY,
        "final_actionability_availability": MIN_ACTIONABLE_AVAILABILITY,
        "max_legs_per_fixture_in_pool": MAX_LEGS_PER_FIXTURE,
        "multi_leg_counts_supported": [2, 3],
        "provider_requests_added": 0,
        "policy": (
            "PERSIST MARKET-BACKED INDIVIDUAL LEGS ACROSS NATURAL TICKS; "
            "ONE LEG PER FIXTURE IN A MULTI; SAME-BOOK COMPONENT QUOTES; "
            "EACH COMPONENT REQUIRES >=1PP EDGE VS ITS VERIFIED PRICE; "
            "DISTINCT-FIXTURE JOINT P USES EXISTING 2_PERCENT_UNCERTAINTY_HAIRCUT; "
            "FINAL PARLAY QUOTE REQUIRED FOR BET"
        ),
    }
    builder["rolling_multi_policy"] = (
        "GALAXY V0.5 ROLLING LEG POOL; CANDIDATES MAY COMBINE LEGS DISCOVERED ON DIFFERENT NATURAL TICKS; "
        "NO EXTRA PROVIDER REQUESTS; NO CANONICAL MODEL OR BET LOGIC CHANGES"
    )
    payload["galaxy_builder"] = builder
    payload["galaxy_builder_active_candidate_count"] = int(builder.get("candidate_count") or 0)
    payload["galaxy_builder_multi_match_count_active"] = len(merged_multis)
    payload["galaxy_builder_rolling_leg_pool_active"] = len(active_legs)

    payload["shortlist_state"] = v6.export_shortlist_state()
    payload["shortlist_state_count"] = len(payload["shortlist_state"])

    payload["v313_provider_requests_added"] = 0
    payload["v313_model_weights_changed"] = False
    payload["v313_canonical_bet_logic_changed"] = False
    payload["v313_sgp_logic_changed"] = False
    payload["v313_multi_match_rolling_leg_pool_added"] = True
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
