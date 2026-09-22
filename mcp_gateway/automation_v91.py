from __future__ import annotations

import os
from collections import Counter
from typing import Any, Awaitable, Callable

from mcp_gateway import automation_v12 as v12
from mcp_gateway import automation_v90 as v90

MODEL_VERSION = v90.MODEL_VERSION
AUTOMATION_VERSION = "3.62.3"

GALAXY_SLATE_FLOOR_MIN_FIXTURES = int(
    os.getenv("SOCCER_EDGE_SLATE_FLOOR_MIN_FIXTURES", "12")
)
RESEARCH_VISIBILITY_ROW_LIMIT = int(os.getenv("SOCCER_EDGE_RESEARCH_VISIBILITY_ROW_LIMIT", "40"))

GalaxySlatePayload = Callable[[str, Any], Awaitable[dict[str, Any] | None]]


def _metric_value(key: str) -> int:
    try:
        return int(v12._SLATE_METRICS.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _restore_metric(key: str, value: int) -> None:
    v12._SLATE_METRICS[key] = int(value)


def _bump_metric(key: str, amount: int = 1) -> None:
    v12._SLATE_METRICS[key] = _metric_value(key) + int(amount)


def _payload_results(payload: dict[str, Any] | None) -> int:
    if not isinstance(payload, dict):
        return 0
    try:
        return int(payload.get("results") or len(payload.get("response") or []))
    except (TypeError, ValueError):
        return 0


def _num(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _round_num(value: Any, digits: int = 3) -> float | None:
    number = _num(value)
    if number is None:
        return None
    return round(number, digits)


def _first_dict(*values: Any) -> dict[str, Any]:
    for value in values:
        if isinstance(value, dict):
            return value
    return {}


def _screen_from_event(event: dict[str, Any]) -> dict[str, Any]:
    return _first_dict(
        event.get("sporting_shortlist"),
        event.get("sporting_screen_refined"),
        event.get("sporting_screen_initial"),
    )


def _raw_projection_scores(event: dict[str, Any]) -> dict[str, Any]:
    raw = event.get("raw_projection")
    if not isinstance(raw, dict):
        return {}
    scores = raw.get("screen_scores")
    return scores if isinstance(scores, dict) else {}


def _score_value(event: dict[str, Any], screen: dict[str, Any], screen_key: str, raw_key: str) -> float | None:
    direct = _round_num(screen.get(screen_key), 1)
    if direct is not None:
        return direct
    raw_scores = _raw_projection_scores(event)
    return _round_num(raw_scores.get(raw_key), 1)


def _tracks(screen: dict[str, Any]) -> list[str]:
    tracks = screen.get("tracks")
    if not isinstance(tracks, list):
        return []
    return [str(track).upper() for track in tracks if track]


def _best_market(event: dict[str, Any]) -> dict[str, Any]:
    best = event.get("best_market")
    if isinstance(best, dict) and best:
        return best
    decision = event.get("market_decision")
    if isinstance(decision, dict):
        best = decision.get("best_decision")
        if isinstance(best, dict):
            return best
    return {}


def _derived_market_from_tracks(tracks: list[str]) -> dict[str, Any]:
    if "GOALS_OVER" in tracks:
        return {
            "market_family": "FT_TOTALS_RESEARCH",
            "market": "Goals Over/Under",
            "selection": "Over research",
        }
    if "GOALS_UNDER" in tracks:
        return {
            "market_family": "FT_TOTALS_RESEARCH",
            "market": "Goals Over/Under",
            "selection": "Under research",
        }
    if "TWO_WAY" in tracks:
        return {
            "market_family": "FT_BTTS_RESEARCH",
            "market": "Both Teams To Score",
            "selection": "BTTS research",
        }
    if "SIDE" in tracks:
        return {
            "market_family": "FT_1X2_RESEARCH",
            "market": "Match Winner / Side",
            "selection": "Side research",
        }
    return {
        "market_family": "SPORTING_SCREEN",
        "market": "Research screen",
        "selection": "No shortlist angle",
    }


def _visible_classification(event: dict[str, Any]) -> str:
    raw = str(event.get("classification") or "WATCH").upper()
    if raw in {"BET", "LEAN"}:
        return raw
    # Product visibility: PASS/CLOSE/POSTGAME/etc. should not make the live
    # digest silent. They remain non-bets, but surface as WATCH rows with the raw
    # classification preserved separately.
    return "WATCH"


def _reason(event: dict[str, Any], screen: dict[str, Any], best: dict[str, Any]) -> str:
    decision = event.get("market_decision")
    if isinstance(decision, dict) and decision.get("reason"):
        return str(decision.get("reason"))
    if best.get("reason"):
        return str(best.get("reason"))
    if screen.get("reason"):
        return str(screen.get("reason"))
    notes = event.get("notes")
    if isinstance(notes, list) and notes:
        return str(notes[0])[:220]
    if event.get("error"):
        return str(event.get("error"))[:220]
    return "NO_BET_RESEARCH_ROW"


def _row_from_event(event: dict[str, Any], index: int) -> dict[str, Any] | None:
    fixture = event.get("fixture")
    if not isinstance(fixture, dict):
        return None
    if str(event.get("event_type") or "").upper() == "DAILY_DISCOVERY":
        return None

    screen = _screen_from_event(event)
    tracks = _tracks(screen)
    best = _best_market(event)
    derived = _derived_market_from_tracks(tracks)
    coverage = event.get("coverage") if isinstance(event.get("coverage"), dict) else {}

    market_family = best.get("family") or derived["market_family"]
    market = best.get("market") or derived["market"]
    selection = best.get("selection") or derived["selection"]
    price = best.get("decimal_price") or best.get("price")

    row = {
        "row_type": "research_visibility",
        "row_index": index,
        "event_type": event.get("event_type"),
        "stage": event.get("stage"),
        "classification": _visible_classification(event),
        "event_classification": str(event.get("classification") or "WATCH").upper(),
        "fixture_id": fixture.get("fixture_id"),
        "kickoff": fixture.get("kickoff"),
        "league": fixture.get("league"),
        "country": fixture.get("country"),
        "home": fixture.get("home_team") or fixture.get("home") or "N/V",
        "away": fixture.get("away_team") or fixture.get("away") or "N/V",
        "status": fixture.get("status"),
        "data_tier": coverage.get("data_tier") or event.get("data_tier") or "N/V",
        "market_family": market_family,
        "market": market,
        "selection": selection,
        "line": best.get("line"),
        "price": _round_num(price, 3),
        "bookmaker": best.get("bookmaker"),
        "tier": best.get("tier") or event.get("tier"),
        "stake_units": _round_num(best.get("stake_units", event.get("stake_units")), 3),
        "bet_eligible": bool(event.get("bet_eligible")),
        "side_score": _score_value(event, screen, "side_edge_score", "side_edge_score"),
        "goals_score": _score_value(event, screen, "goal_environment_score", "goal_environment_score"),
        "two_way_score": _score_value(event, screen, "two_way_scoring_score", "two_way_scoring_score"),
        "shortlist_rank": _round_num(screen.get("rank"), 1),
        "tracks": tracks,
        "prob_edge_pp": _round_num(best.get("prob_edge_pp"), 3),
        "estimated_ev": _round_num(best.get("estimated_ev"), 4),
        "p_market_fair": _round_num(best.get("p_market_fair"), 4),
        "p_shrunk": _round_num(best.get("p_shrunk"), 4),
        "reason": _reason(event, screen, best),
        "notes_count": len(event.get("notes") or []) if isinstance(event.get("notes"), list) else 0,
        "market_use": event.get("market_use"),
    }
    return row


def _sort_row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    class_rank = {"BET": 0, "LEAN": 1, "WATCH": 2}.get(str(row.get("classification") or ""), 9)
    stage_rank = {"T-10": 0, "T-20": 1, "T-40": 2, "CLOSE": 3, "T-60": 4, "T-90": 5, "SLATE_SCREEN": 6}.get(
        str(row.get("stage") or ""),
        9,
    )
    signal_rank = -max(
        _num(row.get("side_score")) or 0.0,
        _num(row.get("goals_score")) or 0.0,
        _num(row.get("two_way_score")) or 0.0,
        _num(row.get("shortlist_rank")) or 0.0,
    )
    kickoff = str(row.get("kickoff") or "")
    return (class_rank, stage_rank, signal_rank, kickoff, row.get("fixture_id") or 0)


def _build_match_table_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    events = payload.get("events") if isinstance(payload, dict) else []
    if not isinstance(events, list):
        return []

    rows: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        row = _row_from_event(event, index)
        if row is not None:
            rows.append(row)

    rows.sort(key=_sort_row_key)
    return rows[: max(RESEARCH_VISIBILITY_ROW_LIMIT, 0)]


def _research_visibility_summary(payload: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    visible_counter = Counter(str(row.get("classification") or "N/V") for row in rows)
    raw_counter = Counter(str(row.get("event_classification") or "N/V") for row in rows)
    family_counter = Counter(str(row.get("market_family") or "N/V") for row in rows)
    stage_counter = Counter(str(row.get("stage") or "N/V") for row in rows)
    track_counter: Counter[str] = Counter()
    for row in rows:
        for track in row.get("tracks") or []:
            track_counter[str(track)] += 1

    return {
        "schema_version": "1.0.0",
        "policy": "ALWAYS_SURFACE_RESEARCH_ROWS_FOR_BET_LEAN_WATCH_PASS_EVENTS",
        "row_count": len(rows),
        "row_limit": RESEARCH_VISIBILITY_ROW_LIMIT,
        "visible_classification_counts": dict(visible_counter),
        "raw_event_classification_counts": dict(raw_counter),
        "market_family_counts": dict(family_counter),
        "stage_counts": dict(stage_counter),
        "track_counts": dict(track_counter),
        "silent_tick_prevention": bool(rows),
        "actionable_refresh_count": payload.get("actionable_refresh_count"),
        "bet_candidate_count": payload.get("bet_candidate_count"),
        "note": (
            "Rows are a visibility layer only. WATCH rows can include raw PASS/CLOSE events so the live digest shows "
            "what was analyzed and why no BET/LEAN was emitted."
        ),
    }


def _annotate_research_visibility(payload: dict[str, Any]) -> None:
    rows = _build_match_table_rows(payload)
    payload["match_table_rows"] = rows
    payload["research_visibility"] = _research_visibility_summary(payload, rows)
    payload["research_visible_count"] = len(rows)
    payload["v3623_research_visibility_added"] = True
    payload["v3623_provider_requests_added"] = 0
    payload["v3623_provider_request_scope"] = "No provider requests; transforms already-returned tick events into visible research rows."
    payload["v3623_model_weights_changed"] = False
    payload["v3623_canonical_bet_logic_changed"] = False
    payload["v3623_runtime_promotion_added"] = False
    payload["v3623_stake_or_tier_change"] = False
    payload["v3623_checkpoint"] = (
        "RESEARCH VISIBILITY BOARD: ticks no longer go visually silent when there are no BET rows. "
        "match_table_rows surfaces BET/LEAN/WATCH plus raw PASS/CLOSE research context, including goals, side, BTTS tracks, "
        "best market family, line, price and no-bet reason where available."
    )


def _annotate_payload(payload: dict[str, Any]) -> None:
    metrics = dict(payload.get("galaxy_first_metrics") or {})
    floor_rejections = _metric_value("galaxy_slate_floor_rejections")
    floor_last_count = _metric_value("galaxy_slate_floor_last_rejected_count")

    metrics["galaxy_slate_floor_min_fixture_count"] = GALAXY_SLATE_FLOOR_MIN_FIXTURES
    metrics["galaxy_slate_floor_rejections"] = floor_rejections
    metrics["galaxy_slate_floor_last_rejected_count"] = floor_last_count or None
    metrics["tiny_galaxy_slate_can_avoid_api"] = False
    metrics["galaxy_slate_floor_policy"] = (
        "FRESH_GALAXY_SLATE_REQUIRES_MIN_FIXTURE_COUNT; OTHERWISE FALL BACK TO API_FOOTBALL DATE SLATE"
    )
    payload["galaxy_first_metrics"] = metrics

    payload["v3622_provider_requests_added"] = 0
    payload["v3622_provider_request_scope"] = (
        "No direct request here; tiny Galaxy slate is rejected so existing API fallback/floor path can run."
    )
    payload["v3622_galaxy_slate_floor_min_fixture_count"] = GALAXY_SLATE_FLOOR_MIN_FIXTURES
    payload["v3622_galaxy_slate_floor_rejections"] = floor_rejections
    payload["v3622_model_weights_changed"] = False
    payload["v3622_canonical_bet_logic_changed"] = False
    payload["v3622_runtime_promotion_added"] = False
    payload["v3622_stake_or_tier_change"] = False
    payload["v3622_checkpoint"] = (
        "GALAXY-FIRST SLATE FLOOR: a fresh Galaxy persisted slate below the fixture floor is not allowed "
        "to avoid the API-Football date slate. This corrects the tiny 2-fixture universe without changing picks, "
        "model weights, thresholds, tiers or stakes."
    )

    _annotate_research_visibility(payload)


async def run_tick() -> dict[str, Any]:
    original_galaxy_slate_payload: GalaxySlatePayload = v12._galaxy_slate_payload

    async def floor_guarded_galaxy_slate_payload(match_date: str, now: Any) -> dict[str, Any] | None:
        before_hits = _metric_value("galaxy_slate_hits")
        before_avoided = _metric_value("api_slate_calls_avoided")
        payload = await original_galaxy_slate_payload(match_date, now)
        count = _payload_results(payload)
        if payload is not None and count < GALAXY_SLATE_FLOOR_MIN_FIXTURES:
            # v12 increments these counters inside _galaxy_slate_payload before returning.
            # Restore them so a rejected tiny Galaxy slate is not counted as a hit or an avoided API call.
            _restore_metric("galaxy_slate_hits", before_hits)
            _restore_metric("api_slate_calls_avoided", before_avoided)
            _bump_metric("galaxy_slate_floor_rejections")
            v12._SLATE_METRICS["galaxy_slate_floor_last_rejected_count"] = count
            return None
        return payload

    v12._galaxy_slate_payload = floor_guarded_galaxy_slate_payload
    try:
        payload = await v90.run_tick()
    finally:
        v12._galaxy_slate_payload = original_galaxy_slate_payload

    _annotate_payload(payload)
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
