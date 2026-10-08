from __future__ import annotations

import asyncio
import math
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

from mcp_gateway import persistence as persistence_base
from mcp_gateway import product_views_v4
from mcp_gateway import public_performance_v4
from mcp_gateway import subscriber_app_v4
from mcp_gateway import subscriber_billing_v2
from mcp_gateway import subscriber_preview_data_v231
from mcp_gateway import subscriber_saved_items_v4
from mcp_gateway import subscriber_ui_contract_v231
from mcp_gateway import subscriber_validation_metrics_v231
from mcp_gateway import subscription_entitlements_v4
from mcp_gateway import supabase_auth_v4

SCHEMA_VERSION = "2.0.0"
MODEL_VERSION = "SOCCER_SUBSCRIBER_CONTRACT_V2_1.0.0"
SOURCE = "POSTGRES_LATEST_PIPELINE_RUN"

_CANONICAL_CLASSIFICATIONS = {"BET", "LEAN", "WATCH", "PASS"}
_WATCH_EXECUTION_STATUSES = {
    "WAIT_MARKET",
    "WAIT_PRICE",
    "WAIT_FRESH_QUOTE",
    "WAIT_XI",
    "WAIT_GK",
    "WAIT_AVAILABILITY",
}

_TERMINAL_FIXTURE_STATUSES = {"FT", "AET", "PEN", "CANC", "PST", "ABD", "AWD", "WO"}
_DECISION_PRIORITY = {"BET": 4, "LEAN": 3, "WATCH": 2, "PASS": 1}
_MARKET_COVERAGE_LABELS = {
    "1X2": "Match Winner",
    "BTTS": "BTTS",
    "FT_TOTALS": "Goals",
    "HOME_TT": "Home Team Total",
    "AWAY_TT": "Away Team Total",
    "1H": "First Half",
    "2H": "Second Half",
    "FT_CORNERS": "Corners",
    "TEAM_CORNERS": "Team Corners",
    "CARDS": "Cards",
    "SHOTS": "Player Shots",
    "SOT": "Shots on Target",
    "GOALSCORER": "Goalscorer",
    "ASSISTS": "Assists",
    "PLAYER_CARDS": "Player Cards",
    "GK_SAVES": "Goalkeeper Saves",
}
_VIEW_NAMES = (
    "strong_sport_signals",
    "value_plays",
    "todays_slate",
    "waiting_for_price",
    "waiting_for_xi",
    "team_totals",
    "first_half",
    "second_half",
    "corners",
    "player_props",
)


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _rows(node: Any) -> list[dict[str, Any]]:
    value = _dict(node).get("rows")
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _first(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _first_with_key(row: dict[str, Any], *keys: str) -> tuple[str | None, Any]:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return key, value
    return None, None


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _probability(value: Any) -> float | None:
    number = _number(value)
    if number is None:
        return None
    if 0.0 <= number <= 1.0:
        return number
    if 0.0 <= number <= 100.0:
        return number / 100.0
    return None


def _pp(value: Any) -> float | None:
    # Only fields explicitly named in percentage-point units are accepted here.
    # Never rescale them heuristically: 0.8 pp must remain 0.8 pp, not 80 pp.
    return _number(value)


def _text_list(row: dict[str, Any], *keys: str) -> list[str]:
    out: list[str] = []
    for key in keys:
        value = row.get(key)
        if isinstance(value, list):
            for item in value:
                text = str(item or "").strip()
                if text and text not in out:
                    out.append(text)
        elif isinstance(value, str):
            text = value.strip()
            if text and text not in out:
                out.append(text)
    return out


def _classification(row: dict[str, Any]) -> str | None:
    for key in (
        "runtime_classification",
        "classification",
        "event_classification",
        "bet_classification",
        "decision_classification",
    ):
        value = str(row.get(key) or "").strip().upper()
        if value in _CANONICAL_CLASSIFICATIONS:
            return value
    return None


def _execution_status(row: dict[str, Any]) -> str | None:
    value = _first(row, "execution_status", "decision_status")
    return str(value).strip().upper() if value not in (None, "") else None

def _display_reason(
    reason: Any,
    execution_status: str | None,
    classification: str | None,
) -> str | None:
    """Translate persisted internal state into concise subscriber language.

    This is presentation only. It never changes the canonical classification.
    """

    raw = str(reason or "").strip()
    status = str(execution_status or "").strip().upper()

    if "Per-tick API budget reached" in raw:
        return (
            "Data refresh deferred by the request budget; required inputs are "
            "not fully verified yet."
        )
    if "PAID_ODDS_RESEARCH_ENTRY" in raw:
        return "Research-only market evaluation; production BET gate not passed."
    if "SPORTING_SCREEN_PASS" in raw:
        return (
            "Sporting screen passed; the verified market/value requirements "
            "for a BET are not yet satisfied."
        )

    status_copy = {
        "WAIT_MARKET": "Waiting for a verified market.",
        "WAIT_PRICE": "Waiting for an actionable verified price.",
        "WAIT_FRESH_QUOTE": "Waiting for a fresh actionable market quote.",
        "WAIT_XI": "Waiting for confirmed starting XI.",
        "WAIT_GK": "Waiting for confirmed goalkeepers.",
        "WAIT_AVAILABILITY": "Waiting for material availability verification.",
        "RESEARCH_ONLY": "Research only; production BET gate not passed.",
        "PASS": "No actionable betting edge in the current verified snapshot.",
    }
    if status in status_copy:
        return status_copy[status]

    if raw:
        # Preserve the source meaning while avoiding code-like underscore labels.
        return raw.replace("_", " ").strip()
    if classification == "WATCH":
        return "Waiting for the remaining verified conditions required for action."
    return None


def _display_bucket(classification: str | None, execution_status: str | None) -> str | None:
    if classification in _CANONICAL_CLASSIFICATIONS:
        return classification
    if str(execution_status or "").upper() in _WATCH_EXECUTION_STATUSES:
        return "WATCH"
    return None


def _fixture(row: dict[str, Any]) -> dict[str, Any]:
    nested = _dict(row.get("fixture"))
    teams = _dict(row.get("teams"))
    home_meta = _dict(teams.get("home"))
    away_meta = _dict(teams.get("away"))

    fixture_id = _first(row, "fixture_id", "id")
    if fixture_id in (None, ""):
        fixture_id = _first(nested, "fixture_id", "id")

    home_id = _first(row, "home_team_id", "home_id")
    if home_id in (None, ""):
        home_id = _first(nested, "home_team_id", "home_id")
    if home_id in (None, ""):
        home_id = home_meta.get("id")

    away_id = _first(row, "away_team_id", "away_id")
    if away_id in (None, ""):
        away_id = _first(nested, "away_team_id", "away_id")
    if away_id in (None, ""):
        away_id = away_meta.get("id")

    home = _first(row, "home_team", "home")
    if home in (None, ""):
        home = _first(nested, "home_team", "home")
    if home in (None, ""):
        home = home_meta.get("name")

    away = _first(row, "away_team", "away")
    if away in (None, ""):
        away = _first(nested, "away_team", "away")
    if away in (None, ""):
        away = away_meta.get("name")

    home_logo = _first(row, "home_team_logo", "home_logo", "home_team_logo_url")
    if home_logo in (None, ""):
        home_logo = home_meta.get("logo")
    away_logo = _first(row, "away_team_logo", "away_logo", "away_team_logo_url")
    if away_logo in (None, ""):
        away_logo = away_meta.get("logo")

    try:
        home_id_int = int(home_id) if home_id not in (None, "") else None
    except (TypeError, ValueError):
        home_id_int = None
    try:
        away_id_int = int(away_id) if away_id not in (None, "") else None
    except (TypeError, ValueError):
        away_id_int = None

    if not (isinstance(home_logo, str) and home_logo.startswith(("https://", "http://"))) and home_id_int:
        home_logo = f"https://media.api-sports.io/football/teams/{home_id_int}.png"
    if not (isinstance(away_logo, str) and away_logo.startswith(("https://", "http://"))) and away_id_int:
        away_logo = f"https://media.api-sports.io/football/teams/{away_id_int}.png"

    return {
        "fixture_id": fixture_id,
        "kickoff": _first(row, "kickoff", "fixture_date", "date") or _first(nested, "kickoff", "date"),
        "league_id": _first(row, "league_id") or _first(nested, "league_id"),
        "league": _first(row, "league_name", "league") or _first(nested, "league"),
        "country": _first(row, "country") or _first(nested, "country"),
        "season": _first(row, "season") or _first(nested, "season"),
        "home_team_id": home_id_int,
        "home_team": home,
        "home_team_logo": home_logo if isinstance(home_logo, str) else None,
        "away_team_id": away_id_int,
        "away_team": away,
        "away_team_logo": away_logo if isinstance(away_logo, str) else None,
    }


def _price(row: dict[str, Any]) -> dict[str, Any]:
    key, value = _first_with_key(row, "decimal_price", "decimal_odds", "price", "odds", "entry_price")
    explicit_format = _first(row, "price_format", "odds_format")
    if explicit_format not in (None, ""):
        price_format = str(explicit_format).upper()
    elif key in {"decimal_price", "decimal_odds"}:
        price_format = "DECIMAL"
    else:
        price_format = "NOT VERIFIED"

    return {
        "value": _number(value),
        "format": price_format,
        "bookmaker": _first(row, "bookmaker"),
        "source": _first(row, "market_source", "odds_source", "provider"),
        "captured_at": _first(
            row,
            "market_captured_at",
            "odds_captured_at",
            "price_captured_at",
            "market_updated_at",
            "odds_updated_at",
            "provider_update",
            "provider_updated_at",
        ),
    }


def _verification_status(value: Any) -> str:
    if value in (None, ""):
        return "NOT VERIFIED"
    if isinstance(value, bool):
        return "CONFIRMED" if value else "NOT CONFIRMED"
    return str(value).strip().upper().replace("_", " ")


def _availability(row: dict[str, Any]) -> dict[str, Any]:
    lineups = _dict(row.get("lineups"))
    availability = _dict(row.get("availability"))
    injuries = row.get("injuries")
    weather = row.get("weather")

    lineup_explicit = _first(row, "lineup_status", "xi_status")
    if lineup_explicit in (None, ""):
        lineup_explicit = availability.get("lineup_status")

    xi_value = lineups.get("both_xi_confirmed")
    gk_value = lineups.get("both_goalkeepers_confirmed")

    injury_status = _first(row, "injury_status", "injuries_status")
    if injury_status in (None, "") and isinstance(injuries, dict):
        injury_status = injuries.get("status")

    weather_status = _first(row, "weather_status")
    if weather_status in (None, "") and isinstance(weather, dict):
        weather_status = weather.get("status")

    confidence = _number(_first(row, "availability_confidence"))
    if confidence is None:
        confidence = _number(availability.get("confidence"))

    coverage = _dict(row.get("coverage"))
    data_tier = _first(row, "data_tier", "quality_tier", "data_quality")
    if data_tier in (None, ""):
        data_tier = coverage.get("data_tier")

    return {
        "confidence": confidence,
        "data_tier": str(data_tier).upper() if data_tier not in (None, "") else "NOT VERIFIED",
        "lineup_status": _verification_status(
            lineup_explicit if lineup_explicit not in (None, "") else xi_value
        ),
        "starting_xi_status": _verification_status(xi_value),
        "goalkeeper_status": _verification_status(gk_value),
        "injury_status": _verification_status(injury_status),
        "weather_status": _verification_status(weather_status),
    }


def adapt_candidate(row: dict[str, Any]) -> dict[str, Any]:
    """Map one persisted row into the V2 subscriber contract without creating a decision."""

    classification = _classification(row)
    execution_status = _execution_status(row)
    tier = str(_first(row, "tier", "stake_tier") or "").strip().upper() or None
    if tier not in {None, "S", "A", "B"}:
        tier = None

    market_family = _first(row, "market_family", "family")
    market_name = _first(row, "market", "market_name")
    selection = _first(row, "selection", "pick")
    period = _first(row, "period")
    line = _number(_first(row, "line", "handicap", "total_line"))

    price = _price(row)
    fair_price = _number(_first(row, "fair_price", "model_fair_price"))

    p_raw = _probability(_first(row, "p_raw", "raw_sport_probability", "raw_probability"))
    p_shrunk = _probability(
        _first(row, "p_shrunk", "market_shrunk_probability", "p_market_shrunk")
    )
    p_calibrated = _probability(
        _first(row, "p_model_calibrated", "calibrated_model_probability", "calibrated_probability")
    )
    p_market = _probability(
        _first(row, "p_market_fair", "p_market_devig", "fair_market_probability")
    )
    p_breakeven = _probability(_first(row, "p_breakeven", "breakeven_probability"))
    edge_pp = _pp(_first(row, "prob_edge_pp", "edge_pp", "raw_edge_vs_market_fair_pp"))
    estimated_ev = _number(_first(row, "estimated_ev"))
    estimated_ev_pct = _number(_first(row, "ev_pct"))

    blockers = _text_list(row, "blockers")
    blocker = _first(row, "blocker")
    if blocker not in (None, "") and str(blocker) not in blockers:
        blockers.append(str(blocker))

    generated_at = _first(
        row,
        "generated_at_utc",
        "generated_at",
        "signal_generated_at",
        "event_generated_at",
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "fixture": _fixture(row),
        "decision": {
            "classification": classification,
            "display_bucket": _display_bucket(classification, execution_status),
            "tier": tier,
            "execution_status": execution_status,
            "stage": _first(row, "stage"),
            "reason_code": _first(row, "reason"),
            "reason_display": _display_reason(
                _first(row, "reason"),
                execution_status,
                classification,
            ),
            "is_canonical_bet": classification == "BET",
            "is_canonical_lean": classification == "LEAN",
        },
        "market": {
            "family": market_family,
            "name": market_name,
            "selection": selection,
            "period": period,
            "line": line,
            "price": price,
            "fair_price": fair_price,
        },
        "projections": {
            "raw_sport_probability": p_raw,
            "market_shrunk_probability": p_shrunk,
            "calibrated_model_probability": p_calibrated,
            "fair_market_probability": p_market,
            "breakeven_probability": p_breakeven,
            "probability_edge_pp": edge_pp,
            "estimated_ev": estimated_ev,
            "estimated_ev_pct": estimated_ev_pct,
        },
        "availability": _availability(row),
        "evidence": {
            "sporting_reasons": _text_list(row, "sporting_reasons", "sport_reasons"),
            "market_reasons": _text_list(row, "market_reasons", "value_reasons"),
            "blockers": blockers,
            "invalidation_conditions": _text_list(
                row, "invalidation_conditions", "invalidators"
            ),
        },
        "stake": {
            "units": _number(_first(row, "stake_units", "recommended_stake_units")),
        },
        "model": {
            "version": _first(row, "model_version"),
            "raw_sport_projection_version": _first(
                row, "raw_sport_projection_version", "sport_projection_version"
            ),
            "market_shrink_version": _first(
                row, "market_shrink_version", "shrinkage_version"
            ),
            "generated_at": generated_at,
        },
        "freshness": {
            "market_fresh": row.get("market_fresh")
            if isinstance(row.get("market_fresh"), bool)
            else None,
            "provider_update": _first(
                row,
                "provider_update",
                "provider_updated_at",
                "market_updated_at",
                "odds_updated_at",
            ),
            "snapshot_generated_at": generated_at,
        },
        "source": SOURCE,
    }


def _candidate_key(candidate: dict[str, Any]) -> tuple[str, ...]:
    fixture = _dict(candidate.get("fixture"))
    market = _dict(candidate.get("market"))
    price = _dict(market.get("price"))
    return (
        str(fixture.get("fixture_id") or ""),
        str(market.get("family") or ""),
        str(market.get("name") or ""),
        str(market.get("selection") or ""),
        str(market.get("period") or ""),
        str(market.get("line") or ""),
        str(price.get("bookmaker") or ""),
        str(price.get("value") or ""),
    )


def _candidate_score(candidate: dict[str, Any]) -> int:
    decision = _dict(candidate.get("decision"))
    projections = _dict(candidate.get("projections"))
    availability = _dict(candidate.get("availability"))
    market = _dict(candidate.get("market"))
    price = _dict(market.get("price"))

    score = 0
    if decision.get("classification") in _CANONICAL_CLASSIFICATIONS:
        score += 20
    if decision.get("tier") in {"S", "A", "B"}:
        score += 3
    for key in (
        "raw_sport_probability",
        "market_shrunk_probability",
        "calibrated_model_probability",
        "fair_market_probability",
        "breakeven_probability",
        "probability_edge_pp",
        "estimated_ev",
        "estimated_ev_pct",
    ):
        if projections.get(key) is not None:
            score += 2
    if price.get("value") is not None:
        score += 2
    if availability.get("confidence") is not None:
        score += 1
    return score


def _all_candidates(views: dict[str, Any]) -> list[dict[str, Any]]:
    best: dict[tuple[str, ...], dict[str, Any]] = {}
    for name in _VIEW_NAMES:
        for raw in _rows(views.get(name)):
            candidate = adapt_candidate(raw)
            key = _candidate_key(candidate)
            prior = best.get(key)
            if prior is None or _candidate_score(candidate) > _candidate_score(prior):
                best[key] = candidate
    return list(best.values())


def _tier_rank(value: Any) -> int:
    return {"S": 0, "A": 1, "B": 2}.get(str(value or "").upper(), 3)


def _sort_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        decision = _dict(row.get("decision"))
        projections = _dict(row.get("projections"))
        fixture = _dict(row.get("fixture"))
        edge = projections.get("probability_edge_pp")
        edge_sort = -float(edge) if edge is not None else float("inf")
        return (
            _tier_rank(decision.get("tier")),
            edge_sort,
            str(fixture.get("kickoff") or ""),
            str(fixture.get("fixture_id") or ""),
        )

    return sorted(rows, key=key)


def _is_watch(candidate: dict[str, Any]) -> bool:
    decision = _dict(candidate.get("decision"))
    if decision.get("classification") == "WATCH":
        return True
    return str(decision.get("execution_status") or "").upper() in _WATCH_EXECUTION_STATUSES


def _public_fixture(row: dict[str, Any]) -> dict[str, Any]:
    fixture = _fixture(row)
    classification = _classification(row)
    execution_status = _execution_status(row)
    status_code = (
        execution_status
        or str(_first(row, "status", "stage") or "NOT VERIFIED").strip().upper()
    )
    if classification in _CANONICAL_CLASSIFICATIONS:
        display_status = classification
    elif status_code in _WATCH_EXECUTION_STATUSES:
        display_status = "WATCH"
    elif status_code == "RESEARCH_ONLY":
        display_status = "RESEARCH"
    else:
        display_status = status_code.replace("_", " ")
    return {
        "fixture": fixture,
        "state": {
            "classification": classification,
            "display_status": display_status,
            "status_code": status_code,
            "stage": _first(row, "stage"),
            "reason_display": _display_reason(
                _first(row, "reason", "blocker"),
                execution_status,
                classification,
            ),
        },
    }


def _public_watch(candidate: dict[str, Any]) -> dict[str, Any]:
    fixture = _dict(candidate.get("fixture"))
    decision = _dict(candidate.get("decision"))
    market = _dict(candidate.get("market"))
    return {
        "fixture": fixture,
        "decision": {
            "classification": decision.get("classification"),
            "display_bucket": decision.get("display_bucket"),
            "execution_status": decision.get("execution_status"),
            "stage": decision.get("stage"),
            "reason_display": decision.get("reason_display"),
        },
        "market": {
            "family": market.get("family"),
            "name": market.get("name"),
            "period": market.get("period"),
        },
        "source": SOURCE,
    }


def _unique_public_slate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        public = _public_fixture(row)
        fixture = _dict(public.get("fixture"))
        key = str(fixture.get("fixture_id") or "")
        if not key:
            key = "|".join(
                str(fixture.get(name) or "")
                for name in ("kickoff", "home_team", "away_team")
            )
        if key in seen:
            continue
        seen.add(key)
        out.append(public)
    out.sort(key=lambda item: str(_dict(item.get("fixture")).get("kickoff") or ""))
    return out



def _date_text(value: Any) -> str | None:
    text = str(value or "").strip()
    if len(text) >= 10 and text[4:5] == "-" and text[7:8] == "-":
        candidate = text[:10]
        parts = candidate.split("-")
        if len(parts) == 3 and all(part.isdigit() for part in parts):
            return candidate
    return None


def _slate_context(payload: dict[str, Any]) -> tuple[str | None, str]:
    timezone_name = str(payload.get("timezone") or "America/Mexico_City").strip() or "America/Mexico_City"
    local_day = _date_text(payload.get("generated_at_local"))
    utc_day = _date_text(payload.get("generated_at_utc"))
    recon = _dict(payload.get("core_slate_floor_reconciliation"))
    scan_dates = [
        value for value in (_date_text(item) for item in (recon.get("scan_dates") or []))
        if value
    ]
    if local_day:
        return local_day, timezone_name
    if scan_dates:
        return scan_dates[0], timezone_name
    return utc_day, timezone_name


def _load_registry_slate(payload: dict[str, Any]) -> dict[str, Any]:
    slate_day, timezone_name = _slate_context(payload)
    if not slate_day:
        return {
            "status": "DATE_NOT_VERIFIED",
            "rows": [],
            "slate_date": None,
            "timezone": timezone_name,
        }

    with persistence_base._connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT fixture_id, kickoff, status, league, country,
                       home_team_id, home_team, away_team_id, away_team
                FROM soccer_fixtures
                WHERE (kickoff AT TIME ZONE %s)::date = %s::date
                  AND COALESCE(status, 'NS') NOT IN
                      ('FT','AET','PEN','CANC','PST','ABD','AWD','WO')
                ORDER BY kickoff, fixture_id
                """,
                (timezone_name, slate_day),
            )
            db_rows = cur.fetchall()

    rows: list[dict[str, Any]] = []
    for db_row in db_rows:
        fixture_id, kickoff, status, league, country, home_id, home, away_id, away = db_row
        rows.append(
            {
                "fixture_id": fixture_id,
                "kickoff": kickoff.isoformat() if hasattr(kickoff, "isoformat") else kickoff,
                "status": status,
                "league": league,
                "country": country,
                "home_team_id": home_id,
                "home_team": home,
                "away_team_id": away_id,
                "away_team": away,
            }
        )
    return {
        "status": "FULL_SLATE_READY",
        "rows": rows,
        "slate_date": slate_day,
        "timezone": timezone_name,
    }


def _market_coverage_label(candidate: dict[str, Any]) -> str | None:
    market = _dict(candidate.get("market"))
    family = str(market.get("family") or "").strip().upper()
    if family in _MARKET_COVERAGE_LABELS:
        return _MARKET_COVERAGE_LABELS[family]
    name = str(market.get("name") or "").strip()
    return name or family or None


def _coverage_by_fixture(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    source = payload.get("match_table_rows")
    raw_rows = [row for row in source if isinstance(row, dict)] if isinstance(source, list) else []

    for raw in raw_rows:
        status = str(raw.get("status") or "").upper()
        stage = str(raw.get("stage") or "").upper()
        if status in _TERMINAL_FIXTURE_STATUSES or stage == "POSTGAME":
            continue

        candidate = adapt_candidate(raw)
        fixture = _dict(candidate.get("fixture"))
        fixture_id = fixture.get("fixture_id")
        if fixture_id in (None, ""):
            continue
        key = str(fixture_id)
        node = grouped.setdefault(
            key,
            {
                "analysis_rows": 0,
                "markets": [],
                "verified_price_markets": [],
                "raw_sport_projection_present": False,
                "availability_confidence_present": False,
                "decision_status": None,
                "decision_reason": None,
                "best_market": None,
            },
        )
        node["analysis_rows"] += 1

        label = _market_coverage_label(candidate)
        if label and label not in node["markets"]:
            node["markets"].append(label)

        market = _dict(candidate.get("market"))
        price = _dict(market.get("price"))
        if (
            label
            and price.get("value") is not None
            and (price.get("bookmaker") or price.get("source"))
            and price.get("captured_at")
            and label not in node["verified_price_markets"]
        ):
            node["verified_price_markets"].append(label)

        projections = _dict(candidate.get("projections"))
        if projections.get("raw_sport_probability") is not None:
            node["raw_sport_projection_present"] = True

        availability = _dict(candidate.get("availability"))
        if availability.get("confidence") is not None:
            node["availability_confidence_present"] = True

        decision = _dict(candidate.get("decision"))
        classification = str(
            decision.get("classification") or decision.get("display_bucket") or ""
        ).upper()
        if not classification and str(decision.get("execution_status") or "").upper() in _WATCH_EXECUTION_STATUSES:
            classification = "WATCH"
        current = str(node.get("decision_status") or "").upper()
        if _DECISION_PRIORITY.get(classification, 0) > _DECISION_PRIORITY.get(current, 0):
            node["decision_status"] = classification
            node["decision_reason"] = decision.get("reason_display")
            node["best_market"] = label

    for node in grouped.values():
        decision_status = str(node.get("decision_status") or "").upper()
        if decision_status in _CANONICAL_CLASSIFICATIONS:
            coverage_status = decision_status
        elif node["analysis_rows"] > 0:
            coverage_status = "PARTIAL_DATA"
        else:
            coverage_status = "INSUFFICIENT_DATA"
        node["coverage_status"] = coverage_status
    return grouped


def _load_registry_fixture(fixture_value: Any) -> dict[str, Any] | None:
    try:
        fixture_id = int(fixture_value)
    except (TypeError, ValueError):
        return None
    with persistence_base._connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT fixture_id, kickoff, status, league, country,
                       home_team_id, home_team, away_team_id, away_team
                FROM soccer_fixtures
                WHERE fixture_id = %s
                LIMIT 1
                """,
                (fixture_id,),
            )
            row = cur.fetchone()
    if not row:
        return None
    fixture_id, kickoff, status, league, country, home_id, home, away_id, away = row
    return {
        "fixture_id": fixture_id,
        "kickoff": kickoff.isoformat() if hasattr(kickoff, "isoformat") else kickoff,
        "status": status,
        "league": league,
        "country": country,
        "home_team_id": home_id,
        "home_team": home,
        "away_team_id": away_id,
        "away_team": away,
    }


def _registry_fixture(row: dict[str, Any]) -> dict[str, Any]:
    home_id = row.get("home_team_id")
    away_id = row.get("away_team_id")
    try:
        home_id_int = int(home_id) if home_id not in (None, "") else None
    except (TypeError, ValueError):
        home_id_int = None
    try:
        away_id_int = int(away_id) if away_id not in (None, "") else None
    except (TypeError, ValueError):
        away_id_int = None
    return {
        "fixture_id": row.get("fixture_id"),
        "kickoff": row.get("kickoff"),
        "league": row.get("league"),
        "country": row.get("country"),
        "home_team_id": home_id_int,
        "home_team": row.get("home_team"),
        "home_team_logo": (
            f"https://media.api-sports.io/football/teams/{home_id_int}.png"
            if home_id_int
            else None
        ),
        "away_team_id": away_id_int,
        "away_team": row.get("away_team"),
        "away_team_logo": (
            f"https://media.api-sports.io/football/teams/{away_id_int}.png"
            if away_id_int
            else None
        ),
        "fixture_status": row.get("status"),
    }


def _attach_full_registry_slate(
    contract_payload: dict[str, Any],
    payload: dict[str, Any],
    registry: dict[str, Any] | None,
) -> dict[str, Any]:
    slate = _dict(contract_payload.get("slate"))
    if not isinstance(registry, dict):
        slate["full_slate"] = False
        slate["registry_status"] = "REGISTRY_UNAVAILABLE"
        slate["source"] = "PERSISTED_ANALYSIS_ROWS_FALLBACK"
        contract_payload["slate"] = slate
        return contract_payload

    coverage = _coverage_by_fixture(payload)
    full_rows: list[dict[str, Any]] = []
    summary = {
        "BET": 0,
        "LEAN": 0,
        "WATCH": 0,
        "PASS": 0,
        "PARTIAL_DATA": 0,
        "INSUFFICIENT_DATA": 0,
    }

    for row in registry.get("rows") or []:
        if not isinstance(row, dict):
            continue
        fixture = _registry_fixture(row)
        key = str(fixture.get("fixture_id") or "")
        cov = dict(coverage.get(key) or {})
        if cov:
            status = str(cov.get("coverage_status") or "PARTIAL_DATA").upper()
            reason = cov.get("decision_reason")
            if not reason:
                reason = (
                    "Persisted analysis exists for "
                    + ", ".join(cov.get("markets") or [])
                    if cov.get("markets")
                    else "Persisted analysis is partial; unsupported or missing fields remain NOT VERIFIED."
                )
        else:
            status = "INSUFFICIENT_DATA"
            cov = {
                "analysis_rows": 0,
                "markets": [],
                "verified_price_markets": [],
                "raw_sport_projection_present": False,
                "availability_confidence_present": False,
                "decision_status": None,
                "decision_reason": None,
                "best_market": None,
                "coverage_status": status,
            }
            reason = "Fixture is in the eligible slate, but deep analysis is not persisted yet."

        summary[status] = summary.get(status, 0) + 1
        full_rows.append(
            {
                "fixture": fixture,
                "state": {
                    "classification": cov.get("decision_status"),
                    "display_status": status.replace("_", " "),
                    "status_code": status,
                    "stage": None,
                    "reason_display": reason,
                },
                "coverage": cov,
            }
        )

    slate.update(
        {
            "rows": full_rows,
            "total": len(full_rows),
            "full_slate": True,
            "eligible_only": True,
            "terminal_statuses_excluded": sorted(_TERMINAL_FIXTURE_STATUSES),
            "registry_status": registry.get("status"),
            "slate_date": registry.get("slate_date"),
            "timezone": registry.get("timezone"),
            "source": "POSTGRES_SOCCER_FIXTURES+PERSISTED_ANALYSIS_COVERAGE",
            "coverage_summary": summary,
        }
    )
    contract_payload["slate"] = slate
    counts = _dict(contract_payload.get("counts"))
    counts["fixtures"] = len(full_rows)
    contract_payload["counts"] = counts
    return contract_payload


def _normalize_entitlement(entitlement: dict[str, Any]) -> dict[str, Any]:
    out = dict(entitlement)
    user = _dict(out.get("user"))
    role = str(user.get("role") or "").upper()
    owner = bool(out.get("owner")) or role == "OWNER"
    admin = bool(out.get("admin")) or owner or role == "ADMIN"
    if owner or admin:
        out["owner"] = owner
        out["admin"] = admin
        out["effective_plan"] = subscription_entitlements_v4.PRO_PLAN
        out["effective_plan_reason"] = "OWNER_ADMIN_PRESENTATION_ACCESS"
        out["feature_access"] = subscription_entitlements_v4.feature_access(
            subscription_entitlements_v4.PRO_PLAN
        )
    return out


def _access(entitlement: dict[str, Any]) -> dict[str, Any]:
    user = _dict(entitlement.get("user"))
    plan = str(
        entitlement.get("effective_plan") or subscription_entitlements_v4.FREE_PLAN
    ).upper()
    owner = bool(entitlement.get("owner")) or str(user.get("role") or "").upper() == "OWNER"
    admin = bool(entitlement.get("admin")) or owner or str(user.get("role") or "").upper() == "ADMIN"
    premium = plan == subscription_entitlements_v4.PRO_PLAN or owner or admin
    return {
        "authenticated": bool(entitlement.get("authenticated")),
        "effective_plan": subscription_entitlements_v4.PRO_PLAN if premium else plan,
        "display_role": "OWNER" if owner else ("ADMIN" if admin else plan),
        "owner": owner,
        "admin": admin,
        "premium_unlocked": premium,
        "feature_access": entitlement.get("feature_access")
        or subscription_entitlements_v4.feature_access(
            subscription_entitlements_v4.PRO_PLAN if premium else plan
        ),
        "effective_plan_reason": entitlement.get("effective_plan_reason"),
    }


def build_contract(payload: dict[str, Any], entitlement: dict[str, Any]) -> dict[str, Any]:
    """Build the read-only V2 subscriber contract from one persisted pipeline snapshot."""

    source = dict(payload)
    source.setdefault("status", "ok")
    source["database_persisted"] = True
    source["database_error"] = None
    product = product_views_v4.build_views(source, limit=product_views_v4.MAX_ROWS_PER_VIEW)
    views = _dict(product.get("views"))

    candidates = _all_candidates(views)
    picks = _sort_candidates(
        [row for row in candidates if _dict(row.get("decision")).get("classification") == "BET"]
    )
    leans = _sort_candidates(
        [row for row in candidates if _dict(row.get("decision")).get("classification") == "LEAN"]
    )
    watches = _sort_candidates([row for row in candidates if _is_watch(row)])
    passes = [
        row for row in candidates
        if _dict(row.get("decision")).get("classification") == "PASS"
    ]

    access = _access(_normalize_entitlement(entitlement))
    premium = bool(access["premium_unlocked"])
    slate_raw = _rows(views.get("todays_slate"))
    slate = _unique_public_slate(slate_raw)

    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": "SUBSCRIBER_CONTRACT_V2_READY",
        "source": SOURCE,
        "generated_at_utc": source.get("generated_at_utc"),
        "generated_at_local": source.get("generated_at_local"),
        "pipeline_version": source.get("version"),
        "runtime_model_version": source.get("model_version"),
        "access": access,
        "counts": {
            "fixtures": len(slate),
            "picks": len(picks),
            "leans": len(leans),
            "watches": len(watches),
            "passes": len(passes),
        },
        "slate": {
            "rows": slate,
            "total": len(slate),
            "premium_values_redacted": not premium,
        },
        "picks": {
            "rows": picks if premium else [],
            "total": len(picks),
            "locked": not premium,
            "classification_policy": "EXPLICIT_PERSISTED_BET_ONLY",
        },
        "leans": {
            "rows": leans if premium else [],
            "total": len(leans),
            "locked": not premium,
            "classification_policy": "EXPLICIT_PERSISTED_LEAN_ONLY",
        },
        "watches": {
            "rows": watches if premium else [_public_watch(row) for row in watches],
            "total": len(watches),
            "locked": False,
            "classification_policy": "EXPLICIT_WATCH_OR_CANONICAL_WAIT_STATE",
        },
        "provider_requests_added": 0,
        "canonical_bet_logic_changed": False,
        "model_weights_changed": False,
        "production_promotion_allowed": False,
    }


async def _resolve_entitlement(request: Request) -> tuple[dict[str, Any] | None, JSONResponse | None]:
    token = supabase_auth_v4.bearer_token(request.headers.get("authorization"))
    if not token:
        return subscriber_app_v4.anonymous_entitlement(), None
    entitlement = await asyncio.to_thread(subscription_entitlements_v4.resolve_entitlement, token)
    if not entitlement.get("ok") or not entitlement.get("authenticated"):
        return None, JSONResponse(
            {"error": entitlement.get("status") or "AUTH_REQUIRED"},
            status_code=401,
        )
    return _normalize_entitlement(entitlement), None


async def _load_snapshot() -> dict[str, Any] | None:
    payload = await asyncio.to_thread(persistence_base.load_latest_pipeline_payload)
    return payload if isinstance(payload, dict) else None


async def _contract_for_request(
    request: Request,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, JSONResponse | None]:
    entitlement, error = await _resolve_entitlement(request)
    if error is not None or entitlement is None:
        return None, None, error
    try:
        payload = await _load_snapshot()
    except Exception as exc:
        return None, None, JSONResponse(
            {"error": "SUBSCRIBER_V2_UNAVAILABLE", "detail": str(exc)[:200]},
            status_code=503,
        )
    if payload is None:
        return None, None, JSONResponse(
            {"error": "NO_PERSISTED_PIPELINE_RUN"},
            status_code=503,
        )
    contract_payload = build_contract(payload, entitlement)
    try:
        registry = await asyncio.to_thread(_load_registry_slate, payload)
    except Exception:
        registry = None
    contract_payload = _attach_full_registry_slate(contract_payload, payload, registry)
    return contract_payload, payload, None


def _no_store(payload: dict[str, Any], status_code: int = 200) -> JSONResponse:
    return JSONResponse(payload, status_code=status_code, headers={"Cache-Control": "no-store"})


async def today(request: Request) -> JSONResponse:
    contract_payload, _, error = await _contract_for_request(request)
    if error is not None:
        return error
    assert contract_payload is not None
    return _no_store(contract_payload)


async def picks(request: Request) -> JSONResponse:
    contract_payload, _, error = await _contract_for_request(request)
    if error is not None:
        return error
    assert contract_payload is not None
    if not _dict(contract_payload.get("access")).get("premium_unlocked"):
        return _no_store(
            {
                "error": "PRO_REQUIRED",
                "resource": "picks",
                "total": _dict(contract_payload.get("picks")).get("total", 0),
            },
            status_code=403,
        )
    return _no_store(
        {
            "schema_version": SCHEMA_VERSION,
            "model_version": MODEL_VERSION,
            "source": SOURCE,
            "generated_at_utc": contract_payload.get("generated_at_utc"),
            "picks": contract_payload.get("picks"),
            "provider_requests_added": 0,
            "canonical_bet_logic_changed": False,
            "model_weights_changed": False,
        }
    )


async def leans(request: Request) -> JSONResponse:
    contract_payload, _, error = await _contract_for_request(request)
    if error is not None:
        return error
    assert contract_payload is not None
    if not _dict(contract_payload.get("access")).get("premium_unlocked"):
        return _no_store(
            {
                "error": "PRO_REQUIRED",
                "resource": "leans",
                "total": _dict(contract_payload.get("leans")).get("total", 0),
            },
            status_code=403,
        )
    return _no_store(
        {
            "schema_version": SCHEMA_VERSION,
            "model_version": MODEL_VERSION,
            "source": SOURCE,
            "generated_at_utc": contract_payload.get("generated_at_utc"),
            "leans": contract_payload.get("leans"),
            "provider_requests_added": 0,
            "canonical_bet_logic_changed": False,
            "model_weights_changed": False,
        }
    )


async def watches(request: Request) -> JSONResponse:
    contract_payload, _, error = await _contract_for_request(request)
    if error is not None:
        return error
    assert contract_payload is not None
    return _no_store(
        {
            "schema_version": SCHEMA_VERSION,
            "model_version": MODEL_VERSION,
            "source": SOURCE,
            "generated_at_utc": contract_payload.get("generated_at_utc"),
            "access": contract_payload.get("access"),
            "watches": contract_payload.get("watches"),
            "provider_requests_added": 0,
            "canonical_bet_logic_changed": False,
            "model_weights_changed": False,
        }
    )


async def account(request: Request) -> JSONResponse:
    entitlement, error = await _resolve_entitlement(request)
    if error is not None:
        return error
    assert entitlement is not None
    access = _access(entitlement)
    return _no_store(
        {
            "schema_version": SCHEMA_VERSION,
            "model_version": MODEL_VERSION,
            "status": "ACCOUNT_ACCESS_READY",
            "access": access,
            "user": entitlement.get("user") if access["authenticated"] else None,
            "subscription_required": entitlement.get("subscription_required"),
            "billing_state": subscriber_billing_v2.account_billing_state(entitlement),
            "provider_requests_added": 0,
            "canonical_bet_logic_changed": False,
            "model_weights_changed": False,
            "production_promotion_allowed": False,
        }
    )


def build_performance_contract(
    track_record: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Any]:
    bet_only = _dict(track_record.get("bet_only"))
    research_lean = _dict(track_record.get("research_lean"))
    settlement = _dict(track_record.get("settlement"))
    true_clv = _dict(track_record.get("true_clv"))

    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": (
            "VERIFIED_PERFORMANCE_READY"
            if track_record.get("status") == "ACTIVE"
            else "VERIFIED_PERFORMANCE_SAMPLE_NOT_READY"
        ),
        "performance_policy": {
            "headline_scope": "CANONICAL_BET_SETTLEMENT_ONLY",
            "leans_in_headline": False,
            "research_oos_is_realized_bet_performance": False,
            "historical_reconstruction_performed": (
                track_record.get("historical_reconstruction_performed") is True
            ),
            "settlement_backfill_performed": (
                track_record.get("settlement_backfill_performed") is True
            ),
        },
        "bet_track_record": {
            "status": track_record.get("status"),
            "sample_status": bet_only.get("sample_status"),
            "directional_minimum": bet_only.get("directional_minimum"),
            "settled": bet_only.get("settled"),
            "win": bet_only.get("win"),
            "loss": bet_only.get("loss"),
            "push": bet_only.get("push"),
            "ungraded": bet_only.get("ungraded"),
            "hit_rate_ex_push": bet_only.get("hit_rate_ex_push"),
            "roi_units": bet_only.get("roi_units"),
            "families": list(bet_only.get("families") or []),
        },
        "research_lean": {
            "classification": research_lean.get("classification"),
            "settled": research_lean.get("settled"),
            "win": research_lean.get("win"),
            "loss": research_lean.get("loss"),
            "push": research_lean.get("push"),
            "ungraded": research_lean.get("ungraded"),
            "hit_rate_ex_push": research_lean.get("hit_rate_ex_push"),
            "roi_units": research_lean.get("roi_units"),
            "families": list(research_lean.get("families") or []),
            "headline_eligible": False,
        },
        "settlement": settlement,
        "true_clv": true_clv,
        "validation_evidence": {
            "status": validation.get("status"),
            "kind": "PERSISTED_VALIDATION_OOS_EVIDENCE",
            "rows": validation.get("rows") or [],
            "weighted_avg_clv_pp": validation.get("weighted_avg_clv_pp"),
            "totals": validation.get("totals") or {},
            "errors": validation.get("errors") or {},
        },
        "notes": [
            (
                "Headline performance uses canonical settled BET rows only. "
                "LEAN rows are displayed separately and never improve the BET headline."
            ),
            (
                "Validation/OOS evidence is research evidence, not realized customer "
                "BET performance."
            ),
            (
                "Small samples remain visibly labeled; no performance claim is upgraded "
                "because of presentation."
            ),
        ],
        "source": {
            "bet_track_record_model_version": track_record.get("model_version"),
            "validation_model_version": validation.get("model_version"),
            "generated_at_utc": track_record.get("generated_at_utc"),
        },
        "provider_requests_added": 0,
        "canonical_bet_logic_changed": False,
        "model_weights_changed": False,
        "production_promotion_allowed": False,
    }


async def performance(request: Request) -> JSONResponse:
    entitlement, error = await _resolve_entitlement(request)
    if error is not None:
        return error
    assert entitlement is not None
    access = _access(entitlement)
    if not access["premium_unlocked"]:
        return _no_store({"error": "PRO_REQUIRED", "resource": "performance"}, status_code=403)

    track_record, validation = await asyncio.gather(
        asyncio.to_thread(public_performance_v4.load_snapshot),
        asyncio.to_thread(subscriber_validation_metrics_v231.load_validation_metrics),
    )
    return _no_store(build_performance_contract(track_record, validation))



def _my_edge_allowed(entitlement: dict[str, Any]) -> bool:
    access = _access(entitlement)
    features = _dict(access.get("feature_access"))
    return bool(access.get("premium_unlocked")) and features.get("favorites_alerts") is True


async def my_edge(request: Request) -> JSONResponse:
    token = supabase_auth_v4.bearer_token(request.headers.get("authorization"))
    if not token:
        return _no_store({"error": "AUTH_REQUIRED", "resource": "my_edge"}, status_code=401)

    entitlement, error = await _resolve_entitlement(request)
    if error is not None:
        return error
    assert entitlement is not None
    if not _my_edge_allowed(entitlement):
        return _no_store({"error": "PRO_REQUIRED", "resource": "my_edge"}, status_code=403)

    user_id = str(_dict(entitlement.get("user")).get("id") or "").strip()
    if not user_id:
        return _no_store({"error": "AUTH_INVALID_USER_PAYLOAD", "resource": "my_edge"}, status_code=401)

    if request.method == "GET":
        result = await asyncio.to_thread(
            subscriber_saved_items_v4.list_saved,
            token,
            verified_user_id=user_id,
        )
    elif request.method == "POST":
        try:
            body = await request.json()
        except Exception:
            return _no_store({"error": "INVALID_JSON"}, status_code=400)
        if not isinstance(body, dict):
            return _no_store({"error": "INVALID_ITEM_PAYLOAD"}, status_code=400)
        result = await asyncio.to_thread(
            subscriber_saved_items_v4.save_item,
            token,
            body,
            verified_user_id=user_id,
        )
    elif request.method == "DELETE":
        item_key = str(request.query_params.get("item_key") or "").strip()
        if not item_key:
            return _no_store({"error": "ITEM_KEY_REQUIRED"}, status_code=400)
        result = await asyncio.to_thread(
            subscriber_saved_items_v4.delete_item,
            token,
            item_key,
            verified_user_id=user_id,
        )
    else:
        return _no_store({"error": "METHOD_NOT_ALLOWED"}, status_code=405)

    if not result.get("ok"):
        status = str(result.get("status") or "MY_EDGE_UNAVAILABLE")
        code = 400 if status in {
            "ITEM_TYPE_NOT_ALLOWED",
            "VALID_FIXTURE_ID_REQUIRED",
            "ITEM_KEY_REQUIRED",
        } else 503
        return _no_store(
            {
                "error": status,
                "resource": "my_edge",
                "provider_requests_added": 0,
            },
            status_code=code,
        )

    return _no_store(
        {
            "schema_version": SCHEMA_VERSION,
            "model_version": MODEL_VERSION,
            "status": result.get("status"),
            "resource": "my_edge",
            "rows": result.get("rows") if isinstance(result.get("rows"), list) else None,
            "row": result.get("row") if isinstance(result.get("row"), dict) else None,
            "item_key": result.get("item_key"),
            "provider_requests_added": 0,
            "canonical_bet_logic_changed": False,
            "model_weights_changed": False,
            "production_promotion_allowed": False,
        }
    )

def _match_market_group(candidate: dict[str, Any]) -> str:
    market = _dict(candidate.get("market"))
    family = str(market.get("family") or "").upper()
    name = str(market.get("name") or "").upper()
    text = f"{family} {name}"
    if "CORNER" in text:
        return "CORNERS"
    if "CARD" in text or "BOOKING" in text:
        return "CARDS"
    if "PLAYER" in text or "PROP" in text:
        return "PLAYERS"
    if (
        family in {"FT_TOTALS", "TEAM_TOTALS", "BTTS", "FIRST_HALF", "SECOND_HALF"}
        or "GOAL" in text
        or "TOTAL" in text
    ):
        return "GOALS"
    return "GENERAL"


def build_match_contract(
    payload: dict[str, Any],
    fixture_value: Any,
    registry_fixture: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Build one fixture intelligence packet from persisted evidence only.

    Registry-only fixtures are returned as explicit INSUFFICIENT_DATA packets so a
    human analyst can inspect every eligible match without fabricating missing inputs.
    """

    raw_rows = subscriber_preview_data_v231._fixture_rows(payload, fixture_value)
    if not raw_rows:
        if not isinstance(registry_fixture, dict):
            return None
        fixture = _registry_fixture(registry_fixture)
        missing = [
            "PERSISTED_ANALYSIS_ROWS",
            "OUTCOME_PROBABILITIES",
            "EXPECTED_GOALS",
            "SCORE_MATRIX",
            "SPORT_PROFILE",
            "AVAILABILITY_CONFIDENCE",
            "EXACT_PRICE",
        ]
        return {
            "schema_version": SCHEMA_VERSION,
            "model_version": MODEL_VERSION,
            "status": "MATCH_INTELLIGENCE_INSUFFICIENT_DATA",
            "source": "POSTGRES_SOCCER_FIXTURES",
            "generated_at_utc": payload.get("generated_at_utc"),
            "fixture": fixture,
            "selected_candidate": None,
            "decision_summary": {
                "classification": None,
                "display_bucket": "INSUFFICIENT DATA",
                "tier": None,
                "execution_status": None,
                "stage": None,
                "reason_display": (
                    "Fixture is in the eligible slate, but no persisted deep-analysis "
                    "rows exist for it yet."
                ),
            },
            "projection_ladder": {
                "raw_sport_probability": None,
                "market_shrunk_probability": None,
                "calibrated_model_probability": None,
                "fair_market_probability": None,
                "breakeven_probability": None,
                "probability_edge_pp": None,
                "estimated_ev": None,
                "estimated_ev_pct": None,
            },
            "availability": _availability({}),
            "evidence": {
                "sporting_reasons": [],
                "market_reasons": [],
                "blockers": ["PERSISTED_ANALYSIS_NOT_AVAILABLE"],
                "invalidation_conditions": [],
            },
            "market_context": {
                "selected": {},
                "candidate_count": 0,
                "candidates": [],
                "groups": {
                    "GOALS": [],
                    "CORNERS": [],
                    "CARDS": [],
                    "PLAYERS": [],
                    "GENERAL": [],
                },
            },
            "sport_context": {
                "outcome_probabilities": None,
                "expected_goals": None,
                "score_matrix": [],
                "sport_profile": [],
            },
            "model_context": {
                "confidence": None,
                "data_quality": None,
                "lineup": None,
                "model_disagreement": None,
                "models_agreeing": None,
                "models_total": None,
                "model_version": None,
                "stage": None,
                "provider_update": None,
                "bookmaker": None,
                "selected_model": {},
                "freshness": {},
            },
            "analyst_review": {
                "status": "INSUFFICIENT_DATA",
                "analysis_rows": 0,
                "available_markets": [],
                "verified_price_markets": [],
                "available_sections": ["FIXTURE_IDENTITY"],
                "missing_sections": missing,
                "human_review_allowed": True,
                "note": (
                    "This screen exposes what is actually persisted for human review. "
                    "No betting edge is implied by fixture presence alone."
                ),
            },
            "data_disclosure": {
                "persisted_fixture_row_count": 0,
                "missing_sections": missing,
                "unknown_policy": "NOT VERIFIED",
                "provider_requests_added": 0,
            },
            "provider_requests_added": 0,
            "canonical_bet_logic_changed": False,
            "model_weights_changed": False,
            "production_promotion_allowed": False,
        }

    candidates = _sort_candidates([adapt_candidate(row) for row in raw_rows])
    selected_raw = max(raw_rows, key=lambda row: _candidate_score(adapt_candidate(row)))
    selected = adapt_candidate(selected_raw)
    selected_v231 = subscriber_ui_contract_v231.adapt_market_row(selected_raw)
    detail = subscriber_preview_data_v231._match_detail(payload, selected_v231) or {}

    groups: dict[str, list[dict[str, Any]]] = {
        "GOALS": [],
        "CORNERS": [],
        "CARDS": [],
        "PLAYERS": [],
        "GENERAL": [],
    }
    for candidate in candidates:
        groups[_match_market_group(candidate)].append(candidate)

    model_context_raw = _dict(detail.get("model_context"))
    model_context = {
        "confidence": model_context_raw.get("confidence"),
        "data_quality": model_context_raw.get("data_quality"),
        "lineup": model_context_raw.get("lineup"),
        "model_disagreement": model_context_raw.get("model_disagreement"),
        "models_agreeing": model_context_raw.get("models_agreeing"),
        "models_total": model_context_raw.get("models_total"),
        "model_version": model_context_raw.get("model_version"),
        "stage": model_context_raw.get("stage"),
        "provider_update": model_context_raw.get("provider_update"),
        "bookmaker": model_context_raw.get("bookmaker"),
    }

    availability = _dict(selected.get("availability"))
    evidence = _dict(selected.get("evidence"))
    projections = _dict(selected.get("projections"))
    market = _dict(selected.get("market"))
    decision = _dict(selected.get("decision"))
    model = _dict(selected.get("model"))
    freshness = _dict(selected.get("freshness"))

    missing: list[str] = []
    if not detail.get("outcome_probabilities"):
        missing.append("OUTCOME_PROBABILITIES")
    if not detail.get("expected_goals"):
        missing.append("EXPECTED_GOALS")
    if not detail.get("score_matrix"):
        missing.append("SCORE_MATRIX")
    if not detail.get("sport_profile"):
        missing.append("SPORT_PROFILE")
    if availability.get("confidence") is None:
        missing.append("AVAILABILITY_CONFIDENCE")
    if market.get("price", {}).get("value") is None:
        missing.append("EXACT_PRICE")

    available_markets: list[str] = []
    verified_price_markets: list[str] = []
    for candidate in candidates:
        label = _market_coverage_label(candidate)
        if label and label not in available_markets:
            available_markets.append(label)
        candidate_market = _dict(candidate.get("market"))
        candidate_price = _dict(candidate_market.get("price"))
        if (
            label
            and candidate_price.get("value") is not None
            and (candidate_price.get("bookmaker") or candidate_price.get("source"))
            and candidate_price.get("captured_at")
            and label not in verified_price_markets
        ):
            verified_price_markets.append(label)

    available_sections = ["FIXTURE_IDENTITY", "MARKET_ROWS"]
    if detail.get("outcome_probabilities"):
        available_sections.append("OUTCOME_PROBABILITIES")
    if detail.get("expected_goals"):
        available_sections.append("EXPECTED_GOALS")
    if detail.get("score_matrix"):
        available_sections.append("SCORE_MATRIX")
    if detail.get("sport_profile"):
        available_sections.append("SPORT_PROFILE")
    if availability.get("confidence") is not None:
        available_sections.append("AVAILABILITY_CONFIDENCE")
    if verified_price_markets:
        available_sections.append("VERIFIED_PRICE")

    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": "MATCH_INTELLIGENCE_READY",
        "source": SOURCE,
        "generated_at_utc": payload.get("generated_at_utc"),
        "fixture": _fixture(selected_raw),
        "selected_candidate": selected,
        "decision_summary": {
            "classification": decision.get("classification"),
            "display_bucket": decision.get("display_bucket"),
            "tier": decision.get("tier"),
            "execution_status": decision.get("execution_status"),
            "stage": decision.get("stage"),
            "reason_display": decision.get("reason_display"),
        },
        "projection_ladder": {
            "raw_sport_probability": projections.get("raw_sport_probability"),
            "market_shrunk_probability": projections.get("market_shrunk_probability"),
            "calibrated_model_probability": projections.get("calibrated_model_probability"),
            "fair_market_probability": projections.get("fair_market_probability"),
            "breakeven_probability": projections.get("breakeven_probability"),
            "probability_edge_pp": projections.get("probability_edge_pp"),
            "estimated_ev": projections.get("estimated_ev"),
            "estimated_ev_pct": projections.get("estimated_ev_pct"),
        },
        "availability": availability,
        "evidence": {
            "sporting_reasons": list(evidence.get("sporting_reasons") or []),
            "market_reasons": list(evidence.get("market_reasons") or []),
            "blockers": list(evidence.get("blockers") or []),
            "invalidation_conditions": list(
                evidence.get("invalidation_conditions") or []
            ),
        },
        "market_context": {
            "selected": market,
            "candidate_count": len(candidates),
            "candidates": candidates,
            "groups": groups,
        },
        "sport_context": {
            "outcome_probabilities": detail.get("outcome_probabilities"),
            "expected_goals": detail.get("expected_goals"),
            "score_matrix": detail.get("score_matrix") or [],
            "sport_profile": detail.get("sport_profile") or [],
        },
        "model_context": {
            **model_context,
            "selected_model": model,
            "freshness": freshness,
        },
        "analyst_review": {
            "status": (
                "ACTIONABLE_MODEL_STATE"
                if str(decision.get("classification") or "").upper() in _CANONICAL_CLASSIFICATIONS
                else "PARTIAL_DATA"
            ),
            "analysis_rows": len(raw_rows),
            "available_markets": available_markets,
            "verified_price_markets": verified_price_markets,
            "available_sections": available_sections,
            "missing_sections": missing,
            "human_review_allowed": True,
            "note": (
                "Human review may inspect all persisted sporting and market evidence, "
                "including partial markets. This view does not create or promote a BET."
            ),
        },
        "data_disclosure": {
            "persisted_fixture_row_count": len(raw_rows),
            "missing_sections": missing,
            "unknown_policy": "NOT VERIFIED",
            "provider_requests_added": 0,
        },
        "provider_requests_added": 0,
        "canonical_bet_logic_changed": False,
        "model_weights_changed": False,
        "production_promotion_allowed": False,
    }


async def match_detail(request: Request) -> JSONResponse:
    entitlement, error = await _resolve_entitlement(request)
    if error is not None:
        return error
    assert entitlement is not None
    access = _access(entitlement)
    if not access["premium_unlocked"]:
        return _no_store({"error": "PRO_REQUIRED", "resource": "match_detail"}, status_code=403)

    fixture_text = str(request.path_params.get("fixture_id") or "").strip()
    if not fixture_text:
        return _no_store({"error": "FIXTURE_ID_REQUIRED"}, status_code=400)
    fixture_value: Any = int(fixture_text) if fixture_text.isdigit() else fixture_text

    try:
        payload = await _load_snapshot()
    except Exception as exc:
        return _no_store(
            {"error": "MATCH_DATA_UNAVAILABLE", "detail": str(exc)[:200]},
            status_code=503,
        )
    if payload is None:
        return _no_store({"error": "NO_PERSISTED_PIPELINE_RUN"}, status_code=503)

    try:
        registry_fixture = await asyncio.to_thread(_load_registry_fixture, fixture_value)
    except Exception:
        registry_fixture = None

    result = build_match_contract(payload, fixture_value, registry_fixture)
    if result is None:
        return _no_store({"error": "FIXTURE_NOT_IN_REGISTRY_OR_PERSISTED_SNAPSHOT"}, status_code=404)
    return _no_store(result)


def contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "resources": ["today", "picks", "leans", "watches", "match", "performance", "my_edge", "account"],
        "raw_sport_probability_is_distinct": True,
        "market_shrunk_probability_is_distinct": True,
        "calibrated_probability_is_distinct": True,
        "fair_market_probability_is_distinct": True,
        "my_edge_persistence": "SUPABASE_RLS_SUBSCRIBER_SAVED_ITEMS",
        "my_edge_model_input_allowed": False,
        "billing_contract": "GUARDED_STRIPE_HOSTED_CHECKOUT_CUSTOMER_PORTAL",
        "billing_model_input_allowed": False,
        "frontend_creates_bet_or_lean": False,
        "missing_verification_policy": "NOT VERIFIED",
        "free_premium_values_redacted": True,
        "provider_requests_added": 0,
        "canonical_bet_logic_changed": False,
        "model_weights_changed": False,
        "production_promotion_allowed": False,
    }
