from __future__ import annotations

import math
import re
from typing import Any

SCHEMA_VERSION = "1.0.0"
SUPPORTED_LINES = (0.5, 1.5, 2.5)


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _decimal(value: Any) -> float | None:
    out = _num(value)
    return out if out is not None and 1.0 < out <= 1000.0 else None


def _poisson_over(lam: float, line: float) -> float | None:
    # v1 intentionally supports only half-goal lines to avoid silently
    # mis-pricing pushes or Asian quarter-line settlement.
    if line not in SUPPORTED_LINES:
        return None
    threshold = int(math.floor(line)) + 1
    cdf = sum(math.exp(-lam) * (lam ** k) / math.factorial(k) for k in range(threshold))
    return max(0.0, min(1.0, 1.0 - cdf))


def _selection(value: Any) -> tuple[str | None, float | None]:
    text = _norm(value)
    match = re.search(r"\b(over|under)\s*([0-9]+(?:\.[0-9]+)?)\b", text)
    if not match:
        return None, None
    return match.group(1).upper(), float(match.group(2))


def _team_role(market_name: Any, fixture: dict[str, Any]) -> str | None:
    name = _norm(market_name)

    # V1 Team Totals is strictly FULL-TIME TEAM GOALS. API-Football exposes
    # many derivative markets with "team total" in the label (cards, corners,
    # shots, half-specific totals, etc.). Those must never be interpreted with
    # a goal lambda or admitted into the FT Team Totals CLV sample.
    period_tokens = ("first half", "1st half", "1h ", "second half", "2nd half", "2h ")
    non_goal_tokens = (
        "corner",
        "card",
        "booking",
        "yellow",
        "red card",
        "shot",
        "offside",
        "throw in",
        "throw-in",
        "foul",
        "save",
        "tackle",
        "goal kick",
    )
    if any(token in name for token in period_tokens):
        return None
    if any(token in name for token in non_goal_tokens):
        return None

    # API-Football canonical FT team-goal markets are bet ids 16/17 and are
    # labelled "Total - Home" / "Total - Away". Treat only those exact generic
    # labels as goal totals when the word "goal" is absent.
    if name in {"total - home", "total home"}:
        return "HOME"
    if name in {"total - away", "total away"}:
        return "AWAY"

    if "goal" not in name:
        return None
    if "team total" not in name and not (
        "total goals" in name and any(token in name for token in ("home team", "away team"))
    ) and not (
        "team goals" in name and any(token in name for token in ("home", "away"))
    ):
        return None

    home_name = _norm(fixture.get("home_team") or fixture.get("home_name"))
    away_name = _norm(fixture.get("away_team") or fixture.get("away_name"))
    home_markers = ("home team", "home total", "team home", "home goals")
    away_markers = ("away team", "away total", "team away", "away goals")

    home_hit = any(token in name for token in home_markers) or (home_name and home_name in name)
    away_hit = any(token in name for token in away_markers) or (away_name and away_name in name)

    if home_hit and not away_hit:
        return "HOME"
    if away_hit and not home_hit:
        return "AWAY"
    return None


def _fair_pair(over_price: float, under_price: float) -> tuple[float, float]:
    oi = 1.0 / over_price
    ui = 1.0 / under_price
    total = oi + ui
    return oi / total, ui / total


def _observed_rows(event: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
    raw = event.get("raw_projection") if isinstance(event.get("raw_projection"), dict) else {}
    market = event.get("market") if isinstance(event.get("market"), dict) else {}
    provenance = event.get("market_provenance") if isinstance(event.get("market_provenance"), dict) else {}

    lambdas = {
        "HOME": _num(raw.get("raw_home_goal_rate")),
        "AWAY": _num(raw.get("raw_away_goal_rate")),
    }
    market_fresh = provenance.get("fresh") is True
    market_source = provenance.get("source") or "NOT_VERIFIED"

    supported: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []

    for market_row in market.get("markets") or []:
        if not isinstance(market_row, dict):
            continue
        role = _team_role(market_row.get("market"), fixture)
        if role is None:
            continue
        lam = lambdas.get(role)
        if lam is None or lam <= 0:
            unsupported.append({
                "market": market_row.get("market"),
                "bookmaker": market_row.get("bookmaker"),
                "reason": "TEAM_LAMBDA_NOT_AVAILABLE",
            })
            continue

        parsed: dict[tuple[str, float], float] = {}
        for value in market_row.get("values") or []:
            if not isinstance(value, dict):
                continue
            side, embedded_line = _selection(value.get("selection"))
            if side is None:
                selection_text = _norm(value.get("selection"))
                if selection_text.startswith("over"):
                    side = "OVER"
                elif selection_text.startswith("under"):
                    side = "UNDER"
            line = _num(value.get("line"))
            if line is None:
                line = embedded_line
            price = _decimal(value.get("decimal_price"))
            if price is None:
                price = _decimal(value.get("price"))
            if side is None or line is None or price is None:
                continue
            parsed[(side, line)] = price

        lines = sorted({line for _, line in parsed})
        for line in lines:
            if line not in SUPPORTED_LINES:
                unsupported.append({
                    "team_role": role,
                    "team": fixture.get("home_team") if role == "HOME" else fixture.get("away_team"),
                    "market": market_row.get("market"),
                    "bookmaker": market_row.get("bookmaker"),
                    "line": line,
                    "reason": "UNSUPPORTED_NON_HALF_LINE_V1",
                })
                continue

            op = parsed.get(("OVER", line))
            up = parsed.get(("UNDER", line))
            p_over = _poisson_over(lam, line)
            if p_over is None:
                continue
            p_under = 1.0 - p_over

            market_fair_over = market_fair_under = None
            if op is not None and up is not None:
                market_fair_over, market_fair_under = _fair_pair(op, up)

            for side, price, p_model, p_market_fair in (
                ("OVER", op, p_over, market_fair_over),
                ("UNDER", up, p_under, market_fair_under),
            ):
                if price is None:
                    continue
                supported.append({
                    "team_role": role,
                    "team_id": fixture.get("home_team_id") if role == "HOME" else fixture.get("away_team_id"),
                    "team": fixture.get("home_team") if role == "HOME" else fixture.get("away_team"),
                    "selection": side,
                    "line": line,
                    "lambda": round(lam, 4),
                    "probability_model": round(p_model, 6),
                    "fair_decimal_model": round(1.0 / p_model, 4) if 0 < p_model < 1 else None,
                    "bookmaker": market_row.get("bookmaker"),
                    "bookmaker_id": market_row.get("bookmaker_id"),
                    "market": market_row.get("market"),
                    "market_id": market_row.get("market_id"),
                    "decimal_price": round(price, 4),
                    "p_breakeven": round(1.0 / price, 6),
                    "p_market_fair": round(p_market_fair, 6) if p_market_fair is not None else None,
                    "raw_edge_vs_market_fair_pp": (
                        round((p_model - p_market_fair) * 100.0, 3)
                        if p_market_fair is not None
                        else None
                    ),
                    "raw_ev_at_observed_price": round(p_model * price - 1.0, 6),
                    "provider_update": market_row.get("provider_update"),
                    "market_source": market_source,
                    "market_fresh": market_fresh,
                    "research_only": True,
                    "actionable": False,
                    "decision_weight": 0.0,
                    "classification": "RESEARCH_ONLY",
                    "promotion_block": "TEAM_TOTALS_NOT_OOS_CALIBRATED_OR_PRODUCTION_APPROVED",
                })

    supported.sort(
        key=lambda row: (
            1.0 if row.get("market_fresh") else 0.0,
            float(row.get("raw_edge_vs_market_fair_pp") or -999.0),
            float(row.get("probability_model") or 0.0),
        ),
        reverse=True,
    )
    return supported[:24], unsupported[:24]


def build(event: dict[str, Any]) -> dict[str, Any]:
    fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
    raw = event.get("raw_projection") if isinstance(event.get("raw_projection"), dict) else {}
    home_lambda = _num(raw.get("raw_home_goal_rate"))
    away_lambda = _num(raw.get("raw_away_goal_rate"))

    canonical_grid: list[dict[str, Any]] = []
    for role, team, lam in (
        ("HOME", fixture.get("home_team"), home_lambda),
        ("AWAY", fixture.get("away_team"), away_lambda),
    ):
        if lam is None or lam <= 0:
            continue
        for line in SUPPORTED_LINES:
            p_over = _poisson_over(lam, line)
            if p_over is None:
                continue
            canonical_grid.append({
                "team_role": role,
                "team": team,
                "line": line,
                "lambda": round(lam, 4),
                "p_over": round(p_over, 6),
                "p_under": round(1.0 - p_over, 6),
                "fair_over_decimal": round(1.0 / p_over, 4) if 0 < p_over < 1 else None,
                "fair_under_decimal": round(1.0 / (1.0 - p_over), 4) if 0 < p_over < 1 else None,
            })

    observed, unsupported = _observed_rows(event)
    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": fixture.get("fixture_id"),
        "stage": event.get("stage"),
        "status": "LIVE_RESEARCH_MODELED" if canonical_grid else "NOT_MODELED_THIS_TICK",
        "model": "CANONICAL_TEAM_LAMBDA_POISSON_DERIVATION_v0.1",
        "model_source": "EXISTING_CANONICAL_RAW_HOME_AWAY_GOAL_RATES",
        "home_lambda": round(home_lambda, 4) if home_lambda is not None else None,
        "away_lambda": round(away_lambda, 4) if away_lambda is not None else None,
        "probability_grid": canonical_grid,
        "observed_exact_market_rows": observed,
        "observed_exact_market_count": len(observed),
        "unsupported_observed_rows": unsupported,
        "supported_lines": list(SUPPORTED_LINES),
        "actionable": False,
        "decision_weight": 0.0,
        "production_status": "RESEARCH_ONLY_NOT_ACTIONABLE",
        "calibration_gate": {
            "minimum_oos_fixtures_for_market_comparison": 100,
            "minimum_oos_fixtures_for_actionable_review": 200,
            "requires": [
                "stable Brier/log-loss by home/away and line",
                "verified team-total market price history and CLV evidence",
                "stable calibration across competitions",
                "no material degradation versus canonical FT-goals calibration",
            ],
        },
        "policy": (
            "SPORT_FIRST; DERIVED_ONLY_FROM_EXISTING_CANONICAL_TEAM_LAMBDAS; "
            "EXACT_OBSERVED_MARKETS_ONLY_FOR_PRICE_COMPARISON; HALF_GOAL_LINES_ONLY_V1; "
            "NO_BET_LEAN_GALAXY_PROMOTION"
        ),
    }


def attach(payload: dict[str, Any]) -> dict[str, int]:
    modeled_events = 0
    observed_events = 0
    observed_rows = 0
    unsupported_rows = 0

    for event in payload.get("events") or []:
        if not isinstance(event, dict) or event.get("stage") == "POSTGAME":
            continue
        if event.get("event_type") not in {"SOCCER_REFRESH", "TEAM_TOTALS_RESEARCH_SPILLOVER"}:
            continue
        intelligence = build(event)
        event["team_totals_intelligence"] = intelligence

        if intelligence.get("status") == "LIVE_RESEARCH_MODELED":
            modeled_events += 1
        count = int(intelligence.get("observed_exact_market_count") or 0)
        if count:
            observed_events += 1
            observed_rows += count
        unsupported_rows += len(intelligence.get("unsupported_observed_rows") or [])

        match_intel = event.get("match_intelligence")
        if isinstance(match_intel, dict):
            areas = match_intel.get("areas")
            if isinstance(areas, dict):
                areas["team_goals"] = {
                    "status": intelligence.get("status"),
                    "home_lambda": intelligence.get("home_lambda"),
                    "away_lambda": intelligence.get("away_lambda"),
                    "probability_grid": intelligence.get("probability_grid") or [],
                    "observed_exact_market_rows": intelligence.get("observed_exact_market_rows") or [],
                    "actionable": False,
                    "decision_weight": 0.0,
                    "production_status": intelligence.get("production_status"),
                    "calibration_gate": intelligence.get("calibration_gate"),
                }

    return {
        "modeled_events": modeled_events,
        "events_with_observed_team_total_markets": observed_events,
        "observed_exact_market_rows": observed_rows,
        "unsupported_observed_rows": unsupported_rows,
    }
