from __future__ import annotations

import re
from collections import Counter
from typing import Any

SCHEMA_VERSION = "0.1.0"


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _line(value: Any) -> float | None:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(value or ""))
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _card_family(name: Any) -> str | None:
    n = _norm(name)
    if not any(token in n for token in ("card", "booking", "bookings")):
        return None
    if "player" in n:
        return "PLAYER_CARDS"
    if "red card" in n or "red cards" in n:
        return "RED_CARDS"
    if any(token in n for token in ("home card", "away card", "team card", "home booking", "away booking", "team booking")):
        return "TEAM_CARDS"
    return "CARDS"


def extract_card_market_groups(event: dict[str, Any]) -> list[dict[str, Any]]:
    if event.get("stage") not in {"T-40", "T-20", "T-10"}:
        return []
    market = event.get("market") if isinstance(event.get("market"), dict) else {}
    groups: list[dict[str, Any]] = []
    for row in market.get("markets") or []:
        if not isinstance(row, dict):
            continue
        family = _card_family(row.get("market"))
        if not family:
            continue
        values = []
        for value in row.get("values") or []:
            if not isinstance(value, dict):
                continue
            try:
                price = float(value.get("price"))
            except (TypeError, ValueError):
                continue
            if price <= 1.0:
                continue
            values.append(
                {
                    "selection": value.get("selection"),
                    "line": _line(value.get("selection")),
                    "decimal_price": round(price, 6),
                }
            )
        if values:
            groups.append(
                {
                    "family": family,
                    "bookmaker": row.get("bookmaker"),
                    "market": row.get("market"),
                    "provider_update": row.get("provider_update"),
                    "values": values,
                }
            )
    return groups


def attach_card_inventory(event: dict[str, Any], captured_at_local: Any) -> tuple[int, int]:
    groups = extract_card_market_groups(event)
    if not groups:
        return 0, 0

    snap = event.get("derivative_research_market_snapshot")
    if not isinstance(snap, dict):
        snap = {
            "schema_version": "1.2.0",
            "captured_at_local": captured_at_local,
            "research_only": True,
            "actionable": False,
            "policy": "EXPLICIT_DERIVATIVE_MODEL_REQUIRED; NEVER_DERIVE_SPORT_PROBABILITY_FROM_MARKET_ODDS",
            "group_count": 0,
            "quote_count": 0,
            "groups": [],
        }
        event["derivative_research_market_snapshot"] = snap

    existing = {
        (
            str(g.get("family")),
            str(g.get("bookmaker")),
            str(g.get("market")),
            str(g.get("provider_update")),
        )
        for g in snap.get("groups") or []
        if isinstance(g, dict)
    }
    added = []
    for group in groups:
        key = (
            str(group.get("family")),
            str(group.get("bookmaker")),
            str(group.get("market")),
            str(group.get("provider_update")),
        )
        if key in existing:
            continue
        existing.add(key)
        added.append(group)

    if not added:
        return 0, 0
    snap.setdefault("groups", []).extend(added)
    quotes = sum(len(g.get("values") or []) for g in added)
    snap["group_count"] = len(snap.get("groups") or [])
    snap["quote_count"] = sum(
        len(g.get("values") or [])
        for g in snap.get("groups") or []
        if isinstance(g, dict)
    )
    snap["contains_card_markets"] = True
    snap["cards_policy"] = (
        "MARKET_INVENTORY_ONLY_UNTIL_EXPLICIT_TEAM_DISCIPLINE_PLUS_REFEREE_MODEL_PASSES_OOS_CALIBRATION"
    )
    return len(added), quotes


def _family_counts(event: dict[str, Any]) -> dict[str, int]:
    snap = event.get("derivative_research_market_snapshot")
    counts: Counter[str] = Counter()
    if not isinstance(snap, dict):
        return {}
    for group in snap.get("groups") or []:
        if not isinstance(group, dict):
            continue
        counts[str(group.get("family") or "UNKNOWN")] += len(
            [x for x in group.get("values") or [] if isinstance(x, dict)]
        )
    return dict(counts)


def _lineup_context(event: dict[str, Any]) -> dict[str, Any]:
    lineup = event.get("lineups")
    if not isinstance(lineup, dict):
        return {
            "status": "NOT_VERIFIED",
            "both_xi_confirmed": False,
            "both_goalkeepers_confirmed": False,
            "teams": [],
        }
    teams = []
    for row in lineup.get("teams") or []:
        if not isinstance(row, dict):
            continue
        teams.append(
            {
                "team_id": row.get("team_id"),
                "team": row.get("team"),
                "formation": row.get("formation"),
                "coach_id": row.get("coach_id"),
                "coach": row.get("coach"),
                "goalkeepers": [
                    {"id": g.get("id"), "name": g.get("name")}
                    for g in row.get("goalkeepers") or []
                    if isinstance(g, dict)
                ],
                "starter_count": len(row.get("starters") or []),
                "substitutes_count": row.get("substitutes_count"),
            }
        )
    return {
        "status": lineup.get("lineup_state") or "PENDING",
        "both_xi_confirmed": bool(lineup.get("both_xi_confirmed")),
        "both_goalkeepers_confirmed": bool(lineup.get("both_goalkeepers_confirmed")),
        "teams": teams,
    }


def _injury_context(event: dict[str, Any]) -> dict[str, Any]:
    injuries = event.get("injuries")
    if isinstance(injuries, list):
        return {"status": "VERIFIED_PROVIDER_REPORT", "count": len(injuries), "rows": injuries[:20]}
    return {"status": "NOT_VERIFIED", "count": None, "rows": []}


def _player_context(event: dict[str, Any]) -> dict[str, Any]:
    research = event.get("player_trends_research")
    if not isinstance(research, dict):
        return {
            "status": "NOT_CAPTURED_THIS_TICK",
            "probability_model": "MISSING_PLAYER_PROP_MODEL",
            "actionable": False,
        }
    return {
        "status": research.get("status"),
        "teams": research.get("teams") or [],
        "probability_model": "MISSING_PLAYER_PROP_MODEL",
        "actionable": False,
        "decision_weight": 0.0,
    }


def build_event_intelligence(event: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    fx = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
    raw = event.get("raw_projection") if isinstance(event.get("raw_projection"), dict) else {}
    screen = event.get("sporting_screen_refined") or event.get("sporting_screen_initial") or event.get("sporting_shortlist") or {}
    if not isinstance(screen, dict):
        screen = {}
    trend = event.get("trend_context") if isinstance(event.get("trend_context"), dict) else {}
    lineup = _lineup_context(event)
    derivative_counts = _family_counts(event)
    market_prov = event.get("market_provenance") if isinstance(event.get("market_provenance"), dict) else {}
    coverage = event.get("coverage") if isinstance(event.get("coverage"), dict) else {}

    raw_home = _num(raw.get("raw_home_goal_rate"))
    raw_away = _num(raw.get("raw_away_goal_rate"))
    raw_total = _num(raw.get("raw_total_goals"))
    p_o25 = _num(raw.get("raw_over_2_5_prob"))
    p_btts = _num(raw.get("raw_btts_yes_prob"))
    p_home = _num(raw.get("raw_home_win_prob"))
    p_draw = _num(raw.get("raw_draw_prob"))
    p_away = _num(raw.get("raw_away_win_prob"))

    coaches = [t.get("coach") for t in lineup.get("teams") or [] if t.get("coach")]
    formations = [t.get("formation") for t in lineup.get("teams") or [] if t.get("formation")]
    referee = fx.get("referee")

    gaps = []
    if payload.get("first_half_market_model") != "ACTIONABLE":
        gaps.append("1H_LIVE_MODEL_NOT_PRODUCTION_APPROVED")
    if payload.get("period_market_model") != "ACTIONABLE":
        gaps.append("2H_LIVE_MODEL_NOT_PRODUCTION_APPROVED")
    gaps.extend(
        [
            "CORNERS_LIVE_PROBABILITY_MODEL_NOT_PRODUCTION_APPROVED",
            "CARDS_TEAM_REFEREE_MODEL_NOT_PRODUCTION_APPROVED",
            "PLAYER_PROP_PROBABILITY_MODELS_PENDING",
            "ADVANCED_XG_NPXG_PPDA_FIELD_TILT_SOURCE_NOT_LIVE",
            "WEATHER_LIVE_SOURCE_NOT_INTEGRATED",
            "SET_PIECE_EXPLICIT_MODEL_PENDING",
            "REST_TRAVEL_CONGESTION_EXPLICIT_FEATURES_PENDING",
            "COACH_REGIME_CHANGE_MODEL_PENDING",
        ]
    )
    if not referee:
        gaps.append("REFEREE_NOT_VERIFIED_FOR_THIS_FIXTURE")

    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": fx.get("fixture_id"),
        "kickoff": fx.get("kickoff"),
        "match": f"{fx.get('home_team')} vs {fx.get('away_team')}",
        "stage": event.get("stage"),
        "data_tier": coverage.get("data_tier"),
        "bet_tier": event.get("tier"),
        "classification": event.get("classification"),
        "availability_confidence": event.get("availability_confidence"),
        "sport_first": True,
        "areas": {
            "side_control": {
                "status": "MODELED_RESEARCH_ONLY" if any(x is not None for x in (p_home, p_draw, p_away)) else "NOT_MODELED_THIS_TICK",
                "side_edge_score": _num(screen.get("side_edge_score")),
                "home_win_prob": p_home,
                "draw_prob": p_draw,
                "away_win_prob": p_away,
            },
            "goals_full_match": {
                "status": "MODELED" if raw_total is not None else "NOT_MODELED_THIS_TICK",
                "home_lambda": raw_home,
                "away_lambda": raw_away,
                "total_lambda": raw_total,
                "over_2_5_prob": p_o25,
                "goal_environment_score": _num(screen.get("goal_environment_score")),
                "scoring_path": raw.get("scoring_path"),
            },
            "btts": {
                "status": "MODELED_RESEARCH_ONLY" if p_btts is not None else "NOT_MODELED_THIS_TICK",
                "yes_probability": p_btts,
                "two_way_scoring_score": _num(screen.get("two_way_scoring_score")),
            },
            "first_half_goals": {
                "status": "RESEARCH_MODEL_EXISTS_OFFLINE_NOT_ATTACHED_TO_LIVE_PROJECTION",
                "observed_quote_count": int(derivative_counts.get("1H_GOALS", 0)),
                "actionable": False,
                "policy": payload.get("first_half_market_model"),
            },
            "second_half_goals": {
                "status": "RESEARCH_MODEL_EXISTS_OFFLINE_NOT_ATTACHED_TO_LIVE_PROJECTION",
                "observed_quote_count": int(derivative_counts.get("2H_GOALS", 0)),
                "actionable": False,
                "policy": payload.get("period_market_model"),
            },
            "corners": {
                "status": "RESEARCH_BASELINE_EXISTS_OFFLINE_NOT_PRODUCTION_APPROVED",
                "observed_quote_count": int(derivative_counts.get("CORNERS", 0)),
                "team_corner_quote_count": int(derivative_counts.get("TEAM_CORNERS", 0)),
                "expected_corner_environment": _num(trend.get("expected_corner_environment")),
                "actionable": False,
            },
            "cards": {
                "status": "RESEARCH_BASELINE_EXISTS_OFFLINE_NOT_ATTACHED_TO_LIVE_PROJECTION",
                "observed_quote_count": int(derivative_counts.get("CARDS", 0)),
                "team_card_quote_count": int(derivative_counts.get("TEAM_CARDS", 0)),
                "red_card_quote_count": int(derivative_counts.get("RED_CARDS", 0)),
                "player_card_quote_count": int(derivative_counts.get("PLAYER_CARDS", 0)),
                "actionable": False,
            },
            "players": _player_context(event),
            "lineups_formations": {
                **lineup,
                "formation_model_status": "RESEARCH_ONLY_FORMATION_INTELLIGENCE",
                "formation_decision_weight": 0.0,
            },
            "goalkeepers": {
                "status": "CONFIRMED" if lineup.get("both_goalkeepers_confirmed") else "NOT_VERIFIED",
                "teams": [
                    {"team": t.get("team"), "goalkeepers": t.get("goalkeepers") or []}
                    for t in lineup.get("teams") or []
                ],
                "explicit_keeper_impact_model": "PENDING",
            },
            "coaches": {
                "status": "VERIFIED_FROM_LINEUP" if len(coaches) == 2 else "PARTIAL_OR_NOT_VERIFIED",
                "coaches": coaches,
                "regime_change_model": "PENDING",
            },
            "referee": {
                "status": "VERIFIED_FIXTURE_ASSIGNMENT" if referee else "NOT_VERIFIED",
                "name": referee,
                "discipline_model": "RESEARCH_BASELINE_EXISTS_OFFLINE_NOT_PRODUCTION_APPROVED",
                "decision_weight": 0.0,
            },
            "injuries_suspensions": _injury_context(event),
            "tactical_style": {
                "status": "LIMITED_VERIFIED_INPUTS",
                "tracks": screen.get("tracks") or [],
                "trend_signals": trend.get("signals") or [],
                "model_disagreements": trend.get("model_disagreements") or [],
                "formation_pair": formations,
            },
            "advanced_metrics": {
                "status": raw.get("advanced_metrics") or "NOT_VERIFIED",
                "required": ["xG", "xGA", "npxG", "npxGA", "PPDA", "field_tilt", "big_chances", "box_entries", "shot_quality"],
            },
            "set_pieces": {"status": "MISSING_EXPLICIT_LIVE_MODEL"},
            "rest_congestion_travel": {"status": "MISSING_EXPLICIT_LIVE_FEATURES"},
            "weather": {"status": "NOT_VERIFIED_NO_LIVE_WEATHER_SOURCE_INTEGRATED"},
            "competition_context": {
                "status": "STRUCTURED_CONTEXT_ONLY",
                "competition": fx.get("league"),
                "round": fx.get("round"),
                "motivation_policy": "NEVER_INFER_MOTIVATION_WITHOUT_VERIFIED_OBJECTIVE_CONTEXT",
            },
            "venue": {"status": "VERIFIED" if fx.get("venue") else "NOT_VERIFIED", "venue": fx.get("venue"), "city": fx.get("city")},
            "market": {
                "status": market_prov.get("source") or "NOT_VERIFIED",
                "fresh": market_prov.get("fresh"),
                "latest_timestamp": market_prov.get("latest_market_timestamp"),
                "market_after_sport": True,
            },
        },
        "coverage_gaps": gaps,
        "policy": "COMPLETE_MATCH_INTELLIGENCE_SURFACE; UNKNOWN_STAYS_NOT_VERIFIED; RESEARCH_NEVER_AUTO_PROMOTES_TO_BET",
    }


def attach(payload: dict[str, Any]) -> dict[str, int]:
    attached = 0
    referee_verified = 0
    coaches_verified = 0
    for event in payload.get("events") or []:
        if not isinstance(event, dict) or event.get("event_type") != "SOCCER_REFRESH" or event.get("stage") == "POSTGAME":
            continue
        intel = build_event_intelligence(event, payload)
        event["match_intelligence"] = intel
        attached += 1
        if ((intel.get("areas") or {}).get("referee") or {}).get("status") == "VERIFIED_FIXTURE_ASSIGNMENT":
            referee_verified += 1
        if ((intel.get("areas") or {}).get("coaches") or {}).get("status") == "VERIFIED_FROM_LINEUP":
            coaches_verified += 1
    return {
        "attached": attached,
        "referee_verified": referee_verified,
        "coaches_verified": coaches_verified,
    }
