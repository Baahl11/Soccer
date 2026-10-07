from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

from mcp_gateway import persistence

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_PROSPECTIVE_EVALUATION_REGISTRY_V1.0.0"
COHORT_START_UTC = "2026-10-07T14:30:00+00:00"
STAGE = "T-10"
MIN_AVAILABILITY = 0.85
DATA_TIERS = {"A", "B"}
TEAM_TOTALS_MIN_RAW_EDGE_PP = 0.0
FINAL_STATUSES = {"FT", "AET", "PEN"}

POLICY = {
    "cohort_start_utc": COHORT_START_UTC,
    "stage": STAGE,
    "point_in_time_source": "soccer_refresh_events.payload",
    "data_tiers": sorted(DATA_TIERS),
    "minimum_availability_confidence": MIN_AVAILABILITY,
    "lineup_gate": "BOTH_XI_AND_BOTH_GOALKEEPERS_CONFIRMED",
    "sporting_gate": "SPORTING_SHORTLIST_SHORTLISTED_TRUE",
    "one_selection_per_fixture_per_track": True,
    "1X2": {
        "market": "Match Winner / Winner",
        "runtime_tier_b": "INCLUDE_IF_EVENT_CLASSIFICATION_BET_AND_TIER_B",
        "advanced_cap_shadow": (
            "INCLUDE_TIER_A_OR_S_WATCH_ONLY_WHEN_ALL_DATA_AVAILABILITY_XI_GK_GATES_PASS "
            "AND_ABS_RAW_MINUS_MARKET_FAIR_LT_12PP"
        ),
        "counts_toward_100_graded_bets": "RUNTIME_TIER_B_ONLY",
    },
    "TEAM_TOTALS": {
        "source": "team_totals_intelligence.observed_exact_market_rows",
        "selection_rule": (
            "WITHIN_FIRST_ELIGIBLE_T10_EVENT_CHOOSE_HIGHEST_POSITIVE_RAW_EDGE_VS_MARKET_FAIR_PP; "
            "TIE_BREAK_TEAM_ROLE_LINE_SELECTION_MARKET_BOOKMAKER"
        ),
        "minimum_raw_edge_pp": TEAM_TOTALS_MIN_RAW_EDGE_PP,
        "counts_toward_100_graded_bets": False,
        "reason": "NO_VALIDATED_TEAM_TOTALS_PRODUCTION_DECISION_RULE_YET",
    },
    "outcomes_never_used_for_selection": True,
    "market_prices_never_used_to_create_raw_sport_projection": True,
    "production_promotion_allowed": False,
}
POLICY_HASH = hashlib.sha256(
    json.dumps(POLICY, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


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
    except ValueError:
        return None
    if out.tzinfo is None:
        out = out.replace(tzinfo=timezone.utc)
    return out.astimezone(timezone.utc)


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _lineup_ok(event: dict[str, Any]) -> bool:
    lineups = event.get("lineups")
    return (
        isinstance(lineups, dict)
        and lineups.get("both_xi_confirmed") is True
        and lineups.get("both_goalkeepers_confirmed") is True
    )


def _sporting_ok(event: dict[str, Any]) -> bool:
    shortlist = event.get("sporting_shortlist")
    return isinstance(shortlist, dict) and shortlist.get("shortlisted") is True


def _event_gate(row: dict[str, Any], event: dict[str, Any]) -> bool:
    availability = _num(
        event.get("availability_confidence")
        if event.get("availability_confidence") is not None
        else row.get("availability_confidence")
    )
    data_tier = str(
        ((event.get("coverage") or {}).get("data_tier"))
        if isinstance(event.get("coverage"), dict)
        else row.get("data_tier") or ""
    ).upper()
    return (
        availability is not None
        and availability >= MIN_AVAILABILITY
        and data_tier in DATA_TIERS
        and _lineup_ok(event)
        and _sporting_ok(event)
    )


def _fixture(event: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    fx = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
    return {
        "fixture_id": int(fx.get("fixture_id") or row.get("fixture_id") or 0),
        "kickoff": fx.get("kickoff"),
        "league_id": fx.get("league_id"),
        "league": fx.get("league"),
        "country": fx.get("country"),
        "season": fx.get("season"),
        "home_team_id": fx.get("home_team_id"),
        "home_team": fx.get("home_team"),
        "away_team_id": fx.get("away_team_id"),
        "away_team": fx.get("away_team"),
    }


def _best_1x2(row: dict[str, Any], event: dict[str, Any]) -> dict[str, Any] | None:
    if not _event_gate(row, event):
        return None
    best = event.get("best_market")
    if not isinstance(best, dict):
        return None
    market = _norm(best.get("market"))
    if market not in {"match winner", "winner"}:
        return None

    price = _num(best.get("decimal_price"))
    p_raw = _num(best.get("p_raw"))
    p_market = _num(best.get("p_market_fair"))
    p_shrunk = _num(best.get("p_shrunk"))
    edge = _num(best.get("prob_edge_pp"))
    if None in {price, p_raw, p_market, p_shrunk, edge} or price <= 1.0:
        return None

    tier = str(best.get("tier") or event.get("tier") or "").upper()
    classification = str(event.get("classification") or "").upper()
    if tier == "B" and classification == "BET":
        candidate_type = "RUNTIME_TIER_B_BET"
        counts_toward_graded_bets = True
    elif tier in {"A", "S"} and classification == "WATCH":
        if abs(float(p_raw) - float(p_market)) >= 0.12:
            return None
        candidate_type = "ADVANCED_METRIC_CAP_SHADOW"
        counts_toward_graded_bets = False
    else:
        return None

    fx = _fixture(event, row)
    if not fx["fixture_id"]:
        return None
    return {
        "schema_version": SCHEMA_VERSION,
        "registry_model_version": MODEL_VERSION,
        "policy_hash": POLICY_HASH,
        "track": "1X2",
        "candidate_type": candidate_type,
        "counts_toward_100_graded_bets": counts_toward_graded_bets,
        "event_id": int(row.get("event_id") or 0),
        "fixture_id": fx["fixture_id"],
        "generated_at_utc": row.get("generated_at").isoformat()
        if isinstance(row.get("generated_at"), datetime)
        else row.get("generated_at"),
        "stage": STAGE,
        **{k: v for k, v in fx.items() if k != "fixture_id"},
        "model_version": event.get("model_version"),
        "data_tier": (event.get("coverage") or {}).get("data_tier")
        if isinstance(event.get("coverage"), dict)
        else row.get("data_tier"),
        "availability_confidence": _num(event.get("availability_confidence")),
        "market": best.get("market"),
        "selection": best.get("selection"),
        "line": _num(best.get("line")),
        "decimal_price": price,
        "bookmaker": best.get("bookmaker"),
        "p_raw": p_raw,
        "p_market_fair": p_market,
        "p_shrunk": p_shrunk,
        "prob_edge_pp": edge,
        "estimated_ev": _num(best.get("estimated_ev")),
        "tier": tier,
        "runtime_classification": classification,
        "sporting_shortlist": event.get("sporting_shortlist"),
        "lineups": event.get("lineups"),
        "selection_frozen": True,
        "outcome_known_at_selection": False,
        "production_promotion_allowed": False,
    }


def _best_team_total(row: dict[str, Any], event: dict[str, Any]) -> dict[str, Any] | None:
    if not _event_gate(row, event):
        return None
    intelligence = event.get("team_totals_intelligence")
    if not isinstance(intelligence, dict):
        return None
    observed = [
        item for item in (intelligence.get("observed_exact_market_rows") or [])
        if isinstance(item, dict)
    ]
    eligible: list[dict[str, Any]] = []
    for item in observed:
        price = _num(item.get("decimal_price"))
        p_model = _num(item.get("probability_model"))
        p_market = _num(item.get("p_market_fair"))
        edge = _num(item.get("raw_edge_vs_market_fair_pp"))
        line = _num(item.get("line"))
        if (
            price is None
            or price <= 1.0
            or p_model is None
            or p_market is None
            or edge is None
            or edge <= TEAM_TOTALS_MIN_RAW_EDGE_PP
            or line not in {0.5, 1.5, 2.5}
            or item.get("market_fresh") is not True
        ):
            continue
        eligible.append(item)
    if not eligible:
        return None

    eligible.sort(
        key=lambda item: (
            -float(_num(item.get("raw_edge_vs_market_fair_pp")) or -999.0),
            str(item.get("team_role") or ""),
            float(_num(item.get("line")) or -1.0),
            str(item.get("selection") or ""),
            str(item.get("market") or ""),
            str(item.get("bookmaker") or ""),
        )
    )
    best = eligible[0]
    fx = _fixture(event, row)
    if not fx["fixture_id"]:
        return None
    return {
        "schema_version": SCHEMA_VERSION,
        "registry_model_version": MODEL_VERSION,
        "policy_hash": POLICY_HASH,
        "track": "TEAM_TOTALS",
        "candidate_type": "PREDECLARED_RESEARCH_TEAM_TOTAL",
        "counts_toward_100_graded_bets": False,
        "event_id": int(row.get("event_id") or 0),
        "fixture_id": fx["fixture_id"],
        "generated_at_utc": row.get("generated_at").isoformat()
        if isinstance(row.get("generated_at"), datetime)
        else row.get("generated_at"),
        "stage": STAGE,
        **{k: v for k, v in fx.items() if k != "fixture_id"},
        "model_version": event.get("model_version"),
        "data_tier": (event.get("coverage") or {}).get("data_tier")
        if isinstance(event.get("coverage"), dict)
        else row.get("data_tier"),
        "availability_confidence": _num(event.get("availability_confidence")),
        "team_role": best.get("team_role"),
        "team_id": best.get("team_id"),
        "team": best.get("team"),
        "market": best.get("market"),
        "selection": best.get("selection"),
        "line": _num(best.get("line")),
        "decimal_price": _num(best.get("decimal_price")),
        "bookmaker": best.get("bookmaker"),
        "bookmaker_id": best.get("bookmaker_id"),
        "market_id": best.get("market_id"),
        "provider_update": best.get("provider_update"),
        "probability_model": _num(best.get("probability_model")),
        "p_market_fair": _num(best.get("p_market_fair")),
        "raw_edge_vs_market_fair_pp": _num(best.get("raw_edge_vs_market_fair_pp")),
        "market_fresh": True,
        "sporting_shortlist": event.get("sporting_shortlist"),
        "lineups": event.get("lineups"),
        "selection_frozen": True,
        "outcome_known_at_selection": False,
        "production_promotion_allowed": False,
    }


def _grade_1x2(row: dict[str, Any], home: int, away: int) -> str:
    actual = "home" if home > away else "away" if away > home else "draw"
    selection = _norm(row.get("selection"))
    home_name = _norm(row.get("home_team"))
    away_name = _norm(row.get("away_team"))
    if selection in {"home", "1", home_name}:
        pick = "home"
    elif selection in {"away", "2", away_name}:
        pick = "away"
    elif selection in {"draw", "x"}:
        pick = "draw"
    else:
        return "UNGRADABLE_SELECTION"
    return "WIN" if pick == actual else "LOSS"


def _grade_team_total(row: dict[str, Any], home: int, away: int) -> str:
    value = home if str(row.get("team_role") or "").upper() == "HOME" else away
    line = _num(row.get("line"))
    if line is None:
        return "UNGRADABLE_LINE"
    selection = _norm(row.get("selection"))
    if selection.startswith("over"):
        return "WIN" if value > line else "LOSS"
    if selection.startswith("under"):
        return "WIN" if value < line else "LOSS"
    return "UNGRADABLE_SELECTION"


def _roi(outcome: str, price: float | None) -> float | None:
    if price is None or price <= 1.0:
        return None
    if outcome == "WIN":
        return round(price - 1.0, 6)
    if outcome == "LOSS":
        return -1.0
    if outcome == "PUSH":
        return 0.0
    return None


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    settled = [row for row in rows if row.get("settled")]
    counts = Counter(str(row.get("outcome") or "UNSETTLED") for row in rows)
    roi_values = [
        float(row["roi_units"])
        for row in settled
        if _num(row.get("roi_units")) is not None
    ]
    return {
        "rows": len(rows),
        "unique_fixtures": len({int(row["fixture_id"]) for row in rows}),
        "settled": len(settled),
        "win": counts["WIN"],
        "loss": counts["LOSS"],
        "push": counts["PUSH"],
        "unsettled": len(rows) - len(settled),
        "roi_units": round(sum(roi_values), 6) if roi_values else 0.0,
        "roi_per_settled_unit": (
            round(sum(roi_values) / len(roi_values), 6) if roi_values else None
        ),
    }


def build_registry() -> dict[str, Any]:
    if not persistence.persistence_configured():
        return {
            "schema_version": SCHEMA_VERSION,
            "model_version": MODEL_VERSION,
            "status": "POSTGRES_NOT_CONFIGURED",
            "policy": POLICY,
            "policy_hash": POLICY_HASH,
            "selection_rows": [],
            "settlement_rows": [],
            "provider_requests_added": 0,
            "production_promotion_allowed": False,
        }

    start = _dt(COHORT_START_UTC)
    assert start is not None
    persistence.ensure_schema()

    with persistence._connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT event_id, fixture_id, generated_at, classification,
                       availability_confidence, data_tier, payload
                FROM soccer_refresh_events
                WHERE generated_at >= %s
                  AND stage = %s
                ORDER BY generated_at ASC, event_id ASC
                """,
                (start, STAGE),
            )
            columns = [desc.name for desc in cur.description]
            event_rows = [dict(zip(columns, raw)) for raw in cur.fetchall()]

        chosen: dict[tuple[int, str], dict[str, Any]] = {}
        for row in event_rows:
            event = row.get("payload")
            if not isinstance(event, dict):
                continue
            try:
                fixture_id = int(row.get("fixture_id"))
            except (TypeError, ValueError):
                continue
            for track, builder in (("1X2", _best_1x2), ("TEAM_TOTALS", _best_team_total)):
                key = (fixture_id, track)
                if key in chosen:
                    continue
                candidate = builder(row, event)
                if candidate is not None:
                    chosen[key] = candidate

        selections = sorted(
            chosen.values(),
            key=lambda row: (
                str(row.get("generated_at_utc") or ""),
                int(row.get("fixture_id") or 0),
                str(row.get("track") or ""),
            ),
        )

        fixture_ids = sorted({int(row["fixture_id"]) for row in selections})
        results: dict[int, tuple[str, int | None, int | None]] = {}
        if fixture_ids:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT fixture_id, final_status, home_goals, away_goals
                    FROM soccer_results
                    WHERE fixture_id = ANY(%s)
                    """,
                    (fixture_ids,),
                )
                for fixture_id, status, home, away in cur.fetchall():
                    try:
                        results[int(fixture_id)] = (
                            str(status or "").upper(),
                            int(home) if home is not None else None,
                            int(away) if away is not None else None,
                        )
                    except (TypeError, ValueError):
                        continue

    settlement_rows: list[dict[str, Any]] = []
    for row in selections:
        status, home, away = results.get(int(row["fixture_id"]), ("", None, None))
        outcome = "UNSETTLED"
        if status in FINAL_STATUSES and home is not None and away is not None:
            outcome = (
                _grade_1x2(row, home, away)
                if row["track"] == "1X2"
                else _grade_team_total(row, home, away)
            )
        settlement_rows.append(
            {
                "policy_hash": POLICY_HASH,
                "track": row["track"],
                "candidate_type": row["candidate_type"],
                "counts_toward_100_graded_bets": row[
                    "counts_toward_100_graded_bets"
                ],
                "fixture_id": row["fixture_id"],
                "selection_event_id": row["event_id"],
                "selection_generated_at_utc": row["generated_at_utc"],
                "market": row.get("market"),
                "selection": row.get("selection"),
                "line": row.get("line"),
                "decimal_price": row.get("decimal_price"),
                "final_status": status or None,
                "final_home_goals": home,
                "final_away_goals": away,
                "outcome": outcome,
                "settled": outcome in {"WIN", "LOSS", "PUSH"},
                "roi_units": _roi(outcome, _num(row.get("decimal_price"))),
                "real_wager_assumed": False,
            }
        )

    by_track: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in settlement_rows:
        by_track[row["track"]].append(row)

    graded_bet_rows = [
        row for row in settlement_rows
        if row.get("counts_toward_100_graded_bets") is True
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": "PROSPECTIVE_EVALUATION_REGISTRY_ACTIVE",
        "policy": POLICY,
        "policy_hash": POLICY_HASH,
        "selection_rows": selections,
        "settlement_rows": settlement_rows,
        "summary": {
            "selection_rows": len(selections),
            "selection_unique_fixtures": len(
                {int(row["fixture_id"]) for row in selections}
            ),
            "by_track": {
                key: _summarize(value) for key, value in sorted(by_track.items())
            },
            "graded_bet_sample": _summarize(graded_bet_rows),
            "graded_bet_target": 100,
            "material_recalibration_allowed": (
                _summarize(graded_bet_rows)["settled"] >= 100
            ),
        },
        "provider_requests_added": 0,
        "runtime_selection_logic_changed": False,
        "model_weights_changed": False,
        "thresholds_changed": False,
        "historical_predictions_rewritten": False,
        "outcomes_used_for_selection": False,
        "production_promotion_allowed": False,
    }
