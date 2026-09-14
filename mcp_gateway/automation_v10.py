from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime, timedelta, timezone as dt_timezone
from typing import Any

import httpx

from mcp_gateway import automation_v2 as v2
from mcp_gateway import automation_v4 as v4
from mcp_gateway import automation_v5 as v5
from mcp_gateway import automation_v8 as v8
from mcp_gateway.galaxyparlay_client import configured as galaxy_configured
from mcp_gateway.galaxyparlay_client import get_fixture_shadow

MODEL_VERSION = "SOCCER EDGE ENGINE v1.1"
AUTOMATION_VERSION = "2.0.0"

GALAXY_SPORT_MAX_AGE = timedelta(
    minutes=int(os.getenv("GALAXYPARLAY_SPORT_MAX_AGE_MINUTES", "720"))
)
GALAXY_ODDS_MAX_AGE = timedelta(
    minutes=int(os.getenv("GALAXYPARLAY_ODDS_MAX_AGE_MINUTES", "20"))
)

SPORT_FIRST_MARKETS = {
    "match_winner_v2_shadow",
    "over_under_1_5",
    "over_under_2_5",
    "over_under_3_5",
    "both_teams_score",
}

_ORIGINAL_PRIORITY_EVENT = v5._priority_event
_ORIGINAL_ODDS_7M = v5._odds_7m
_ORIGINAL_V8_RESEARCH_EVALUATE = v8._research_only_side_evaluate

_GALAXY_CLIENT: httpx.AsyncClient | None = None
_GALAXY_CACHE: dict[int, dict[str, Any]] = {}
_METRICS: dict[str, int] = {}


def _reset_metrics() -> None:
    _METRICS.clear()
    _METRICS.update(
        galaxy_fixture_reads=0,
        galaxy_fixture_hits=0,
        galaxy_fixture_misses=0,
        galaxy_raw_used=0,
        galaxy_raw_fallbacks=0,
        api_team_stats_calls_avoided=0,
        api_recent_calls_avoided=0,
        api_odds_calls_avoided=0,
        galaxy_odds_used=0,
        galaxy_odds_stale_or_missing=0,
    )


def _bump(key: str, amount: int = 1) -> None:
    _METRICS[key] = int(_METRICS.get(key, 0)) + amount


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _dt(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        result = value
    else:
        text = str(value).strip().replace("Z", "+00:00")
        try:
            result = datetime.fromisoformat(text)
        except ValueError:
            return None
    if result.tzinfo is None:
        result = result.replace(tzinfo=dt_timezone.utc)
    return result.astimezone(dt_timezone.utc)


def _fresh(value: Any, now: datetime, max_age: timedelta) -> bool:
    stamp = _dt(value)
    if stamp is None:
        return False
    age = now - stamp
    return timedelta(minutes=-5) <= age <= max_age


def _prediction_json(row: dict[str, Any]) -> dict[str, Any]:
    payload = row.get("prediction_json")
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    prediction = row.get("prediction")
    return prediction if isinstance(prediction, dict) else {}


def _market_key(value: Any) -> str:
    return str(value or "").strip().lower().replace(".", "_")


def _latest_predictions(shadow: dict[str, Any], now: datetime) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in shadow.get("predictions") or []:
        if not isinstance(row, dict):
            continue
        key = _market_key(row.get("market_key"))
        if key not in SPORT_FIRST_MARKETS:
            continue
        if not _fresh(row.get("predicted_at"), now, GALAXY_SPORT_MAX_AGE):
            continue
        prior = latest.get(key)
        if prior is None or str(row.get("predicted_at") or "") > str(prior.get("predicted_at") or ""):
            latest[key] = row
    return latest


def _poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * (lam**k) / math.factorial(k)


def _total_dist(lam: float, max_goals: int = 12) -> dict[int, float]:
    dist = {k: _poisson_pmf(k, lam) for k in range(max_goals + 1)}
    mass = sum(dist.values()) or 1.0
    return {k: value / mass for k, value in dist.items()}


def _over_from_dist(dist: dict[int, float], line: float) -> float:
    threshold = math.floor(line) + 1
    return sum(prob for goals, prob in dist.items() if goals >= threshold)


def _lambda_from_over(line: float, over_probability: float) -> float:
    target = _clamp(over_probability, 0.01, 0.99)
    lo, hi = 0.10, 7.00
    for _ in range(60):
        mid = (lo + hi) / 2.0
        p = _over_from_dist(_total_dist(mid), line)
        if p < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _line_prediction(
    predictions: dict[str, dict[str, Any]], line: float
) -> tuple[float | None, dict[str, Any] | None]:
    key = f"over_under_{str(line).replace('.', '_')}"
    row = predictions.get(key)
    if not row:
        return None, None
    payload = _prediction_json(row)
    over = _num(payload.get("over"))
    if over is None:
        return None, row
    return _clamp(over, 0.001, 0.999), row


def _model_actionable(row: dict[str, Any] | None) -> bool:
    if not isinstance(row, dict):
        return False
    if row.get("publishable") is False:
        return False
    state = str(row.get("publication_state") or "").strip().lower()
    if state in {"blocked", "disabled", "research", "research_only", "watchlist", "sampling"}:
        return False
    calibration = str(row.get("calibration_status") or "").strip().lower()
    if calibration in {"failed", "bad", "blocked", "invalid"}:
        return False
    return True


def _galaxy_raw_projection(
    shadow: dict[str, Any], fx: dict[str, Any], now: datetime
) -> dict[str, Any] | None:
    if shadow.get("status") != "AVAILABLE":
        return None
    predictions = _latest_predictions(shadow, now)

    line = 2.5
    over, anchor = _line_prediction(predictions, 2.5)
    if over is None:
        line = 1.5
        over, anchor = _line_prediction(predictions, 1.5)
    if over is None:
        line = 3.5
        over, anchor = _line_prediction(predictions, 3.5)
    if over is None:
        return None

    lam = _lambda_from_over(line, over)
    dist = _total_dist(lam)

    over_15, _ = _line_prediction(predictions, 1.5)
    over_25, anchor_25 = _line_prediction(predictions, 2.5)
    over_35, _ = _line_prediction(predictions, 3.5)
    over_15 = over_15 if over_15 is not None else _over_from_dist(dist, 1.5)
    over_25 = over_25 if over_25 is not None else _over_from_dist(dist, 2.5)
    over_35 = over_35 if over_35 is not None else _over_from_dist(dist, 3.5)

    one_x_two = predictions.get("match_winner_v2_shadow")
    one_x_two_json = _prediction_json(one_x_two or {})
    home_p = _num(one_x_two_json.get("home_win"))
    draw_p = _num(one_x_two_json.get("draw"))
    away_p = _num(one_x_two_json.get("away_win"))
    if home_p is None or draw_p is None or away_p is None or home_p + draw_p + away_p <= 0:
        home_p, draw_p, away_p = 0.365, 0.27, 0.365
    else:
        total = home_p + draw_p + away_p
        home_p, draw_p, away_p = home_p / total, draw_p / total, away_p / total

    home_share = _clamp(0.50 + 0.45 * (home_p - away_p), 0.28, 0.72)
    home_lam = max(0.15, lam * home_share)
    away_lam = max(0.15, lam - home_lam)

    btts_row = predictions.get("both_teams_score")
    btts_json = _prediction_json(btts_row or {})
    btts_yes = _num(btts_json.get("yes"))
    if btts_yes is None:
        btts_yes = (1.0 - math.exp(-home_lam)) * (1.0 - math.exp(-away_lam))
    btts_yes = _clamp(btts_yes, 0.001, 0.999)

    stronger_prob = max(home_p, away_p)
    lambda_gap = abs(home_lam - away_lam)
    side_score = _clamp(
        50 + lambda_gap * 13 + max(0.0, stronger_prob - 0.45) * 70, 0, 100
    )
    goal_env = _clamp(50 + (lam - 2.35) * 28, 0, 100)
    two_way = _clamp(btts_yes * 100.0, 0, 100)

    anchor_row = anchor_25 or anchor
    actionable = _model_actionable(anchor_row)
    model_versions = sorted(
        {
            str(row.get("model_version"))
            for row in predictions.values()
            if row.get("model_version")
        }
    )

    return {
        "status": "MODELED_LIMITED",
        "model_version": MODEL_VERSION,
        "projection_model": "GALAXYPARLAY_SPORT_FIRST_v1",
        "sport_source": "GALAXYPARLAY_PERSISTED",
        "market_independent": True,
        "galaxy_contract_version": shadow.get("contract_version"),
        "galaxy_model_versions": model_versions,
        "galaxy_data_cutoff": shadow.get("data_cutoff"),
        "galaxy_actionable_total_model": actionable,
        "galaxy_actionable_reason": (
            "PUBLISHED_OR_NOT_BLOCKED"
            if actionable
            else "PUBLICATION_OR_CALIBRATION_GATE_NOT_PASSED"
        ),
        "model_limitations": [
            "GalaxyParlay persisted SPORT-FIRST predictions are reused to avoid duplicate API-Football team-stat/recent requests.",
            "1X2 remains research-only in Soccer Edge v1.1.",
            "Home/away goal split is diagnostic only; FT totals use Galaxy probability vectors directly.",
            "Lineups, goalkeeper and injuries are verified separately when material.",
        ],
        "advanced_metrics": "GALAXYPARLAY_PERSISTED_MODEL_LAYER",
        "sample": {
            "minimum_split_sample": 0,
            "source_sample": "NOT EXPOSED BY INTEGRATION CONTRACT",
        },
        "raw_home_goal_rate": round(home_lam, 4),
        "raw_away_goal_rate": round(away_lam, 4),
        "raw_total_goals": round(lam, 4),
        "raw_home_xg": "NOT EXPOSED BY INTEGRATION CONTRACT",
        "raw_away_xg": "NOT EXPOSED BY INTEGRATION CONTRACT",
        "raw_home_win_prob": round(home_p, 6),
        "raw_draw_prob": round(draw_p, 6),
        "raw_away_win_prob": round(away_p, 6),
        "raw_btts_yes_prob": round(btts_yes, 6),
        "raw_over_1_5_prob": round(over_15, 6),
        "raw_over_2_5_prob": round(over_25, 6),
        "raw_over_3_5_prob": round(over_35, 6),
        "top_scorelines": [],
        "scoring_path": (
            "TWO-WAY OPEN GAME"
            if lam >= 2.85 and btts_yes >= 0.58
            else "HIGH TOTAL ENVIRONMENT"
            if lam >= 2.85
            else "MUTUAL SUPPRESSION UNDER"
            if lam <= 2.05
            else "MIXED / NO STRONG SCORING PATH"
        ),
        "screen_scores": {
            "side_edge_score": round(side_score, 1),
            "goal_environment_score": round(goal_env, 1),
            "two_way_scoring_score": round(two_way, 1),
            "corners_opportunity_score": "NOT MODELED",
            "score_status": "GALAXYPARLAY_SPORT_FIRST",
        },
        "_total_dist": dist,
    }


def _public_raw(raw: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in raw.items() if not key.startswith("_")}


def _odds_timestamp(row: dict[str, Any]) -> Any:
    return row.get("snapshot_at") or row.get("updated_at") or row.get("captured_at")


def _line_from_market_key(key: str) -> float | None:
    match = re.search(r"over_under_([0-9]+)[_.]([0-9]+)", key)
    if not match:
        return None
    return float(f"{match.group(1)}.{match.group(2)}")


def _galaxy_odds_market(
    shadow: dict[str, Any], now: datetime
) -> dict[str, Any] | None:
    rows: list[dict[str, Any]] = []
    for item in shadow.get("odds") or []:
        if not isinstance(item, dict) or not _fresh(
            _odds_timestamp(item), now, GALAXY_ODDS_MAX_AGE
        ):
            continue
        key = _market_key(item.get("market_key"))
        odds_data = item.get("odds_data") or {}
        if not isinstance(odds_data, dict):
            continue
        book = item.get("bookmaker")
        stamp = _odds_timestamp(item)
        normalized = {str(k).strip().lower(): v for k, v in odds_data.items()}

        if key in {"match_winner", "match_winner_v2_shadow"}:
            vals = []
            for source, selection in (
                ("home", "Home"),
                ("draw", "Draw"),
                ("away", "Away"),
            ):
                price = _num(normalized.get(source))
                if price and price > 1:
                    vals.append({"selection": selection, "price": price})
            if len(vals) == 3:
                rows.append(
                    {
                        "bookmaker_id": None,
                        "bookmaker": book,
                        "market_id": item.get("provider_market_id"),
                        "market": "Match Winner",
                        "values": vals,
                        "provider_update": stamp,
                    }
                )
            continue

        if key.startswith("over_under_"):
            line = _line_from_market_key(key)
            if line is None:
                continue
            over_price = _num(
                normalized.get(f"over {line:g}") or normalized.get("over")
            )
            under_price = _num(
                normalized.get(f"under {line:g}") or normalized.get("under")
            )
            if over_price and under_price and over_price > 1 and under_price > 1:
                rows.append(
                    {
                        "bookmaker_id": None,
                        "bookmaker": book,
                        "market_id": item.get("provider_market_id"),
                        "market": "Goals Over/Under",
                        "values": [
                            {"selection": f"Over {line:g}", "price": over_price},
                            {"selection": f"Under {line:g}", "price": under_price},
                        ],
                        "provider_update": stamp,
                    }
                )
            continue

        if key in {"both_teams_score", "both_teams_to_score", "btts"}:
            yes = _num(normalized.get("yes"))
            no = _num(normalized.get("no"))
            if yes and no and yes > 1 and no > 1:
                rows.append(
                    {
                        "bookmaker_id": None,
                        "bookmaker": book,
                        "market_id": item.get("provider_market_id"),
                        "market": "Both Teams To Score",
                        "values": [
                            {"selection": "Yes", "price": yes},
                            {"selection": "No", "price": no},
                        ],
                        "provider_update": stamp,
                    }
                )

    if not rows:
        return None
    return {
        "markets": rows[:60],
        "market_count": len(rows),
        "truncated": len(rows) > 60,
    }


async def _galaxy_fixture(fixture_id: int) -> dict[str, Any]:
    cached = _GALAXY_CACHE.get(int(fixture_id))
    if cached is not None:
        return cached
    if not galaxy_configured() or _GALAXY_CLIENT is None:
        result = {
            "status": "UNAVAILABLE",
            "fixture_id": fixture_id,
            "error": "GALAXY_NOT_CONFIGURED",
        }
    else:
        _bump("galaxy_fixture_reads")
        result = await get_fixture_shadow(_GALAXY_CLIENT, int(fixture_id))
    _GALAXY_CACHE[int(fixture_id)] = result
    if result.get("status") == "AVAILABLE":
        _bump("galaxy_fixture_hits")
    else:
        _bump("galaxy_fixture_misses")
    return result


async def _galaxy_aware_odds_7m(
    fixture_id: int, now: datetime
) -> dict[str, Any]:
    shadow = await _galaxy_fixture(fixture_id)
    galaxy_market = _galaxy_odds_market(shadow, now)
    if galaxy_market is not None:
        _bump("galaxy_odds_used")
        _bump("api_odds_calls_avoided")
        return galaxy_market
    _bump("galaxy_odds_stale_or_missing")
    return await _ORIGINAL_ODDS_7M(fixture_id, now)


def _downgrade_row(
    row: dict[str, Any], reason: str, status_key: str
) -> bool:
    row[status_key] = "RESEARCH_ONLY_PENDING_VALIDATION"
    reasons = list(row.get("reasons") or [])
    if reason not in reasons:
        reasons.append(reason)
    row["reasons"] = reasons
    if row.get("classification") in {"BET", "LEAN"}:
        row["classification"] = "WATCH"
        row["stake_units"] = 0.0
        return True
    return False


def _research_only_side_btts_and_galaxy_gate(
    raw: dict[str, Any],
    market: Any,
    coverage: dict[str, Any],
    availability: float | None,
    stage: str,
    lineup: Any,
) -> dict[str, Any]:
    decision = _ORIGINAL_V8_RESEARCH_EVALUATE(
        raw, market, coverage, availability, stage, lineup
    )
    if not isinstance(decision, dict):
        return decision
    out = dict(decision)
    rows = [
        dict(row)
        for row in (out.get("decisions") or [])
        if isinstance(row, dict)
    ]
    changed_btts = 0
    changed_gate = 0
    for row in rows:
        if row.get("family") == "BTTS":
            if _downgrade_row(
                row,
                "BTTS_RESEARCH_ONLY: current calibration sample is insufficient for automatic BET/LEAN promotion.",
                "btts_model_status",
            ):
                changed_btts += 1
        if (
            raw.get("sport_source") == "GALAXYPARLAY_PERSISTED"
            and not raw.get("galaxy_actionable_total_model")
        ):
            if _downgrade_row(
                row,
                "GALAXY_PUBLICATION_GATE: Galaxy model is fresh but has not passed its publication/calibration gate; research only.",
                "galaxy_publication_gate",
            ):
                changed_gate += 1

    rank = {"BET": 4, "WATCH": 3, "LEAN": 2, "PASS": 1}
    rows.sort(
        key=lambda item: (
            rank.get(str(item.get("classification")), 0),
            item.get("prob_edge_pp")
            if isinstance(item.get("prob_edge_pp"), (int, float))
            else -999,
            item.get("estimated_ev")
            if isinstance(item.get("estimated_ev"), (int, float))
            else -999,
        ),
        reverse=True,
    )
    best = rows[0] if rows else out.get("best_decision")
    out["decisions"] = rows[:20]
    out["best_decision"] = best
    out["status"] = (
        best.get("classification")
        if isinstance(best, dict)
        else out.get("status", "WATCH")
    )
    out["btts_market_mode"] = "RESEARCH_ONLY_PENDING_VALIDATION"
    out["btts_decisions_downgraded"] = changed_btts
    out["galaxy_publication_gate_downgraded"] = changed_gate
    return out


async def _galaxy_first_priority_event(
    fx: dict[str, Any],
    stage: str,
    coverage: dict[str, Any],
    now: datetime,
) -> dict[str, Any]:
    if stage == "POSTGAME":
        return await _ORIGINAL_PRIORITY_EVENT(fx, stage, coverage, now)

    shadow = await _galaxy_fixture(int(fx["fixture_id"]))

    if stage == "CLOSE":
        prior = v5._shortlist_get(fx["fixture_id"], now)
        if not prior or not prior.get("shortlisted"):
            return await _ORIGINAL_PRIORITY_EVENT(fx, stage, coverage, now)
        event = {
            "event_type": "SOCCER_REFRESH",
            "stage": stage,
            "fixture": fx,
            "coverage": coverage,
            "model_version": MODEL_VERSION,
            "classification": "CLOSE",
            "tier": None,
            "stake_units": 0.0,
            "bet_eligible": False,
            "availability_confidence": None,
            "sporting_shortlist": prior,
            "notes": [],
        }
        galaxy_market = _galaxy_odds_market(shadow, now)
        if galaxy_market is not None:
            event["market"] = galaxy_market
            event["market_source"] = "GALAXYPARLAY_PERSISTED"
            event["market_use"] = "CLOSING_SNAPSHOT_FROM_GALAXY_CACHE"
            _bump("galaxy_odds_used")
            _bump("api_odds_calls_avoided")
            return event
        return await _ORIGINAL_PRIORITY_EVENT(fx, stage, coverage, now)

    raw_internal = _galaxy_raw_projection(shadow, fx, now)
    if raw_internal is None:
        _bump("galaxy_raw_fallbacks")
        return await _ORIGINAL_PRIORITY_EVENT(fx, stage, coverage, now)

    _bump("galaxy_raw_used")
    _bump("api_team_stats_calls_avoided", 2)

    event: dict[str, Any] = {
        "event_type": "SOCCER_REFRESH",
        "stage": stage,
        "fixture": fx,
        "coverage": coverage,
        "model_version": MODEL_VERSION,
        "classification": "WATCH",
        "tier": None,
        "stake_units": 0.0,
        "bet_eligible": False,
        "availability_confidence": None,
        "notes": [
            "GALAXY FIRST: fresh persisted sport projection reused; duplicate API-Football team-stat/recent requests skipped."
        ],
        "galaxy_source": {
            "status": shadow.get("status"),
            "contract_version": shadow.get("contract_version"),
            "data_cutoff": shadow.get("data_cutoff"),
            "external_api_calls": shadow.get("external_api_calls"),
        },
    }

    prior_shortlist = v5._shortlist_get(fx["fixture_id"], now)
    shortlisted = bool(prior_shortlist and prior_shortlist.get("shortlisted"))

    if stage in v2.SPORTING_STAGES:
        screen = v5._screen_shortlist(raw_internal)
        event["sporting_screen_initial"] = screen
        event["sporting_screen_refined"] = screen
        v5._shortlist_set(fx["fixture_id"], screen, now)
        event["sporting_shortlist"] = screen
        shortlisted = bool(screen.get("shortlisted"))
        event["raw_projection"] = _public_raw(raw_internal)
        if shortlisted:
            _bump("api_recent_calls_avoided", 2)
        if not shortlisted:
            event["classification"] = "PASS"
            event["market_skipped_by_sport_screen"] = stage in v2.MARKET_STAGES
            event["notes"].append(
                "Galaxy SPORT-FIRST screen below shortlist threshold; availability/market refresh skipped."
            )
            return event
    else:
        event["sporting_shortlist"] = prior_shortlist

    if not shortlisted:
        event["classification"] = "PASS"
        event["notes"].append("No active sporting shortlist for this refresh.")
        return event

    if stage in v2.INJURY_STAGES and coverage.get("injuries"):
        event["injuries"] = await v5._injuries_2h(fx["fixture_id"], now)
    elif stage in v2.INJURY_STAGES:
        event["injuries"] = "NOT VERIFIED"

    if stage in v2.LINEUP_STAGES and coverage.get("lineups"):
        lineup = await v5._lineup_cached(fx["fixture_id"], now)
        event["lineups"] = lineup
        if lineup.get("both_xi_confirmed") and lineup.get(
            "both_goalkeepers_confirmed"
        ):
            event["availability_confidence"] = 0.90
        else:
            event["availability_confidence"] = (
                0.70 if stage in {"T-60", "T-40"} else 0.60
            )
            event["notes"].append(
                "Material lineup/goalkeeper information remains NOT VERIFIED."
            )
    elif stage in v2.LINEUP_STAGES:
        event["lineups"] = "NOT VERIFIED"
        event["availability_confidence"] = 0.60

    event["raw_projection"] = _public_raw(raw_internal)

    if stage in v2.MARKET_STAGES:
        galaxy_market = _galaxy_odds_market(shadow, now)
        if galaxy_market is not None:
            event["market"] = galaxy_market
            event["market_source"] = "GALAXYPARLAY_PERSISTED"
            event["market_use"] = "MARKET_COMPARISON_AFTER_SPORTING_SHORTLIST"
            _bump("galaxy_odds_used")
            _bump("api_odds_calls_avoided")
        elif coverage.get("odds"):
            event["market"] = await _ORIGINAL_ODDS_7M(
                fx["fixture_id"], now
            )
            event["market_source"] = (
                "API_FOOTBALL_FALLBACK_GALAXY_STALE_OR_MISSING"
            )
            event["market_use"] = "MARKET_COMPARISON_AFTER_SPORTING_SHORTLIST"
            _bump("galaxy_odds_stale_or_missing")
        else:
            event["market"] = "NOT VERIFIED"
            _bump("galaxy_odds_stale_or_missing")

    if stage in {"T-40", "T-20", "T-10"}:
        decision = v4._safe_evaluate_market(
            raw_internal,
            event.get("market"),
            coverage,
            event.get("availability_confidence"),
            stage,
            event.get("lineups"),
        )
        event["market_decision"] = decision
        best = decision.get("best_decision") or {}
        event["classification"] = decision.get("status") or "WATCH"
        event["tier"] = best.get("tier")
        event["stake_units"] = best.get("stake_units", 0.0)
        event["bet_eligible"] = event["classification"] == "BET"
        if best:
            event["best_market"] = v5._best_decision_fields(best)

    if stage == "T-20":
        lineup = event.get("lineups")
        if (
            not isinstance(lineup, dict)
            or not lineup.get("both_xi_confirmed")
            or not lineup.get("both_goalkeepers_confirmed")
        ):
            event["bet_eligible"] = False
            event["classification"] = "WATCH"
            event["notes"].append(
                "T-20 lineup/GK gate failed: BET eligibility blocked."
            )

    return event


async def run_tick() -> dict[str, Any]:
    global _GALAXY_CLIENT
    _reset_metrics()
    _GALAXY_CACHE.clear()

    previous_priority = v5._priority_event
    previous_odds = v5._odds_7m
    previous_v8_eval = v8._research_only_side_evaluate

    if galaxy_configured():
        timeout = float(os.getenv("GALAXYPARLAY_TIMEOUT", "6"))
        _GALAXY_CLIENT = httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
        )
    else:
        _GALAXY_CLIENT = None

    v5._priority_event = _galaxy_first_priority_event
    v5._odds_7m = _galaxy_aware_odds_7m
    v8._research_only_side_evaluate = (
        _research_only_side_btts_and_galaxy_gate
    )

    try:
        payload = await v8.run_tick()
    finally:
        v5._priority_event = previous_priority
        v5._odds_7m = previous_odds
        v8._research_only_side_evaluate = previous_v8_eval
        if _GALAXY_CLIENT is not None:
            await _GALAXY_CLIENT.aclose()
            _GALAXY_CLIENT = None

    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    payload["galaxy_first_enabled"] = galaxy_configured()
    payload["galaxy_first_policy"] = (
        "GALAXY_PERSISTED_SPORT_AND_FRESH_ODDS_FIRST; "
        "API_FOOTBALL_ONLY_FOR_MISSING_OR_STALE_DATA_AND_MATERIAL_AVAILABILITY"
    )
    payload["galaxy_first_metrics"] = dict(_METRICS)
    payload["actionable_model_scope"] = (
        "FT_TOTALS_ONLY; 1X2_AND_BTTS_RESEARCH_ONLY"
    )
    payload["duplicate_request_policy"] = {
        "team_stats": "SKIP_API_FOOTBALL_WHEN_FRESH_GALAXY_SPORT_PROJECTION_EXISTS",
        "recent_form": "SKIP_API_FOOTBALL_WHEN_FRESH_GALAXY_SPORT_PROJECTION_EXISTS",
        "odds": "SKIP_API_FOOTBALL_WHEN_FRESH_GALAXY_ODDS_EXIST",
        "lineups": "API_FOOTBALL_ALLOWED_UNTIL_GALAXY_CONTRACT_EXPOSES_VERIFIED_CURRENT_XI",
        "injuries": "API_FOOTBALL_ALLOWED_UNTIL_GALAXY_CONTRACT_EXPOSES_VERIFIED_CURRENT_AVAILABILITY",
        "slate": "API_FOOTBALL_RECONCILIATION_STILL_ACTIVE_FOR_AUTHORITATIVE_CURRENT_STATUS",
    }
    return payload
