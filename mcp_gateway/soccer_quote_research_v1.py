"""Instrument-exact SPORT-THEN-MARKET research evaluation. No live quote access.

A provider proxy is not a real executable ticket. All outputs WATCH/PASS,
zero stakes. No caller supplied 'verified' flag can promote a bet.
"""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import Any
import math

MODEL_VERSION = "SOCCER_QUOTE_RESEARCH_v1.0"
ALLOWED_FAMILIES = frozenset({
    "1X2", "BTTS", "FT_TOTALS", "TEAM_TOTALS", "1H_1X2",
    "1H_TOTALS", "2H_1X2", "2H_TOTALS", "FT_CORNERS",
    "TEAM_CORNERS", "FT_CARDS", "PLAYER_SHOTS",
})
THRESHOLDS = {
    "SIDE": (3.0, 5.0, 7.5),
    "GOALS": (3.5, 5.5, 8.0),
    "DERIVATIVE": (4.0, 6.0, 9.0),
}


def _dt(raw: Any) -> datetime | None:
    if not isinstance(raw, str): return None
    try: t = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError: return None
    return t.astimezone(timezone.utc) if t.tzinfo else None


def _price(raw: Any) -> float | None:
    if type(raw) not in (int, float) or not math.isfinite(raw):
        return None
    return float(raw) if 1.0 < raw <= 500 else None


def _prob(raw: Any) -> float | None:
    if type(raw) not in (int,float) or not math.isfinite(raw): return None
    return float(raw) if 0 <= raw <= 1 else None


def _family_bucket(family: str) -> str:
    return "SIDE" if family == "1X2" else (
        "GOALS" if family in {"FT_TOTALS", "BTTS"} else "DERIVATIVE")


def _result(reason: str, **extra: Any) -> dict[str, Any]:
    return {
        "status": "WATCH", "classification": "WATCH",
        "reason": reason, "model_version": MODEL_VERSION,
        "bet_eligible": False, "stake_units": 0.0,
        "direct_executable_quote_verified": False,
        "true_clv_verified": False, "commercial_promotion_allowed": False,
        **extra,
    }


def analyze_frozen_sport_and_quote(
    frozen_sport: dict[str, Any],
    market_snapshot: dict[str, Any] | None,
    *,
    evaluated_at_utc: str,
    shrink_weight: float = 0.25,
) -> dict[str, Any]:
    """Market comes second. Requires complete same-book same-line quote set."""
    if not isinstance(frozen_sport, dict):
        return _result("RAW_SPORT_NOT_VERIFIED")
    family = frozen_sport.get("family")
    if family not in ALLOWED_FAMILIES:
        return _result("FAMILY_NOT_MODELED_OR_NOT_SUPPORTED")
    raw = frozen_sport.get("selection_probabilities")
    if (not isinstance(raw, dict) or not raw or
            not all(isinstance(k,str) and _prob(v) is not None for k,v in raw.items())):
        return _result("FROZEN_RAW_SPORT_PROBABILITIES_MISSING")
    model_at = _dt(frozen_sport.get("frozen_at_utc"))
    kickoff = _dt(frozen_sport.get("kickoff_utc"))
    now = _dt(evaluated_at_utc)
    if not model_at or not kickoff or not now or model_at >= kickoff - timedelta(minutes=5):
        return _result("RAW_SPORT_NOT_FROZEN_BEFORE_MATCH")
    if not 0 <= shrink_weight <= 0.55:
        return _result("SHRINK_WEIGHT_OUTSIDE_LIMITED_RESEARCH_RANGE")
    if not isinstance(market_snapshot, dict):
        return _result("NO_VERIFIED_MARKET_SOURCE")
    if market_snapshot.get("fixture_id") != frozen_sport.get("fixture_id"):
        return _result("FIXTURE_ID_MISMATCH")
    if market_snapshot.get("family") != family:
        return _result("MARKET_FAMILY_MISMATCH")
    if market_snapshot.get("source_type") not in {"AGGREGATOR_PROXY","DIRECT_BOOKMAKER_RESEARCH"}:
        return _result("BOOKMAKER_SOURCE_NOT_VERIFIED")
    book, threshold, selection = (
        market_snapshot.get("bookmaker"), market_snapshot.get("line"), market_snapshot.get("selection"))
    if not isinstance(book,str) or not book.strip() or not isinstance(selection,str):
        return _result("EXACT_BOOK_AND_SELECTION_REQUIRED")
    if family in {"TEAM_TOTALS","TEAM_CORNERS","PLAYER_SHOTS"}:
        if type(market_snapshot.get("entity_id")) is not int or market_snapshot["entity_id"]<=0:
            return _result("EXACT_TEAM_OR_PLAYER_ID_REQUIRED")
        if frozen_sport.get("entity_id") != market_snapshot["entity_id"]:
            return _result("PLAYER_OR_TEAM_INSTRUMENT_MISMATCH")
    if family in {"1H_1X2","1H_TOTALS","2H_1X2","2H_TOTALS"}:
        if market_snapshot.get("period") != family[:2] or frozen_sport.get("period") != family[:2]:
            return _result("PERIOD_MISMATCH")
    if family in {"FT_TOTALS","TEAM_TOTALS","1H_TOTALS","2H_TOTALS","FT_CORNERS","TEAM_CORNERS","FT_CARDS","PLAYER_SHOTS"}:
        if type(threshold) not in (float,int) or not math.isfinite(threshold) or not (
                abs(2 * threshold - round(2 * threshold)) < 1e-8 and int(round(2 * threshold)) % 2 == 1):
            return _result("BINARY_HALF_GOAL_OR_COUNT_LINE_REQUIRED")
    elif threshold is not None:
        return _result("UNEXPECTED_THRESHOLD_FOR_NOLINE_MARKET")
    captured = _dt(market_snapshot.get("captured_at_utc"))
    source_at = _dt(market_snapshot.get("source_updated_at_utc"))
    if not captured or not source_at or source_at > captured or captured < model_at or captured >= kickoff:
        return _result("QUOTE_SOURCE_ASOF_NOT_VERIFIED")
    if now < captured or now - captured > timedelta(minutes=10):
        return _result("QUOTE_NOT_FRESH")
    # Only the SAME half-line is compared, never prices at different goals.
    selection_probs = raw
    if threshold is not None:
        selection_probs = {k:v for k,v in raw.items()
                           if k in {f"over_{threshold:g}",f"under_{threshold:g}"}}
        if len(selection_probs)!=2:
            return _result("RAW_NOT_MODELED_AT_EXACT_THRESHOLD")
    selections = market_snapshot.get("outcomes")
    if not isinstance(selections, dict) or set(selections) != set(selection_probs):
        return _result("COMPLETE_SAME_LINE_TWO_WAY_OR_THREE_WAY_REQUIRED")
    prices = {key:_price(v) for key,v in selections.items()}
    if any(p is None for p in prices.values()):
        return _result("INVALID_OR_MISSING_OPPOSITE_PRICE")
    if len(prices) != (3 if family in {"1X2","1H_1X2","2H_1X2"} else 2):
        return _result("INCOMPLETE_INSTRUMENT")
    if selection not in prices:
        return _result("SELECTION_NOT_PRICED")
    if not isinstance(market_snapshot.get("snapshot_id"),str) or not market_snapshot["snapshot_id"]:
        return _result("QUOTE_SNAPSHOT_ID_REQUIRED")
    if not isinstance(market_snapshot.get("market_id"),str) or not market_snapshot["market_id"]:
        return _result("EXACT_MARKET_ID_REQUIRED")
    # Exact same-line and same-market book instrument is assumed only if a
    # single snapshot object includes all sides. We never pool sources.
    implied={k:1/p for k,p in prices.items()}
    vig=sum(implied.values())
    if vig < 0.99 or vig > 1.60:
        return _result("QUOTE_VIG_OUTSIDE_SANITY_RANGE")
    fair={k:implied[k]/vig for k in prices}
    raw_p=selection_probs[selection]
    fair_p=fair[selection]
    shrunk=shrink_weight*raw_p+(1-shrink_weight)*fair_p
    edge_pp=(shrunk-fair_p)*100
    ev=shrunk * prices[selection]-1
    b,a,s=THRESHOLDS[_family_bucket(family)]
    tier="S" if edge_pp>=s else "A" if edge_pp>=a else "B" if edge_pp>=b else "NONE"
    return _result(
        "DIRECT_EXECUTABLE_BOOKMAKER_BETSLIP_NOT_VERIFIED",
        fixture_id=frozen_sport["fixture_id"],family=family,
        bookmaker=book,market_id=market_snapshot["market_id"],
        snapshot_id=market_snapshot["snapshot_id"],line=threshold,
        selection=selection,source_type=market_snapshot["source_type"],
        price=prices[selection],quoted_at_utc=captured.isoformat(),
        market_source_updated_at=source_at.isoformat(),
        p_raw=round(raw_p, 8),p_breakeven=round(implied[selection],8),
        p_market_fair=round(fair_p,8),p_shrunk_research=round(shrunk,8),
        shrink_weight=shrink_weight,prob_edge_pp=round(edge_pp,4),
        estimated_ev_research=round(ev,6),theoretical_tier=tier,
        quote_is_provider_proxy=market_snapshot["source_type"]=="AGGREGATOR_PROXY",
        family_oos_gate_promoted=False,
        research_only=True,
    )


def compare_close_same_instrument(opening: dict[str,Any], close: dict[str,Any]) -> dict[str, Any]:
    """No claimed CLV without immutable same-instrument verified bookmaker quotes."""
    identity=("fixture_id","family","bookmaker","market_id","selection","line","entity_id","period")
    if any(opening.get(k)!=close.get(k) for k in identity):
        return {"status":"NOT_VERIFIED","reason":"CLOSING_QUOTE_INSTRUMENT_MISMATCH","true_clv":None}
    if opening.get("source_type")!="DIRECT_BOOKMAKER_EXECUTABLE_VERIFIED" or close.get("source_type")!="DIRECT_BOOKMAKER_EXECUTABLE_VERIFIED":
        return {"status":"NOT_VERIFIED","reason":"DIRECT_EXECUTABLE_CLOSE_NOT_VERIFIED","true_clv":None}
    a,b=_price(opening.get("price")),_price(close.get("price"))
    start,end=_dt(opening.get("captured_at_utc")),_dt(close.get("captured_at_utc"))
    if not a or not b or not start or not end or end<=start:
        return {"status":"NOT_VERIFIED","reason":"CLOSE_PRICE_OR_CHRONOLOGY_NOT_VERIFIED","true_clv":None}
    # Even purported direct-verified status must be independently audited.
    return {"status":"REQUIRES_INDEPENDENT_BOOKMAKER_AUDIT",
            "reason":"CLOSE_PROVENANCE_AND_EXECUTABILITY_NOT_EXTERNALLY_ATTESTED",
            "research_price_change":round(a/b-1,6),"true_clv":None}
