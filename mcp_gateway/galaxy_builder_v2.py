from __future__ import annotations

import itertools
import math
from collections import Counter
from typing import Any

from mcp_gateway import galaxy_builder as v1

SCHEMA_VERSION = "0.2.0"
TARGET_DECIMAL = 2.10
TARGET_EDGE_PP = 3.5
MIN_AVAILABILITY = 0.70
MAX_SGP = 6
MAX_MULTI = 6


def _num(v: Any) -> float | None:
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _american(d: float | None) -> int | None:
    if d is None or d <= 1:
        return None
    return int(round((d - 1) * 100)) if d >= 2 else int(round(-100 / (d - 1)))


def _matrix(home_lam: float, away_lam: float, max_goals: int = 10) -> list[tuple[int, int, float]]:
    rows = []
    mass = 0.0
    for h in range(max_goals + 1):
        hp = math.exp(-home_lam) * home_lam**h / math.factorial(h)
        for a in range(max_goals + 1):
            p = hp * math.exp(-away_lam) * away_lam**a / math.factorial(a)
            rows.append((h, a, p)); mass += p
    return [(h, a, p / mass) for h, a, p in rows] if mass > 0 else []


def _pred(leg: dict[str, Any], h: int, a: int) -> bool:
    fam = leg["family"]; sel = leg["selection"]; line = leg.get("line")
    if fam == "FT_GOALS": return h + a > line if sel == "OVER" else h + a < line
    if fam == "BTTS": return (h > 0 and a > 0) if sel == "YES" else (h == 0 or a == 0)
    if fam == "DOUBLE_CHANCE":
        return h >= a if sel == "1X" else a >= h if sel == "X2" else h != a
    return False


def _prob(matrix: list[tuple[int, int, float]], legs: list[dict[str, Any]]) -> float:
    return max(0.0, min(1.0, sum(p for h, a, p in matrix if all(_pred(leg, h, a) for leg in legs))))


def _quotes(event: dict[str, Any], leg: dict[str, Any]) -> list[dict[str, Any]]:
    market = event.get("market") if isinstance(event.get("market"), dict) else {}
    out = []
    for row in market.get("markets") or []:
        if not isinstance(row, dict): continue
        name = v1._normalize(row.get("market"))
        fam = leg["family"]; sel = leg["selection"]; line = leg.get("line")
        if fam == "FT_GOALS" and not v1._is_full_match_goal_market(row.get("market")): continue
        if fam == "BTTS" and not ("both teams" in name and "score" in name): continue
        if fam == "DOUBLE_CHANCE" and not ("double chance" in name or "doble oportunidad" in name): continue
        for value in row.get("values") or []:
            if not isinstance(value, dict): continue
            text = v1._normalize(value.get("selection")); ok = False
            if fam == "FT_GOALS":
                parsed = v1._line_from_text(value.get("selection")); ok = sel.lower() in text and parsed is not None and abs(parsed - line) < 1e-6
            elif fam == "BTTS": ok = text in ({"yes", "si", "sí"} if sel == "YES" else {"no"})
            else:
                compact = text.replace(" ", "")
                aliases = {"1X": ("1x", "homeordraw", "homedraw"), "X2": ("x2", "draworaway", "drawaway"), "12": ("12", "homeoraway", "homeaway")}
                ok = any(x in compact for x in aliases[sel])
            price = _num(value.get("price"))
            if ok and price and price > 1:
                out.append({"bookmaker": row.get("bookmaker"), "price": round(price, 4), "provider_update": row.get("provider_update"), "market": row.get("market"), "selection_text": value.get("selection")})
    return out


def _legs(event: dict[str, Any]) -> tuple[list[dict[str, Any]], list[tuple[int, int, float]]]:
    raw = event.get("raw_projection") if isinstance(event.get("raw_projection"), dict) else {}
    hl = _num(raw.get("raw_home_goal_rate")); al = _num(raw.get("raw_away_goal_rate"))
    if not hl or not al: return [], []
    matrix = _matrix(hl, al); rows = []
    for line in (0.5, 1.5, 2.5, 3.5, 4.5, 5.5):
        for sel in ("OVER", "UNDER"):
            leg = {"family": "FT_GOALS", "selection": sel, "line": line}; p = _prob(matrix, [leg])
            if p >= (0.66 if line == 2.5 else 0.70):
                leg.update(probability=round(p, 6), actionable_model=True, probability_source="DIRECT_SCORE_MATRIX", quotes=_quotes(event, leg)); rows.append(leg)
    for sel in ("YES", "NO"):
        leg = {"family": "BTTS", "selection": sel, "line": None}; p = _prob(matrix, [leg])
        if p >= 0.66:
            leg.update(probability=round(p, 6), actionable_model=False, probability_source="DIRECT_SCORE_MATRIX", research_only_reason="BTTS_RESEARCH_ONLY", quotes=_quotes(event, leg)); rows.append(leg)
    for sel in ("1X", "X2", "12"):
        leg = {"family": "DOUBLE_CHANCE", "selection": sel, "line": None}; p = _prob(matrix, [leg])
        if p >= 0.72:
            leg.update(probability=round(p, 6), actionable_model=False, probability_source="PROTECTIVE_TRANSFORM_OF_SCORE_MATRIX", research_only_reason="SIDE_MODEL_RESEARCH_ONLY; USE_DOUBLE_CHANCE_AS_PROTECTIVE_SHADOW", quotes=_quotes(event, leg)); rows.append(leg)
    return rows, matrix


def _same_book_ref(legs: list[dict[str, Any]]) -> dict[str, Any] | None:
    books = None; price_maps = []
    for leg in legs:
        prices = {}
        for q in leg.get("quotes") or []:
            b = str(q.get("bookmaker") or ""); p = _num(q.get("price"))
            if b and p and p > 1: prices[b] = max(prices.get(b, 0), p)
        if not prices: return None
        price_maps.append(prices); books = set(prices) if books is None else books & set(prices)
    if not books: return None
    book = max(books, key=lambda b: math.prod(m[b] for m in price_maps)); d = math.prod(m[book] for m in price_maps)
    return {"bookmaker": book, "component_product_decimal_reference": round(d, 4), "component_product_american_reference": _american(d), "warning": "REFERENCE_ONLY_NOT_FINAL_SGP_QUOTE"}


def _public_leg(leg: dict[str, Any]) -> dict[str, Any]:
    return dict(leg)


def _sgp(event: dict[str, Any]) -> list[dict[str, Any]]:
    coverage = event.get("coverage") if isinstance(event.get("coverage"), dict) else {}
    tier = event.get("tier") or coverage.get("data_tier"); av = _num(event.get("availability_confidence"))
    if tier not in {"A", "B"} or av is None or av < MIN_AVAILABILITY: return []
    legs, matrix = _legs(event)
    if len(legs) < 2: return []
    fx = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}; out = []
    for n in (2, 3):
        for combo in itertools.combinations(legs, n):
            if len({x["family"] for x in combo}) != n: continue
            p = _prob(matrix, list(combo)); edge110 = (p - 1 / TARGET_DECIMAL) * 100
            if p < 0.515 or edge110 < TARGET_EDGE_PP: continue
            all_actionable = all(bool(x.get("actionable_model")) for x in combo)
            out.append({
                "type": "SAME_GAME_PARLAY", "fixture_id": fx.get("fixture_id"), "kickoff": fx.get("kickoff"), "country": fx.get("country"), "league": fx.get("league"),
                "match": f"{fx.get('home_team') or fx.get('home_name')} vs {fx.get('away_team') or fx.get('away_name')}", "data_tier": tier, "availability_confidence": round(av, 3),
                "legs": [_public_leg(x) for x in combo], "joint_model_probability": round(p, 6), "fair_decimal": round(1 / p, 4), "fair_american": _american(1 / p),
                "edge_if_exact_quote_is_plus_110_pp": round(edge110, 2), "component_price_reference": _same_book_ref(list(combo)), "exact_parlay_quote": None,
                "status": "GALAXY SGP WATCH — EXACT SGP QUOTE NEEDED", "bet_eligible": False, "all_leg_models_actionable": all_actionable,
                "block_reasons": ["EXACT_CORRELATION_ADJUSTED_SGP_QUOTE_NOT_EXPOSED", *([] if av >= 0.85 else ["AVAILABILITY_BELOW_0.85"]), *([] if all_actionable else ["RESEARCH_ONLY_LEG_PRESENT"])],
                "joint_probability_method": "DIRECT_SCORE_MATRIX_CORRELATION_AWARE",
            })
    out.sort(key=lambda r: (r["all_leg_models_actionable"], r["component_price_reference"] is not None, r["joint_model_probability"]), reverse=True)
    return out[:3]


def _multi(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pool = []
    for e in events:
        if e.get("event_type") != "SOCCER_REFRESH" or e.get("stage") in {"POSTGAME", "CLOSE"}: continue
        prov = e.get("market_provenance") if isinstance(e.get("market_provenance"), dict) else {}
        if prov.get("source") not in {"GALAXY_ODDS", "API_FALLBACK_ODDS"}: continue
        av = _num(e.get("availability_confidence")); fx = e.get("fixture") if isinstance(e.get("fixture"), dict) else {}
        if av is None or av < MIN_AVAILABILITY: continue
        legs, _ = _legs(e)
        for leg in legs:
            p = float(leg.get("probability") or 0)
            if p < 0.70 or (leg["family"] == "DOUBLE_CHANCE" and p < 0.76): continue
            by_book = {}
            for q in leg.get("quotes") or []:
                b = str(q.get("bookmaker") or ""); price = _num(q.get("price"))
                if b and price and price > 1: by_book[b] = max(by_book.get(b, 0), price)
            if by_book: pool.append({"fixture_id": fx.get("fixture_id"), "match": f"{fx.get('home_team') or fx.get('home_name')} vs {fx.get('away_team') or fx.get('away_name')}", "kickoff": fx.get("kickoff"), "league": fx.get("league"), "availability": av, "leg": leg, "by_book": by_book})
    out = []
    for a, b in itertools.combinations(pool, 2):
        if a["fixture_id"] == b["fixture_id"]: continue
        for book in set(a["by_book"]) & set(b["by_book"]):
            d = a["by_book"][book] * b["by_book"][book]
            if not (TARGET_DECIMAL <= d <= 3.50): continue
            raw_p = a["leg"]["probability"] * b["leg"]["probability"]; p = raw_p * 0.98; edge = (p - 1 / d) * 100
            if edge < TARGET_EDGE_PP: continue
            action = bool(a["leg"].get("actionable_model")) and bool(b["leg"].get("actionable_model"))
            out.append({"type": "MULTI_MATCH_PARLAY", "bookmaker": book, "legs": [
                {"fixture_id": a["fixture_id"], "match": a["match"], "kickoff": a["kickoff"], "league": a["league"], "availability_confidence": round(a["availability"], 3), **_public_leg(a["leg"]), "decimal_price": round(a["by_book"][book], 4)},
                {"fixture_id": b["fixture_id"], "match": b["match"], "kickoff": b["kickoff"], "league": b["league"], "availability_confidence": round(b["availability"], 3), **_public_leg(b["leg"]), "decimal_price": round(b["by_book"][book], 4)}],
                "component_product_decimal_reference": round(d, 4), "component_product_american_reference": _american(d), "conservative_joint_probability": round(p, 6), "probability_edge_vs_component_product_pp": round(edge, 2),
                "exact_parlay_quote": None, "status": "GALAXY MULTI WATCH — FINAL PARLAY QUOTE NEEDED", "bet_eligible": False, "all_leg_models_actionable": action,
                "block_reasons": ["FINAL_PARLAY_QUOTE_NOT_VERIFIED", *([] if action else ["RESEARCH_ONLY_LEG_PRESENT"])], "joint_probability_method": "DISTINCT_FIXTURES_INDEPENDENCE_WITH_2_PERCENT_UNCERTAINTY_HAIRCUT"})
    out.sort(key=lambda r: (r["all_leg_models_actionable"], r["probability_edge_vs_component_product_pp"], r["conservative_joint_probability"]), reverse=True)
    return out[:MAX_MULTI]


def _derivative_inventory(events: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(); total = 0
    for e in events:
        snap = e.get("derivative_research_market_snapshot")
        if not isinstance(snap, dict): continue
        for group in snap.get("groups") or []:
            if not isinstance(group, dict): continue
            values = [v for v in group.get("values") or [] if isinstance(v, dict)]; n = len(values); total += n; counts[str(group.get("family") or group.get("market") or "UNKNOWN")] += n
    return {"observed_quote_count": total, "observed_family_counts": dict(counts), "policy": "CORNERS_CARDS_PERIOD_PLAYER_AND_OTHER_MARKETS_ARE_INVENTORIED_NOW; THEY_ENTER_SCORED_COMBOS ONLY AFTER AN EXPLICIT SPORT_MODEL PROBABILITY EXISTS"}


def build(payload: dict[str, Any]) -> dict[str, Any]:
    base = v1.build(payload); events = [e for e in payload.get("events") or [] if isinstance(e, dict)]; sgps = []
    for e in events:
        if e.get("event_type") == "SOCCER_REFRESH" and e.get("stage") not in {"POSTGAME", "CLOSE"}: sgps.extend(_sgp(e))
    sgps.sort(key=lambda r: (r["all_leg_models_actionable"], r["component_price_reference"] is not None, r["joint_model_probability"]), reverse=True); sgps = sgps[:MAX_SGP]
    multis = _multi(events)
    return {**base, "schema_version": SCHEMA_VERSION, "target": {"minimum_decimal": TARGET_DECIMAL, "minimum_american": 110, "target_probability_edge_pp": TARGET_EDGE_PP},
            "candidate_count": len(sgps) + len(multis), "same_game_candidate_count": len(sgps), "multi_match_candidate_count": len(multis), "same_game_candidates": sgps, "multi_match_candidates": multis,
            "derivative_inventory": _derivative_inventory(events), "family_policy": {"FT_GOALS": "PRIMARY", "BTTS": "RESEARCH_ONLY_SHADOW", "DOUBLE_CHANCE": "1X_X2_12_PROTECTIVE_SHADOW_FOR_WEAK_1X2", "OTHER_MARKETS": "AUTO_ADD_WHEN_EXPLICIT_SPORT_MODEL_EXISTS; NEVER_FABRICATE_PROBABILITY"},
            "actionable_gate": "FINAL_SPORTSBOOK_QUOTE + ALL_LEGS_MODELED + AVAILABILITY>=0.85 + EDGE_GATE", "provider_requests_added": 0, "model_weights_changed": False, "canonical_bet_logic_changed": False,
            "policy": "SPORT_FIRST; MARKET_SECOND; COMBINATION_THIRD; TARGET +110 OR BETTER; CORRELATION_AWARE SGP; CROSS_MATCH PARLAYS ALLOWED; NEVER INVENT PRICES OR PROBABILITIES"}
