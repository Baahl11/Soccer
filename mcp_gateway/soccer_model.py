from __future__ import annotations

import math
import re
from typing import Any

MODEL_VERSION = "SOCCER EDGE ENGINE v1.0"
PROJECTION_MODEL = "GOAL_RATE_POISSON_LIMITED_v0.1"

SIDE_THRESHOLDS = {"B": 3.0, "A": 5.0, "S": 7.5}
GOALS_THRESHOLDS = {"B": 3.5, "A": 5.5, "S": 8.0}
DERIV_THRESHOLDS = {"B": 4.0, "A": 6.0, "S": 9.0}


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _nested(d: Any, *keys: str) -> Any:
    cur = d
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _rate(stats: dict[str, Any], side: str, which: str) -> float | None:
    value = _nested(stats, "goals", which, "average", side)
    if value is None:
        value = _nested(stats, "goals", which, "average", "total")
    return _num(value)


def _played(stats: dict[str, Any], side: str) -> int:
    value = _nested(stats, "fixtures", "played", side)
    if value is None:
        value = _nested(stats, "fixtures", "played", "total")
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _recent_rates(matches: list[dict[str, Any]], team_id: int | None) -> tuple[float | None, float | None, int]:
    gf = ga = 0.0
    n = 0
    if not team_id:
        return None, None, 0
    for fx in matches or []:
        goals = fx.get("goals") or {}
        hg = _num(goals.get("home"))
        ag = _num(goals.get("away"))
        if hg is None or ag is None:
            continue
        if fx.get("home_team_id") == team_id:
            gf += hg
            ga += ag
            n += 1
        elif fx.get("away_team_id") == team_id:
            gf += ag
            ga += hg
            n += 1
    if not n:
        return None, None, 0
    return gf / n, ga / n, n


def _blend(primary: float | None, recent: float | None, recent_n: int) -> float | None:
    if primary is None:
        return recent
    if recent is None or recent_n < 3:
        return primary
    recent_weight = min(0.25, 0.04 * recent_n)
    return (1.0 - recent_weight) * primary + recent_weight * recent


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * (lam**k) / math.factorial(k)


def _score_matrix(home_lam: float, away_lam: float, max_goals: int = 10) -> list[list[float]]:
    hp = [_poisson_pmf(i, home_lam) for i in range(max_goals + 1)]
    ap = [_poisson_pmf(i, away_lam) for i in range(max_goals + 1)]
    return [[h * a for a in ap] for h in hp]


def _probabilities(home_lam: float, away_lam: float) -> dict[str, Any]:
    matrix = _score_matrix(home_lam, away_lam, 10)
    home = draw = away = btts = 0.0
    total_dist: dict[int, float] = {}
    score_probs: list[tuple[float, int, int]] = []
    for h, row in enumerate(matrix):
        for a, p in enumerate(row):
            if h > a:
                home += p
            elif h == a:
                draw += p
            else:
                away += p
            if h > 0 and a > 0:
                btts += p
            total_dist[h + a] = total_dist.get(h + a, 0.0) + p
            score_probs.append((p, h, a))
    mass = home + draw + away
    if mass > 0:
        home, draw, away = home / mass, draw / mass, away / mass
    total_mass = sum(total_dist.values()) or 1.0
    for k in list(total_dist):
        total_dist[k] /= total_mass
    score_probs.sort(reverse=True)
    top_scores = [{"home": h, "away": a, "prob": round(p / total_mass, 6)} for p, h, a in score_probs[:6]]
    return {
        "home_win": home,
        "draw": draw,
        "away_win": away,
        "btts_yes": btts / total_mass,
        "btts_no": 1.0 - (btts / total_mass),
        "over_1_5": sum(p for t, p in total_dist.items() if t >= 2),
        "over_2_5": sum(p for t, p in total_dist.items() if t >= 3),
        "over_3_5": sum(p for t, p in total_dist.items() if t >= 4),
        "total_dist": total_dist,
        "top_scores": top_scores,
    }


def _over_prob(total_dist: dict[int, float], line: float) -> float | None:
    if abs((line * 2) - round(line * 2)) > 1e-8 or int(round(line * 2)) % 2 == 0:
        return None
    threshold = math.floor(line) + 1
    return sum(p for goals, p in total_dist.items() if goals >= threshold)


def _quality_scores(home_lam: float, away_lam: float, probs: dict[str, Any], availability: float | None) -> dict[str, Any]:
    stronger_prob = max(probs["home_win"], probs["away_win"])
    lambda_gap = abs(home_lam - away_lam)
    side_score = _clamp(50 + lambda_gap * 13 + max(0.0, stronger_prob - 0.45) * 70, 0, 100)
    total_lam = home_lam + away_lam
    goal_env = _clamp(50 + (total_lam - 2.35) * 28, 0, 100)
    weaker_lam = min(home_lam, away_lam)
    two_way = _clamp(35 + weaker_lam * 37 + max(0.0, total_lam - 2.4) * 12, 0, 100)
    if availability is not None and availability < 0.85:
        side_score = min(side_score, 79.0)
        goal_env = min(goal_env, 79.0)
        two_way = min(two_way, 79.0)
    return {
        "side_edge_score": round(side_score, 1),
        "goal_environment_score": round(goal_env, 1),
        "two_way_scoring_score": round(two_way, 1),
        "corners_opportunity_score": "NOT MODELED",
        "score_status": "LIMITED_VERIFIED_INPUTS",
    }


def build_raw_projection(fx: dict[str, Any], sporting: dict[str, Any], availability_confidence: float | None = None) -> dict[str, Any]:
    home_stats = sporting.get("home_stats") or {}
    away_stats = sporting.get("away_stats") or {}
    home_recent = sporting.get("home_recent") or []
    away_recent = sporting.get("away_recent") or []

    h_for = _rate(home_stats, "home", "for")
    h_against = _rate(home_stats, "home", "against")
    a_for = _rate(away_stats, "away", "for")
    a_against = _rate(away_stats, "away", "against")

    h_recent_for, h_recent_against, h_recent_n = _recent_rates(home_recent, fx.get("home_team_id"))
    a_recent_for, a_recent_against, a_recent_n = _recent_rates(away_recent, fx.get("away_team_id"))

    h_attack = _blend(h_for, h_recent_for, h_recent_n)
    h_def_concede = _blend(h_against, h_recent_against, h_recent_n)
    a_attack = _blend(a_for, a_recent_for, a_recent_n)
    a_def_concede = _blend(a_against, a_recent_against, a_recent_n)

    missing = []
    for name, value in [("home_scoring_rate", h_attack), ("home_concede_rate", h_def_concede), ("away_scoring_rate", a_attack), ("away_concede_rate", a_def_concede)]:
        if value is None:
            missing.append(name)
    if missing:
        return {
            "status": "NOT MODELED",
            "model_version": MODEL_VERSION,
            "projection_model": PROJECTION_MODEL,
            "reason": "INSUFFICIENT_VERIFIED_GOAL_RATE_INPUTS",
            "missing": missing,
            "advanced_metrics": "NOT VERIFIED",
        }

    home_lam = _clamp((h_attack + a_def_concede) / 2.0, 0.15, 4.5)
    away_lam = _clamp((a_attack + h_def_concede) / 2.0, 0.15, 4.5)
    probs = _probabilities(home_lam, away_lam)
    scores = _quality_scores(home_lam, away_lam, probs, availability_confidence)
    home_played = _played(home_stats, "home")
    away_played = _played(away_stats, "away")
    sample_min = min(home_played, away_played)

    weaker = min(home_lam, away_lam)
    total = home_lam + away_lam
    if total >= 2.85 and weaker >= 1.05:
        scoring_path = "TWO-WAY OPEN GAME"
    elif total >= 2.85 and max(home_lam, away_lam) >= 2.0:
        scoring_path = "FAVORITE-CARRY OVER"
    elif total <= 2.05 and weaker <= 0.75:
        scoring_path = "MUTUAL SUPPRESSION UNDER"
    else:
        scoring_path = "MIXED / NO STRONG SCORING PATH"

    return {
        "status": "MODELED_LIMITED",
        "model_version": MODEL_VERSION,
        "projection_model": PROJECTION_MODEL,
        "model_limitations": [
            "API-Football goal rates are used as a verified baseline; xG/npxG/PPDA/field tilt are NOT VERIFIED in this automated layer.",
            "Goals are not labeled or substituted as xG.",
            "Lineup quality deltas are not quantified without a verified player-impact model.",
        ],
        "advanced_metrics": "NOT VERIFIED",
        "sample": {"home_home_played": home_played, "away_away_played": away_played, "minimum_split_sample": sample_min},
        "raw_home_goal_rate": round(home_lam, 4),
        "raw_away_goal_rate": round(away_lam, 4),
        "raw_total_goals": round(total, 4),
        "raw_home_xg": "NOT VERIFIED",
        "raw_away_xg": "NOT VERIFIED",
        "raw_home_win_prob": round(probs["home_win"], 6),
        "raw_draw_prob": round(probs["draw"], 6),
        "raw_away_win_prob": round(probs["away_win"], 6),
        "raw_btts_yes_prob": round(probs["btts_yes"], 6),
        "raw_over_1_5_prob": round(probs["over_1_5"], 6),
        "raw_over_2_5_prob": round(probs["over_2_5"], 6),
        "raw_over_3_5_prob": round(probs["over_3_5"], 6),
        "top_scorelines": probs["top_scores"],
        "scoring_path": scoring_path,
        "screen_scores": scores,
        "_total_dist": probs["total_dist"],
    }


def _decimal_price(value: Any) -> float | None:
    p = _num(value)
    if p is None or p <= 1.0 or p > 1000:
        return None
    return p


def _market_family(name: str) -> str | None:
    n = (name or "").strip().lower()
    if n in {"match winner", "winner"} or "match winner" in n:
        return "1X2"
    if "both teams" in n and ("score" in n or "to score" in n):
        return "BTTS"
    if "goals over/under" in n or n in {"over/under", "goals over under"}:
        return "TOTAL"
    return None


def _selection_key(selection: str) -> tuple[str | None, float | None]:
    s = (selection or "").strip().lower()
    if s in {"home", "1"}:
        return "home", None
    if s in {"draw", "x"}:
        return "draw", None
    if s in {"away", "2"}:
        return "away", None
    if s in {"yes", "btts yes"}:
        return "yes", None
    if s in {"no", "btts no"}:
        return "no", None
    m = re.match(r"^(over|under)\s*([0-9]+(?:\.[0-9]+)?)$", s)
    if m:
        return m.group(1), float(m.group(2))
    return None, None


def _fair_probs(prices: dict[str, float]) -> dict[str, float] | None:
    if not prices:
        return None
    implied = {k: 1.0 / v for k, v in prices.items() if v > 1.0}
    if len(implied) != len(prices):
        return None
    total = sum(implied.values())
    if total <= 0:
        return None
    return {k: v / total for k, v in implied.items()}


def _tier(edge_pp: float, family: str) -> str | None:
    thresholds = SIDE_THRESHOLDS if family == "SIDE" else GOALS_THRESHOLDS if family == "GOALS" else DERIV_THRESHOLDS
    if edge_pp >= thresholds["S"]:
        return "S"
    if edge_pp >= thresholds["A"]:
        return "A"
    if edge_pp >= thresholds["B"]:
        return "B"
    return None


def _weight(raw: dict[str, Any], data_tier: str, availability: float | None) -> float:
    sample = int(_nested(raw, "sample", "minimum_split_sample") or 0)
    if sample < 5:
        base = 0.18
    elif sample < 10:
        base = 0.22
    else:
        base = 0.25
    if data_tier == "A":
        base += 0.02
    elif data_tier == "C":
        base = min(base, 0.18)
    if availability is not None:
        if availability < 0.75:
            base = min(base, 0.16)
        elif availability < 0.85:
            base = min(base, 0.20)
        elif availability >= 0.90:
            base += 0.01
    return round(_clamp(base, 0.15, 0.28), 3)


def _gate(data_tier: str, availability: float | None, raw: dict[str, Any], stage: str, lineup: Any) -> tuple[bool, list[str]]:
    reasons = []
    if data_tier not in {"A", "B"}:
        reasons.append("Data Tier A/B required for automated BET.")
    if raw.get("status") != "MODELED_LIMITED":
        reasons.append("Raw soccer projection unavailable.")
    sample = int(_nested(raw, "sample", "minimum_split_sample") or 0)
    if sample < 5:
        reasons.append("Minimum home/away split sample <5.")
    if availability is None or availability < 0.85:
        reasons.append("Availability Confidence <0.85.")
    if stage in {"T-40", "T-20", "T-10"}:
        if not isinstance(lineup, dict) or not lineup.get("both_xi_confirmed") or not lineup.get("both_goalkeepers_confirmed"):
            reasons.append("Material XI/goalkeeper verification incomplete.")
    return not reasons, reasons


def evaluate_market(raw: dict[str, Any], market: Any, coverage: dict[str, Any], availability: float | None, stage: str, lineup: Any) -> dict[str, Any]:
    if raw.get("status") != "MODELED_LIMITED":
        return {"status": "WATCH", "reason": "RAW_PROJECTION_NOT_AVAILABLE", "decisions": []}
    if not isinstance(market, dict):
        return {"status": "WATCH", "reason": "NO_VERIFIED_MARKET", "decisions": []}

    data_tier = coverage.get("data_tier") or "D"
    w = _weight(raw, data_tier, availability)
    total_dist = raw.get("_total_dist") or {}
    gate_ok, gate_reasons = _gate(data_tier, availability, raw, stage, lineup)
    decisions: list[dict[str, Any]] = []

    for row in market.get("markets") or []:
        family = _market_family(row.get("market") or "")
        if not family:
            continue
        parsed: dict[str, float] = {}
        lines: dict[str, float | None] = {}
        for v in row.get("values") or []:
            key, line = _selection_key(str(v.get("selection") or ""))
            price = _decimal_price(v.get("price"))
            if key and price:
                if family == "TOTAL" and line is None:
                    continue
                parsed[key if line is None else f"{key}:{line:g}"] = price
                lines[key if line is None else f"{key}:{line:g}"] = line

        candidates: list[tuple[str, str, float | None, float, dict[str, float]]] = []
        if family == "1X2" and all(k in parsed for k in ("home", "draw", "away")):
            fair = _fair_probs({k: parsed[k] for k in ("home", "draw", "away")})
            if fair:
                for sel in ("home", "draw", "away"):
                    candidates.append((sel, "SIDE", None, parsed[sel], fair))
        elif family == "BTTS" and all(k in parsed for k in ("yes", "no")):
            fair = _fair_probs({k: parsed[k] for k in ("yes", "no")})
            if fair:
                for sel in ("yes", "no"):
                    candidates.append((sel, "GOALS", None, parsed[sel], fair))
        elif family == "TOTAL":
            thresholds = sorted({line for line in lines.values() if line is not None})
            for line in thresholds:
                ok = f"over:{line:g}"
                uk = f"under:{line:g}"
                if ok in parsed and uk in parsed:
                    fair = _fair_probs({"over": parsed[ok], "under": parsed[uk]})
                    if fair and _over_prob(total_dist, line) is not None:
                        candidates.append(("over", "GOALS", line, parsed[ok], fair))
                        candidates.append(("under", "GOALS", line, parsed[uk], fair))

        for selection, threshold_family, line, price, fair in candidates:
            if family == "1X2":
                raw_p = raw[f"raw_{'home_win' if selection == 'home' else 'draw' if selection == 'draw' else 'away_win'}_prob"]
                fair_p = fair[selection]
            elif family == "BTTS":
                raw_p = raw["raw_btts_yes_prob"] if selection == "yes" else 1.0 - raw["raw_btts_yes_prob"]
                fair_p = fair[selection]
            else:
                over_p = _over_prob(total_dist, float(line))
                if over_p is None:
                    continue
                raw_p = over_p if selection == "over" else 1.0 - over_p
                fair_p = fair[selection]

            shrunk = w * raw_p + (1.0 - w) * fair_p
            edge_pp = (shrunk - fair_p) * 100.0
            ev = shrunk * (price - 1.0) - (1.0 - shrunk)
            tier = _tier(edge_pp, threshold_family)
            discrepancy = False
            discrepancy_reason = None
            if family == "1X2" and abs(raw_p - fair_p) >= 0.12:
                discrepancy = True
                discrepancy_reason = "1X2 raw-vs-market gap >=12pp requires RECHECK."
            if family == "TOTAL" and line is not None and abs(raw.get("raw_total_goals", 0.0) - line) >= 0.75:
                discrepancy = True
                discrepancy_reason = "Goal-total raw-vs-market center gap >=0.75 requires RECHECK."

            classification = "PASS"
            reasons = list(gate_reasons)
            if tier:
                if discrepancy:
                    classification = "WATCH"
                    reasons.append(discrepancy_reason or "Discrepancy recheck required.")
                elif not gate_ok:
                    classification = "WATCH"
                else:
                    classification = "BET"
            elif edge_pp > 0.0:
                classification = "LEAN" if gate_ok else "WATCH"
                reasons.append("Positive edge below Tier B threshold.")
            else:
                reasons.append("No positive market edge after shrinkage.")

            if classification == "BET" and tier in {"S", "A"}:
                classification = "WATCH"
                reasons.append("Tier A/S blocked until advanced-metric layer is verified; automated model is limited goal-rate baseline.")

            stake = 0.0
            if classification == "BET" and tier == "B":
                stake = round(0.40 * float(availability or 0.0), 2)

            decisions.append({
                "family": family,
                "market": row.get("market"),
                "market_id": row.get("market_id"),
                "bookmaker": row.get("bookmaker"),
                "bookmaker_id": row.get("bookmaker_id"),
                "provider_update": row.get("provider_update"),
                "selection": selection,
                "line": line,
                "decimal_price": price,
                "p_breakeven": round(1.0 / price, 6),
                "p_market_fair": round(fair_p, 6),
                "p_raw": round(raw_p, 6),
                "shrink_weight": w,
                "p_shrunk": round(shrunk, 6),
                "prob_edge_pp": round(edge_pp, 3),
                "estimated_ev": round(ev, 6),
                "tier": tier,
                "classification": classification,
                "stake_units": stake,
                "discrepancy_recheck": discrepancy,
                "reasons": reasons,
            })

    rank = {"BET": 4, "WATCH": 3, "LEAN": 2, "PASS": 1}
    decisions.sort(key=lambda d: (rank.get(d["classification"], 0), d.get("prob_edge_pp") or -999, d.get("estimated_ev") or -999), reverse=True)
    best = decisions[0] if decisions else None
    return {
        "status": best["classification"] if best else "WATCH",
        "model_version": MODEL_VERSION,
        "projection_model": PROJECTION_MODEL,
        "advanced_metrics": "NOT VERIFIED",
        "automated_tier_cap": "B until advanced-metric layer is verified",
        "best_decision": best,
        "decisions": decisions[:20],
        "decision_count": len(decisions),
    }


def public_raw_projection(raw: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in raw.items() if not k.startswith("_")}
