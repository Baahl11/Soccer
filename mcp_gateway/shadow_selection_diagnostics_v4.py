from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from typing import Any

from mcp_gateway import evaluate_postgame as ep
from mcp_gateway import shadow_settlement_v4 as shadow

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_1X2_SHADOW_SELECTION_DIAGNOSTICS_V4_1.1.0"
DIRECTIONAL_MIN = 20
REVIEW_MIN = 50


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _pick_side(row: dict[str, Any], best: dict[str, Any]) -> str:
    selection = _norm(best.get("selection"))
    if selection in {"home", "1"} or selection == _norm(row.get("home_team")):
        return "HOME"
    if selection in {"away", "2"} or selection == _norm(row.get("away_team")):
        return "AWAY"
    if selection in {"draw", "x"}:
        return "DRAW"
    return "OTHER"


def _price_band(value: Any) -> str:
    price = _num(value)
    if price is None:
        return "MISSING"
    if price < 1.80:
        return "LT_1_80"
    if price < 2.50:
        return "1_80_TO_2_49"
    if price < 3.50:
        return "2_50_TO_3_49"
    return "GE_3_50"


def _edge_band(value: Any) -> str:
    edge = _num(value)
    if edge is None:
        return "MISSING"
    if edge < 0:
        return "NEGATIVE"
    if edge < 2:
        return "0_TO_1_99PP"
    if edge < 5:
        return "2_TO_4_99PP"
    if edge < 10:
        return "5_TO_9_99PP"
    return "GE_10PP"


def _ev_band(value: Any) -> str:
    ev = _num(value)
    if ev is None:
        return "MISSING"
    if ev < 0:
        return "NEGATIVE"
    if ev < 5:
        return "0_TO_4_99PCT"
    if ev < 10:
        return "5_TO_9_99PCT"
    return "GE_10PCT"


def _gap_band(value: Any) -> str:
    gap = _num(value)
    if gap is None:
        return "MISSING"
    if gap < 5:
        return "LT_5PP"
    if gap < 12:
        return "5_TO_11_99PP"
    if gap < 20:
        return "12_TO_19_99PP_RECHECK"
    return "GE_20PP_RECHECK"


def _watch_quality_bucket(row: dict[str, Any], best: dict[str, Any]) -> str:
    raw = _num(best.get("p_raw"))
    fair = _num(best.get("p_market_fair"))
    gap_pp = abs(raw - fair) * 100.0 if raw is not None and fair is not None else None
    if gap_pp is not None and gap_pp >= 12.0:
        return "DISCREPANCY_RECHECK"

    tier = str(best.get("tier") or "").upper()
    if tier in {"S", "A"}:
        return "ADVANCED_TIER_CAP"

    availability = _num(row.get("availability_confidence"))
    lineup = row.get("lineups") if isinstance(row.get("lineups"), dict) else {}
    xi_ok = lineup.get("both_xi_confirmed") is True and lineup.get("both_goalkeepers_confirmed") is True
    if availability is None or availability < 0.85 or not xi_ok:
        return "AVAILABILITY_OR_XI_GATE"

    if tier == "B":
        return "TIER_B_SHADOW"
    return "OTHER_WATCH"


def _summary(group: list[dict[str, Any]]) -> dict[str, Any]:
    settled = [row for row in group if row.get("outcome") in {"WIN", "LOSS", "PUSH"}]
    wins = sum(1 for row in settled if row.get("outcome") == "WIN")
    losses = sum(1 for row in settled if row.get("outcome") == "LOSS")
    pushes = sum(1 for row in settled if row.get("outcome") == "PUSH")
    roi_values = [float(row["roi"]) for row in settled if row.get("roi") is not None]
    n = len(settled)
    if n >= REVIEW_MIN:
        sample_status = "SHADOW_REVIEW_READY"
    elif n >= DIRECTIONAL_MIN:
        sample_status = "DIRECTIONAL_SHADOW"
    else:
        sample_status = "DATA_BLOCKED"
    return {
        "rows": len(group),
        "settled": n,
        "win": wins,
        "loss": losses,
        "push": pushes,
        "hit_rate_ex_push": round(wins / (wins + losses), 6) if wins + losses else None,
        "roi_units": round(sum(roi_values), 6) if roi_values else 0.0,
        "roi_per_settled_unit": round(sum(roi_values) / n, 6) if roi_values and n else None,
        "sample_status": sample_status,
    }


def _segments(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key) or "MISSING")].append(row)
    return {name: _summary(group) for name, group in sorted(groups.items())}


def build(signal_rows: list[dict[str, Any]]) -> dict[str, Any]:
    finals: dict[int, dict[str, Any]] = {}
    for row in signal_rows:
        if not isinstance(row, dict):
            continue
        try:
            fid = int(row.get("fixture_id"))
        except (TypeError, ValueError):
            continue
        if isinstance(row.get("result"), dict) and row.get("result"):
            finals[fid] = row["result"]

    selected = shadow._latest_watch_by_fixture_family(signal_rows)
    rows: list[dict[str, Any]] = []
    for row in selected:
        best = row.get("best_market")
        if not isinstance(best, dict) or ep.market_family(best) != "FT_1X2":
            continue
        fid = int(row["fixture_id"])
        result = finals.get(fid)
        outcome = ep.grade_market(
            best,
            result,
            row.get("home_team") or "",
            row.get("away_team") or "",
            row.get("home_team_id"),
            row.get("away_team_id"),
        )
        price = _num(best.get("decimal_price"))
        p_raw = _num(best.get("p_raw"))
        p_market_fair = _num(best.get("p_market_fair"))
        raw_market_gap_pp = (
            abs(p_raw - p_market_fair) * 100.0
            if p_raw is not None and p_market_fair is not None
            else None
        )
        rows.append({
            "fixture_id": fid,
            "stage": str(row.get("stage") or "UNKNOWN").upper(),
            "league": row.get("league"),
            "selection_side": _pick_side(row, best),
            "decimal_price": price,
            "price_band": _price_band(price),
            "prob_edge_pp": _num(best.get("prob_edge_pp")),
            "edge_band": _edge_band(best.get("prob_edge_pp")),
            "ev_pct": _num(best.get("ev_pct")),
            "ev_band": _ev_band(best.get("ev_pct")),
            "p_market_fair": p_market_fair,
            "p_shrunk": _num(best.get("p_shrunk")),
            "p_raw": p_raw,
            "raw_market_gap_pp": raw_market_gap_pp,
            "raw_market_gap_band": _gap_band(raw_market_gap_pp),
            "discrepancy_recheck": raw_market_gap_pp is not None and raw_market_gap_pp >= 12.0,
            "tier": str(best.get("tier") or "NONE").upper(),
            "watch_quality_bucket": _watch_quality_bucket(row, best),
            "availability_confidence": _num(row.get("availability_confidence")),
            "outcome": outcome,
            "roi": ep.roi_units(outcome, price, 1.0),
        })

    by_stage_raw: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stage_raw[row["stage"]].append(row)

    by_stage: dict[str, Any] = {}
    for stage, group in sorted(by_stage_raw.items()):
        by_stage[stage] = {
            "overall": _summary(group),
            "by_selection_side": _segments(group, "selection_side"),
            "by_price_band": _segments(group, "price_band"),
            "by_edge_band": _segments(group, "edge_band"),
            "by_ev_band": _segments(group, "ev_band"),
            "by_raw_market_gap_band": _segments(group, "raw_market_gap_band"),
            "by_tier": _segments(group, "tier"),
            "by_watch_quality_bucket": _segments(group, "watch_quality_bucket"),
            "discrepancy_recheck_rows": sum(1 for row in group if row.get("discrepancy_recheck")),
            "edge_available_rows": sum(1 for row in group if row.get("prob_edge_pp") is not None),
            "ev_available_rows": sum(1 for row in group if row.get("ev_pct") is not None),
            "avg_prob_edge_pp": round(
                sum(row["prob_edge_pp"] for row in group if row.get("prob_edge_pp") is not None)
                / sum(1 for row in group if row.get("prob_edge_pp") is not None),
                6,
            ) if any(row.get("prob_edge_pp") is not None for row in group) else None,
            "avg_ev_pct": round(
                sum(row["ev_pct"] for row in group if row.get("ev_pct") is not None)
                / sum(1 for row in group if row.get("ev_pct") is not None),
                6,
            ) if any(row.get("ev_pct") is not None for row in group) else None,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": "SHADOW_SELECTION_DIAGNOSTICS_ACTIVE",
        "market_family": "FT_1X2",
        "rows": len(rows),
        "unique_fixtures": len({row["fixture_id"] for row in rows}),
        "overall": _summary(rows),
        "by_stage": by_stage,
        "sample_policy": {
            "directional_minimum": DIRECTIONAL_MIN,
            "review_minimum": REVIEW_MIN,
        },
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
        "runtime_logic_changed": False,
        "notes": [
            "Diagnostics use the same latest pre-kickoff WATCH per fixture/market-family selector as the shadow settlement ledger.",
            "Price, edge, and EV bands are descriptive only and do not create runtime filters.",
            "Segments below 20 settled rows are not treated as directional evidence.",
            "WATCH quality buckets separate raw-vs-market discrepancy rechecks, tier caps, and availability/XI gates so intentionally blocked candidates are not mistaken for promotion-quality shadow evidence.",
        ],
    }


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path or not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose 1X2 WATCH selection and price quality by stage.")
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = build(_load_jsonl(args.ledger))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({
        "model_version": report["model_version"],
        "status": report["status"],
        "rows": report["rows"],
        "unique_fixtures": report["unique_fixtures"],
        "stage_overall": {
            stage: value["overall"] for stage, value in report["by_stage"].items()
        },
        "provider_requests_added": report["provider_requests_added"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
