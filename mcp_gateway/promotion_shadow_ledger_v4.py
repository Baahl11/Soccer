from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Iterable

from mcp_gateway import evaluate_postgame as ep

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_PROMOTION_SHADOW_LEDGER_V4_1.0.0"
PREGAME_STAGES = {"T-40", "T-20", "T-10"}
DIRECTIONAL_MIN = 20
REVIEW_MIN = 50
FINAL_STATUSES = {"FT", "AET", "PEN"}


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        out = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return out if out.tzinfo is not None else None


def _result(event: dict[str, Any], fixture: dict[str, Any]) -> dict[str, Any] | None:
    result = event.get("result")
    if isinstance(result, dict) and result:
        return result
    status = str(fixture.get("status") or "").upper()
    goals = fixture.get("goals") if isinstance(fixture.get("goals"), dict) else {}
    if status in FINAL_STATUSES and goals.get("home") is not None and goals.get("away") is not None:
        return {"goals": goals, "score": fixture.get("score"), "status": status}
    return None


def _clean_tier_cap_watch(decision: dict[str, Any]) -> bool:
    if str(decision.get("classification") or "").upper() != "WATCH":
        return False
    if str(decision.get("tier") or "").upper() not in {"A", "S"}:
        return False
    reasons = [str(value) for value in (decision.get("reasons") or []) if value]
    if len(reasons) != 1:
        return False
    reason = reasons[0].lower()
    return "tier a/s blocked" in reason and "advanced-metric" in reason


def _event_candidates(event: dict[str, Any]) -> list[dict[str, Any]]:
    md = event.get("market_decision")
    if not isinstance(md, dict):
        return []
    decisions = md.get("decisions")
    if not isinstance(decisions, list):
        return []
    return [
        decision for decision in decisions
        if isinstance(decision, dict) and _clean_tier_cap_watch(decision)
    ]


def _candidate_rank(decision: dict[str, Any]) -> tuple[float, float]:
    edge = _num(decision.get("prob_edge_pp"))
    ev = _num(decision.get("estimated_ev"))
    return (edge if edge is not None else -999.0, ev if ev is not None else -999.0)


def _sample_status(n: int) -> str:
    if n >= REVIEW_MIN:
        return "PROMOTION_SHADOW_REVIEW_READY"
    if n >= DIRECTIONAL_MIN:
        return "DIRECTIONAL_PROMOTION_SHADOW"
    return "DATA_BLOCKED"


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    settled_rows = [row for row in rows if row.get("settled")]
    statuses = Counter(str(row.get("settlement_status") or "UNKNOWN") for row in rows)
    wins = statuses["WIN"]
    losses = statuses["LOSS"]
    roi_values = [float(row["roi_units"]) for row in settled_rows if row.get("roi_units") is not None]
    settled = len(settled_rows)
    return {
        "rows": len(rows),
        "unique_fixtures": len({row["fixture_id"] for row in rows}),
        "settled": settled,
        "win": wins,
        "loss": losses,
        "push": statuses["PUSH"],
        "ungraded": len(rows) - settled,
        "hit_rate_ex_push": round(wins / (wins + losses), 6) if wins + losses else None,
        "roi_units": round(sum(roi_values), 6) if roi_values else 0.0,
        "roi_per_settled_unit": round(sum(roi_values) / settled, 6) if roi_values and settled else None,
        "sample_status": _sample_status(settled),
    }


def build_from_ticks(ticks: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    finals: dict[int, dict[str, Any]] = {}
    chosen: dict[tuple[int, str], tuple[datetime, tuple[float, float], dict[str, Any], dict[str, Any], dict[str, Any]]] = {}
    raw_eligible_decisions = 0

    for tick in ticks:
        if not isinstance(tick, dict):
            continue
        generated = _parse_dt(tick.get("generated_at_utc") or tick.get("generated_at_local"))
        for event in tick.get("events") or []:
            if not isinstance(event, dict):
                continue
            fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
            try:
                fixture_id = int(fixture.get("fixture_id"))
            except (TypeError, ValueError):
                continue

            final = _result(event, fixture)
            if final is not None:
                finals[fixture_id] = final

            stage = str(event.get("stage") or "").upper()
            if stage not in PREGAME_STAGES or generated is None:
                continue
            kickoff = _parse_dt(fixture.get("kickoff"))
            if kickoff is None or generated >= kickoff:
                continue

            eligible = _event_candidates(event)
            raw_eligible_decisions += len(eligible)
            by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for decision in eligible:
                by_family[ep.market_family(decision)].append(decision)

            for family, decisions in by_family.items():
                best = max(decisions, key=_candidate_rank)
                key = (fixture_id, family)
                rank = _candidate_rank(best)
                prior = chosen.get(key)
                if prior is None or generated > prior[0] or (generated == prior[0] and rank > prior[1]):
                    chosen[key] = (generated, rank, event, fixture, best)

    ledger: list[dict[str, Any]] = []
    for (fixture_id, family), (generated, _, event, fixture, decision) in sorted(
        chosen.items(), key=lambda item: (item[1][0], item[0][0], item[0][1])
    ):
        result = finals.get(fixture_id)
        if result is None:
            continue
        outcome = ep.grade_market(
            decision,
            result,
            fixture.get("home_team") or "",
            fixture.get("away_team") or "",
            fixture.get("home_team_id"),
            fixture.get("away_team_id"),
        )
        price = _num(decision.get("decimal_price"))
        roi = ep.roi_units(outcome, price, 1.0)
        ledger.append({
            "schema_version": SCHEMA_VERSION,
            "fixture_id": fixture_id,
            "generated_at": generated.isoformat(),
            "kickoff": fixture.get("kickoff"),
            "stage": event.get("stage"),
            "league": fixture.get("league"),
            "home_team": fixture.get("home_team"),
            "away_team": fixture.get("away_team"),
            "market_family": family,
            "market": decision.get("market"),
            "selection": decision.get("selection"),
            "line": _num(decision.get("line")),
            "decimal_price": price,
            "bookmaker": decision.get("bookmaker"),
            "tier": decision.get("tier"),
            "prob_edge_pp": _num(decision.get("prob_edge_pp")),
            "estimated_ev": _num(decision.get("estimated_ev")),
            "p_raw": _num(decision.get("p_raw")),
            "p_shrunk": _num(decision.get("p_shrunk")),
            "p_market_fair": _num(decision.get("p_market_fair")),
            "reasons": list(decision.get("reasons") or []),
            "promotion_shadow_policy": "CLEAN_TIER_A_S_CAP_ONLY",
            "settlement_status": outcome,
            "settled": outcome in {"WIN", "LOSS", "PUSH"},
            "roi_units": roi,
            "real_wager_assumed": False,
            "bankroll_impact": 0.0,
            "result": result,
        })

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ledger:
        by_family[row["market_family"]].append(row)

    family_summary: dict[str, Any] = {}
    for family, group in sorted(by_family.items()):
        by_stage_raw: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in group:
            by_stage_raw[str(row.get("stage") or "UNKNOWN")].append(row)
        family_summary[family] = {
            **_summary(group),
            "by_stage": {
                stage: _summary(stage_rows)
                for stage, stage_rows in sorted(by_stage_raw.items())
            },
        }

    summary = {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": "PROMOTION_SHADOW_LEDGER_ACTIVE",
        "source": "soccer_edge_state/history/*.jsonl",
        "eligibility_policy": {
            "classification": "WATCH",
            "tiers": ["A", "S"],
            "required_reason": "Tier A/S blocked until advanced-metric layer is verified",
            "exactly_one_reason_required": True,
            "discrepancy_rechecks_excluded": True,
            "availability_or_lineup_gate_failures_excluded": True,
            "one_latest_candidate_per_fixture_family": True,
        },
        "raw_eligible_decisions_seen": raw_eligible_decisions,
        "rows": len(ledger),
        "unique_fixtures": len({row["fixture_id"] for row in ledger}),
        "family_count": len(family_summary),
        "by_market_family": family_summary,
        "provider_requests_added": 0,
        "real_wagers_assumed": False,
        "bankroll_impact": 0.0,
        "production_promotion_allowed": False,
        "notes": [
            "This ledger is separate from WATCH alerts/rechecks and from real bet settlements.",
            "Only decisions whose sole WATCH reason was the explicit Tier A/S safety cap are promotion-evaluable.",
            "The latest eligible pre-kickoff decision per fixture and market family is retained to avoid snapshot duplication.",
        ],
    }
    return ledger, summary


def _load_ticks(history_dir: str) -> list[dict[str, Any]]:
    ticks: list[dict[str, Any]] = []
    for path in sorted(glob.glob(os.path.join(history_dir, "*.jsonl"))):
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
                    ticks.append(value)
    return ticks


def main() -> None:
    parser = argparse.ArgumentParser(description="Build clean promotion-shadow ledger from raw historical market decisions.")
    parser.add_argument("--history-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    args = parser.parse_args()

    ledger, summary = build_from_ticks(_load_ticks(args.history_dir))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        for row in ledger:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    with open(args.summary_output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
