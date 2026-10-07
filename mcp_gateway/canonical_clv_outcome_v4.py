from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Iterable

from mcp_gateway import evaluate_postgame as ep

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_CANONICAL_CLV_OUTCOME_V4_1.0.0"
SUPPORTED_FAMILIES = {"1X2", "HOME_TT", "AWAY_TT"}
PROJECT_RECALIBRATION_MIN_GRADED_BETS = 100


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


def _fixture_id(row: dict[str, Any]) -> int | None:
    try:
        return int(row.get("fixture_id"))
    except (TypeError, ValueError):
        return None


def _family(row: dict[str, Any]) -> str:
    return str(row.get("market_family") or "").strip().upper()


def _signal_price(row: dict[str, Any]) -> float | None:
    return _num(row.get("signal_price")) or _num(row.get("entry_price"))


def _signal_probability(row: dict[str, Any]) -> float | None:
    value = _num(row.get("signal_fair_probability"))
    return value if value is not None else _num(row.get("entry_fair_probability"))


def _signal_line(row: dict[str, Any]) -> float | None:
    value = _num(row.get("entry_line"))
    return value if value is not None else _num(row.get("line"))


def _entry_timestamp(row: dict[str, Any]) -> str:
    return str(row.get("entry_timestamp") or row.get("signal_timestamp") or "")


def _exact_key(row: dict[str, Any]) -> tuple[Any, ...] | None:
    fixture_id = _fixture_id(row)
    family = _family(row)
    if fixture_id is None or family not in SUPPORTED_FAMILIES:
        return None
    selection = str(row.get("selection") or "").strip().upper()
    if family == "1X2":
        return fixture_id, family, selection
    line = _signal_line(row)
    if line is None:
        return None
    return fixture_id, family, selection, float(line)


def _dedupe_exact_signals(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    chosen: dict[tuple[Any, ...], dict[str, Any]] = {}
    eligible = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = _exact_key(row)
        if key is None:
            continue
        eligible += 1
        prior = chosen.get(key)
        if prior is None:
            chosen[key] = row
            continue
        current_order = (
            _entry_timestamp(row),
            str(row.get("bookmaker_at_signal") or row.get("bookmaker") or ""),
        )
        prior_order = (
            _entry_timestamp(prior),
            str(prior.get("bookmaker_at_signal") or prior.get("bookmaker") or ""),
        )
        if current_order < prior_order:
            chosen[key] = row
    ordered = sorted(
        chosen.values(),
        key=lambda row: (
            str(row.get("kickoff") or ""),
            _fixture_id(row) or 0,
            _family(row),
            str(row.get("selection") or ""),
            _signal_line(row) if _signal_line(row) is not None else -1.0,
        ),
    )
    return ordered, max(0, eligible - len(ordered))


def _finals(signal_rows: Iterable[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    finals: dict[int, dict[str, Any]] = {}
    latest: dict[int, datetime] = {}
    for row in signal_rows:
        if not isinstance(row, dict):
            continue
        fixture_id = _fixture_id(row)
        result = row.get("result")
        if fixture_id is None or not isinstance(result, dict) or not result:
            continue
        stamp = _parse_dt(row.get("generated_at_utc") or row.get("generated_at_local"))
        prior_stamp = latest.get(fixture_id)
        if fixture_id not in finals or (
            stamp is not None and (prior_stamp is None or stamp >= prior_stamp)
        ):
            finals[fixture_id] = result
            if stamp is not None:
                latest[fixture_id] = stamp
    return finals


def _normalized_market(row: dict[str, Any]) -> dict[str, Any]:
    family = _family(row)
    price = _signal_price(row)
    if family == "1X2":
        return {
            "market": "Match Winner",
            "selection": row.get("selection"),
            "decimal_price": price,
            "bookmaker": row.get("bookmaker_at_signal") or row.get("bookmaker"),
        }

    side = "home" if family == "HOME_TT" else "away"
    team_name = row.get("home_team") if side == "home" else row.get("away_team")
    return {
        "market": "Team Total",
        "selection": row.get("selection"),
        "line": _signal_line(row),
        "decimal_price": price,
        "bookmaker": row.get("bookmaker_at_signal") or row.get("bookmaker"),
        "team": team_name,
    }


def _final_score(result: dict[str, Any] | None) -> dict[str, Any]:
    result = result if isinstance(result, dict) else {}
    goals = result.get("goals")
    goals = goals if isinstance(goals, dict) else {}
    return {"home": goals.get("home"), "away": goals.get("away")}


def _brier(p: float, y: int) -> float:
    return (p - y) ** 2


def _logloss(p: float, y: int) -> float:
    p = max(1e-9, min(1.0 - 1e-9, p))
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def _grade_row(row: dict[str, Any], result: dict[str, Any] | None) -> dict[str, Any]:
    family = _family(row)
    canonical_family = "1X2" if family == "1X2" else "TEAM_TOTALS"
    best = _normalized_market(row)
    fixture_id = _fixture_id(row)
    outcome = ep.grade_market(
        best,
        result,
        str(row.get("home_team") or ""),
        str(row.get("away_team") or ""),
    ) if result else "NO_FINAL"
    price = _signal_price(row)
    roi = ep.roi_units(outcome, price, 1.0)
    probability = _signal_probability(row)
    scored = outcome in {"WIN", "LOSS"} and probability is not None
    y = 1 if outcome == "WIN" else 0 if outcome == "LOSS" else None
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "fixture_id": fixture_id,
        "kickoff": row.get("kickoff"),
        "league": row.get("league"),
        "home_team": row.get("home_team"),
        "away_team": row.get("away_team"),
        "canonical_family": canonical_family,
        "source_market_family": family,
        "team_role": "HOME" if family == "HOME_TT" else "AWAY" if family == "AWAY_TT" else None,
        "market": row.get("market"),
        "selection": row.get("selection"),
        "line": _signal_line(row),
        "signal_price": price,
        "signal_fair_probability": probability,
        "entry_timestamp": row.get("entry_timestamp"),
        "stage": row.get("stage"),
        "classification": row.get("classification"),
        "bookmaker": row.get("bookmaker_at_signal") or row.get("bookmaker"),
        "signal_source": row.get("signal_source"),
        "close_price": _num(row.get("close_price")),
        "closing_line": _num(row.get("closing_line")),
        "closing_timestamp": row.get("closing_timestamp"),
        "clv_probability_pp": _num(row.get("clv_probability_pp")),
        "final_score": _final_score(result),
        "settlement_status": outcome,
        "settled": outcome in {"WIN", "LOSS", "PUSH"},
        "hypothetical_roi_units": roi,
        "brier": _brier(probability, y) if scored and y is not None else None,
        "log_loss": _logloss(probability, y) if scored and y is not None else None,
        "real_wager_assumed": False,
    }


def _fixture_equal_weight_roi(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_fixture: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        fixture_id = _fixture_id(row)
        roi = _num(row.get("hypothetical_roi_units"))
        if fixture_id is not None and row.get("settled") and roi is not None:
            by_fixture[fixture_id].append(roi)
    fixture_values = [sum(values) / len(values) for values in by_fixture.values() if values]
    return {
        "unique_fixtures": len(fixture_values),
        "mean_hypothetical_roi_units_per_fixture": (
            round(sum(fixture_values) / len(fixture_values), 6) if fixture_values else None
        ),
        "sum_fixture_equal_weight_units": (
            round(sum(fixture_values), 6) if fixture_values else 0.0
        ),
        "policy": "MEAN_OF_EXACT_SIGNAL_ROI_WITHIN_FIXTURE_THEN_EQUAL_WEIGHT_ACROSS_FIXTURES",
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = Counter(str(row.get("settlement_status") or "UNKNOWN") for row in rows)
    settled = [row for row in rows if row.get("settled")]
    decided = [row for row in rows if row.get("settlement_status") in {"WIN", "LOSS"}]
    priced = [
        row for row in settled
        if _num(row.get("hypothetical_roi_units")) is not None
    ]
    scored = [
        row for row in decided
        if _num(row.get("signal_fair_probability")) is not None
        and _num(row.get("brier")) is not None
        and _num(row.get("log_loss")) is not None
    ]
    clv = [
        float(row["clv_probability_pp"])
        for row in rows
        if _num(row.get("clv_probability_pp")) is not None
    ]
    roi = [float(row["hypothetical_roi_units"]) for row in priced]
    probabilities = [float(row["signal_fair_probability"]) for row in scored]
    observed = [
        1 if row.get("settlement_status") == "WIN" else 0
        for row in scored
    ]
    briers = [float(row["brier"]) for row in scored]
    losses = [float(row["log_loss"]) for row in scored]
    return {
        "rows": len(rows),
        "unique_fixtures": len({
            _fixture_id(row) for row in rows if _fixture_id(row) is not None
        }),
        "settled": len(settled),
        "settled_unique_fixtures": len({
            _fixture_id(row) for row in settled if _fixture_id(row) is not None
        }),
        "win": statuses["WIN"],
        "loss": statuses["LOSS"],
        "push": statuses["PUSH"],
        "ungraded": len(rows) - len(settled),
        "hit_rate_ex_push": (
            round(statuses["WIN"] / (statuses["WIN"] + statuses["LOSS"]), 6)
            if statuses["WIN"] + statuses["LOSS"] else None
        ),
        "priced_settled_rows": len(priced),
        "hypothetical_roi_units_exact_observations": round(sum(roi), 6) if roi else 0.0,
        "hypothetical_roi_per_priced_settled_observation": (
            round(sum(roi) / len(roi), 6) if roi else None
        ),
        "fixture_equal_weight_roi": _fixture_equal_weight_roi(rows),
        "probability_scored_rows": len(scored),
        "mean_signal_fair_probability": (
            round(sum(probabilities) / len(probabilities), 6) if probabilities else None
        ),
        "observed_selected_outcome_rate": (
            round(sum(observed) / len(observed), 6) if observed else None
        ),
        "brier": round(sum(briers) / len(briers), 6) if briers else None,
        "log_loss": round(sum(losses) / len(losses), 6) if losses else None,
        "calibration_gap_pp": (
            round((sum(probabilities) / len(probabilities) - sum(observed) / len(observed)) * 100.0, 6)
            if probabilities and observed else None
        ),
        "true_clv_rows": len(clv),
        "avg_true_clv_probability_pp": round(sum(clv) / len(clv), 6) if clv else None,
        "positive_true_clv_rate": (
            round(sum(1 for value in clv if value > 0) / len(clv), 6) if clv else None
        ),
    }


def _group_summary(rows: list[dict[str, Any]], key_fn) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(key_fn(row))].append(row)
    return {key: _summary(group) for key, group in sorted(groups.items())}


def build(
    clv_rows: list[dict[str, Any]],
    signal_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_supported = [
        row for row in clv_rows
        if isinstance(row, dict) and _family(row) in SUPPORTED_FAMILIES
    ]
    exact, duplicates_collapsed = _dedupe_exact_signals(raw_supported)
    finals = _finals(signal_rows)
    ledger = [
        _grade_row(row, finals.get(_fixture_id(row) or -1))
        for row in exact
    ]

    one_x_two = [row for row in ledger if row.get("canonical_family") == "1X2"]
    team_totals = [row for row in ledger if row.get("canonical_family") == "TEAM_TOTALS"]

    tt_summary = _summary(team_totals)
    tt_summary["by_team_role"] = _group_summary(
        team_totals, lambda row: row.get("team_role") or "UNKNOWN"
    )
    tt_summary["by_line"] = _group_summary(
        team_totals,
        lambda row: (
            f"{float(row['line']):.1f}" if _num(row.get("line")) is not None else "MISSING"
        ),
    )
    tt_summary["by_selection"] = _group_summary(
        team_totals, lambda row: str(row.get("selection") or "MISSING").upper()
    )

    one_summary = _summary(one_x_two)
    one_summary["by_selection"] = _group_summary(
        one_x_two, lambda row: str(row.get("selection") or "MISSING").upper()
    )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": "RESEARCH_ONLY_CANONICAL_CLV_OUTCOME",
        "policy": (
            "GRADE_ONLY_CAPTURED_EXACT_TRUE_CLV_SIGNALS; NO_SYNTHETIC_SELECTIONS; "
            "TEAM_TOTALS_DEDUPED_BY_FIXTURE_ROLE_SELECTION_LINE; "
            "MULTIPLE_TEAM_TOTAL_LINES_WITHIN_ONE_FIXTURE_ARE_CORRELATED; "
            "SPORT_FIRST_RAW_PROJECTION_UNCHANGED"
        ),
        "production_promotion_allowed": False,
        "manual_review_required": True,
        "real_wagers_assumed": False,
        "provider_requests_added": 0,
        "runtime_logic_changed": False,
        "model_weights_changed": False,
        "canonical_bet_logic_changed": False,
        "raw_supported_clv_rows": len(raw_supported),
        "exact_signal_rows_after_dedup": len(exact),
        "duplicate_exact_rows_collapsed": duplicates_collapsed,
        "fixtures_with_final_result_in_signal_ledger": len(finals),
        "families": {
            "1X2": one_summary,
            "TEAM_TOTALS": tt_summary,
        },
        "recalibration_gate": {
            "project_minimum_graded_bets": PROJECT_RECALIBRATION_MIN_GRADED_BETS,
            "material_recalibration_allowed": False,
            "reason": (
                "These are captured exact-market shadow observations, not a predeclared independent bet-selection sample. "
                "Outcome evidence may diagnose calibration and market behavior but cannot by itself trigger weight changes."
            ),
        },
        "notes": [
            "1X2 true-CLV rows are one exact captured selection per fixture in the current canonical cohort.",
            "Team Totals can contain multiple sides/lines in the same fixture, so exact-row ROI is diagnostic rather than portfolio ROI.",
            "Fixture-equal-weight ROI is reported to reduce domination by fixtures with many Team Total observations.",
            "Brier/log-loss score the captured selected outcome probability only when a final result exists.",
            "No betting price is used to construct or modify the raw sporting projection in this report.",
        ],
    }
    return ledger, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grade canonical 1X2 and Team Totals true-CLV observations against final results."
    )
    parser.add_argument("--clv-tracking", required=True)
    parser.add_argument("--signal-ledger", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    args = parser.parse_args()

    ledger, summary = build(
        _load_jsonl(args.clv_tracking),
        _load_jsonl(args.signal_ledger),
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        for row in ledger:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    os.makedirs(os.path.dirname(args.summary_output) or ".", exist_ok=True)
    with open(args.summary_output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
