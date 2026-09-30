from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from statistics import mean, median
from typing import Any, Iterable

MODEL_VERSION = "SOCCER_MARKET_RESIDUAL_CHALLENGER_V4_1.1.0"
SCHEMA_VERSION = "1.1.0"


FAMILY_ALIASES = {
    "FT_1X2": "1X2",
    "MATCH_WINNER": "1X2",
    "FT_BTTS": "BTTS",
    "FT_BTTS_RESEARCH": "BTTS",
    "FT_TOTALS_RESEARCH": "FT_TOTALS",
    "TOTAL": "FT_TOTALS",
}


def _prob(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        return None
    return number


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _family(row: dict[str, Any]) -> str:
    value = row.get("market_family") or row.get("family") or row.get("market") or "UNKNOWN"
    family = str(value).strip().upper() or "UNKNOWN"
    return FAMILY_ALIASES.get(family, family)


def _selection(value: Any) -> str:
    text = " ".join(str(value or "").strip().upper().split())
    aliases = {
        "1": "HOME",
        "HOME WIN": "HOME",
        "HOME TEAM": "HOME",
        "X": "DRAW",
        "TIE": "DRAW",
        "2": "AWAY",
        "AWAY WIN": "AWAY",
        "AWAY TEAM": "AWAY",
    }
    for suffix in (" RESEARCH",):
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()
    return aliases.get(text, text)


def _line_key(value: Any) -> float | None:
    number = _number(value)
    return round(number, 6) if number is not None else None


def _stage(value: Any) -> str:
    return " ".join(str(value or "").strip().upper().split())


def _binary_outcome(row: dict[str, Any]) -> float | None:
    """Read only explicit outcome fields already attached to the source row.

    Generic score/result payloads are deliberately ignored here. Historical
    settlement enrichment, when requested, is handled by a separate market-aware
    adapter below so runtime rows never silently infer their own outcomes.
    """
    for key in ("binary_outcome", "settled_binary_outcome", "outcome_binary"):
        value = row.get(key)
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric in (0.0, 1.0):
            return numeric
    return None


def _explicit_outcome_class(row: dict[str, Any]) -> str | None:
    for key in (
        "multiclass_outcome",
        "settled_outcome_class",
        "outcome_class",
        "winner_selection",
    ):
        value = _selection(row.get(key))
        if value in {"HOME", "DRAW", "AWAY"}:
            return value
    return None


def _log_loss(p: float, y: float) -> float:
    clipped = min(max(p, 1e-12), 1.0 - 1e-12)
    return -(y * math.log(clipped) + (1.0 - y) * math.log(1.0 - clipped))


def _bucket(p: float) -> str:
    low = min(int(p * 10.0) * 10, 90)
    high = low + 10
    return f"{low:02d}-{high:02d}%"


def _round(value: float | None, digits: int = 8) -> float | None:
    return round(value, digits) if value is not None and math.isfinite(value) else None


def _signal_key(row: dict[str, Any], *, include_stage: bool) -> tuple[Any, ...]:
    base: tuple[Any, ...] = (
        row.get("fixture_id"),
        _family(row),
        _selection(row.get("selection")),
        _line_key(row.get("line")),
    )
    return (*base, _stage(row.get("stage"))) if include_stage else base


def _settlement_binary_index(rows: Iterable[dict[str, Any]]) -> dict[tuple[Any, ...], float]:
    index: dict[tuple[Any, ...], float] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("settled") is not True:
            continue
        status = str(row.get("settlement_status") or "").strip().upper()
        if status not in {"WIN", "LOSS"}:
            continue
        selection = _selection(row.get("selection"))
        if not selection:
            continue
        index[_signal_key(row, include_stage=False)] = 1.0 if status == "WIN" else 0.0
    return index


def _settlement_1x2_outcome_index(rows: Iterable[dict[str, Any]]) -> dict[Any, str]:
    """Derive canonical FT 1X2 class only from the dedicated settlement ledger.

    We require a settled canonical FT Match Winner row with explicit full-time
    score. Period/AET/PEN/non-canonical families are not used for RPS labels.
    """
    index: dict[Any, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("settled") is not True or row.get("canonical_ft_market") is not True:
            continue
        if _family(row) != "1X2":
            continue
        result = row.get("result")
        if not isinstance(result, dict) or str(result.get("status") or "").upper() != "FT":
            continue
        score = result.get("score")
        fulltime = score.get("fulltime") if isinstance(score, dict) else None
        if not isinstance(fulltime, dict):
            continue
        home = _number(fulltime.get("home"))
        away = _number(fulltime.get("away"))
        if home is None or away is None:
            continue
        outcome = "HOME" if home > away else "AWAY" if away > home else "DRAW"
        fixture_id = row.get("fixture_id")
        if fixture_id is not None:
            index[fixture_id] = outcome
    return index


def _strict_close_index(rows: Iterable[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    """Index only canonical true-close rows without changing strict-close semantics."""
    index: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("is_true_closing_line") is not True:
            continue
        if row.get("probability_comparable_same_line") is False:
            continue
        key = _signal_key(row, include_stage=True)
        previous = index.get(key)
        if previous is None or str(row.get("closing_timestamp") or "") > str(previous.get("closing_timestamp") or ""):
            index[key] = row
    return index


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "comparable_rows": 0,
            "settled_binary_rows": 0,
            "mean_market_residual": None,
            "median_market_residual": None,
            "mean_absolute_market_residual": None,
            "model_brier": None,
            "market_brier": None,
            "brier_delta_model_minus_market": None,
            "model_log_loss": None,
            "market_log_loss": None,
            "log_loss_delta_model_minus_market": None,
        }

    residuals = [float(row["market_residual"]) for row in rows]
    settled = [row for row in rows if row.get("binary_outcome") is not None]
    output = {
        "comparable_rows": len(rows),
        "settled_binary_rows": len(settled),
        "mean_market_residual": _round(mean(residuals)),
        "median_market_residual": _round(median(residuals)),
        "mean_absolute_market_residual": _round(mean(abs(value) for value in residuals)),
        "model_brier": None,
        "market_brier": None,
        "brier_delta_model_minus_market": None,
        "model_log_loss": None,
        "market_log_loss": None,
        "log_loss_delta_model_minus_market": None,
    }
    if settled:
        model_brier = mean((row["p_model_calibrated"] - row["binary_outcome"]) ** 2 for row in settled)
        market_brier = mean((row["p_market_devig"] - row["binary_outcome"]) ** 2 for row in settled)
        model_log = mean(_log_loss(row["p_model_calibrated"], row["binary_outcome"]) for row in settled)
        market_log = mean(_log_loss(row["p_market_devig"], row["binary_outcome"]) for row in settled)
        output.update(
            {
                "model_brier": _round(model_brier),
                "market_brier": _round(market_brier),
                "brier_delta_model_minus_market": _round(model_brier - market_brier),
                "model_log_loss": _round(model_log),
                "market_log_loss": _round(market_log),
                "log_loss_delta_model_minus_market": _round(model_log - market_log),
            }
        )
    return output


def _rps(probabilities: dict[str, float], outcome: str) -> float:
    ordered = ("HOME", "DRAW", "AWAY")
    observed = {key: 1.0 if key == outcome else 0.0 for key in ordered}
    cumulative_p = 0.0
    cumulative_o = 0.0
    squared = 0.0
    for key in ordered[:-1]:
        cumulative_p += probabilities[key]
        cumulative_o += observed[key]
        squared += (cumulative_p - cumulative_o) ** 2
    return squared / (len(ordered) - 1)


def _rps_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["family"] != "1X2":
            continue
        key = (
            row.get("fixture_id"),
            row.get("stage"),
            row.get("market"),
            _line_key(row.get("line")),
        )
        groups[key].append(row)

    scored: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    for key, items in groups.items():
        by_selection: dict[str, dict[str, Any]] = {}
        for item in items:
            selection = _selection(item.get("selection"))
            if selection in {"HOME", "DRAW", "AWAY"} and selection not in by_selection:
                by_selection[selection] = item
        if set(by_selection) != {"HOME", "DRAW", "AWAY"}:
            excluded["INCOMPLETE_1X2_VECTOR"] += 1
            continue

        outcomes = {item.get("outcome_class") for item in by_selection.values() if item.get("outcome_class")}
        if len(outcomes) != 1:
            excluded["MISSING_OR_AMBIGUOUS_1X2_OUTCOME"] += 1
            continue
        outcome = next(iter(outcomes))
        if outcome not in {"HOME", "DRAW", "AWAY"}:
            excluded["INVALID_1X2_OUTCOME"] += 1
            continue

        model = {selection: by_selection[selection]["p_model_calibrated"] for selection in by_selection}
        market = {selection: by_selection[selection]["p_market_devig"] for selection in by_selection}
        model_sum = sum(model.values())
        market_sum = sum(market.values())
        if abs(model_sum - 1.0) > 0.02:
            excluded["MODEL_1X2_PROBABILITY_SUM_OUTSIDE_TOLERANCE"] += 1
            continue
        if abs(market_sum - 1.0) > 0.02:
            excluded["MARKET_1X2_PROBABILITY_SUM_OUTSIDE_TOLERANCE"] += 1
            continue

        model_rps = _rps(model, outcome)
        market_rps = _rps(market, outcome)
        scored.append(
            {
                "fixture_id": key[0],
                "stage": key[1],
                "outcome_class": outcome,
                "model_rps": model_rps,
                "market_rps": market_rps,
                "rps_delta_model_minus_market": model_rps - market_rps,
                "model_probability_sum": model_sum,
                "market_probability_sum": market_sum,
            }
        )

    if not scored:
        return {
            "eligible_1x2_groups": len(groups),
            "scored_1x2_groups": 0,
            "model_rps": None,
            "market_rps": None,
            "rps_delta_model_minus_market": None,
            "excluded_groups": dict(sorted(excluded.items())),
            "formula": "normalized RPS=(sum cumulative squared error across HOME,DRAW boundaries)/(3-1)",
        }

    return {
        "eligible_1x2_groups": len(groups),
        "scored_1x2_groups": len(scored),
        "model_rps": _round(mean(item["model_rps"] for item in scored)),
        "market_rps": _round(mean(item["market_rps"] for item in scored)),
        "rps_delta_model_minus_market": _round(mean(item["rps_delta_model_minus_market"] for item in scored)),
        "excluded_groups": dict(sorted(excluded.items())),
        "formula": "normalized RPS=(sum cumulative squared error across HOME,DRAW boundaries)/(3-1)",
    }


def _clv_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matched = [row for row in rows if isinstance(row.get("strict_close"), dict)]
    if not matched:
        return {
            "matched_true_close_rows": 0,
            "mean_probability_clv_pp": None,
            "positive_probability_clv_rate": None,
            "mean_price_clv_pct": None,
            "mean_model_residual_to_true_close": None,
            "mean_absolute_model_residual_to_true_close": None,
            "by_family": {},
            "policy": "EXACT_SIGNAL_IDENTITY_AND_STAGE; CANONICAL_TRUE_CLOSE_ROWS_ONLY",
        }

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        probability_clv = [item.get("probability_clv_pp") for item in items]
        probability_clv = [value for value in probability_clv if value is not None]
        price_clv = [item.get("price_clv_pct") for item in items]
        price_clv = [value for value in price_clv if value is not None]
        model_close = [item.get("model_residual_to_true_close") for item in items]
        model_close = [value for value in model_close if value is not None]
        return {
            "rows": len(items),
            "mean_probability_clv_pp": _round(mean(probability_clv), 6) if probability_clv else None,
            "positive_probability_clv_rate": _round(sum(value > 0 for value in probability_clv) / len(probability_clv), 6) if probability_clv else None,
            "mean_price_clv_pct": _round(mean(price_clv), 6) if price_clv else None,
            "mean_model_residual_to_true_close": _round(mean(model_close)) if model_close else None,
            "mean_absolute_model_residual_to_true_close": _round(mean(abs(value) for value in model_close)) if model_close else None,
        }

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in matched:
        by_family[row["family"]].append(row)
    overall = summarize(matched)
    return {
        "matched_true_close_rows": len(matched),
        **{key: value for key, value in overall.items() if key != "rows"},
        "by_family": {family: summarize(items) for family, items in sorted(by_family.items())},
        "policy": "EXACT_SIGNAL_IDENTITY_AND_STAGE; CANONICAL_TRUE_CLOSE_ROWS_ONLY",
    }


def build_report(
    source_rows: Iterable[dict[str, Any]],
    *,
    settlement_rows: Iterable[dict[str, Any]] | None = None,
    clv_rows: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    source = [row for row in source_rows if isinstance(row, dict)]
    settlements = [row for row in (settlement_rows or []) if isinstance(row, dict)]
    closes = [row for row in (clv_rows or []) if isinstance(row, dict)]

    settlement_binary = _settlement_binary_index(settlements)
    settlement_1x2 = _settlement_1x2_outcome_index(settlements)
    close_index = _strict_close_index(closes)

    comparable: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    family_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in source:
        p_model = _prob(row.get("p_model_calibrated"))
        if p_model is None:
            excluded["MISSING_CALIBRATED_MODEL_PROBABILITY"] += 1
            continue
        p_market = _prob(
            row.get("p_market_fair")
            if row.get("p_market_fair") is not None
            else row.get("p_market_devig")
        )
        if p_market is None:
            excluded["MISSING_DEVIG_MARKET_PROBABILITY"] += 1
            continue

        family = _family(row)
        residual = p_model - p_market
        outcome = _binary_outcome(row)
        if outcome is None and settlement_binary:
            outcome = settlement_binary.get(_signal_key(row, include_stage=False))
        outcome_class = _explicit_outcome_class(row)
        if outcome_class is None and family == "1X2":
            outcome_class = settlement_1x2.get(row.get("fixture_id"))

        item = {
            "fixture_id": row.get("fixture_id"),
            "family": family,
            "market": row.get("market"),
            "selection": row.get("selection"),
            "line": row.get("line"),
            "stage": row.get("stage"),
            "p_model_calibrated": p_model,
            "p_market_devig": p_market,
            "market_residual": residual,
            "binary_outcome": outcome,
            "outcome_class": outcome_class,
        }

        close = close_index.get(_signal_key(row, include_stage=True))
        if close is not None:
            probability_clv = _number(
                close.get("probability_clv")
                if close.get("probability_clv") is not None
                else close.get("clv_probability_pp")
            )
            price_clv = _number(
                close.get("price_clv")
                if close.get("price_clv") is not None
                else close.get("clv_price_pct")
            )
            closing_fair = _prob(
                close.get("closing_fair_probability")
                if close.get("closing_fair_probability") is not None
                else close.get("close_fair_probability")
            )
            item["strict_close"] = {
                "closing_timestamp": close.get("closing_timestamp"),
                "closing_provider_update": close.get("closing_provider_update"),
                "probability_clv_pp": probability_clv,
                "price_clv_pct": price_clv,
                "closing_fair_probability": closing_fair,
            }
            item["probability_clv_pp"] = probability_clv
            item["price_clv_pct"] = price_clv
            item["model_residual_to_true_close"] = (
                p_model - closing_fair if closing_fair is not None else None
            )

        comparable.append(item)
        family_rows[family].append(item)
        buckets[_bucket(p_model)].append(item)

    family_metrics = {
        family: _summarize(rows)
        for family, rows in sorted(family_rows.items())
    }
    reliability_buckets: list[dict[str, Any]] = []
    for bucket_name, rows in sorted(buckets.items()):
        settled = [row for row in rows if row.get("binary_outcome") is not None]
        reliability_buckets.append(
            {
                "model_probability_bucket": bucket_name,
                "rows": len(rows),
                "mean_p_model_calibrated": _round(mean(row["p_model_calibrated"] for row in rows)),
                "mean_p_market_devig": _round(mean(row["p_market_devig"] for row in rows)),
                "mean_market_residual": _round(mean(row["market_residual"] for row in rows)),
                "settled_binary_rows": len(settled),
                "observed_rate": _round(mean(row["binary_outcome"] for row in settled)) if settled else None,
            }
        )

    status = "RESEARCH_ONLY" if comparable else "NOT_VERIFIED_NO_COMPARABLE_ROWS"
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": status,
        "formula": "p_model_calibrated - p_market_devig",
        "source_rows": len(source),
        "comparable_rows": len(comparable),
        "excluded_rows": dict(sorted(excluded.items())),
        "overall": _summarize(comparable),
        "family_metrics": family_metrics,
        "reliability_buckets": reliability_buckets,
        "rps_1x2": _rps_report(comparable),
        "clv": _clv_report(comparable),
        "evaluation_sources": {
            "settlement_rows": len(settlements),
            "strict_close_rows": len(closes),
            "settlement_policy": "DEDICATED_SETTLEMENT_LEDGER_ONLY; WIN/LOSS_FOR_BINARY; CANONICAL_FT_1X2_FOR_RPS",
            "clv_policy": "EXISTING_STRICT_CLOSE_TRACKING_ONLY; NO_CLOSE_REDEFINITION",
        },
        "interpretation_policy": (
            "DESCRIPTIVE_RESEARCH_ONLY; NEGATIVE BRIER/LOGLOSS/RPS DELTAS MEAN LOWER LOSS THAN "
            "MARKET BASELINE; CLV RETAINS THE EXISTING STRICT-CLOSE DEFINITION; NO PROMOTION OR "
            "BET DECISION MAY BE DERIVED FROM THIS REPORT ALONE"
        ),
        "provider_requests_added": 0,
        "provider_budget_changed": False,
        "decision_weight": 0.0,
        "production_promotion_allowed": False,
        "model_weights_changed": False,
        "thresholds_changed": False,
        "gates_changed": False,
        "canonical_bet_logic_changed": False,
        "strict_close_semantics_changed": False,
    }


def _load_jsonl(path: str | None) -> list[dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _load_source_json(path: str, key: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        return []
    rows = payload.get(key)
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def run(
    *,
    source_json: str | None,
    source_jsonl: str | None,
    source_key: str,
    settlement_jsonl: str | None,
    clv_jsonl: str | None,
    output_path: str,
) -> dict[str, Any]:
    if source_json:
        source = _load_source_json(source_json, source_key)
    elif source_jsonl:
        source = _load_jsonl(source_jsonl)
    else:
        raise ValueError("source_json or source_jsonl is required")

    report = build_report(
        source,
        settlement_rows=_load_jsonl(settlement_jsonl),
        clv_rows=_load_jsonl(clv_jsonl),
    )
    report["source"] = {
        "source_json": source_json,
        "source_jsonl": source_jsonl,
        "source_key": source_key,
        "settlement_jsonl": settlement_jsonl,
        "clv_jsonl": clv_jsonl,
    }
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-json")
    parser.add_argument("--source-jsonl")
    parser.add_argument("--source-key", default="match_table_rows")
    parser.add_argument("--settlement-jsonl")
    parser.add_argument("--clv-jsonl")
    parser.add_argument("--output", default="artifacts/v211_market_residual_challenger.json")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                source_json=args.source_json,
                source_jsonl=args.source_jsonl,
                source_key=args.source_key,
                settlement_jsonl=args.settlement_jsonl,
                clv_jsonl=args.clv_jsonl,
                output_path=args.output,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
