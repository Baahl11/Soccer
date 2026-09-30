from __future__ import annotations

import math
from collections import Counter, defaultdict
from statistics import mean, median
from typing import Any, Iterable

MODEL_VERSION = "SOCCER_MARKET_RESIDUAL_CHALLENGER_V4_1.0.0"
SCHEMA_VERSION = "1.0.0"


def _prob(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        return None
    return number


def _family(row: dict[str, Any]) -> str:
    value = row.get("market_family") or row.get("family") or row.get("market") or "UNKNOWN"
    return str(value).strip().upper() or "UNKNOWN"


def _binary_outcome(row: dict[str, Any]) -> float | None:
    """Read only explicit point-in-time binary settlement fields.

    Never infer an outcome from score text or selection names here. That keeps
    the challenger leakage-safe and prevents family-specific settlement rules
    from being silently approximated.
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


def _log_loss(p: float, y: float) -> float:
    clipped = min(max(p, 1e-12), 1.0 - 1e-12)
    return -(y * math.log(clipped) + (1.0 - y) * math.log(1.0 - clipped))


def _bucket(p: float) -> str:
    low = min(int(p * 10.0) * 10, 90)
    high = low + 10
    return f"{low:02d}-{high:02d}%"


def _round(value: float | None, digits: int = 8) -> float | None:
    return round(value, digits) if value is not None and math.isfinite(value) else None


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


def build_report(source_rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    comparable: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    family_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)

    source_count = 0
    for row in source_rows:
        if not isinstance(row, dict):
            continue
        source_count += 1
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
            "binary_outcome": _binary_outcome(row),
        }
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
        "source_rows": source_count,
        "comparable_rows": len(comparable),
        "excluded_rows": dict(sorted(excluded.items())),
        "overall": _summarize(comparable),
        "family_metrics": family_metrics,
        "reliability_buckets": reliability_buckets,
        "interpretation_policy": (
            "DESCRIPTIVE_RESEARCH_ONLY; NEGATIVE SCORE DELTAS MEAN LOWER LOSS THAN MARKET BASELINE; "
            "NO PROMOTION OR BET DECISION MAY BE DERIVED FROM THIS REPORT ALONE"
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
