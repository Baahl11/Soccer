from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v91 as v91

MODEL_VERSION = v91.MODEL_VERSION
AUTOMATION_VERSION = "4.0.0-v4.003"

MODEL_SIGNALS = {
    "VERY_STRONG",
    "STRONG",
    "MODERATE",
    "WEAK",
    "NEUTRAL",
    "MODEL_DISAGREEMENT",
    "INSUFFICIENT_DATA",
    "NOT_MODELED",
}

EXECUTION_STATUSES = {
    "READY",
    "WAIT_PRICE",
    "WAIT_FRESH_QUOTE",
    "WAIT_XI",
    "WAIT_GK",
    "WAIT_AVAILABILITY",
    "WAIT_MARKET",
    "PRICE_TOO_LOW",
    "MODEL_DISAGREEMENT",
    "STALE_QUOTE",
    "DATA_TOO_WEAK",
    "RESEARCH_ONLY",
}


def _num(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _derive_model_signal(row: dict[str, Any]) -> str:
    """Price-independent v1 sporting signal for visibility only.

    Existing sporting screen scores are used as-is; no odds, price, edge or EV
    field participates. Thresholds are presentation semantics, not promotion or
    canonical betting thresholds.
    """
    scores = [
        _num(row.get("side_score")),
        _num(row.get("goals_score")),
        _num(row.get("two_way_score")),
    ]
    usable = [score for score in scores if score is not None]
    if not usable:
        return "INSUFFICIENT_DATA"

    score = max(usable)
    if score >= 80.0:
        return "VERY_STRONG"
    if score >= 70.0:
        return "STRONG"
    if score >= 60.0:
        return "MODERATE"
    if score >= 50.0:
        return "WEAK"
    return "NEUTRAL"


def _reason_text(row: dict[str, Any]) -> str:
    return str(row.get("reason") or "").upper()


def _derive_execution_status(row: dict[str, Any]) -> str:
    """Execution readiness derived separately from sporting strength."""
    reason = _reason_text(row)
    classification = str(row.get("classification") or "WATCH").upper()
    raw_classification = str(row.get("event_classification") or classification).upper()

    if "STALE" in reason and ("QUOTE" in reason or "ODDS" in reason or "PRICE" in reason):
        return "STALE_QUOTE"
    if "WAIT_FRESH_QUOTE" in reason or "FRESH_QUOTE" in reason:
        return "WAIT_FRESH_QUOTE"
    if "WAIT_XI" in reason or "LINEUP" in reason or "CONFIRMED_XI" in reason:
        return "WAIT_XI"
    if "WAIT_GK" in reason or "GOALKEEPER" in reason:
        return "WAIT_GK"
    if "AVAILABILITY" in reason or "INJURY" in reason or "SUSPENSION" in reason:
        return "WAIT_AVAILABILITY"
    if "MODEL_DISAGREEMENT" in reason:
        return "MODEL_DISAGREEMENT"
    if "DATA_TOO_WEAK" in reason or "INSUFFICIENT_DATA" in reason:
        return "DATA_TOO_WEAK"
    if "PRICE_TOO_LOW" in reason or "PRICE_BELOW" in reason or "MIN_PRICE" in reason:
        return "PRICE_TOO_LOW"

    has_market = bool(row.get("market")) and str(row.get("market")) not in {"Research screen", "—"}
    has_price = _num(row.get("price")) is not None

    if classification == "BET" and bool(row.get("bet_eligible")) and has_price:
        return "READY"
    if not has_market:
        return "WAIT_MARKET"
    if not has_price:
        return "WAIT_PRICE"
    if raw_classification in {"PASS", "CLOSE", "POSTGAME"}:
        return "RESEARCH_ONLY"
    return "RESEARCH_ONLY"


def _blockers(row: dict[str, Any], execution_status: str) -> list[str]:
    if execution_status == "READY":
        return []
    reason = str(row.get("reason") or "").strip()
    blockers = [execution_status]
    if reason and reason not in blockers:
        blockers.append(reason[:220])
    return blockers


def _annotate_decision_separation(payload: dict[str, Any]) -> None:
    rows = payload.get("match_table_rows")
    if not isinstance(rows, list):
        rows = []
        payload["match_table_rows"] = rows

    model_counts: dict[str, int] = {}
    execution_counts: dict[str, int] = {}

    for row in rows:
        if not isinstance(row, dict):
            continue
        model_signal = _derive_model_signal(row)
        execution_status = _derive_execution_status(row)
        row["model_signal"] = model_signal
        row["model_signal_score"] = max([score for score in (_num(row.get("side_score")), _num(row.get("goals_score")), _num(row.get("two_way_score"))) if score is not None], default=None)
        row["execution_status"] = execution_status
        row["blockers"] = _blockers(row, execution_status)
        row["model_signal_basis"] = "SPORTING_SCREEN_ONLY_NO_PRICE"
        row["execution_status_basis"] = "READINESS_BLOCKERS_AND_MARKET_AVAILABILITY"
        model_counts[model_signal] = model_counts.get(model_signal, 0) + 1
        execution_counts[execution_status] = execution_counts.get(execution_status, 0) + 1

    payload["decision_separation"] = {
        "schema_version": "1.1.0",
        "policy": "MODEL_SIGNAL_INDEPENDENT_OF_PRICE_EXECUTION_STATUS_INDEPENDENT_OF_SPORT_STRENGTH",
        "row_count": len([row for row in rows if isinstance(row, dict)]),
        "model_signal_counts": model_counts,
        "execution_status_counts": execution_counts,
        "valid_model_signals": sorted(MODEL_SIGNALS),
        "valid_execution_statuses": sorted(EXECUTION_STATUSES),
        "model_signal_inputs": ["side_score", "goals_score", "two_way_score"],
        "shortlist_rank_is_not_model_strength": True,
    }
    payload["v4_003_model_signal_execution_status"] = True
    payload["v4_003_provider_requests_added"] = 0
    payload["v4_003_model_weights_changed"] = False
    payload["v4_003_canonical_bet_logic_changed"] = False
    payload["v4_003_runtime_promotion_added"] = False
    payload["v4_003_checkpoint"] = (
        "MODEL_SIGNAL and EXECUTION_STATUS are now explicit independent fields on research rows. "
        "MODEL_SIGNAL uses only existing sporting-screen scores; EXECUTION_STATUS uses readiness/market blockers. "
        "No provider calls, model weights, canonical BET logic, tiers or stakes changed."
    )


async def run_tick() -> dict[str, Any]:
    payload = await v91.run_tick()
    _annotate_decision_separation(payload)
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
