from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

SCHEMA_VERSION = "1.1.0"
MODEL_VERSION = "SOCCER_MATURATION_WATCHDOGS_V4_1.1.0"
STAGNATION_WINDOW_SECONDS = 48 * 60 * 60

_ARTIFACT_TIMESTAMP_KEYS = (
    "report_updated_at_utc",
    "report_generated_at_utc",
    "artifact_updated_at_utc",
    "updated_at_utc",
    "updated_at",
)
_EVIDENCE_TIMESTAMP_KEYS = (
    "last_evidence_at_utc",
    "last_evidence_at_local",
    "last_generated_at_utc",
    "last_generated_at_local",
)


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _timestamp_from_keys(report: dict[str, Any], keys: tuple[str, ...]) -> datetime | None:
    for key in keys:
        parsed = _parse_timestamp(report.get(key))
        if parsed is not None:
            return parsed
    return None


def _artifact_timestamp(report: dict[str, Any]) -> datetime | None:
    return _timestamp_from_keys(report, _ARTIFACT_TIMESTAMP_KEYS)


def _evidence_timestamp(report: dict[str, Any]) -> datetime | None:
    return _timestamp_from_keys(report, _EVIDENCE_TIMESTAMP_KEYS)


def _watch(
    status: str,
    reason: str,
    *,
    source: str,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "reason": reason,
        "source": source,
        "evidence": evidence or {},
        "decision_weight": 0.0,
        "production_promotion_allowed": False,
    }


def evidence_counters(reports: dict[str, dict[str, Any]]) -> dict[str, int]:
    counters: dict[str, int] = {}

    for key in ("one_x_two", "btts", "team_totals", "one_h", "two_h"):
        report = _dict(reports.get(key))
        clv = _dict(report.get("true_clv"))
        rows = _int(clv.get("rows"))
        fixtures = _int(clv.get("unique_fixtures"))
        if rows is not None:
            counters[f"{key}.true_clv_rows"] = rows
        if fixtures is not None:
            counters[f"{key}.true_clv_fixtures"] = fixtures

    corners = _dict(reports.get("corners"))
    corners_ft = _dict(corners.get("ft_corners"))
    formation_rows = _int(corners_ft.get("formation_adjusted_evaluations"))
    if formation_rows is not None:
        counters["corners.formation_adjusted_evaluations"] = formation_rows

    cards = _dict(reports.get("cards"))
    cards_market = _dict(_dict(cards.get("market_evidence")).get("match_cards"))
    for field in ("market_snapshot_rows", "priced_value_rows", "unique_fixtures"):
        value = _int(cards_market.get(field))
        if value is not None:
            counters[f"cards.match_cards.{field}"] = value
    cards_clv = _dict(cards.get("true_clv"))
    if _int(cards_clv.get("rows")) is not None:
        counters["cards.true_clv_rows"] = int(cards_clv["rows"])

    props = _dict(reports.get("player_props"))
    for family, node in _dict(props.get("prop_families")).items():
        market = _dict(_dict(node).get("market_evidence"))
        for field in (
            "market_snapshot_rows",
            "priced_value_rows",
            "confirmed_xi_pre_kickoff_unique_fixtures",
            "confirmed_xi_player_aligned_unique_fixtures",
        ):
            value = _int(market.get(field))
            if value is not None:
                counters[f"player_props.{family}.{field}"] = value

    signal_summary = _dict(reports.get("signal_summary"))
    stages = _dict(signal_summary.get("stage_counts"))
    close_rows = _int(stages.get("CLOSE"))
    if close_rows is not None:
        counters["signals.close_stage_rows"] = close_rows

    settlement = _dict(reports.get("settlement_coverage"))
    settlement_rows = _int(settlement.get("settlement_rows"))
    if settlement_rows is not None:
        counters["settlement.rows"] = settlement_rows

    return counters


def provider_call_counter(reports: dict[str, dict[str, Any]]) -> int | None:
    efficiency = _dict(reports.get("api_efficiency"))
    totals = _dict(efficiency.get("totals"))
    return _int(totals.get("api_calls_this_tick"))


def _growth_delta(previous: dict[str, int], current: dict[str, int]) -> tuple[int, list[str], list[str]]:
    growth = 0
    increased: list[str] = []
    decreased: list[str] = []
    shared = set(previous) & set(current)
    for key in sorted(shared):
        delta = current[key] - previous[key]
        if delta > 0:
            growth += delta
            increased.append(key)
        elif delta < 0:
            decreased.append(key)
    return growth, increased, decreased


def build_watchdogs(
    reports: dict[str, dict[str, Any]],
    *,
    baseline: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    now_utc = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    now_ts = now_utc.timestamp()
    counters = evidence_counters(reports)
    provider_calls = provider_call_counter(reports)

    previous = baseline if isinstance(baseline, dict) else {}
    previous_counters = {
        str(key): int(value)
        for key, value in _dict(previous.get("counters")).items()
        if _int(value) is not None
    }
    previous_calls = _int(previous.get("provider_calls"))
    since_ts = _float(previous.get("since_ts"))

    growth, increased, decreased = _growth_delta(previous_counters, counters)
    if not previous_counters or since_ts is None or decreased or growth > 0:
        next_baseline = {
            "since_ts": now_ts,
            "counters": dict(counters),
            "provider_calls": provider_calls,
        }
        stagnation_seconds = 0.0
    else:
        next_baseline = {
            "since_ts": since_ts,
            "counters": dict(previous_counters),
            "provider_calls": previous_calls,
        }
        stagnation_seconds = max(0.0, now_ts - since_ts)

    watchdogs: dict[str, dict[str, Any]] = {}

    signal = _dict(reports.get("signal_summary"))
    artifact_ts = _artifact_timestamp(signal)
    evidence_ts = _evidence_timestamp(signal)
    stages = _dict(signal.get("stage_counts"))
    close_rows = _int(stages.get("CLOSE"))
    if not signal:
        watchdogs["signal_close_freshness"] = _watch(
            "NOT_VERIFIED", "SIGNAL_LEDGER_SUMMARY_UNAVAILABLE", source="signal_ledger_summary.json"
        )
        watchdogs["signal_evidence_age"] = _watch(
            "NOT_VERIFIED", "SIGNAL_EVIDENCE_UNAVAILABLE", source="signal_ledger_summary.json"
        )
    else:
        if artifact_ts is None:
            watchdogs["signal_close_freshness"] = _watch(
                "NOT_VERIFIED",
                "SIGNAL_REPORT_ARTIFACT_TIMESTAMP_UNAVAILABLE",
                source="signal_ledger_summary.json",
                evidence={"close_stage_rows": close_rows},
            )
        else:
            artifact_age = max(0.0, now_ts - artifact_ts.timestamp())
            watchdogs["signal_close_freshness"] = _watch(
                "WATCH" if artifact_age >= STAGNATION_WINDOW_SECONDS else "OK",
                "SIGNAL_REPORT_ARTIFACT_STALE_48H" if artifact_age >= STAGNATION_WINDOW_SECONDS else "SIGNAL_REPORT_ARTIFACT_FRESH",
                source="signal_ledger_summary.json",
                evidence={
                    "report_updated_at": artifact_ts.isoformat(),
                    "age_hours": round(artifact_age / 3600.0, 2),
                    "close_stage_rows": close_rows,
                },
            )
        if evidence_ts is None:
            watchdogs["signal_evidence_age"] = _watch(
                "NOT_VERIFIED",
                "SIGNAL_LAST_EVIDENCE_TIMESTAMP_UNAVAILABLE",
                source="signal_ledger_summary.json",
                evidence={"close_stage_rows": close_rows},
            )
        else:
            evidence_age = max(0.0, now_ts - evidence_ts.timestamp())
            watchdogs["signal_evidence_age"] = _watch(
                "WATCH" if evidence_age >= STAGNATION_WINDOW_SECONDS else "OK",
                "SIGNAL_EVIDENCE_AGE_OVER_48H" if evidence_age >= STAGNATION_WINDOW_SECONDS else "SIGNAL_EVIDENCE_RECENT",
                source="signal_ledger_summary.json",
                evidence={
                    "last_evidence_at": evidence_ts.isoformat(),
                    "age_hours": round(evidence_age / 3600.0, 2),
                    "close_stage_rows": close_rows,
                },
            )

    two_h = _dict(reports.get("two_h"))
    two_h_clv = _dict(two_h.get("true_clv"))
    two_h_rows = _int(two_h_clv.get("rows"))
    two_h_fixtures = _int(two_h_clv.get("unique_fixtures"))
    if not two_h or two_h_rows is None:
        watchdogs["two_h_market_maturation"] = _watch(
            "NOT_VERIFIED", "2H_MATURATION_SOURCE_UNAVAILABLE", source="v4_021_2h_oos_validation.json"
        )
    else:
        watchdogs["two_h_market_maturation"] = _watch(
            "WATCH" if two_h_rows <= 0 else "OK",
            "2H_STRICT_CLOSE_EVIDENCE_ZERO" if two_h_rows <= 0 else "2H_STRICT_CLOSE_EVIDENCE_PRESENT",
            source="v4_021_2h_oos_validation.json",
            evidence={"true_clv_rows": two_h_rows, "true_clv_unique_fixtures": two_h_fixtures, "minimum_rows": _int(two_h_clv.get("minimum_rows"))},
        )

    corners = _dict(reports.get("corners"))
    formation_rows = _int(_dict(corners.get("ft_corners")).get("formation_adjusted_evaluations"))
    if not corners or formation_rows is None:
        watchdogs["corners_formation_join"] = _watch(
            "NOT_VERIFIED", "CORNERS_FORMATION_SOURCE_UNAVAILABLE", source="v4_022_corners_oos_validation.json"
        )
    else:
        watchdogs["corners_formation_join"] = _watch(
            "WATCH" if formation_rows <= 0 else "OK",
            "CORNERS_FORMATION_JOINS_ZERO" if formation_rows <= 0 else "CORNERS_FORMATION_JOINS_PRESENT",
            source="v4_022_corners_oos_validation.json",
            evidence={"formation_adjusted_evaluations": formation_rows},
        )

    cards = _dict(reports.get("cards"))
    match_cards = _dict(_dict(cards.get("market_evidence")).get("match_cards"))
    card_priced = _int(match_cards.get("priced_value_rows"))
    card_snapshots = _int(match_cards.get("market_snapshot_rows"))
    settlement = _dict(reports.get("settlement_coverage"))
    settlement_families = _dict(settlement.get("by_market_family_reason"))
    card_settlement_families = sorted(str(name) for name in settlement_families if "CARD" in str(name).upper())
    if not cards:
        watchdogs["cards_settlement"] = _watch(
            "NOT_VERIFIED", "CARDS_MATURATION_SOURCE_UNAVAILABLE", source="phase14_cards_referee_validation.json"
        )
    elif card_priced is None:
        watchdogs["cards_settlement"] = _watch(
            "NOT_VERIFIED", "CARD_PRICE_COUNTER_UNAVAILABLE", source="phase14_cards_referee_validation.json"
        )
    elif card_priced <= 0:
        watchdogs["cards_settlement"] = _watch(
            "WATCH",
            "CARD_SETTLEMENT_BLOCKED_NO_OBSERVED_PRICE_HISTORY",
            source="phase14_cards_referee_validation.json",
            evidence={"match_card_priced_value_rows": card_priced, "match_card_market_snapshot_rows": card_snapshots, "settlement_card_families": card_settlement_families},
        )
    elif not settlement:
        watchdogs["cards_settlement"] = _watch(
            "NOT_VERIFIED", "SETTLEMENT_COVERAGE_SOURCE_UNAVAILABLE", source="settlement_coverage_report.json", evidence={"match_card_priced_value_rows": card_priced}
        )
    else:
        watchdogs["cards_settlement"] = _watch(
            "WATCH" if not card_settlement_families else "OK",
            "CARD_SETTLEMENT_ROWS_NOT_OBSERVED" if not card_settlement_families else "CARD_SETTLEMENT_EVIDENCE_PRESENT",
            source="settlement_coverage_report.json",
            evidence={"match_card_priced_value_rows": card_priced, "settlement_card_families": card_settlement_families},
        )

    props = _dict(reports.get("player_props"))
    prop_families = _dict(props.get("prop_families"))
    prop_gaps: list[str] = []
    prop_evidence: dict[str, Any] = {}
    for family, node in sorted(prop_families.items()):
        market = _dict(_dict(node).get("market_evidence"))
        priced = _int(market.get("priced_value_rows")) or 0
        xi_fixture = _int(market.get("confirmed_xi_pre_kickoff_unique_fixtures")) or 0
        xi_player = _int(market.get("confirmed_xi_player_aligned_unique_fixtures")) or 0
        prop_evidence[str(family)] = {"priced_value_rows": priced, "confirmed_xi_fixtures": xi_fixture, "player_xi_aligned_fixtures": xi_player}
        if priced > 0 and (xi_fixture <= 0 or xi_player <= 0):
            prop_gaps.append(str(family))
    if not prop_families:
        watchdogs["player_props_xi_confirmed"] = _watch(
            "NOT_VERIFIED", "PLAYER_PROP_FAMILY_EVIDENCE_UNAVAILABLE", source="phase15_player_props_validation.json"
        )
    else:
        watchdogs["player_props_xi_confirmed"] = _watch(
            "WATCH" if prop_gaps else "OK",
            "PLAYER_PROP_XI_ALIGNMENT_GAPS" if prop_gaps else "PLAYER_PROP_XI_ALIGNMENT_PRESENT",
            source="phase15_player_props_validation.json",
            evidence={"families_with_gaps": prop_gaps, "by_family": prop_evidence},
        )

    baseline_source = str(previous.get("source") or "persistent_watchdog_baseline")
    if not previous_counters or since_ts is None:
        watchdogs["evidence_growth_48h"] = _watch(
            "NOT_VERIFIED", "MATURATION_BASELINE_INITIALIZED", source=baseline_source, evidence={"counter_count": len(counters)}
        )
    else:
        stalled = stagnation_seconds >= STAGNATION_WINDOW_SECONDS and growth <= 0 and not decreased
        watchdogs["evidence_growth_48h"] = _watch(
            "WATCH" if stalled else "OK",
            "NO_MATURATION_EVIDENCE_GROWTH_48H" if stalled else "MATURATION_EVIDENCE_MONITORING",
            source=baseline_source,
            evidence={"stagnation_hours": round(stagnation_seconds / 3600.0, 2), "increased_counters": increased, "counter_count": len(counters)},
        )

    if provider_calls is None:
        watchdogs["provider_requests_without_evidence_growth"] = _watch(
            "NOT_VERIFIED", "PROVIDER_CALL_COUNTER_UNAVAILABLE", source="api_efficiency.json"
        )
    elif previous_calls is None or not previous_counters or since_ts is None:
        watchdogs["provider_requests_without_evidence_growth"] = _watch(
            "NOT_VERIFIED", "PROVIDER_EFFICIENCY_BASELINE_INITIALIZED", source=baseline_source, evidence={"provider_calls": provider_calls}
        )
    else:
        request_delta = provider_calls - previous_calls
        stalled = request_delta > 0 and growth <= 0 and not decreased and stagnation_seconds >= STAGNATION_WINDOW_SECONDS
        watchdogs["provider_requests_without_evidence_growth"] = _watch(
            "WATCH" if stalled else "OK",
            "PROVIDER_REQUESTS_CONSUMED_WITHOUT_EVIDENCE_GROWTH_48H" if stalled else "PROVIDER_EFFICIENCY_MONITORING",
            source=baseline_source,
            evidence={"provider_calls": provider_calls, "provider_call_delta": request_delta, "evidence_growth_delta": growth, "stagnation_hours": round(stagnation_seconds / 3600.0, 2)},
        )

    statuses = [node["status"] for node in watchdogs.values()]
    watch_count = statuses.count("WATCH")
    not_verified_count = statuses.count("NOT_VERIFIED")
    overall = "WATCH" if watch_count else ("NOT_VERIFIED" if not_verified_count == len(statuses) else "OK")
    return (
        {
            "schema_version": SCHEMA_VERSION,
            "model_version": MODEL_VERSION,
            "status": overall,
            "generated_at_utc": now_utc.isoformat(),
            "watchdogs": watchdogs,
            "watch_count": watch_count,
            "ok_count": statuses.count("OK"),
            "not_verified_count": not_verified_count,
            "stagnation_window_hours": 48,
            "provider_requests_added": 0,
            "production_promotion_allowed": False,
            "canonical_bet_logic_changed": False,
            "model_weights_changed": False,
            "thresholds_changed": False,
            "gates_changed": False,
            "decision_weights_changed": False,
        },
        next_baseline,
    )
