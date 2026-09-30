from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import Counter
from typing import Any, Iterable

MODEL_VERSION = "SOCCER_FROZEN_CALIBRATION_AUDIT_V4_1.1.0"
SCHEMA_VERSION = "1.1.0"


def _prob(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        return None
    return number


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().upper().split())


def _artifact_from_provenance(provenance: dict[str, Any]) -> dict[str, Any]:
    for key in (
        "phase16_calibrator_artifact",
        "calibrator_artifact",
        "calibration_artifact",
    ):
        value = provenance.get(key)
        if isinstance(value, dict) and value:
            return value
    return {}


def _artifact(row: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    direct = row.get("phase16_calibrator_artifact")
    if isinstance(direct, dict) and direct:
        return direct
    return _artifact_from_provenance(provenance)


def _artifact_fingerprint(artifact: dict[str, Any]) -> str | None:
    for key in ("fingerprint_sha256", "sha256", "artifact_id", "fingerprint"):
        value = artifact.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _exact_artifact_payload(artifact: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("exact_calibrator_payload", "calibrator_payload"):
        value = artifact.get(key)
        if isinstance(value, dict) and value:
            return value
    return None


def _source_model_version(provenance: dict[str, Any], artifact: dict[str, Any]) -> str | None:
    for value in (
        artifact.get("source_model_version"),
        provenance.get("source_model_version"),
    ):
        if value is not None and str(value).strip():
            return str(value).strip()

    binary = provenance.get("binary_calibration_diagnostics")
    if isinstance(binary, dict):
        value = binary.get("source_model_version")
        if value is not None and str(value).strip():
            return str(value).strip()

    one_x_two = provenance.get("one_x_two_class_discrimination_diagnostics")
    if isinstance(one_x_two, dict):
        for item in one_x_two.values():
            if not isinstance(item, dict):
                continue
            value = item.get("source_model_version")
            if value is not None and str(value).strip():
                return str(value).strip()
    return None


def _model_version_match(provenance: dict[str, Any], artifact: dict[str, Any]) -> bool | None:
    value = artifact.get("source_model_version_matches")
    if isinstance(value, bool):
        return value

    binary = provenance.get("binary_calibration_diagnostics")
    if isinstance(binary, dict) and isinstance(binary.get("source_model_version_matches"), bool):
        return bool(binary["source_model_version_matches"])

    one_x_two = provenance.get("one_x_two_class_discrimination_diagnostics")
    if isinstance(one_x_two, dict):
        states = [
            item.get("source_model_version_matches")
            for item in one_x_two.values()
            if isinstance(item, dict) and isinstance(item.get("source_model_version_matches"), bool)
        ]
        if states:
            return all(states)
    return None


def _calibrated_probability(row: dict[str, Any], provenance: dict[str, Any]) -> float | None:
    for key in (
        "p_model_calibrated",
        "p_calibrated",
        "calibrated_probability",
        "model_probability_calibrated",
    ):
        value = _prob(row.get(key))
        if value is not None:
            return value
    fields = provenance.get("calibrated_probability_fields")
    if isinstance(fields, dict):
        for key in (
            "p_model_calibrated",
            "p_calibrated",
            "calibrated_probability",
            "model_probability_calibrated",
        ):
            value = _prob(fields.get(key))
            if value is not None:
                return value
    return None


def _is_calibrated(row: dict[str, Any], provenance: dict[str, Any]) -> bool:
    status = str(
        row.get("phase16_calibration_status")
        or provenance.get("calibration_status")
        or ""
    ).strip().upper()
    if status == "RESEARCH_CALIBRATION_APPLIED":
        return True
    return _calibrated_probability(row, provenance) is not None


def _provider_update(mapping: Any) -> Any:
    if not isinstance(mapping, dict):
        return None
    for key in (
        "provider_update",
        "provider_updated_at",
        "provider_last_update",
        "provider_update_at",
        "odds_update",
        "odds_updated_at",
        "last_update",
        "updated_at",
    ):
        value = mapping.get(key)
        if value is not None and str(value).strip():
            return value
    return None


def _row_probability(row: dict[str, Any], key: str) -> float | None:
    value = _prob(row.get(key))
    if value is not None:
        return value
    best = row.get("best_market")
    if isinstance(best, dict):
        return _prob(best.get(key))
    return None


def audit_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    total_rows = 0
    calibrated_rows = 0
    complete_rows = 0
    calibrated_with_artifact = 0
    calibrated_with_fingerprint = 0
    calibrated_with_exact_payload = 0
    gap_counts: Counter[str] = Counter()
    match_counts: Counter[str] = Counter()

    for row in rows:
        if not isinstance(row, dict):
            continue
        total_rows += 1
        provenance = row.get("phase16_calibration_provenance")
        provenance = provenance if isinstance(provenance, dict) else {}
        artifact = _artifact(row, provenance)
        calibrated = _is_calibrated(row, provenance)
        if calibrated:
            calibrated_rows += 1

        complete = True
        if not str(row.get("model_version") or "").strip():
            gap_counts["MISSING_MODEL_VERSION"] += 1
            complete = False
        if not str(row.get("stage") or "").strip():
            gap_counts["MISSING_STAGE"] += 1
            complete = False
        if _row_probability(row, "p_raw") is None:
            gap_counts["MISSING_P_RAW"] += 1
            complete = False
        if _row_probability(row, "p_market_fair") is None:
            gap_counts["MISSING_P_MARKET_FAIR"] += 1
            complete = False
        provider_update = row.get("provider_update")
        if provider_update is None:
            provider_update = _provider_update(row)
        if provider_update is None:
            best = row.get("best_market")
            provider_update = _provider_update(best)
        if provider_update is None:
            gap_counts["MISSING_PROVIDER_UPDATE"] += 1
            complete = False

        if calibrated:
            if _calibrated_probability(row, provenance) is None:
                gap_counts["MISSING_P_MODEL_CALIBRATED"] += 1
                complete = False
            if artifact:
                calibrated_with_artifact += 1
            else:
                gap_counts["MISSING_CALIBRATOR_ARTIFACT"] += 1
                complete = False
            if artifact and _artifact_fingerprint(artifact):
                calibrated_with_fingerprint += 1
            else:
                gap_counts["MISSING_IMMUTABLE_ARTIFACT_FINGERPRINT"] += 1
                complete = False
            if artifact and _exact_artifact_payload(artifact):
                calibrated_with_exact_payload += 1
            else:
                gap_counts["MISSING_EXACT_CALIBRATOR_PAYLOAD"] += 1
                complete = False
            if not _source_model_version(provenance, artifact):
                gap_counts["MISSING_SOURCE_MODEL_VERSION"] += 1
                complete = False
            match = _model_version_match(provenance, artifact)
            match_counts["MATCH" if match is True else "MISMATCH" if match is False else "UNKNOWN"] += 1

        if complete:
            complete_rows += 1

    if total_rows == 0:
        status = "NOT_VERIFIED_NO_HISTORICAL_SIGNALS"
    elif complete_rows == total_rows:
        status = "OK_FROZEN_POINT_IN_TIME_COMPLETE"
    else:
        status = "WATCH_FROZEN_POINT_IN_TIME_GAPS"

    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": status,
        "signal_definition": "HISTORICAL_EVENT_WITH_CAPTURED_BEST_MARKET",
        "historical_signal_rows": total_rows,
        "point_in_time_complete_rows": complete_rows,
        "point_in_time_complete_coverage": round(complete_rows / total_rows, 8) if total_rows else None,
        "calibrated_signal_rows": calibrated_rows,
        "calibrated_rows_with_calibrator_artifact": calibrated_with_artifact,
        "calibrated_rows_with_immutable_artifact_fingerprint": calibrated_with_fingerprint,
        "calibrated_rows_with_exact_calibrator_payload": calibrated_with_exact_payload,
        "source_model_version_match_counts": dict(sorted(match_counts.items())),
        "gap_counts": dict(sorted(gap_counts.items())),
        "required_point_in_time_fields": [
            "model_version",
            "stage",
            "p_raw",
            "p_model_calibrated_when_applied",
            "p_market_fair",
            "provider_update",
            "exact_calibrator_artifact_when_applied",
        ],
        "historical_rows_mutated": False,
        "historical_probabilities_recomputed": False,
        "historical_rows_recalibrated": False,
        "provider_requests_added": 0,
        "provider_budget_changed": False,
        "decision_weight": 0.0,
        "production_promotion_allowed": False,
        "model_weights_changed": False,
        "thresholds_changed": False,
        "gates_changed": False,
        "strict_close_semantics_changed": False,
        "policy": (
            "AUDIT_IMMUTABLE_POINT_IN_TIME_HISTORY_ONLY; NO_RETROACTIVE_RECALIBRATION; "
            "NO_PROVIDER_CALLS; NO_DECISION_WEIGHT; NO_PROMOTION"
        ),
    }


def _same_direction(left: Any, right: Any) -> bool:
    a = _norm(left)
    b = _norm(right)
    if not a or not b:
        return True
    for token in ("OVER", "UNDER", "YES", "NO", "HOME", "DRAW", "AWAY"):
        a_has = token in a
        b_has = token in b
        if a_has or b_has:
            return a_has == b_has
    return a == b


def _family_matches(best_family: Any, row_family: Any) -> bool:
    best = _norm(best_family)
    row = _norm(row_family)
    if not best or not row or best == row:
        return True
    aliases = {
        "FT_TOTALS": {"FT_TOTALS", "TOTAL", "FT_TOTALS_RESEARCH"},
        "TOTAL": {"FT_TOTALS", "TOTAL", "FT_TOTALS_RESEARCH"},
        "FT_TOTALS_RESEARCH": {"FT_TOTALS", "TOTAL", "FT_TOTALS_RESEARCH"},
        "BTTS": {"BTTS", "FT_BTTS", "FT_BTTS_RESEARCH"},
        "FT_BTTS": {"BTTS", "FT_BTTS", "FT_BTTS_RESEARCH"},
        "1X2": {"1X2", "FT_1X2", "FT_1X2_RESEARCH", "MATCH_WINNER"},
        "FT_1X2": {"1X2", "FT_1X2", "FT_1X2_RESEARCH", "MATCH_WINNER"},
    }
    return row in aliases.get(best, {best})


def _match_table_row(tick: dict[str, Any], fixture_id: int, best: dict[str, Any]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    best_price = best.get("decimal_price")
    try:
        best_price_f = float(best_price) if best_price is not None else None
    except (TypeError, ValueError):
        best_price_f = None

    for row in tick.get("match_table_rows") or []:
        if not isinstance(row, dict):
            continue
        try:
            row_fixture = int(row.get("fixture_id"))
        except (TypeError, ValueError):
            continue
        if row_fixture != int(fixture_id):
            continue
        if not _family_matches(best.get("family"), row.get("market_family")):
            continue
        if not _same_direction(best.get("selection"), row.get("selection")):
            continue
        if best.get("line") is not None or row.get("line") is not None:
            try:
                row_line = float(row.get("line")) if row.get("line") is not None else None
                best_line = float(best.get("line")) if best.get("line") is not None else None
            except (TypeError, ValueError):
                row_line = best_line = None
            if row_line is None or best_line is None or abs(row_line - best_line) > 1e-9:
                continue
        candidates.append(row)

    if not candidates:
        return {}

    def distance(row: dict[str, Any]) -> float:
        try:
            row_price = float(row.get("price"))
        except (TypeError, ValueError):
            return 999999.0
        if best_price_f is None:
            return 999999.0
        return abs(row_price - best_price_f)

    return min(candidates, key=distance)


def _first_probability(primary: dict[str, Any], secondary: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for mapping in (primary, secondary):
        for key in keys:
            value = _prob(mapping.get(key))
            if value is not None:
                return value
    return None


def iter_history_signal_rows(history_dir: str) -> Iterable[dict[str, Any]]:
    """Yield exact same-tick signal snapshots from immutable state history.

    Only values already persisted in each historical tick are copied. No
    provider is called, no calibrator is refit, and no probability is recomputed.
    """
    for path in sorted(glob.glob(os.path.join(history_dir, "*.jsonl"))):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    tick = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(tick, dict):
                    continue
                for event in tick.get("events") or []:
                    if not isinstance(event, dict):
                        continue
                    best = event.get("best_market")
                    fixture = event.get("fixture")
                    if not isinstance(best, dict) or not best or not isinstance(fixture, dict):
                        continue
                    try:
                        fixture_id = int(fixture.get("fixture_id"))
                    except (TypeError, ValueError):
                        continue
                    matched = _match_table_row(tick, fixture_id, best)
                    artifact = matched.get("phase16_calibrator_artifact") if matched else None
                    calibrated = _first_probability(
                        matched,
                        {},
                        (
                            "p_model_calibrated",
                            "p_calibrated",
                            "calibrated_probability",
                            "model_probability_calibrated",
                        ),
                    )
                    yield {
                        "generated_at_local": tick.get("generated_at_local"),
                        "fixture_id": fixture_id,
                        "model_version": tick.get("model_version") or event.get("model_version"),
                        "stage": event.get("stage"),
                        "market_family": best.get("family") or matched.get("market_family"),
                        "market": best.get("market") or matched.get("market"),
                        "selection": best.get("selection") or matched.get("selection"),
                        "line": best.get("line") if best.get("line") is not None else matched.get("line"),
                        "p_raw": _first_probability(best, matched, ("p_raw", "model_probability", "probability")),
                        "p_model_calibrated": calibrated,
                        "p_market_fair": _first_probability(best, matched, ("p_market_fair", "p_market_devig")),
                        "provider_update": _provider_update(matched) or _provider_update(best) or _provider_update(event),
                        "phase16_calibration_status": matched.get("phase16_calibration_status"),
                        "phase16_calibrator_artifact": artifact if isinstance(artifact, dict) else None,
                        "phase16_calibration_provenance": {
                            "calibration_status": matched.get("phase16_calibration_status"),
                            "source_model_version": (artifact or {}).get("source_model_version") if isinstance(artifact, dict) else None,
                            "phase16_calibrator_artifact": artifact if isinstance(artifact, dict) else None,
                        },
                    }


def _iter_jsonl(path: str) -> Iterable[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def run(output_path: str, *, input_path: str | None = None, history_dir: str | None = None) -> dict[str, Any]:
    if history_dir:
        rows = iter_history_signal_rows(history_dir)
        source = f"IMMUTABLE_HISTORY:{history_dir}"
    elif input_path:
        rows = _iter_jsonl(input_path)
        source = f"JSONL:{input_path}"
    else:
        raise ValueError("input_path or history_dir is required")

    report = audit_rows(rows)
    report["audit_source"] = source
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--history-dir")
    parser.add_argument(
        "--output",
        default="soccer_edge_state/analysis/frozen_calibration_audit_v4.json",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.output, input_path=args.input, history_dir=args.history_dir),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
