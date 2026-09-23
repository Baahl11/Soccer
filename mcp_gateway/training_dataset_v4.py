from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Iterable

from mcp_gateway import feature_snapshot_v4

SCHEMA_VERSION = "4.0.0"
DATASET_VERSION = "4.0.0"
POLICY = (
    "LATEST_VALID_FEATURE_SNAPSHOT_AT_OR_BEFORE_KICKOFF;"
    "FINAL_RESULT_TARGETS_ONLY;NO_MARKET_FIELDS;NO_POST_KICKOFF_FEATURES"
)


def _as_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def row_fingerprint(row: dict[str, Any]) -> str:
    payload = {key: value for key, value in row.items() if key != "row_fingerprint"}
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def build_row(
    *,
    snapshot: dict[str, Any],
    kickoff: str,
    final_status: str,
    home_goals: int,
    away_goals: int,
) -> dict[str, Any]:
    errors = feature_snapshot_v4.validate(snapshot)
    if errors:
        raise ValueError(f"INVALID_FEATURE_SNAPSHOT:{','.join(errors)}")

    captured_at = str(snapshot.get("captured_at") or "")
    if not captured_at or not kickoff:
        raise ValueError("CAPTURED_AT_AND_KICKOFF_REQUIRED")

    captured_dt = datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
    kickoff_dt = datetime.fromisoformat(str(kickoff).replace("Z", "+00:00"))
    if captured_dt > kickoff_dt:
        raise ValueError("FEATURE_SNAPSHOT_AFTER_KICKOFF")

    home = _as_int(home_goals)
    away = _as_int(away_goals)
    if home is None or away is None:
        raise ValueError("FINAL_GOALS_REQUIRED")

    values: dict[str, Any] = {}
    missing: dict[str, bool] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for key in sorted((snapshot.get("features") or {}).keys()):
        envelope = snapshot["features"][key]
        values[key] = envelope.get("value")
        missing[key] = envelope.get("value") is None
        provenance[key] = {
            "source": envelope.get("source"),
            "captured_at": envelope.get("captured_at"),
            "sample_n": envelope.get("sample_n"),
            "freshness": envelope.get("freshness"),
            "confidence": envelope.get("confidence"),
            "missing_reason": envelope.get("missing_reason"),
        }

    total = home + away
    row = {
        "dataset_version": DATASET_VERSION,
        "feature_schema_version": snapshot.get("schema_version"),
        "fixture_id": snapshot.get("fixture_id"),
        "feature_captured_at": captured_at,
        "kickoff": kickoff,
        "stage": snapshot.get("stage"),
        "model_version": snapshot.get("model_version"),
        "data_tier": snapshot.get("data_tier"),
        "features": values,
        "feature_missing": missing,
        "feature_provenance": provenance,
        "targets": {
            "final_status": final_status,
            "home_goals": home,
            "away_goals": away,
            "total_goals": total,
            "home_win": int(home > away),
            "draw": int(home == away),
            "away_win": int(away > home),
            "btts": int(home > 0 and away > 0),
            "over_1_5": int(total >= 2),
            "over_2_5": int(total >= 3),
            "over_3_5": int(total >= 4),
        },
        "leakage_policy": POLICY,
        "market_fields_included": False,
    }
    row["row_fingerprint"] = row_fingerprint(row)
    return row


def validate_row(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if row.get("dataset_version") != DATASET_VERSION:
        errors.append("DATASET_VERSION_MISMATCH")
    if row.get("feature_schema_version") != feature_snapshot_v4.SCHEMA_VERSION:
        errors.append("FEATURE_SCHEMA_VERSION_MISMATCH")
    if not row.get("fixture_id"):
        errors.append("FIXTURE_ID_REQUIRED")
    if row.get("market_fields_included") is not False:
        errors.append("MARKET_FIELDS_MUST_BE_EXCLUDED")
    if row.get("leakage_policy") != POLICY:
        errors.append("LEAKAGE_POLICY_MISMATCH")
    if not isinstance(row.get("features"), dict) or not row.get("features"):
        errors.append("FEATURE_VALUES_REQUIRED")
    if not isinstance(row.get("targets"), dict):
        errors.append("TARGETS_REQUIRED")
    try:
        captured_dt = datetime.fromisoformat(str(row.get("feature_captured_at")).replace("Z", "+00:00"))
        kickoff_dt = datetime.fromisoformat(str(row.get("kickoff")).replace("Z", "+00:00"))
        if captured_dt > kickoff_dt:
            errors.append("FEATURE_SNAPSHOT_AFTER_KICKOFF")
    except Exception:
        errors.append("INVALID_TIMESTAMPS")
    expected = row_fingerprint(row)
    if row.get("row_fingerprint") != expected:
        errors.append("ROW_FINGERPRINT_MISMATCH")
    return errors


def dataset_fingerprint(rows: Iterable[dict[str, Any]]) -> str:
    canonical = [
        {key: value for key, value in row.items() if key != "row_fingerprint"}
        for row in sorted(rows, key=lambda item: (int(item.get("fixture_id") or 0), str(item.get("feature_captured_at") or "")))
    ]
    return hashlib.sha256(_canonical_json(canonical).encode("utf-8")).hexdigest()


def load_rows(conn: Any, *, cutoff: str | None = None) -> list[dict[str, Any]]:
    params: list[Any] = [feature_snapshot_v4.SCHEMA_VERSION]
    cutoff_clause = ""
    if cutoff:
        cutoff_clause = " AND s.captured_at <= %s"
        params.append(cutoff)

    query = f"""
        SELECT DISTINCT ON (f.fixture_id)
            f.fixture_id,
            f.kickoff,
            s.payload,
            r.final_status,
            r.home_goals,
            r.away_goals
        FROM soccer_fixtures f
        JOIN soccer_results r ON r.fixture_id = f.fixture_id
        JOIN soccer_feature_snapshots s ON s.fixture_id = f.fixture_id
        WHERE s.schema_version = %s
          AND s.captured_at <= f.kickoff
          AND r.home_goals IS NOT NULL
          AND r.away_goals IS NOT NULL
          {cutoff_clause}
        ORDER BY f.fixture_id, s.captured_at DESC
    """

    rows: list[dict[str, Any]] = []
    with conn.cursor() as cur:
        cur.execute(query, tuple(params))
        for fixture_id, kickoff, snapshot, final_status, home_goals, away_goals in cur.fetchall():
            if not isinstance(snapshot, dict):
                continue
            row = build_row(
                snapshot=snapshot,
                kickoff=kickoff.isoformat() if hasattr(kickoff, "isoformat") else str(kickoff),
                final_status=str(final_status or ""),
                home_goals=int(home_goals),
                away_goals=int(away_goals),
            )
            if row.get("fixture_id") != fixture_id:
                continue
            if not validate_row(row):
                rows.append(row)
    return rows


def manifest(rows: list[dict[str, Any]], *, cutoff: str | None = None) -> dict[str, Any]:
    return {
        "dataset_version": DATASET_VERSION,
        "feature_schema_version": feature_snapshot_v4.SCHEMA_VERSION,
        "policy": POLICY,
        "row_count": len(rows),
        "fixture_count": len({row.get("fixture_id") for row in rows}),
        "cutoff": cutoff,
        "dataset_fingerprint": dataset_fingerprint(rows),
        "market_fields_included": False,
        "target_fields": [
            "home_goals",
            "away_goals",
            "total_goals",
            "home_win",
            "draw",
            "away_win",
            "btts",
            "over_1_5",
            "over_2_5",
            "over_3_5",
        ],
    }
