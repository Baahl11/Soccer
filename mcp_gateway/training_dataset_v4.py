from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone as dt_timezone
from typing import Any, Iterable

from mcp_gateway import feature_snapshot_v4
from mcp_gateway import persistence as persistence_base

SCHEMA_VERSION = "4.0.0"
DATASET_VERSION = "4.0.0"
POLICY = (
    "LATEST_VALID_FEATURE_SNAPSHOT_STRICTLY_BEFORE_KICKOFF;"
    "FINAL_RESULT_GRADED_BY_BUILD_CUTOFF_ONLY;NO_MARKET_FIELDS;"
    "NO_POST_KICKOFF_FEATURES;DETERMINISTIC_FIXTURE_ORDER"
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
    if captured_dt >= kickoff_dt:
        raise ValueError("FEATURE_SNAPSHOT_NOT_STRICTLY_BEFORE_KICKOFF")

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
        if captured_dt >= kickoff_dt:
            errors.append("FEATURE_SNAPSHOT_NOT_STRICTLY_BEFORE_KICKOFF")
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
    build_cutoff = cutoff or datetime.now(dt_timezone.utc).isoformat()
    params: list[Any] = [
        feature_snapshot_v4.SCHEMA_VERSION,
        build_cutoff,
        build_cutoff,
    ]

    query = """
        SELECT DISTINCT ON (f.fixture_id)
            f.fixture_id,
            f.kickoff,
            s.snapshot_id,
            s.payload,
            r.final_status,
            r.home_goals,
            r.away_goals
        FROM soccer_fixtures f
        JOIN soccer_results r ON r.fixture_id = f.fixture_id
        JOIN soccer_feature_snapshots s ON s.fixture_id = f.fixture_id
        WHERE s.schema_version = %s
          AND s.captured_at < f.kickoff
          AND s.captured_at <= %s
          AND r.graded_at IS NOT NULL
          AND r.graded_at <= %s
          AND r.home_goals IS NOT NULL
          AND r.away_goals IS NOT NULL
        ORDER BY f.fixture_id, s.captured_at DESC, s.snapshot_id DESC
    """

    rows: list[dict[str, Any]] = []
    with conn.cursor() as cur:
        cur.execute(query, tuple(params))
        for fixture_id, kickoff, snapshot_id, snapshot, final_status, home_goals, away_goals in cur.fetchall():
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
            row["snapshot_id"] = int(snapshot_id)
            row["row_fingerprint"] = row_fingerprint(row)
            if not validate_row(row):
                rows.append(row)
    rows.sort(key=lambda item: (int(item.get("fixture_id") or 0), int(item.get("snapshot_id") or 0)))
    return rows


def manifest(rows: list[dict[str, Any]], *, cutoff: str | None = None) -> dict[str, Any]:
    feature_names = sorted({
        key
        for row in rows
        for key in (row.get("features") or {}).keys()
    })
    missing_counts = {
        key: sum(1 for row in rows if (row.get("features") or {}).get(key) is None)
        for key in feature_names
    }
    return {
        "dataset_version": DATASET_VERSION,
        "feature_schema_version": feature_snapshot_v4.SCHEMA_VERSION,
        "policy": POLICY,
        "row_count": len(rows),
        "fixture_count": len({row.get("fixture_id") for row in rows}),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "missing_counts": missing_counts,
        "cutoff": cutoff,
        "dataset_fingerprint": dataset_fingerprint(rows),
        "market_fields_included": False,
        "post_kickoff_features_allowed": False,
        "critical_missingness_imputed": False,
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


def backfill_feature_snapshots(conn: Any, *, limit: int = 5000) -> dict[str, int]:
    """Derive missing v4 snapshots from historical point-in-time refresh payloads.

    No provider calls and no future information are used. Only refresh events
    without an equivalent v4 feature snapshot are selected, newest first, so
    current OOS evidence cannot be starved by an old fixed prefix of events.
    Missing advanced features remain explicitly missing under feature_snapshot_v4.
    """
    scanned = valid = inserted = invalid = 0
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT e.event_id, e.generated_at, e.payload
            FROM soccer_refresh_events e
            WHERE e.event_type='SOCCER_REFRESH'
              AND e.stage <> 'POSTGAME'
              AND NOT EXISTS (
                  SELECT 1
                  FROM soccer_feature_snapshots s
                  WHERE s.fixture_id = e.fixture_id
                    AND s.captured_at = e.generated_at
                    AND COALESCE(s.stage, '') = COALESCE(e.stage, '')
                    AND s.schema_version = %s
              )
            ORDER BY e.generated_at DESC, e.event_id DESC
            LIMIT %s
            """,
            (feature_snapshot_v4.SCHEMA_VERSION, int(limit)),
        )
        source_rows = cur.fetchall()

    insert_rows: list[tuple[Any, ...]] = []
    for _event_id, generated_at, payload in source_rows:
        scanned += 1
        if not isinstance(payload, dict):
            invalid += 1
            continue
        tick = {
            "generated_at_utc": generated_at.isoformat() if hasattr(generated_at, "isoformat") else str(generated_at),
            "model_version": payload.get("model_version"),
        }
        snapshot = feature_snapshot_v4.build(tick, payload)
        errors = feature_snapshot_v4.validate(snapshot)
        if errors:
            invalid += 1
            continue
        valid += 1
        insert_rows.append((
            snapshot.get("fixture_id"),
            snapshot.get("captured_at"),
            snapshot.get("stage"),
            snapshot.get("schema_version"),
            snapshot.get("model_version"),
            snapshot.get("data_tier"),
            snapshot.get("feature_count", 0),
            snapshot.get("missing_feature_count", 0),
            _canonical_json(snapshot),
        ))

    if insert_rows:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO soccer_feature_snapshots (
                    fixture_id, captured_at, stage, schema_version,
                    model_version, data_tier, feature_count,
                    missing_feature_count, payload
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                ON CONFLICT (fixture_id, captured_at, stage, schema_version) DO NOTHING
                """,
                insert_rows,
            )
            inserted = max(int(cur.rowcount or 0), 0)

    return {
        "scanned_refresh_events": scanned,
        "valid_snapshots": valid,
        "inserted_snapshots": inserted,
        "invalid_snapshots": invalid,
        "backfill_limit": int(limit),
    }


def persist_materialized_dataset(
    conn: Any,
    rows: list[dict[str, Any]],
    dataset_manifest: dict[str, Any],
) -> str:
    cutoff = str(dataset_manifest.get("cutoff") or "")
    digest = str(dataset_manifest.get("dataset_fingerprint") or "")
    compact_cutoff = (
        cutoff.replace("-", "").replace(":", "").replace("+00:00", "Z").replace(".", "")
        or "NO_CUTOFF"
    )
    build_id = f"v4-{compact_cutoff}-{digest[:12]}"

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO soccer_training_dataset_builds (
                build_id, dataset_version, feature_schema_version, as_of,
                row_count, feature_count, dataset_sha256, selection_policy, manifest
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
            ON CONFLICT (build_id) DO UPDATE SET
                row_count=EXCLUDED.row_count,
                feature_count=EXCLUDED.feature_count,
                dataset_sha256=EXCLUDED.dataset_sha256,
                selection_policy=EXCLUDED.selection_policy,
                manifest=EXCLUDED.manifest
            """,
            (
                build_id,
                DATASET_VERSION,
                feature_snapshot_v4.SCHEMA_VERSION,
                cutoff,
                dataset_manifest.get("row_count", 0),
                dataset_manifest.get("feature_count", 0),
                digest,
                POLICY,
                _canonical_json(dataset_manifest),
            ),
        )
        cur.execute("DELETE FROM soccer_training_dataset_rows WHERE build_id=%s", (build_id,))
        for index, row in enumerate(rows, start=1):
            cur.execute(
                """
                INSERT INTO soccer_training_dataset_rows (
                    build_id, row_number, fixture_id, snapshot_id,
                    snapshot_captured_at, kickoff, league_id, season, stage,
                    features, missingness, provenance, targets, row_payload, row_sha256
                ) VALUES (
                    %s,%s,%s,%s,%s,%s,%s,%s,%s,
                    %s::jsonb,%s::jsonb,%s::jsonb,%s::jsonb,%s::jsonb,%s
                )
                """,
                (
                    build_id,
                    index,
                    row.get("fixture_id"),
                    row.get("snapshot_id"),
                    row.get("feature_captured_at"),
                    row.get("kickoff"),
                    None,
                    None,
                    row.get("stage"),
                    _canonical_json(row.get("features") or {}),
                    _canonical_json(row.get("feature_missing") or {}),
                    _canonical_json(row.get("feature_provenance") or {}),
                    _canonical_json(row.get("targets") or {}),
                    _canonical_json(row),
                    row.get("row_fingerprint"),
                ),
            )
    return build_id


def build_and_persist(*, cutoff: str | None = None, backfill_limit: int = 5000) -> dict[str, Any]:
    build_cutoff = cutoff or datetime.now(dt_timezone.utc).isoformat()
    persistence_base.ensure_schema()
    with persistence_base._connect() as conn:
        backfill = backfill_feature_snapshots(conn, limit=backfill_limit)
        rows = load_rows(conn, cutoff=build_cutoff)
        dataset_manifest = manifest(rows, cutoff=build_cutoff)
        build_id = persist_materialized_dataset(conn, rows, dataset_manifest)

    return {
        "status": "OK",
        "build_id": build_id,
        "manifest": dataset_manifest,
        "backfill": backfill,
        "persisted_row_count": len(rows),
        "provider_requests_added": 0,
        "market_fields_included": False,
    }
