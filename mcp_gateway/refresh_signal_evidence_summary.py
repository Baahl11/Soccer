from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


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


def _iter_ticks(paths: Iterable[Path]):
    for path in paths:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    tick = json.loads(line)
                except Exception:
                    continue
                if isinstance(tick, dict):
                    yield tick


def latest_persisted_event_tick(paths: Iterable[Path]) -> dict[str, Any] | None:
    latest: dict[str, Any] | None = None
    latest_dt: datetime | None = None
    for tick in _iter_ticks(paths):
        try:
            event_count = int(tick.get("event_count") or 0)
        except (TypeError, ValueError):
            event_count = 0
        if event_count <= 0 or tick.get("database_persisted") is not True:
            continue
        candidate_dt = _parse_timestamp(tick.get("generated_at_utc") or tick.get("generated_at_local"))
        if candidate_dt is None:
            continue
        if latest_dt is None or candidate_dt > latest_dt:
            latest_dt = candidate_dt
            latest = tick
    return latest


def refresh_summary(summary: dict[str, Any], latest_tick: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(summary)
    # Evidence freshness is an additive metadata refresh. Never downgrade a
    # newer summary schema produced by the canonical ledger materializer.
    out.setdefault("schema_version", "1.5.0")
    out.setdefault("last_materialized_ledger_row_at_local", out.get("last_generated_at_local"))
    out.setdefault("last_materialized_ledger_row_at_utc", out.get("last_generated_at_utc"))

    previous_value = out.get("last_evidence_at_utc") or out.get("last_evidence_at_local") or out.get("last_generated_at_utc") or out.get("last_generated_at_local")
    previous_dt = _parse_timestamp(previous_value)
    evidence_advanced = False

    if latest_tick is not None:
        latest_utc = latest_tick.get("generated_at_utc")
        latest_local = latest_tick.get("generated_at_local")
        latest_dt = _parse_timestamp(latest_utc or latest_local)
        if latest_dt is not None and (previous_dt is None or latest_dt > previous_dt):
            out["last_evidence_at_utc"] = latest_utc
            out["last_evidence_at_local"] = latest_local
            out["last_evidence_source"] = "COMPACT_HISTORY_EVENT_COUNT_WITH_POSTGRES_PERSISTED_TRUE"
            out["last_evidence_event_count"] = int(latest_tick.get("event_count") or 0)
            out["last_evidence_database_persisted"] = True
            out["last_evidence_pipeline_version"] = latest_tick.get("version")
            out["last_evidence_model_version"] = latest_tick.get("model_version")
            evidence_advanced = True

    if not out.get("last_evidence_at_local"):
        out["last_evidence_at_local"] = out.get("last_generated_at_local")
    if not out.get("last_evidence_at_utc"):
        out["last_evidence_at_utc"] = out.get("last_generated_at_utc")
    if not out.get("last_evidence_source") and (out.get("last_evidence_at_local") or out.get("last_evidence_at_utc")):
        out["last_evidence_source"] = "MATERIALIZED_SIGNAL_LEDGER_ROW"

    # Artifact freshness is intentionally separate from eligible-evidence age.
    # Only stamp the artifact when this refresh actually advances persisted
    # eligible evidence; do not mirror the evidence timestamp or synthesize it.
    if evidence_advanced:
        out["report_updated_at_utc"] = datetime.now(timezone.utc).isoformat()

    out["evidence_freshness_semantics"] = (
        "last_generated_at tracks the newest materialized signal-ledger row; "
        "last_evidence_at may advance from a compact-history tick only when "
        "event_count>0 and database_persisted=true; report_updated_at_utc tracks "
        "the artifact refresh itself. No event/CLOSE/result row is synthesized."
    )
    out["provider_requests_added"] = 0
    out["canonical_bet_logic_changed"] = False
    out["model_weights_changed"] = False
    out["historical_probabilities_recomputed"] = False
    out["historical_rows_recalibrated"] = False
    out["synthetic_close_rows_added"] = 0
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--history-file", action="append", default=[])
    args = parser.parse_args()

    summary_path = Path(args.summary)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    paths = [Path(value) for value in args.history_file]
    latest = latest_persisted_event_tick(paths)
    updated = refresh_summary(summary, latest)
    summary_path.write_text(json.dumps(updated, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema_version": updated.get("schema_version"),
        "report_updated_at_utc": updated.get("report_updated_at_utc"),
        "last_materialized_ledger_row_at_local": updated.get("last_materialized_ledger_row_at_local"),
        "last_evidence_at_local": updated.get("last_evidence_at_local"),
        "last_evidence_at_utc": updated.get("last_evidence_at_utc"),
        "last_evidence_source": updated.get("last_evidence_source"),
        "last_evidence_event_count": updated.get("last_evidence_event_count"),
        "synthetic_close_rows_added": updated.get("synthetic_close_rows_added"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
