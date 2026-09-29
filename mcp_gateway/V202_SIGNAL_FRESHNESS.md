# v202 — Signal freshness semantics

Closed the false-stale ambiguity in maturation telemetry without changing any production decision path.

## Changes

- `signal_close_freshness` now measures the persisted report artifact timestamp, not the timestamp of the newest historical evidence row.
- The latest Git commit time for `soccer_edge_state/analysis/signal_ledger_summary.json` is attached in-memory as `report_updated_at_utc`.
- Historical ledger timing remains preserved as `first_evidence_at_local` / `last_evidence_at_local`.
- Added a separate `signal_evidence_age` watchdog so a freshly rebuilt report can truthfully coexist with old evidence.
- If artifact commit metadata cannot be verified, freshness becomes `NOT_VERIFIED`; it is never inferred from evidence age.

## Safety invariants

- `provider_requests_added = 0`
- `decision_weight = 0`
- `production_promotion_allowed = false`
- no model, threshold, gate, strict-close, scheduling or provider-budget changes
