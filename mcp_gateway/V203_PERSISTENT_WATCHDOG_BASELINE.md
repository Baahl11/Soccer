# v203 — Persistent maturation watchdog baseline

The 48h maturation baseline now survives Render restarts/deploys.

- Added Postgres-backed `maturation_watchdog_state` with one keyed JSONB baseline.
- Load/save is fail-open and observability-only.
- The store creates its table with `CREATE TABLE IF NOT EXISTS` on first use.
- The in-process cache remains an optimization; Postgres is the durable source across restarts.
- Store health is surfaced in `watchdog_baseline_store`.

Safety invariants: provider_requests_added=0, production_promotion_allowed=false, decision_weight=0, no model/threshold/gate/budget changes.
