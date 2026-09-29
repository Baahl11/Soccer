# v201.15 — Frontend maturity wiring

This checkpoint wires the persisted maturation control plane into the Control Tower UI.

## Runtime wiring

- Render provides `STATE_REPO=Baahl11/Soccer` and `STATE_BRANCH=soccer-edge-state`.
- `maturity_snapshot_v4` remains read-only and loads persisted validation reports from the state branch.
- Existing Model Maturity gates now expose their real persisted counters instead of silently falling back to `N/V` when the state source is configured.
- The dashboard now renders the v201.14 maturation watchdog bundle, including status, reason, compact evidence, and source.
- The Model Maturity header exposes snapshot health and report coverage so broken state wiring is visible immediately.

## Safety invariants

- `provider_requests_added = 0`
- `production_promotion_allowed = false`
- no model changes
- no threshold changes
- no gate changes
- no decision-weight changes
- no provider-budget changes
- no strict-close semantic changes
