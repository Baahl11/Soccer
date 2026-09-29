# v204 — Maturation Control Tower

Closed the family-level maturation view in the read-only product dashboard.

## Families

1X2, BTTS, Team Totals, 1H, 2H, Corners, Cards and Player Props now expose verified evidence kind, current/target counters, unique fixtures when available, maturation status, source and explicit blocker when a watchdog applies.

## Monitoring

The control plane also carries report freshness, evidence age, 48h evidence growth and persistent-baseline store health. The frontend renders these counters in a dedicated `Maturation Control Tower` section before the roadmap.

## Safety invariants

- `provider_requests_added = 0`
- `decision_weight = 0`
- `production_promotion_allowed = false`
- no model, threshold, gate, strict-close, scheduler or provider-budget changes
