# V209 — Post-v208 Maturation Verification

## Purpose

Verify from the final runtime tick that the v208 monotonic hard provider cap remains respected while reserved price-resolution capacity is actually usable.

This is an observability-only checkpoint. It does not add provider requests, create maturation candidates, change models, thresholds, gates, tiers, stakes, promotion state, or strict-close semantics.

## Pre-v209 live evidence

The first persisted tick after v208 was generated at `2026-09-29T16:55:51.852069-06:00` and showed:

- `api_calls_this_tick = 23`
- `max_api_calls_per_tick = 45`
- `effective_max_api_calls_per_tick = 45`
- `price_resolution_v4.api_calls_added = 5`
- `price_resolution_v4.primary_clv_maturation_api_calls_added = 1`
- one BTTS paid-entry research row was surfaced from an already-paid `/odds` response
- database persistence succeeded

Therefore v208 did not reproduce the previous 55/55 upstream starvation pattern: the hard cap was respected and price-resolution work still executed.

## Runtime output

`automation_v126` adds `v209_post_reserve_maturation_verification` and also nests the same report at `price_resolution_v4.v209_post_reserve_verification`, which means the existing compact scheduler-state projection persists it without a main-workflow schema change.

Status values:

- `LIVE_VERIFIED` — cap telemetry exists, the cap was respected, and price-resolution calls executed.
- `WAITING_PRICE_ACTIVITY` — cap was respected but no price-resolution call was needed in that tick.
- `CAP_VIOLATION_DETECTED` — final provider calls exceeded the effective/configured cap.
- `NOT_VERIFIED_NO_CAP_TELEMETRY` — required cap telemetry was absent.

## Safety invariants

- `provider_requests_added = 0`
- `provider_budget_changed = false`
- `decision_weight = 0.0`
- `production_promotion_allowed = false`
- `model_weights_changed = false`
- `thresholds_changed = false`
- `gates_changed = false`
- `canonical_bet_logic_changed = false`
- `strict_close_semantics_changed = false`
