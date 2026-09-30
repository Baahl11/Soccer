# V209 — Post-v208 Maturation Verification

## Purpose

Verify from the final runtime tick that the v208 monotonic hard provider cap remains respected while reserved price-resolution capacity is actually usable.

This is an observability-only checkpoint. It does not add provider requests, create maturation candidates, change models, thresholds, gates, tiers, stakes, promotion state, or strict-close semantics.

## Pre-v209 evidence

The first persisted tick after v208 was generated at `2026-09-29T16:55:51.852069-06:00` and showed:

- `api_calls_this_tick = 23`
- `max_api_calls_per_tick = 45`
- `effective_max_api_calls_per_tick = 45`
- `price_resolution_v4.api_calls_added = 5`
- `price_resolution_v4.primary_clv_maturation_api_calls_added = 1`
- one BTTS paid-entry research row surfaced from an already-paid `/odds` response
- database persistence succeeded

A later stress tick exposed an intermittent cap-path inconsistency, which led to v209.1-v209.3 observability and explicit phase-release work. The system was not declared closed on that anomalous tick.

## Final live verification

The persisted tick generated at `2026-09-29T17:58:55.383632-06:00` closes v209:

- `v209_post_reserve_verification.status = LIVE_VERIFIED`
- `api_calls_this_tick = 13`
- `configured_cap = 45`
- `effective_cap = 45`
- `cap_respected = true`
- `provider_headroom_after_tick = 32`
- `price_resolution_v4.api_calls_added = 4`
- `primary_clv_maturation_api_calls_added = 3`
- reserved price phase release executed: `25 -> 45`
- release occurred at provider-call count `9`
- roadmap-priority backlog visible in the same tick: `1X2=2`, `BTTS=2`, `FT_TOTALS=3`
- database persistence succeeded

This is the required proof that reserved capacity can be released for the price-resolution phase while the final tick stays inside the same global cap.

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

## Roadmap consequence

V209 is closed. Continue natural maturation in the existing order (`1X2 -> BTTS -> FT Totals -> Team Totals -> 1H -> Corners -> 2H -> Cards/Player Props`) while the zero-call research work from `V209_RESEARCH_LEDGER_AND_MARKET_BASELINE.md` proceeds in parallel. The first parallel research block is the frozen point-in-time calibration/provenance audit.