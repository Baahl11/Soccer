# V210 — Frozen Calibration Provenance

Date: 2026-09-29

## Objective

Prevent future evaluation leakage by freezing the identity of the exact calibration artifact used at decision time. Historical rows are never recalibrated with a newer artifact and no historical probability is recomputed.

## Historical audit

The v210 audit rebuilds the point-in-time signal ledger from the immutable scheduler history and then evaluates calibration provenance without mutating state.

Latest audited snapshot:

- ticks read: `1536`
- signal-ledger rows: `10758`
- unique fixtures: `2419`
- rows with Phase16 calibration provenance: `806`
- rows with applied Phase16 calibration in the rebuilt historical ledger: `0`
- rows with frozen calibrator artifact identity: `0`
- audit status: `NOT_VERIFIED_NO_CALIBRATED_ROWS`

This is intentionally not backfilled. Existing historical rows do not contain enough point-in-time evidence to prove which exact calibrator artifact was applied, so v210 refuses to infer or attach today's calibrator to old decisions.

## Prospective runtime behavior

`automation_v127` wraps the existing Phase16 calibration call. When and only when calibration is actually applied, it:

1. receives the exact in-memory calibration state used by the decision;
2. isolates the exact calibrator payload used for that family;
3. canonicalizes that payload as stable JSON;
4. computes a SHA-256 fingerprint;
5. attaches only immutable identity metadata to the same decision row;
6. marks the artifact as frozen at decision time.

Supported initial families:

- BTTS — binary Platt calibrator;
- FT Totals / Over 2.5 — binary Platt calibrator;
- 1X2 — multiclass temperature calibrator.

Raw calibrator parameters are not copied into the persistent signal ledger. The ledger retains the fingerprint, artifact kind/target, source model version, match flag, calibrator status, parameter-key names, and fingerprint basis.

## Signal-ledger schema

`build_signal_ledger.py` schema is now `1.4.0` and records:

- Phase16 calibration source;
- Phase16 calibration policy;
- calibrated probability fields available in the same tick;
- source-model diagnostics;
- sanitized frozen artifact identity;
- `calibrator_artifact_frozen_at_decision_time`;
- explicit flags that history was not recalibrated or recomputed.

## Verification

- V4 runtime suite #611: success — artifact fingerprint implementation.
- V4 runtime suite #612: success — `automation_v127` activated in `tick_worker`.
- V4 runtime suite #613: success — frozen artifact persistence in signal ledger.
- V210 dedicated audit workflow #4: success — tests + rebuilt historical ledger + read-only audit artifact.
- Render deploy `dep-dau5cjlg1s2s73bh4qd0`: live on the v210 HEAD.
- Render startup: Uvicorn started and application startup completed successfully.

The first post-deploy natural scheduler tick is intentionally left to the normal scheduler cadence so this verification does not consume additional provider calls. A tick with no applied calibration may legitimately report zero frozen artifacts; an artifact must never be fabricated merely to exercise the path.

## Safety invariants

- `provider_requests_added = 0`
- `provider_budget_changed = false`
- `decision_weight = 0.0`
- `production_promotion_allowed = false`
- `historical_rows_mutated = false`
- `historical_probabilities_recomputed = false`
- `model_weights_changed = false`
- `thresholds_changed = false`
- `gates_changed = false`
- `canonical_bet_logic_changed = false`
- `strict_close_semantics_changed = false`

## Next research block

V211 is the Market Residual Challenger. It will treat the de-vig market probability as an explicit benchmark and measure the residual `p_model_calibrated - p_market_devig` using only already-captured point-in-time rows. It remains research-only with zero decision weight and zero provider requests.