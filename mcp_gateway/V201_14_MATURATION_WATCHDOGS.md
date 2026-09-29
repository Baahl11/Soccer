# v201.14 — Maturation watchdogs

This checkpoint closes the v201 maturation-control observability loop without changing the production decision engine.

## Watchdogs

The maturity snapshot now exposes read-only watchdogs for:

- stale signal/close telemetry (48-hour window when a source timestamp is available);
- 2H strict-close evidence remaining at zero;
- Corners formation-adjusted joins remaining at zero;
- Cards settlement maturity blocked by missing observed match-card price history or missing settlement evidence;
- Player Props families with priced rows but missing confirmed-XI/player alignment;
- no maturation evidence growth for 48 hours;
- provider requests increasing for 48 hours without maturation evidence growth.

Temporal growth watchdogs initialize an in-process baseline and remain `NOT_VERIFIED` until a comparable observation exists. Evidence growth or counter resets restart the observation window. A process restart also restarts that baseline, so the watchdog never fabricates a 48-hour claim it cannot prove.

## Data sources

Only persisted state reports are read:

- V4 validation reports for 1X2, BTTS, Team Totals, 1H, 2H and Corners;
- Phase14 Cards validation;
- Phase15 Player Props validation;
- signal ledger summary;
- settlement coverage report;
- API-efficiency report.

No API-Football call is made by the watchdog layer.

## Invariants

- provider request cap unchanged
- `provider_requests_added = 0`
- canonical bet logic unchanged
- model weights unchanged
- thresholds and gates unchanged
- decision weights unchanged
- strict-close semantics unchanged
- production promotion remains disabled/manual
- missing evidence returns `NOT_VERIFIED`; it is never inferred
