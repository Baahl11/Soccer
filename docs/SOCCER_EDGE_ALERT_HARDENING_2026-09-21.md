# Soccer Edge Alert Hardening — 2026-09-21

## Objective

Reduce noisy weekend backend alerts while preserving real failures from the natural scheduler. The monitor must read persisted state from the explicit GitHub branch `soccer-edge-state` and must never use the repository default branch as scheduler state.

## Source of truth

Primary heartbeat:

```text
ref: soccer-edge-state
path: soccer_edge_state/health.json
```

Secondary failure signal:

```text
ref: soccer-edge-state
path: soccer_edge_state/last_error.json
```

Optional analytical context only:

```text
ref: soccer-edge-state
path: soccer_edge_state/latest.json
```

`latest.json` must not be treated as the scheduler heartbeat. It may legitimately stay old when there are zero relevant events.

## Alert rules retained

Alert only when one of these is true:

1. `health.json` is missing or unreadable on `soccer-edge-state`.
2. `health.generated_at_local` is more than 75 minutes old during 07:00–23:00 America/Mexico_City.
3. `health.version` is below `3.60.0`.
4. `health.status` is not `ok`.
5. `last_error.json` exists on `soccer-edge-state` and is newer than the healthy heartbeat.
6. A newer PIPELINE_ERROR, HTTP 5xx/tick failure, rate-limit failure, or GitHub state-persistence failure is evident.
7. Natural scheduler advancement stops.

## Non-alert signals

These must not alert by themselves:

- `database_persisted=false`
- zero bets
- zero alerts
- zero events
- old `latest.json`
- missing state on the repository default branch

## Severity model

| Severity | Meaning | Examples |
|---|---|---|
| P1 | Scheduler/backend health is broken or stale | missing health, stale heartbeat, status not ok |
| P2 | Real backend failure after healthy heartbeat | HTTP 5xx, tick_failed, PIPELINE_ERROR, GitHub persistence failure |
| P3 | Provider/budget degradation | 429, rate-limit, request-limit pressure |
| OK | Current heartbeat is healthy | fresh `status=ok` health and no newer error |

## Dedupe model

Alerts are fingerprinted by normalized cause, not by timestamp. This avoids repeating the same `_american`/HTTP 500 error every check.

Default dedupe window: 4 hours.

A repeated alert is suppressed unless:

- the cause changes,
- severity changes,
- the dedupe window expires,
- the backend recovers and later fails again,
- or the monitor crosses a stronger stale threshold.

## New monitor

Added:

```text
scripts/soccer_edge_health_monitor.py
```

The script:

- reads `health.json` first from `soccer-edge-state`,
- optionally reads `last_error.json` from the same branch,
- intentionally ignores `latest.json` as a health signal,
- prints an alert only for real unhealthy/stale conditions,
- returns exit code `2` when an alert should notify,
- returns exit code `0` when healthy or duplicate-suppressed,
- supports JSON output with `--json`,
- supports `--no-dedupe` for diagnostics,
- does not trigger ticks, provider requests, deployments, or code changes.

Example:

```bash
python scripts/soccer_edge_health_monitor.py --repo Baahl11/Soccer --ref soccer-edge-state --json
```

## Tests

Added pure unit tests:

```text
tests/test_soccer_edge_health_monitor.py
```

Covered cases:

- fresh `status=ok` health does not alert with zero bets/events and `database_persisted=false`,
- missing health alerts,
- stale health alerts during active window,
- version below `3.60.0` alerts,
- non-ok health status alerts,
- newer HTTP 500/tick failure alerts,
- older `last_error.json` does not alert,
- 429/rate-limit classified as P3,
- fingerprint normalizes timestamps,
- duplicate alerts are suppressed,
- old `latest.json` is not part of health evaluation.

## Remaining backend issue

The weekend error:

```text
module 'mcp_gateway.automation_v2' has no attribute '_american'
```

was visible in persisted `last_error.json`, but the defining backend module was not present in this repository's visible default-branch code. The scheduler workflow calls the Render endpoint:

```text
https://soccer-edge-api.onrender.com/internal/tick
```

Therefore this PR hardens monitoring and alert quality in this repo, but the actual `_american` attribute bug must be patched in the backend service/repository that implements `/internal/tick`.
