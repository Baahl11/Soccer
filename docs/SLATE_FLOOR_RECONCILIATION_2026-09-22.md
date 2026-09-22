# Soccer Edge Slate Floor Reconciliation — 2026-09-22

## Problem

A healthy scheduler tick can still become operationally blind when the primary date slate is tiny. The observed failure mode was:

- `fixture_scan_count` was only `2`.
- Both fixtures failed the sport-first shortlist screen.
- No market calls were made, which is correct after the sport screen.
- The user-facing result was still bad because the engine had almost no universe to scan.

The issue is slate coverage, not market evaluation, staking, tiers, or model weights.

## Fix

`automation_v89` wraps the scheduler's paced API getter at the same point used by `automation_v6` adaptive pacing.

When the scheduler requests the raw API-Football date slate:

```text
fixtures?date=<today>&timezone=America/Mexico_City
```

it now checks the primary slate count. If the primary slate is below `SOCCER_EDGE_SLATE_FLOOR_MIN_FIXTURES` and budget allows, it performs one extra raw slate request:

```text
fixtures?date=<tomorrow>&timezone=America/Mexico_City
```

Then it merges both responses by `fixture.id` before the existing due-stage, coverage, sport-screen, shortlist, lineup and market logic runs.

## Defaults

```text
SOCCER_EDGE_SLATE_FLOOR_MIN_FIXTURES=12
SOCCER_EDGE_SLATE_FLOOR_MIN_DAILY_REMAINING=4000
```

The reconciliation is blocked when:

- the call is not the raw `fixtures?date` slate;
- the call is a team recent-form, fixture-id, live, league or other scoped fixtures request;
- the primary date is not today;
- local time is already `>= 22:00`, where the legacy path already includes tomorrow;
- the primary slate already meets the floor;
- the per-tick budget is exhausted;
- the daily remaining quota is at or below the configured floor.

## Safety guarantees

This change:

- adds at most one extra provider request per tick;
- only calls `fixtures?date=<tomorrow>`;
- does not call odds/markets;
- does not change model weights;
- does not change canonical bet logic;
- does not promote tiers;
- does not increase stakes;
- does not bypass sport-first screening.

## New telemetry

The tick payload now includes:

```text
slate_floor_reconciliation
slate_floor_triggered
slate_floor_reason
slate_floor_min_fixture_count
api_slate_reconciliation_calls
api_raw_slate_count
api_raw_reconciliation_slate_count
merged_slate_count
slate_source_policy
v362_provider_requests_added
v362_provider_request_scope
```

Expected behavior on a tiny morning slate:

```json
{
  "slate_floor_triggered": true,
  "slate_floor_reason": "PRIMARY_SLATE_BELOW_FLOOR_RECONCILED_WITH_NEXT_DATE",
  "api_slate_reconciliation_calls": 1,
  "api_raw_slate_count": 2,
  "api_raw_reconciliation_slate_count": 25,
  "merged_slate_count": 27
}
```

## Validation

Unit coverage lives in:

```text
tests/test_slate_floor_reconciliation.py
```

It verifies:

- only raw date slates are eligible;
- team recent-form `/fixtures` calls are excluded;
- fixture rows are deduped by `fixture.id`;
- tiny today slates reconcile when budget allows;
- slates meeting the floor do not reconcile;
- reduced daily budget blocks reconciliation.
