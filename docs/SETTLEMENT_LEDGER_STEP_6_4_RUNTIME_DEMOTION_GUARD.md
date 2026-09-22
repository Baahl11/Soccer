# Step 6.4 — Runtime BET Demotion Guard

Date: 2026-09-22
Branch: `fix/runtime-bet-demotion-guard-20260922`

## Purpose

After Step 6.3 backfilled final results and Step 6 settlement review rebuilt the settlement outputs, the data showed that several BET-classified markets should not remain actionable at runtime.

This step adds a conservative runtime guard that demotes targeted BET rows to WATCH/research.

It does **not** promote anything.

## Settlement basis

Latest settlement review after final-result backfill:

| Segment | Settled | Record | ROI |
|---|---:|---:|---:|
| FT_TOTALS BET | 7 | 3-4 | -0.5652u |
| 2H_TOTALS BET | 2 | 0-2 | -0.72u |
| 2H_BTTS BET | 1 | 0-1 | -0.36u |
| 1H_OTHER BET | 0 | 1 ungraded | 0.0u |
| FT_TOTALS LEAN | 8 | 8-0 | +3.91u |

Promotion review still says all market families are `HOLD_OR_DEMOTE_REVIEW` and blocked by sample size or non-positive ROI.

## Runtime behavior

Targeted rows are demoted only when:

- `classification == BET`
- market family is one of:
  - `FT_TOTALS`
  - `2H_TOTALS`
  - `2H_BTTS`
  - `1H_OTHER`

The guard sets:

- `classification = WATCH`
- `original_classification = BET`
- `bet_eligible = false`
- `tier = null`
- `stake_units = 0.0` when present
- appends `SETTLEMENT_GUARD_RUNTIME_DEMOTION_NEGATIVE_OR_INSUFFICIENT_SAMPLE` to notes/block reasons
- attaches `runtime_demotion_guard` metadata

## Explicit non-goals

This does not:

- promote LEAN to BET
- promote Tier B/A/S
- increase stakes
- change model weights
- call providers
- call odds endpoints
- call `/internal/tick`
- manually deploy Render

## LEAN preservation

`FT_TOTALS LEAN` remains untouched because it is positive so far but still below the minimum 20 settled decisions for Tier B review.

The correct behavior is to continue accumulating sample, not promote.

## Files changed

- `mcp_gateway/runtime_bet_demotion_guard.py`
- `mcp_gateway/automation_v88.py`
- `mcp_gateway/tick_worker.py`
- `tests/test_runtime_bet_demotion_guard.py`

## Validation expectations

Unit tests cover:

- FT_TOTALS BET demoted to WATCH
- 2H_TOTALS / 2H_BTTS / 1H_OTHER BET demoted
- LEAN rows preserved
- presentation `match_table_rows` demoted too
- non-target BET families preserved
