# Settlement Ledger Step 6.1 — Coverage Audit

## Purpose

Step 6 showed that no runtime promotion is allowed yet because the settled sample is below the Tier B minimum:

- FT_1X2: 10 settled
- FT_TOTALS: 14 settled
- Tier B review minimum: 20 settled

Step 6.1 does **not** promote tiers. It increases observability so we can grow the settleable sample correctly.

## New script

```bash
python engine/mcp_gateway/analyze_settlement_coverage.py \
  --ledger state/soccer_edge_state/analysis/signal_ledger.jsonl \
  --postgame-evaluation state/soccer_edge_state/analysis/postgame_evaluation.jsonl \
  --settlement-ledger state/soccer_edge_state/analysis/bet_settlement_ledger.jsonl \
  --output state/soccer_edge_state/analysis/settlement_coverage_report.json \
  --backlog-output state/soccer_edge_state/analysis/settlement_backlog.jsonl
```

## Outputs

### `settlement_coverage_report.json`

Aggregates BET/LEAN settlement coverage:

- source actionable BET/LEAN rows
- rows already in settlement ledger
- coverage rate
- backlog count
- reason counts
- coverage by classification
- coverage by market family + reason
- recommended next actions

### `settlement_backlog.jsonl`

One row per BET/LEAN source row not yet represented as a settled decision.

Reasons include:

- `PENDING_FINAL_RESULT`
- `MISSING_BEST_MARKET_WITH_FINAL`
- `PENDING_FINAL_AND_MISSING_MARKET`
- `DUPLICATE_OLDER_SNAPSHOT`
- `UNSETTLED_<reason>`
- `FINAL_WITH_MARKET_NOT_EVALUATED`
- `FINAL_WITH_MARKET_NOT_IN_SETTLEMENT_LEDGER`

## Safety

This step is diagnostic only.

It does not:

- call `/internal/tick`
- call API-Football/provider APIs
- deploy Render
- promote tiers
- change stake sizing
- change runtime pick classification
- change model weights

## How this helps sample growth

The system currently has more BET/LEAN source rows than settled decisions. This audit makes the missing rows actionable:

- pending finals should settle naturally after final result arrives
- missing market metadata should be fixed at signal emission time
- duplicate older snapshots are expected and should not count twice
- unsupported/ungraded markets require explicit grading metadata before they can contribute to ROI

## Promotion rule remains unchanged

No segment should move to runtime promotion until `settlement_promotion_review.json` reports a Tier B/A/S manual-review candidate with enough sample, positive ROI, non-negative CLV and stability checks.
