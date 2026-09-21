# Settlement Ledger Step 6.2 — Postgame Final-Result Backfill Queue

## Purpose

Step 6 showed no runtime promotion is allowed yet because no market family reached the Tier B minimum of 20 settled decisions. Step 6.1 showed the remaining unsettled BET/LEAN backlog is mostly healthy:

- rows already in settlement ledger
- duplicate older snapshots
- a small number of `PENDING_FINAL_RESULT` rows

Step 6.2 turns only the `PENDING_FINAL_RESULT` rows into a fixture-level backfill queue.

## New script

```bash
python mcp_gateway/build_postgame_backfill_queue.py \
  --backlog soccer_edge_state/analysis/settlement_backlog.jsonl \
  --queue-output soccer_edge_state/analysis/postgame_final_backfill_queue.jsonl \
  --summary-output soccer_edge_state/analysis/postgame_final_backfill_queue_summary.json
```

## Outputs

### `postgame_final_backfill_queue.jsonl`

One JSON row per unique `fixture_id` needing final-result backfill.

Each row includes:

- fixture id
- kickoff
- league
- teams
- source classifications
- market families
- source decision rows
- priority score
- provider hint: `api-football` / `fixtures` / `id=<fixture_id>`
- safety flags

### `postgame_final_backfill_queue_summary.json`

Summary includes:

- pending backlog rows
- unique fixture count
- max provider calls needed
- classification counts
- market-family counts
- priority policy

## Safety

This step is queue generation only.

It does **not**:

- call API-Football or provider APIs
- call `/internal/tick`
- deploy Render
- change runtime picks
- change tiers
- change stakes
- change model weights

## Next guarded step

A later, separate PR can add a budget-guarded runner that consumes this queue and fetches final results using:

```text
GET fixtures?id=<fixture_id>
```

That runner must have provider budget guards and should write final results back to state/history before the next settlement review.

## Promotion rule remains unchanged

No runtime promotion is allowed until `settlement_promotion_review.json` shows a market family/classification has:

- 20+ settled decisions for Tier B review
- positive ROI
- CLV review
- stage/league/odds-band stability

