# Soccer Edge Settlement Ledger — Step 6.3 Guarded Final-Result Backfill Runner

Date: 2026-09-21
Branch target: `soccer-edge-mcp-v1`

## Purpose

Step 6.3 consumes the queue generated in Step 6.2:

- `soccer_edge_state/analysis/postgame_final_backfill_queue.jsonl`

and can fetch missing final results for pending BET/LEAN settlement rows by fixture id.

The runner is intentionally narrow:

- provider: API-Football
- endpoint: `fixtures`
- params: `id=<fixture_id>`
- maximum calls: 5
- no odds requests
- no `/internal/tick`
- no Render deployment
- no runtime pick/stake/tier/model changes

## New script

```bash
python mcp_gateway/run_postgame_final_backfill.py \
  --queue soccer_edge_state/analysis/postgame_final_backfill_queue.jsonl \
  --history-output soccer_edge_state/history/postgame_final_backfill.jsonl \
  --report-output soccer_edge_state/analysis/postgame_final_backfill_run_report.json \
  --max-calls 5
```

By default, this is a dry-run. It writes only the report and makes **zero provider calls**.

## Execute mode

Actual API-Football calls require both:

1. `API_FOOTBALL_KEY` in the environment.
2. Explicit `--execute` flag.

```bash
API_FOOTBALL_KEY=... python mcp_gateway/run_postgame_final_backfill.py \
  --queue soccer_edge_state/analysis/postgame_final_backfill_queue.jsonl \
  --history-output soccer_edge_state/history/postgame_final_backfill.jsonl \
  --report-output soccer_edge_state/analysis/postgame_final_backfill_run_report.json \
  --max-calls 5 \
  --execute
```

## Outputs

### `postgame_final_backfill_run_report.json`

Includes:

- selected fixture ids
- provider calls attempted
- provider calls succeeded
- final results appended
- skipped fixtures
- per-fixture errors
- safety note

### `history/postgame_final_backfill.jsonl`

Only written when final results are returned.

Each appended history tick contains events with:

- `event_type=POSTGAME_FINAL_BACKFILL`
- `classification=POSTGAME`
- `stage=POSTGAME`
- `bet_eligible=false`
- `stake_units=null`
- `result.goals`
- `result.score`
- `fixture.status`

This format is compatible with `build_signal_ledger.py`, which can convert final fixture results into settlement-ready `result` rows.

## Safety contract

This runner must never:

- call odds endpoints
- create new picks
- promote/demote runtime tiers
- change stakes
- change model weights
- call `/internal/tick`
- trigger Render deploys manually

If `--execute` is omitted, provider calls must remain zero.

If `API_FOOTBALL_KEY` is missing in execute mode, the runner writes a blocked report and exits without provider calls.

## Promotion policy remains unchanged

Even after backfill succeeds, runtime promotion remains blocked until the next settlement review confirms:

- 20+ settled decisions for Tier B review
- positive ROI
- CLV review
- stage/league/odds-band stability

Step 6.3 only closes missing results; it does not promote anything.
