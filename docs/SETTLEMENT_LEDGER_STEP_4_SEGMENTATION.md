# Settlement Ledger Step 4 — Performance Segmentation

Date: 2026-09-21
Branch: `fix/settlement-ledger-v1-20260921`
Base: `soccer-edge-mcp-v1`

## Goal

Step 4 converts the normalized settlement ledger into diagnostic segment reports. This is required before promoting any market family, classification or stage window into Tier B/A/S.

The report is intentionally **research-only**. It does not upgrade picks, stake sizes or classifications by itself.

## New script

`mcp_gateway/analyze_settlement_segments.py`

Default input:

```text
soccer_edge_state/analysis/bet_settlement_ledger.jsonl
```

Default output:

```text
soccer_edge_state/analysis/settlement_segments_summary.json
```

## Segments generated

The script summarizes settled performance by:

- classification
- market family
- market family + classification
- market family + stage bucket
- market family + odds band
- market family + data tier
- market family + league
- market family + stage bucket + odds band
- market family + classification + stage bucket

## Stage buckets

- `T_MINUS_0_30`
- `T_MINUS_31_60`
- `T_MINUS_61_90`
- `T_MINUS_91_PLUS`
- `EARLY_RESEARCH`
- `CLOSE`
- `POSTGAME`
- `UNKNOWN_STAGE`

## Odds bands

- `LT_1_50`
- `1_50_1_79`
- `1_80_2_19`
- `2_20_2_99`
- `GE_3_00`
- `NO_PRICE`

## Segment review policy

Each segment receives a conservative diagnostic status:

- `INSUFFICIENT_SAMPLE`: fewer than 10 settled decisions.
- `DIRECTIONAL_ONLY`: 10-19 settled decisions.
- `HOLD_NO_DECIDED_OUTCOMES`: no win/loss after pushes.
- `HOLD_OR_DEMOTE_CANDIDATE`: 20+ settled decisions with non-positive ROI.
- `PROMOTION_REVIEW_INPUT`: 20+ settled decisions and positive ROI.

`PROMOTION_REVIEW_INPUT` is **not** a promotion. It means the segment can be reviewed with CLV, league stability, odds-band stability and manual sanity checks.

## Safety

- No provider requests.
- No manual tick.
- No deploy.
- No automatic tier promotion.
- No stake-size changes.

## Workflow note

This script is ready to be wired after `evaluate_postgame.py` has produced `bet_settlement_ledger.jsonl`. If the active GitHub Actions workflow is managed from another branch, add this step after postgame evaluation:

```yaml
- name: Segment settlement performance
  run: python engine/mcp_gateway/analyze_settlement_segments.py --settlement-ledger state/soccer_edge_state/analysis/bet_settlement_ledger.jsonl --output state/soccer_edge_state/analysis/settlement_segments_summary.json
```

The output path is under `soccer_edge_state/analysis/`, so the existing `git add -f soccer_edge_state/analysis/` persist step will include it.
