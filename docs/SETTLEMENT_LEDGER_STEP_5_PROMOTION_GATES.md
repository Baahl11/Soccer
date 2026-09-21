# Settlement Ledger Step 5 — Promotion/Demotion Review Gates

Date: 2026-09-21
Branch: `fix/settlement-ledger-v1-20260921`
Target base: `soccer-edge-mcp-v1`

## Purpose

Step 5 converts settled performance segmentation into **review candidates** for Tier B/A/S. It does not alter runtime pick tiers, model weights, classifications or stakes.

The goal is to prevent premature promotion from small samples or noisy ROI.

## New script

```bash
python engine/mcp_gateway/review_settlement_promotion_gates.py \
  --segments-summary state/soccer_edge_state/analysis/settlement_segments_summary.json \
  --market-summary state/soccer_edge_state/analysis/market_performance_summary.json \
  --true-clv-summary state/soccer_edge_state/analysis/true_clv_summary.json \
  --clv-summary state/soccer_edge_state/analysis/clv_summary.json \
  --output state/soccer_edge_state/analysis/settlement_promotion_review.json
```

## Output

```text
soccer_edge_state/analysis/settlement_promotion_review.json
```

The output contains:

- global policy thresholds
- input status
- recommendation counts
- one review record per market family
- blockers
- warnings
- stage stability
- league stability
- CLV review status

## Conservative gate policy

| Gate | Minimum |
|---|---:|
| Tier B review | 20 settled decisions |
| Tier A review | 50 settled decisions + non-negative true CLV directional sample |
| Tier S review | 100 settled decisions + true CLV model-change sample + stability |
| True CLV directional read | 50 rows |
| True CLV model-change read | 200 rows |

## Recommendation statuses

### `HOLD_OR_DEMOTE_REVIEW`

Used when any hard blocker exists:

- sample below Tier B minimum
- no decided win/loss sample
- non-positive ROI
- negative true CLV

### `TIER_B_CANDIDATE_MANUAL_REVIEW`

Used when:

- 20+ settled decisions
- positive ROI
- no hard blocker
- but CLV or stability is not strong enough for Tier A/S

### `TIER_A_CANDIDATE_MANUAL_REVIEW`

Used when:

- 50+ settled decisions
- positive ROI
- non-negative true CLV directional sample
- still requires manual review before implementation

### `TIER_S_CANDIDATE_MANUAL_REVIEW`

Used when:

- 100+ settled decisions
- positive ROI
- true CLV is large enough for model-change consideration
- no stage/league stability warnings
- still requires a separate implementation PR

## CLV handling

CLV is intentionally conservative:

- Proxy CLV can diagnose only.
- True CLV under 50 rows is not enough for promotion.
- True CLV 50-199 rows can support Tier A review but not Tier S.
- True CLV 200+ rows can support model-change/Tier S review.
- Negative true CLV blocks promotion.

## Workflow step to add after segmentation

```yaml
- name: Review settlement promotion gates
  run: python engine/mcp_gateway/review_settlement_promotion_gates.py --segments-summary state/soccer_edge_state/analysis/settlement_segments_summary.json --market-summary state/soccer_edge_state/analysis/market_performance_summary.json --true-clv-summary state/soccer_edge_state/analysis/true_clv_summary.json --clv-summary state/soccer_edge_state/analysis/clv_summary.json --output state/soccer_edge_state/analysis/settlement_promotion_review.json
```

The output is under `soccer_edge_state/analysis/`, so the existing persist step should pick it up.

## Safety

This script must not be used to auto-promote tiers. It creates review candidates only. Any production change to runtime classifiers, thresholds, model weights, or stake sizing must be made in a separate PR after reviewing the generated report.
