# Soccer Edge Settlement Ledger — Step 3 Corners/Cards

Date: 2026-09-21
Branch: `fix/settlement-ledger-v1-20260921`
Base: `soccer-edge-mcp-v1`

## Purpose

Extend postgame settlement beyond canonical FT markets while avoiding false grading.

## What changed

### 1. Preserve tactical stats in the signal ledger

`mcp_gateway/build_signal_ledger.py` now preserves `postgame_tactical_stats` inside `result.tactical_stats` when the history event provides it separately from `event.result`.

This makes derivative settlement possible without re-reading raw history during postgame evaluation.

### 2. Corners settlement

`mcp_gateway/evaluate_postgame.py` can now grade corners markets when `result.tactical_stats.teams[]` includes corner counts.

Supported examples:

- `Corners Over/Under` + `Over 9.5` → total match corners
- `Corners Over/Under` + `Home Over 5.5` → home/team corners if team side can be inferred

If tactical stats are missing, the settlement status is `NO_TACTICAL_STATS`.

### 3. Cards settlement

Only explicit yellow/red-card markets are graded:

- `Yellow Cards Over/Under`
- `Red Cards Over/Under`
- team-specific versions when side can be inferred

Generic `Cards Over/Under` remains blocked as `UNSUPPORTED_CARD_RULES`, because books can score cards differently, for example yellow = 1 and red = 2, or yellow-card-only, or player/team-specific card markets.

### 4. Still blocked

The following remain unpromoted until explicitly mapped:

- generic cards without scoring rule
- player props
- ambiguous card-point markets
- corners/cards without tactical stats

## Safety

- No manual ticks.
- No provider requests.
- No merge.
- No Render deploy.
- No automatic promotion to BET/Tier A/S.

## Expected outputs after merge + scheduled history analysis

- `soccer_edge_state/analysis/bet_settlement_ledger.jsonl`
- `soccer_edge_state/analysis/market_performance_summary.json`
- updated `signal_ledger_summary.json` with `rows_with_tactical_stats`
- updated `postgame_summary.json` schema `1.2.0`

## Next step

Step 4 should segment settled performance by:

- market family
- classification BET/LEAN
- league
- stage window
- odds band
- data tier

This is required before any Tier B/A/S promotion.
