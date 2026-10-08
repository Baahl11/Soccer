# Soccer Edge Golden Master Freeze v1.0

Status: **FROZEN FOR IMPLEMENTATION**

## Frozen visual reference

- Route: `/design-lab/match-center`
- Source: `mcp_gateway/ui_golden_master_v1.py`
- Frozen after the final hero, typography, spacing, matrix, probability, xG, Edge Gap and O/U polish passes.
- The design-lab values are illustrative sample data and are **not** a live prediction.

## Architecture target

The approved Match Center is now being migrated to:

- React 18
- TypeScript
- Vite
- existing Python API as the data source of truth

The stable `/app` surface is not replaced by this migration.

## Product firewall

The frontend migration must not:

- change model weights, thresholds, gates or betting logic;
- fabricate missing football data;
- turn market data into the sporting projection;
- promote sample values as live;
- hide `NOT VERIFIED`, `INSUFFICIENT_DATA` or `MARKET_DATA_ONLY` states.

## Implementation gates

1. Freeze visual reference.
2. Extract the Match Center into typed React components.
3. Validate Vite/TypeScript build.
4. Map `/app/api/v2/match/{fixture_id}` into a frontend view model.
5. Implement explicit missing-data states.
6. Run visual parity QA against the Golden Master.
7. Promote only after parity and data-contract QA.

## Current state

`web_v3/` now contains the formal Vite + React + TypeScript project and a typed component extraction using explicit sample data. Live API binding is intentionally the next gate.
