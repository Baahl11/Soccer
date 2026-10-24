# Soccer Edge — Frontend Maturity Wiring Audit

**Date:** 2026-10-09  
**Scope:** Soccer customer frontend (/app), subscriber contract v2 and persisted scientific validation reports.  
**Status:** IMPLEMENTATION ON REVIEW BRANCH — NOT VERIFIED IN PRODUCTION.

## Audited current state

- Active customer route `/app` calls `subscriber_frontend_v2.app_page` on the `soccer-edge-mcp-v1` runtime branch.
- `/app/api/v2/performance` is **BET-only commercial performance** plus a smaller `validation_evidence.rows` sample. That does not expose the canonical multi-gate maturity model.
- Existing `subscriber_maturity_v232.load_maturity_evidence()` already reads the scientific reports and strict CLV counts from the **persisted state branch**; however the primary app did not consume it.
- Owner/operator Maturation Control Tower, located in `maturity_snapshot_v4`, is a distinct product/observability surface and must not be mislabeled as customer BET permission.

## Implemented on the proposed change

1. `GET /app/api/v2/maturity`: authenticated PRO-only, no-store, read-only scientific resource. The source is persisted reports, not fabricated market data. It carries no production promotion permission.
2. New `Maturity` tab on the current `/app` frontend, including mobile navigation; refreshed when entering the tab if the prior fetch is older than five minutes, or manually via Refresh.
3. Per parent market: model/OOS sample with unit and optional target, mapped selection rows, observed priced rows, strict True CLV rows and target, unique CLV fixtures, research stage, next gate, full source-reported blockers, exact source artifact, report timestamp if provided.
4. Preserve child market segments independently: `1X2`, `BTTS`, `FT_TOTALS`, `HOME_TT`, `AWAY_TT`, `1H`, `FT_CORNERS`, `TEAM_CORNERS`, `2H`, `CARDS`, `SHOTS`, `SOT`, `GOALSCORER`, `ASSISTS`, `PLAYER_CARDS`, `GK_SAVES` (16 segments across nine aggregated families).
5. Missing fields stay **NOT VERIFIED**, never synthetic zero; OOS counts and market-price observations must not be conflated with CLV. Each card explicitly says **Production BET approval: NO**.

## What is still NOT complete (critical honesty boundary)

- The nine covered parent families do **not** equal every research market. Double Chance, Draw No Bet, Asian Handicap, Correct Score and the separately validated Red Cards market are not yet individually included as maturity families in this customer API. These have research artifacts, but have not been normalized into this multi-gate contract.
- Full operator Maturation Control Tower watchdogs, research freshness/coverage and fixture-level evidence/provenance are not yet exposed through this customer-safe contract.
- Child market rows expose independently recorded mapped/priced/True CLV counts, but their model/OOS samples remain group-level when the source does not provide a trustworthy per-child denominator.
- The new frontend cannot resolve missing real late market snapshots, unconfirmed lineup/availability, blocked promotion policy or underpowered OOS samples.
- Persisted report generation timestamps may be missing. Such reports are **NOT VERIFIED** for generation time; frontend loading time is not a source-freshness timestamp.
- No live browser/mobile E2E, authenticated route smoke test, deploy or production switch has been asserted merely by committing this change.

## Architectural policy

**Sport First, Market Second. Maturity is the evidence base for model governance, never a direct BET trigger.** Raw projections are created sport-first. Calibration, OOS, provider data, strict CLV, fixture-level independence and formal review determine whether a model is eligible to become a promoted version. Promotion requires explicit backend governance. Showing a market as research-READY must never silently change engine weights, thresholds, staking or canonical BET classifications.

## Release QA gate

- Compile the four modified Python modules.
- Run subscriber contract, frontend and maturation regression suites.
- Verify auth: anonymous and FREE cannot read PRO maturity endpoint; PRO can.
- Verify 9 parent rows and all 16 segments with a real persisted snapshot.
- Verify missing sources, stale reports, 0 CLV, absent generation date, and healthy sample with research blocker render honestly.
- Verify mobile nav and responsive cards.
- Verify `Performance` BET-only totals remain unchanged.
- Close remaining all-market source coverage/lineage and watchdog gaps before claiming **complete all-market maturity**.

This document is not permission to promote a model or release a paid frontend.
