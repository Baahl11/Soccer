# Soccer Edge — React V3 commercial frontend completion, provider API paused

**Date:** 2026-10-13
**Baseline:** `fix/fe-maturity-evidence-20261009` (open PR #68)
**Implementation branch:** `feat/react-v3-offline-commercial-readonly-20261013`
**State:** OFFLINE DEV / NOT DEPLOYED / NO PROVIDER REQUESTS
**Authoritative plans:** `18_SOCCER_MATCH_CENTER_UI_BLUEPRINT_v1.0.md`, `19_SOCCER_EDGE_FRONTEND_REBUILD_PLAN_v1.0.md`, `20_SOCCER_EDGE_PIXEL_FIRST_UI_WORKFLOW_v1.0.md`, `21_SOCCER_EDGE_GOLDEN_MASTER_FREEZE_v1.0.md`, `SOCCER_EDGE_COMMERCIAL_ROADMAP.md`.
**Release debts:** Issues #64, #66, #67, #69, #70 and commercial V226/227.

## Locked policies

1. SPORT FIRST / MARKET SECOND. No creation of projections, probabilities, prices, result scores, lineups or injuries in frontend.
2. A persisted BET is not automatically a current, actionable bet. Never display stale prices as active tips. No manual "picks" or editing original predictions.
3. BET-only canonical settlement drives customer track record; research LEAN, OOS and market maturation must never inflate BET ROI/win rate.
4. 0 is a reported zero; null/absent is NOT VERIFIED. Never reconstruct fixture-level historical selections from aggregate performance.
5. Authenticate server-side. All premium resources fail closed on 401, 403, 503. Refreshing sessions must clear previously authorized content.
6. API-Football subscription unavailable: do not re-enable main scheduler or add any paid provider call, including CI or previews.
7. `main` is currently in cost-saving mode (28 cron schedules removed in `bad7f431`). Keep this branch isolated; do not merge or deploy while the release gate is red.

## Existing surfaces: verified from React code and backend contracts

| Surface | Reality | Next acceptance |
| --- | --- | --- |
| Today / Matches | React V3 navigable, real persisted fixture registry when API bridge available | Stable date/league/search filters, correct empty/stale/cancelled states, tablet/mobile parity |
| Match Center / Matchups | React V3 sport-first panels for probability, goal rates, score matrix, team form, observed stats/events/players and market/model evidence | Availability as a first-class panel; verify period-specific notes, improve safe states, snapshot provenance and chart parity |
| Picks / LEAN | Backend already exposes PRO/OWNER gated `/app/api/v2/picks` + `/leans`. **This branch wires separate read-only React pages.** | Real entitlement E2E, evidence completeness, stale readout, exact bookmaker+timestamp, original immutable decision |
| Historical Results | Backend `/performance` contains BET settlement *aggregate*, not a fixture-level list; this branch shows that summary with explicit NOT VERIFIED per-fixture history | New authoritative paginated read-only results/settlement endpoints; official FT/AET/PEN score, kickoff/date/timezone, fixture ID, original selection/odds/model version and final grade; no joining after kickoff to fabricate pregame decisions |
| Performance | Backend PRO/OWNER `/app/api/v2/performance`, canonical settled BET metrics + separate research LEAN | This branch adds React BET-only summary, sample warning, True CLV and settlement coverage; no OOS-as-real-ROI |
| Edge Feed | React V3 still disabled. Legacy `/app` has an earlier implementation | Feed from persisted, *server-authorized* classifications. Separate BET/LEAN/WATCH/PASS and price/source/freshness, never synthesize edge |
| My Edge | React V3 disabled. Backend authenticated `/app/api/v2/my-edge` has favorites/alerts entitlement rules | UI and role-proof; avoid cross-account cache/data leaks; no wagering functionality without approved sources |
| Market Maturity | Existing PR #68 21-market research surface with OOS/CLV provenance | Scientific generated-at, fixture lineage, exact close and real role E2E; DO NOT merge as production-ready from green UI tests alone |
| Control Tower / Research Lab | Existing operator interfaces are separate from new React subscriber UX | Integrate an operator-only route if warranted; no admin secrets in customer bundle |
| Pricing / Billing | Existing Stripe/Supabase infrastructure is wired but V226 activation held | Verify signed webhook, real checkout/portal, cancel/past_due → entitlement downgrade, regional price verification before charging |
| Accessibility / QA | Screenshot and mobile 390/desktop 1440 smoke existed for Match Center/Maturity | Expand to 360/390/430/768/1440, keyboard focus, CTA state and high-data vs no-data, independent date/time zones |

## Work order, no paid API required

### P0 — Navigation and controlled customer surface
- Complete React V3 read-only Picks, Leans, Performance and historical **settlement summary** screens from existing V2 endpoints. Unit/browser fixtures are explicitly mocked in tests and never exposed as live results.
- Reconcile E2E 401/403/503 and anonymous→PRO→revoked transitions, no stale premium rows.
- Preserve `/app` old commercial routes while React V3 is under development.

### P1 — Historical results endpoint and actual fixture lists
- Build server-side `GET /app/api/v2/results` as bounded/date-paginated, read-only official result rows from already-persisted `soccer_results` with `soccer_fixtures`. Each row requires original source/status/observed timestamp and stable fixture ID.
- Build separate PRO-gated `GET /app/api/v2/pick-history` using canonical historical BET ledger joined only by original immutable selection identifiers, with original timestamp, price, bookmaker, outcome and grading. Do **not** infer/retro-label past picks from today’s prediction. No reconstruction from report aggregates.
- UI filters: date interval, league, verified status, market family, source; minimum sample warnings.
- Mock and DB-fixture contract tests can be written offline; **actual database/role E2E must wait for an approved isolated test**.

### P2 — Matchup polish + Dashboard / Edge Feed
- Finish independent Availability visual, sport-first cards, goal chart and evidence gaps; ensure no xG label where only a Poisson scoring λ exists.
- Finish actual dashboard sport coverage, fixture status, missing-data families, bookmarks; use V3 design tokens, not a parallel V2 rebuild.
- Bring Edge Feed online only from server-gated persisted classifications. Add separate WATCH/PASS display; no synthetic opportunities during provider outage.

### P3 — Commercial and operating quality
- My Edge, Performance cohort drilldowns, operator Research Lab, 4 known FE2 assertions, tests for all market families and full responsive accessibility.
- Pixel-first screenshot compare vs approved Golden Master; no mock numbers in customer route.
- FREE/PRO/OWNER live review and compiled `web_v3/dist` release artifact parity, frontend beta review, Stripe V226 verification.

## Exit criteria
UI green != scientific green. Frontend can be developed and merged into a reviewed release branch without reactivating the provider. **Production launch only after:** maturity/CLV and independent prospective gates green; server stable, authenticated roles, no stale prices, final scores and original picks independently verified, billing test completed, budget guardrails in place.

## Current isolated changes in this branch
- `web_v3/src/CommercialPages.tsx`: read-only consumer for `/picks`, `/leans`, `/performance`; historical results uses ONLY legitimate performance aggregate and explicitly states missing fixture-level API.
- `web_v3/src/App.tsx`: desktop/mobile links to new surfaces, keeping Edge Feed and My Edge disabled until proven.
- `web_v3/src/app.css`: legible laptop layout and horizontally scrollable mobile navigation.
- `web_v3/scripts/e2e-commercial-smoke.mjs`: fixture-only regression for BET/LEAN isolation, small sample, canonical 0/null, stale/non-actionable, 401/403/503 and mobile overflow.

Build/browser CI remains a separate gate. No release, merge, or provider calls authorized by this work.
