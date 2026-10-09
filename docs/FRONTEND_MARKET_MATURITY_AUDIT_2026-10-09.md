# FE Market Maturity — Frontend-to-Research Audit (2026-10-09)

## Verified route topology
- `/app` and `/app-v2`: current customer-facing V2 app; contains BET-only Performance and a limited OOS validation table, but previously lacked a market maturation route/page.
- `/app-preview/maturity`: authenticated, PRO/OWNER-gated V232 family-level research endpoint.
- `/app-v3-react`: React/Vite preview. Performance nav remains disabled and previously had no maturity view.
- Operator Control Tower: uses `maturity_snapshot_v4`, persisted-state watchdogs, evidence freshness and report baseline. Do **not** conflate this operator plane with customer BET performance.

## Delivered in this change
1. `GET /app/api/v2/maturity` is a read-only protected alias of the existing maturity endpoint (same entitlement check; no public raw data leak).
2. `subscriber_maturity_v232` retains its nine family aggregates and adds 21 explicit market inventory rows:
   1X2, BTTS, FT Totals, Home/Away Team Totals, 1H, 2H, FT/Team Corners, Yellow/Red Cards, six player props (Shots, Shots on Target, Goalscorer, Assists, Player Cards, Goalkeeper Saves), Double Chance, Draw No Bet, Asian Handicap and Correct Score.
3. Every market row displays model/OOS evidence when independently sourced, mapped count, observed priced count, strict True CLV, parent research phase, next blocking gate and source file. Unknown child-model samples and unknown strict closes are **NOT VERIFIED**, never inherited from a parent report.
4. Customer V2 navigation and React V3 preview navigation render market maturity separately from realized BET performance.
5. New regression suite requires no synthetic CLV, no parent-to-child OOS sample cloning, no implicit production promotion, route/auth coverage, Python compilation, render-time JavaScript syntax and React build.

## Still *not* solved by a frontend patch
- Raw evidence freshness for reports and fixture-level artifact lineage must be verified live after a real workflow run. Loading a JSON report does not prove it is fresh.
- True CLV requires a later **real same-book strict pre-kickoff close**. No close must remain zero only if the canonical report confirms zero; absent report must remain NOT VERIFIED.
- Full calibration metrics (Brier, log loss, ECE, reliability buckets), actual model version history, per-league/line/venue OOS slices and lookahead leakage checks need dedicated back-end data contracts for each family, not generic UI assumptions.
- Derived markets (Double Chance, DNB, Asian Handicap, Correct Score) are listed transparently but currently do **not** have family-level research reports wired into this V232 source; their independent model evidence is NOT VERIFIED.
- Cards and player-prop per-market report fields may be partial or absent; no aggregation is interpreted as an independent 0.
- No market in this change receives BET eligibility, calibration weight changes, stake changes or automatic production promotion.
- Paid entitlements, exact prices, live deployment status and rendered mobile UX require post-deploy browser E2E confirmation.

## Governing decision rule
```
LIVE SPORT EVIDENCE
  -> PROSPECTIVE FIXTURE SNAPSHOT
  -> OUT-OF-SAMPLE MATURATION (independent sample, temporal provenance)
  -> MARKET PRICING + STRICT TRUE CLV
  -> CALIBRATION / BASELINE / BLOCKER REVIEW
  -> EXPLICIT MANUAL MODEL ELIGIBILITY (never triggered by frontend)
  -> CANONICAL BET / LEAN / WATCH / PASS
  -> CUSTOMER UI
```

Maturation is the **evidence governance system**, not a replacement for Sport First / Market Second. The subscriber UI is a read-only display of verified state. The current model gate and historical ledger remain authoritative and unchanged.

## Release acceptance
1. Validate Python + rendered JS + React TypeScript/Vite through `FE Market Maturity Contract` CI on the pull request.
2. Merge only after passing CI and verifying no route conflicts.
3. Confirm the source branch and state variables on the actual backend deployment.
4. Smoke-test `/app/api/v2/maturity`: anonymous 401, FREE 403, PRO/OWNER 200, missing state transparent.
5. Browser-validate `/app` and `/app-v3-react` on desktop and mobile, including restricted/partial/missing reports.
6. Ensure the frontend build for V3 is refreshed and deployed. The source change alone does **not** update the committed `web_v3/dist`.
7. Reconcile observed market rows against canonical artifacts and then expand per-market calibration/freshness. Do not assert scientific maturity from UI completeness.
